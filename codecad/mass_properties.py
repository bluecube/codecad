import collections
import functools
import operator
import math
import time
import logging

import numpy
import pyopencl
from pyopencl import cltypes

from . import util
from . import cl_util
from . import subdivision
from .cl_util import opencl_manager
from . import nodes

TREE_SIZE = 2  # Cube root of tree children count (2 => octree)
WORK_SIZE = 2**15
MAX_SPLITS = TREE_SIZE**3 * WORK_SIZE

assert TREE_SIZE**3 < 32, "Set of node children must be representable by uint bitmask"
assert MAX_SPLITS <= 2**23, "Maximum number of split nodes per run must be exactly representable as float"

c = opencl_manager.add_compile_unit()
c.append_define("TREE_SIZE", TREE_SIZE)
c.append_file("mass_properties.cl")

logger = logging.getLogger(__name__)


class MassProperties(collections.namedtuple("MassProperties", "volume centroid inertia_tensor")):
    """ Contains volume of a body, position of its centroid and its inertia tensor.
    The inertia tensor is referenced to the centroid, not origin! """
    __slots__ = ()


def mass_properties(shape, allowedError=1e-3):
    """ Calculate volume, centroid and inertia tensor of the shape.
    Iteratively subdivides the shape until
    abs(actual_volume - computed_volume) < allowedError """
    # Inertia tensor info:
    # http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node64.html
    #
    # Some useful integrals:
    # integral(x, a, a + s, integral(y, b, b + s, integral(z, c, c + s, x))) = s**3 * (a + s / 2)
    # integral(x, a, a + s, integral(y, b, b + s, integral(z, c, c + s, x * y))) = s**3 * (a + s / 2) * (b + s / 2)

    #TODO: Support relative errors

    assert shape.dimension() == 3, "2D objects are not supported yet"

    box = shape.bounding_box()
    box_size = box.size()
    initial_step_size = box_size.max() / TREE_SIZE
    # TODO: Figure out correct max_depth
    # final_step_size = ??? shape.get_epsilon() / 2 ???
    # max_depth = 1 + math.log(initial_step_size / final_step_size, TREE_SIZE)
    max_depth = 10
    max_locations = TREE_SIZE**3 * WORK_SIZE * max_depth
    print("max_locations", max_locations)

    program_buffer = nodes.make_program_buffer(shape)
    locations = cl_util.Buffer(cltypes.float4, max_locations, pyopencl.mem_flags.READ_WRITE)
    allowed_errors = cl_util.Buffer(cltypes.float, max_locations, pyopencl.mem_flags.READ_WRITE)
    temp_locations = cl_util.Buffer(cltypes.float4, WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    next_allowed_errors = cl_util.Buffer(cltypes.float, WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    integral1 = cl_util.Buffer(cltypes.float4, WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    integral2 = cl_util.Buffer(cltypes.float4, WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    integral3 = cl_util.Buffer(cltypes.float2, WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    split_counts = cl_util.Buffer(cltypes.uint, WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    split_masks = cl_util.Buffer(cltypes.uint, WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    output_buffer = cl_util.Buffer(cltypes.float, 11, pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.ALLOC_HOST_PTR)
    assert_buffer = cl_util.AssertBuffer()

    # Initialize the first location
    with locations.map(pyopencl.map_flags.WRITE, shape=1) as mapped:
        mapped[0]["x"] = box.a.x
        mapped[0]["y"] = box.a.y
        mapped[0]["z"] = box.a.z
        mapped[0]["w"] = initial_step_size
    with allowed_errors.map(pyopencl.map_flags.WRITE, shape=1) as mapped:
        mapped[0] = allowedError
    locations_to_process = 1

    integral_one = util.KahanSummation()
    integral_x = util.KahanSummation()
    integral_y = util.KahanSummation()
    integral_z = util.KahanSummation()
    integral_xx = util.KahanSummation()
    integral_yy = util.KahanSummation()
    integral_zz = util.KahanSummation()
    integral_xy = util.KahanSummation()
    integral_xz = util.KahanSummation()
    integral_yz = util.KahanSummation()

    time_computing = 0
    iteration_count = 0
    locations_processed = 0
    start_time = time.time()

    while locations_to_process > 0:
        work_size = min(WORK_SIZE, locations_to_process)
        start_offset = locations_to_process - work_size  # TODO: Divisible by 16!
        ev1 = opencl_manager.k.mass_properties_stage1([work_size], None,
                                                      program_buffer,
                                                      cltypes.uint(start_offset),
                                                      locations, allowed_errors,
                                                      temp_locations, next_allowed_errors,
                                                      integral1, integral2, integral3,
                                                      split_counts, split_masks,
                                                      assert_buffer)
        ev2 = opencl_manager.k.mass_properties_stage2([1], None,
                                                      integral1, integral2, integral3,
                                                      split_counts,
                                                      assert_buffer,
                                                      wait_for=[ev1])
        ev3 = opencl_manager.k.mass_properties_stage3([work_size], None,
                                                      cltypes.uint(start_offset),
                                                      locations, allowed_errors,
                                                      temp_locations, next_allowed_errors,
                                                      integral1, integral2, integral3,
                                                      split_counts, split_masks,
                                                      output_buffer,
                                                      assert_buffer,
                                                      wait_for=[ev2])
        with output_buffer.map(pyopencl.map_flags.READ, wait_for=[ev3]) as mapped:
            integral_one += mapped[0]
            integral_x += mapped[1]
            integral_y += mapped[2]
            integral_z += mapped[3]
            integral_xx += mapped[4]
            integral_yy += mapped[5]
            integral_zz += mapped[6]
            integral_yz += mapped[7]
            integral_xz += mapped[8]
            integral_xy += mapped[9]
            locations_to_process += int(mapped[10])
        if locations_to_process > max_locations:
            raise Exception("Maximum location buffer size exceeded. "
                            "This is likely a bug in CodeCad. "
                            "As a workaround try specifying lower precision.")

        time_computing_this_iteration = (ev3.profile.end - ev1.profile.start) * 1e-9

        time_computing += time_computing_this_iteration
        iteration_count += 1
        locations_processed += work_size

        logger.debug("Iteration finished: time_computing: %f s, locations to process: %i",
                     time_computing_this_iteration, locations_to_process)

    time_spent = time.time() - start_time
    logger.info("Mass properties finished: "
                "time elapsed: %f s, opencl compute time: %f s, efficiency: %f, "
                "iteration count: %f (%f iterations / s), "
                "locations processed: %f (%f locations / iteration, %f locations / s)",
                time_spent, time_computing, time_computing / time_spent,
                iteration_count, iteration_count / time_spent,
                locations_processed, locations_processed / iteration_count, locations_processed / time_spent)

    # Unwrap the integral values from the KahanSummation objects
    integral_one = integral_one.result
    integral_x = integral_x.result
    integral_y = integral_y.result
    integral_z = integral_z.result
    integral_xx = integral_xx.result
    integral_yy = integral_yy.result
    integral_zz = integral_zz.result
    integral_xy = integral_xy.result
    integral_xz = integral_xz.result
    integral_yz = integral_yz.result

    volume = integral_one
    if volume == 0:
        return MassProperties(0, util.Vector.splat(0), numpy.zeros((3, 3)))
    centroid = util.Vector(integral_x, integral_y, integral_z) / integral_one

    # Prepare the inertia tensor based on the integrals, but make it referenced
    # to object centroid instead of origin
    shifted_integral_xx = integral_xx - 2 * centroid.x * integral_x + centroid.x * centroid.x * integral_one
    shifted_integral_yy = integral_yy - 2 * centroid.y * integral_y + centroid.y * centroid.y * integral_one
    shifted_integral_zz = integral_zz - 2 * centroid.z * integral_z + centroid.z * centroid.z * integral_one
    shifted_integral_xy = integral_xy - centroid.x * integral_y - centroid.y * integral_x + centroid.x * centroid.y * integral_one
    shifted_integral_xz = integral_xz - centroid.x * integral_z - centroid.z * integral_x + centroid.x * centroid.z * integral_one
    shifted_integral_yz = integral_yz - centroid.y * integral_z - centroid.z * integral_y + centroid.y * centroid.z * integral_one
    I_xx = shifted_integral_yy + shifted_integral_zz
    I_yy = shifted_integral_xx + shifted_integral_zz
    I_zz = shifted_integral_xx + shifted_integral_yy
    I_xy = -shifted_integral_xy
    I_xz = -shifted_integral_xz
    I_yz = -shifted_integral_yz
    inertia_tensor = numpy.array([[I_xx, I_xy, I_xz],
                                  [I_xy, I_yy, I_yz],
                                  [I_xz, I_yz, I_zz]])

    return MassProperties(volume, centroid, inertia_tensor)
