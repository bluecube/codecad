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

TREE_SIZE = 3  # Cube root of tree children count (2 => octree)
MAX_WORK_SIZE = 2**16
TREE_CHILD_COUNT = TREE_SIZE**3

assert TREE_CHILD_COUNT < 32, "Set of node children must be representable by uint bitmask"
assert TREE_CHILD_COUNT * MAX_WORK_SIZE <= 2**23, "Maximum number of split nodes per run must be exactly representable as float"

c = opencl_manager.add_compile_unit()
c.append_define("TREE_SIZE", TREE_SIZE)
c.append_file("mass_properties.cl")

logger = logging.getLogger(__name__)


class MassProperties(collections.namedtuple("MassProperties", "volume centroid inertia_tensor volume_error")):
    """ Contains volume of a body, position of its centroid and its inertia tensor.
    The inertia tensor is referenced to the centroid, not origin! """
    __slots__ = ()


def mass_properties(shape, allowed_error=1e-3):
    """ Calculate volume, centroid and inertia tensor of the shape.
    Iteratively subdivides the shape until
    abs(actual_volume - computed_volume) < allowed_error """
    # Inertia tensor info:
    # http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node64.html

    #TODO: Support relative errors

    assert shape.dimension() == 3, "2D objects are not supported yet"

    box = shape.bounding_box()
    box_size = box.size()
    initial_step_size = box_size.max() / TREE_SIZE
    volume_to_process = box_size.max()**3
    # TODO: Figure out correct max_depth
    # final_step_size = ??? shape.get_epsilon() / 2 ???
    # max_depth = 1 + math.log(initial_step_size / final_step_size, TREE_SIZE)
    max_depth = 20
    max_locations1 = MAX_WORK_SIZE + MAX_WORK_SIZE * (TREE_CHILD_COUNT - 1) * max_depth
    max_locations2 = max_locations1 + MAX_WORK_SIZE * (TREE_CHILD_COUNT - 1)
    logger.debug("max_location1: %i, max_locations2: %i", max_locations1, max_locations2)

    program_buffer = nodes.make_program_buffer(shape)
    locations = cl_util.Buffer(cltypes.float4, max_locations2, pyopencl.mem_flags.READ_WRITE)
    temp_locations = cl_util.Buffer(cltypes.float4, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    integral1 = cl_util.Buffer(cltypes.float4, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    integral2 = cl_util.Buffer(cltypes.float4, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    integral3 = cl_util.Buffer(cltypes.float2, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    split_counts = cl_util.Buffer(cltypes.uint, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    split_masks = cl_util.Buffer(cltypes.uint, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
    output_buffer = cl_util.Buffer(cltypes.float, 13, pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.ALLOC_HOST_PTR)
    assert_buffer = cl_util.AssertBuffer()

    # Initialize the first location
    prepare_next_ev = locations.enqueue_write(box.a.as_float4(initial_step_size))
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
    integral_all = util.KahanSummation()
    total_error = util.KahanSummation()

    time_computing = util.KahanSummation()
    iteration_count = 0
    locations_processed = 0
    start_time = time.time()

    prev_prepare_next_ev = None

    while locations_to_process > 0:
        work_size = min(MAX_WORK_SIZE, locations_to_process)

        start_offset = locations_to_process - work_size  # TODO: Divisible by 16!

        if start_offset + work_size * TREE_CHILD_COUNT >= max_locations1:
            work_size = (max_locations2 - locations_to_process) // (TREE_CHILD_COUNT - 1)
            work_size = min(work_size, MAX_WORK_SIZE, locations_to_process)
            work_size = 0
            if work_size == 0:
                raise Exception("Maximum location buffer size exceeded. "
                                "This is likely a bug in CodeCad. "
                                "As a workaround try specifying lower precision.")

            logger.warning("Limiting work size to %i to avoid exceeding location buffer size. "
                           "This is likely a bug in CodeCad, but should only affect performance.",
                           work_size)
            start_offset = locations_to_process - work_size  # TODO: Divisible by 16!

        assert work_size > 0
        assert work_size <= MAX_WORK_SIZE
        assert start_offset >= 0
        assert start_offset + work_size == locations_to_process
        assert start_offset + work_size * TREE_CHILD_COUNT <= max_locations2

        current_allowed_error = 0.5 * (allowed_error - total_error.result) / locations_to_process

        evaluate_ev = opencl_manager.k.mass_properties_evaluate([work_size], None,
                                                                program_buffer,
                                                                cltypes.uint(start_offset),
                                                                cltypes.float(current_allowed_error),
                                                                locations, temp_locations,
                                                                integral1, integral2, integral3,
                                                                split_counts, split_masks,
                                                                assert_buffer,
                                                                wait_for=[prepare_next_ev])
        sum_ev = opencl_manager.k.mass_properties_sum([work_size], None,
                                                      integral1, integral2, integral3,
                                                      split_counts,
                                                      assert_buffer,
                                                      wait_for=[evaluate_ev])
        output_ev = opencl_manager.k.mass_properties_output([1], None,
                                                            cltypes.uint(work_size),
                                                            integral1, integral2, integral3,
                                                            split_counts,
                                                            output_buffer,
                                                            wait_for=[sum_ev])
        prepare_next_ev = opencl_manager.k.mass_properties_prepare_next([work_size], None,
                                                                        cltypes.uint(start_offset),
                                                                        locations, temp_locations,
                                                                        split_counts, split_masks,
                                                                        assert_buffer,
                                                                        wait_for=[sum_ev])
        with output_buffer.map(pyopencl.map_flags.READ, wait_for=[output_ev]) as mapped:
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
            integral_all += mapped[10]
            total_error += mapped[11]
            splits = int(mapped[12])
        #assert_buffer.check()

        assert splits <= work_size * TREE_CHILD_COUNT
        locations_to_process += splits - work_size

        #with locations.map(pyopencl.map_flags.READ, wait_for=[prepare_next_ev]) as mapped:
        #    logger.debug("locations after: %s", str(mapped[:locations_to_process]))
        #with allowed_errors.map(pyopencl.map_flags.READ, wait_for=[prepare_next_ev]) as mapped:
        #    logger.debug("allowed_errors after: %s", str(mapped[:locations_to_process]))

        # Computing time spent in OpenCL is a bit tricky here.
        # Because we try to give stage 4 as much time as possible, it might be
        # still running at this point, so we need to measure its time times one step behind.
        # The other stages are calculated immediately
        time_computing += cl_util.event_time_spent(evaluate_ev) + cl_util.event_time_spent(sum_ev) + cl_util.event_time_spent(output_ev)
        if prev_prepare_next_ev is not None:
            time_computing += cl_util.event_time_spent(prev_prepare_next_ev)
        iteration_count += 1
        locations_processed += work_size

        prev_prepare_next_ev = prepare_next_ev

        progress = integral_all.result / volume_to_process

        time_spent = time.time() - start_time
        logger.debug("Iteration finished: %.4f%%, total error used: %f, split count: %i, locations to process: %i, "
                     "time elapsed: %f s, opencl compute time: %f s, efficiency: %f, "
                     "iteration count: %i (%f iterations/s), "
                     "locations processed: %i (%f locations/iteration, %f locations/s)",
                     100 * progress, total_error.result, splits, locations_to_process,
                     time_spent, time_computing.result, time_computing.result / time_spent,
                     iteration_count, iteration_count / time_spent,
                     locations_processed, locations_processed / iteration_count, locations_processed / time_spent)

        #break

    assert_buffer.check()

    prev_prepare_next_ev.wait() # Wait for the last event to finalize timing stats
    time_computing += cl_util.event_time_spent(prev_prepare_next_ev)

    time_spent = time.time() - start_time
    logger.info("Mass properties finished: "
                "time elapsed: %f s, opencl compute time: %f s, efficiency: %f, "
                "iteration count: %i (%f iterations/s), "
                "locations processed: %i (%f locations/iteration, %f locations/s)",
                time_spent, time_computing.result, time_computing.result / time_spent,
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

    return MassProperties(volume, centroid, inertia_tensor, total_error.result)
