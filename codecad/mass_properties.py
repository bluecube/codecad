import collections
import functools
import operator
import math
import time

import numpy
import pyopencl
import pyopencl.cltypes

from . import util
from . import cl_util
from . import subdivision
from .cl_util import opencl_manager
from . import nodes

INNER_LOOP_SIDE = 3  # Side of sample cube that gets calculated within private memory
GRID_SIZE = INNER_LOOP_SIDE * 32  # Max allowed grid size
LOCAL_GROUP_SIDE = 4 # LOCAL_GROUP_SIDE**3 work items in work group


def _max_local_sum(m, n):
    """ calculate maximum possible value of index sum, where index < n
    summed over a region m**3 large.
    Expressions here are derived in the notebook in docs. """
    return m**3 * (2 * m**2 - 6 * m*n + 3 * m + 6 * n**2 - 6 * n + 1) // 6


MAX_WEIGHT_MULTIPLIER = int((2**32 - 1) // _max_local_sum(INNER_LOOP_SIDE * LOCAL_GROUP_SIDE, GRID_SIZE))

assert INNER_LOOP_SIDE % 2 == 1, "Inner loop side has to be odd to allow skipping space based on midpoint value"
assert GRID_SIZE >= 2, "Grid side size needs to be >= 2"
assert GRID_SIZE % INNER_LOOP_SIDE == 0, "Grid size must be divisible by {}".format(INNER_LOOP_SIDE)
assert GRID_SIZE < 2**8, "Grid coordinates must fit uint8"
assert MAX_WEIGHT_MULTIPLIER > 127, "Index sum multiplier needs enough resolution"
assert MAX_WEIGHT_MULTIPLIER * _max_local_sum(INNER_LOOP_SIDE * LOCAL_GROUP_SIDE, GRID_SIZE) < 2**32, \
    "Local index sum must fit uint32"
assert MAX_WEIGHT_MULTIPLIER * _max_local_sum(GRID_SIZE, GRID_SIZE) < 2**64, \
    "Global index sum must fit in uint64"

c = opencl_manager.add_compile_unit()
c.append_define("INNER_LOOP_SIDE", INNER_LOOP_SIDE)
c.append_define("MAX_WEIGHT_MULTIPLIER", MAX_WEIGHT_MULTIPLIER)
c.append_define("LOCAL_GROUP_SIDE", LOCAL_GROUP_SIDE)
c.append_file("mass_properties.cl")


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

    program_buffer = nodes.make_program_buffer(shape)

    box = shape.bounding_box()
    box_size = box.size()
    initial_step = box_size.max() / GRID_SIZE
    initial_grid = [min(int(util.round_up_to(s / initial_step,
                                             LOCAL_GROUP_SIDE * INNER_LOOP_SIDE)),
                        GRID_SIZE)
                    for s in box_size]
    initial_step = min(bs / gs for bs, gs in zip(box_size, initial_grid))
    assert all(gs * initial_step >= bs for bs, gs in zip(box_size, initial_grid))
    initial_allowed_error = allowedError / initial_step**3

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
    kernel_invocations = 0

    local_work_size = [LOCAL_GROUP_SIDE] * 3

    def job(job_id):
        nonlocal integral_one, integral_x, integral_y, integral_z, \
                 integral_xx, integral_yy, integral_zz, \
                 integral_xy, integral_xz, integral_yz
        nonlocal time_computing, kernel_invocations

        current_corner, current_step, current_grid, current_allowed_error = job_id
        assert all(x <= GRID_SIZE for x in current_grid)

        counters = cl_util.Buffer(numpy.uint32, 22, pyopencl.mem_flags.READ_WRITE)
        values = cl_util.Buffer(numpy.float32, [GRID_SIZE]*3,
                                pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.ALLOC_HOST_PTR)
        split_list = cl_util.Buffer(pyopencl.cltypes.uchar3, GRID_SIZE**3,
                                    pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.ALLOC_HOST_PTR)
        fill_ev = counters.enqueue_zero_fill_compatible()

        shifted_corner = current_corner + util.Vector.splat(current_step / 2)

        assert all(x % INNER_LOOP_SIDE == 0 for x in current_grid)
        global_work_size = [x // INNER_LOOP_SIDE for x in current_grid]

        ev1 = opencl_manager.k.mass_properties_stage1(global_work_size, local_work_size,
                                                      program_buffer,
                                                      shifted_corner.as_float4(), numpy.float32(current_step),
                                                      values, counters,
                                                      wait_for=[fill_ev])
        ev2 = opencl_manager.k.mass_properties_stage2(global_work_size, local_work_size,
                                                      numpy.float32(current_allowed_error),
                                                      values, split_list, counters,
                                                      wait_for=[ev1])
        yield ev2

        tt = (ev2.profile.end - ev1.profile.start) / 1e9

        time_computing += tt
        kernel_invocations += 1

        s = current_step
        s2 = s * s
        s3 = s * s2
        b = shifted_corner

        # Order is defined in mass_properties.cl
        with counters.map(pyopencl.map_flags.READ) as array:
            intersecting_count = array[0]
            split_count = array[1]
            index_sums = [(a + 2**32 * b) / MAX_WEIGHT_MULTIPLIER
                          for a, b in zip(array[2:len(array):2], array[3:len(array):2])]

        # print("Spent {} s in kernels, {} intersections, splitting {}, current allowed error {}, {}ki/s" \
        #       .format(tt, intersecting_count, split_count,  current_allowed_error, kernel_invocations / time_computing))

        sum_xx, sum_xy, sum_xz, sum_x, sum_yy, sum_yz, \
            sum_y, sum_zz, sum_z, sum_one = index_sums

        tmp_one = sum_one
        tmp_x = s * sum_x
        tmp_y = s * sum_y
        tmp_z = s * sum_z
        tmp_xx = s2 * sum_xx
        tmp_yy = s2 * sum_yy
        tmp_zz = s2 * sum_zz
        tmp_xy = s2 * sum_xy
        tmp_xz = s2 * sum_xz
        tmp_yz = s2 * sum_yz

        integral_one += s3 * sum_one
        integral_x  += s3 * (tmp_one * b.x + tmp_x)
        integral_y  += s3 * (tmp_one * b.y + tmp_y)
        integral_z  += s3 * (tmp_one * b.z + tmp_z)
        integral_xx += s3 * (tmp_one * (b.x * b.x + s2 / 12) + 2 * b.x * tmp_x + tmp_xx)
        integral_yy += s3 * (tmp_one * (b.y * b.y + s2 / 12) + 2 * b.y * tmp_y + tmp_yy)
        integral_zz += s3 * (tmp_one * (b.z * b.z + s2 / 12) + 2 * b.z * tmp_z + tmp_zz)
        integral_xy += s3 * (tmp_one * b.x * b.y + b.x * tmp_y + b.y * tmp_x + tmp_xy)
        integral_xz += s3 * (tmp_one * b.x * b.z + b.x * tmp_z + b.z * tmp_x + tmp_xz)
        integral_yz += s3 * (tmp_one * b.y * b.z + b.y * tmp_z + b.z * tmp_y + tmp_yz)

        if split_count == 0:
            return []

        next_step = current_step / GRID_SIZE
        next_grid = [GRID_SIZE,] * 3
        next_allowed_error = current_allowed_error * GRID_SIZE**3 / split_count
        next_jobs = []

        with split_list.map(pyopencl.map_flags.READ, shape=split_count) as array:
            assert len(array) == split_count
            for i, j, k, l in array:
                corner = util.Vector(i * s, j * s, k * s) + current_corner
                next_jobs.append((corner, next_step, next_grid, next_allowed_error))

        return next_jobs

    start_time = time.time()
    cl_util.interleave2(job, [(box.a, initial_step, initial_grid, initial_allowed_error)])
    end_time = time.time()
    print("time spent", end_time - start_time, "opencl compute time", time_computing,
          "efficiency", time_computing / (end_time - start_time))

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
