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
assert GRID_SIZE > 2, "Grid side size needs to be > 2"
assert GRID_SIZE % INNER_LOOP_SIDE == 0, "Grid size must be divisible by {}".format(INNER_LOOP_SIDE)
assert GRID_SIZE < 2**7, "Grid coordinates must fit int8"
assert GRID_SIZE**3 * (GRID_SIZE + 1)**2 / 2 <= 2**32, \
    "Centroid coordinate sums must fit uint32"

c = opencl_manager.add_compile_unit()
c.append_define("INNER_LOOP_SIDE", INNER_LOOP_SIDE)
c.append_file("mass_properties.cl")


class MassProperties(collections.namedtuple("MassProperties", "volume centroid inertia_tensor max_volume_error")):
    """
    Contains volume of a body, position of its centroid and its inertia tensor.
    The inertia tensor is referenced to the centroid, not origin!
    max_volume_error returns maximum deviation of the volume field from the actual
    volume of the shape.
    """
    __slots__ = ()


def mass_properties(shape, precision=1e-4):
    """ Calculate volume, centroid and inertia tensor of the shape.
    Iteratively subdivides the shape until
    abs(actual_volume - computed_volume) < precision * actual_volume """
    # Inertia tensor info:
    # http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node64.html
    #
    # Some useful integrals:
    # integral(x, a, a + s, integral(y, b, b + s, integral(z, c, c + s, x))) = s**3 * (a + s / 2)
    # integral(x, a, a + s, integral(y, b, b + s, integral(z, c, c + s, x * y))) = s**3 * (a + s / 2) * (b + s / 2)

    assert shape.dimension() == 3, "2D objects are not supported yet"

    program_buffer = nodes.make_program_buffer(shape)

    box = shape.bounding_box()
    box_size = box.size()
    initial_step = box_size.max() / GRID_SIZE
    initial_grid = [int(util.round_up_to(s / initial_step, INNER_LOOP_SIDE)) for s in box_size]

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
    total_error = util.KahanSummation()

    time_computing = 0
    kernel_invocations = 0
    function_evaluations = 0

    def job(job_id):
        nonlocal integral_one, integral_x, integral_y, integral_z, \
                 integral_xx, integral_yy, integral_zz, \
                 integral_xy, integral_xz, integral_yz, total_error
        nonlocal time_computing

        nonlocal kernel_invocations, function_evaluations

        current_corner, current_step, current_grid, current_allowed_error = job_id
        assert all(x <= GRID_SIZE for x in current_grid)

        current_evaluations = functools.reduce(operator.mul, current_grid)

        index_sums = cl_util.Buffer(numpy.uint32, 10, pyopencl.mem_flags.READ_WRITE)
        intersecting_counter = cl_util.Buffer(numpy.uint32, 1, pyopencl.mem_flags.READ_WRITE)
        intersecting_list = cl_util.Buffer(pyopencl.cltypes.char4,
                                           GRID_SIZE**3,
                                           pyopencl.mem_flags.WRITE_ONLY)

        # Enqueue write instead of fill to work around pyopencl bug #168
        fill_ev = index_sums.enqueue_write(numpy.zeros(10, index_sums.dtype))
        fill_ev = intersecting_counter.enqueue_write(numpy.zeros(1, intersecting_counter.dtype), wait_for=[fill_ev])

        shifted_corner = current_corner + util.Vector.splat(current_step / 2)

        assert all(x % INNER_LOOP_SIDE == 0 for x in current_grid)
        global_work_size = [x // INNER_LOOP_SIDE for x in current_grid]

        ev = opencl_manager.k.mass_properties(global_work_size, None,
                                              program_buffer,
                                              shifted_corner.as_float4(), numpy.float32(current_step),
                                              index_sums,
                                              intersecting_counter, intersecting_list,
                                              wait_for=[fill_ev])
        yield ev

        time_computing += ev.profile.end - ev.profile.start
        kernel_invocations += 1
        function_evaluations += current_evaluations

        intersecting_count = intersecting_counter.read()[0]
        intersecting_event = intersecting_list.enqueue_read()

        # For all the functions in question, convert `sum f(I)` (where I are indices
        # of occupied cells) to `integral f(X)` over all occupied cells.

        s = current_step
        s2 = s * s
        s3 = s * s2
        b = shifted_corner

        # Order is defined in mass_properties.cl
        sum_xx, sum_xy, sum_xz, sum_x, sum_yy, sum_yz, \
            sum_y, sum_zz, sum_z, n = index_sums.read()

        tmp_x = s * sum_x
        tmp_y = s * sum_y
        tmp_z = s * sum_z
        tmp_xx = s2 * sum_xx
        tmp_yy = s2 * sum_yy
        tmp_zz = s2 * sum_zz
        tmp_xy = s2 * sum_xy
        tmp_xz = s2 * sum_xz
        tmp_yz = s2 * sum_yz

        integral_one += s3 * n
        integral_x += s3 * (n * b.x + tmp_x)
        integral_y += s3 * (n * b.y + tmp_y)
        integral_z += s3 * (n * b.z + tmp_z)
        integral_xx += s3 * (n * (b.x * b.x + s2 / 12) + 2 * b.x * tmp_x + tmp_xx)
        integral_yy += s3 * (n * (b.y * b.y + s2 / 12) + 2 * b.y * tmp_y + tmp_yy)
        integral_zz += s3 * (n * (b.z * b.z + s2 / 12) + 2 * b.z * tmp_z + tmp_zz)
        integral_xy += s3 * (n * b.x * b.y + b.x * tmp_y + b.y * tmp_x + tmp_xy)
        integral_xz += s3 * (n * b.x * b.z + b.x * tmp_z + b.z * tmp_x + tmp_xz)
        integral_yz += s3 * (n * b.y * b.z + b.y * tmp_z + b.z * tmp_y + tmp_yz)

        # The adaptive part:
        # Everything in this part is calculated in blocks instead of in volumes
        # (Everything is divided by s3).

        # Go through all intersecting sub blocks and calculate volume estimate
        # and error bound based on their unbounding volumes
        estimate = util.KahanSummation()
        estimate += n
        error_bound = util.KahanSummation()
        processed_intersecting = []
        intersecting_event.wait()
        for i, j, k, l in intersecting_list[:intersecting_count]:
            corner = util.Vector(i, j, k) * s + current_corner
            volume_fraction = l / 127

            weight = (1 - volume_fraction) / 2
            error = (1 - abs(volume_fraction)) / 2

            estimate += weight
            error_bound += error

            processed_intersecting.append((corner, weight, error))

        estimate = estimate.result
        error_bound = error_bound.result

        # TODO: The precision setting will not work correctly on mostly empty
        # models (eg. a grid of tiny balls at vertices of the initial grid)
        if current_allowed_error is None:
            # Handle first kernel launch
            current_allowed_error = precision * estimate

        need_splitting = error_bound > current_allowed_error
        current_allowed_error_per_intersecting = current_allowed_error / intersecting_count
        next_step = current_step / GRID_SIZE
        next_grid = [GRID_SIZE,] * 3
        next_allowed_error = current_allowed_error_per_intersecting * GRID_SIZE**3
        next_jobs = []

        for corner, weight, error in processed_intersecting:
            if error < current_allowed_error_per_intersecting or not need_splitting:
                weight *= s3
                center = corner + util.Vector.splat(s / 2)

                integral_one += weight
                integral_x +=  weight * center.x
                integral_y +=  weight * center.y
                integral_z +=  weight * center.z
                integral_xx += weight * center.x**2
                integral_yy += weight * center.y**2
                integral_zz += weight * center.z**2
                integral_xy += weight * center.x * center.y
                integral_xz += weight * center.x * center.z
                integral_yz += weight * center.y * center.z
                total_error += s3 * error
            else:
                next_jobs.append((corner, next_step, next_grid, next_allowed_error))

        return next_jobs

    start_time = time.time()
    cl_util.interleave2(job, [(box.a, initial_step, initial_grid, None)])
    end_time = time.time()
    #print("time spent", end_time - start_time, "opencl compute time", time_computing,
    #      "efficiency", time_computing / (end_time - start_time))

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
