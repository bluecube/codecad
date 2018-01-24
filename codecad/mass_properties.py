import collections
import functools
import operator
import math

import numpy
import pyopencl
import pyopencl.cltypes

from . import util
from . import cl_util
from . import subdivision
from .cl_util import opencl_manager
from . import nodes

INNER_LOOP_SIDE = 3

c = opencl_manager.add_compile_unit()
c.append_define("INNER_LOOP_SIDE", INNER_LOOP_SIDE)
c.append_file("mass_properties.cl")


class MassProperties(collections.namedtuple("MassProperties", "volume centroid inertia_tensor")):
    """
    Contains volume of a body, position of its centroid and its inertia tensor.
    The inertia tensor is referenced to the centroid, not origin!
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

    grid_size = INNER_LOOP_SIDE * 32  # Max allowed grid size
    initial_grid_size = INNER_LOOP_SIDE * 16  # Only to cut away large chunks of empty space

    assert shape.dimension() == 3, "2D objects are not supported yet"
    assert grid_size > 2, "Grid side size needs to be > 2"
    assert grid_size % INNER_LOOP_SIDE == 0, "Grid size must be divisible by {}".format(INNER_LOOP_SIDE)
    assert grid_size < 2**8, "Grid coordinates must fit uint8"
    assert 0.25 * grid_size**3 * (grid_size + 1)**2 <= 2**32, \
        "Centroid coordinate sums must fit uint32"

    program_buffer = nodes.make_program_buffer(shape)

    box = shape.bounding_box()
    box_size = box.size()
    initial_step = box_size.max() / initial_grid_size
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

    kernel_invocations = 0
    function_evaluations = 0

    def job(job_id):
        nonlocal integral_one, integral_x, integral_y, integral_z, \
                 integral_xx, integral_yy, integral_zz, \
                 integral_xy, integral_xz, integral_yz
        nonlocal kernel_invocations, function_evaluations

        current_corner, current_step, current_grid, current_allowed_error = job_id
        assert all(x <= grid_size for x in current_grid)

        current_evaluations = functools.reduce(operator.mul, current_grid)

        index_sums = cl_util.Buffer(numpy.uint32, 10, pyopencl.mem_flags.READ_WRITE)
        intersecting_counter = cl_util.Buffer(numpy.uint32, 1, pyopencl.mem_flags.READ_WRITE)
        intersecting_list = cl_util.Buffer(pyopencl.cltypes.uchar4,
                                           grid_size**3,
                                           pyopencl.mem_flags.WRITE_ONLY)

        # Enqueue write instead of fill to work around pyopencl bug #168
        fill_ev = index_sums.enqueue_write(numpy.zeros(10, index_sums.dtype))
        fill_ev = intersecting_counter.enqueue_write(numpy.zeros(1, intersecting_counter.dtype), wait_for=[fill_ev])

        shifted_corner = current_corner + util.Vector.splat(current_step / 2)
        distance_threshold = current_step * math.sqrt(3) / 2

        assert all(x % INNER_LOOP_SIDE == 0 for x in current_grid)
        global_work_size = [x // INNER_LOOP_SIDE for x in current_grid]

        yield opencl_manager.k.mass_properties(global_work_size, None,
                                               program_buffer,
                                               shifted_corner.as_float4(), numpy.float32(current_step),
                                               numpy.float32(distance_threshold),
                                               index_sums,
                                               intersecting_counter, intersecting_list,
                                               wait_for=[fill_ev])
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
        # TODO: The precision setting will not work correctly on mostly empty
        # models (eg. a grid of tiny balls at vertices of the initial grid)
        if current_allowed_error is None:
            # Handle first kernel launch
            estimate = n + intersecting_count / 2  # Volume integral estimate for this block
            current_allowed_error = precision * estimate
        error_bound = intersecting_count / 2

        intersecting_event.wait()
        sub_block_corners = (util.Vector(i, j, k) * s + current_corner
                           for i, j, k, l in intersecting_list[:intersecting_count])
        if error_bound < current_allowed_error:
            # The error is small, we can stop subdividing

            # Add the intersecting sub blocks to the integral with 0.5 weight
            for sub_block_corner in sub_block_corners:
                weight = 0.5 * s3

                sub_block_center = sub_block_corner + util.Vector.splat(s)
                integral_one += weight
                integral_x +=  weight * sub_block_center.x
                integral_y +=  weight * sub_block_center.y
                integral_z +=  weight * sub_block_center.z
                integral_xx += weight * sub_block_center.x**2
                integral_yy += weight * sub_block_center.y**2
                integral_zz += weight * sub_block_center.z**2
                integral_xy += weight * sub_block_center.x * sub_block_center.y
                integral_xz += weight * sub_block_center.x * sub_block_center.z
                integral_yz += weight * sub_block_center.y * sub_block_center.z

            # No more jobs from here
            return []
        else:
            # Must subdivide
            next_step = current_step / grid_size
            next_grid = [grid_size,] * 3
            next_allowed_error = current_allowed_error * grid_size**3 / intersecting_count

            next_jobs = [(next_corner, next_step, next_grid, next_allowed_error)
                         for next_corner in sub_block_corners]
            return next_jobs

    cl_util.interleave2(job, [(box.a, initial_step, initial_grid, None)])

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

    # Relative speedup compared to just blindly dividing the bounding box to the
    # final resolution in one step
    #direct_evaluations = functools.reduce(operator.mul, (box.size() / resolution).applyfunc(math.ceil))
    #speedup = 1 - function_evaluations / direct_evaluations
    #print("speedup:", speedup)

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
