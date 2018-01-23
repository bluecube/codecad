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


def mass_properties(shape, resolution, grid_size=None):
    # Inertia tensor info:
    # http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node64.html

    if grid_size is None:
        # TODO: Determine default grid size
        grid_size = INNER_LOOP_SIDE * 32

    assert shape.dimension() == 3, "2D objects are not supported yet"
    assert resolution > 0, "Non-positive resolution makes no sense"
    assert grid_size > 2, "Grid needs to be > 2"
    assert grid_size % INNER_LOOP_SIDE == 0, "Grid size must be divisible by {}".format(INNER_LOOP_SIDE)
    assert 0.5 * grid_size**5 <= 2**32, "Centroid coordinate sums would overflow uint32"

    program_buffer = nodes.make_program_buffer(shape)

    box = shape.bounding_box()
    block_sizes = subdivision.calculate_block_sizes(box,
                                                    shape.dimension(),
                                                    resolution,
                                                    grid_size,
                                                    overlap=False,
                                                    level_size_multiplier=INNER_LOOP_SIDE)
    block_sizes = [(resolution * cell_size, level_size)
                   for cell_size, level_size in block_sizes]

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

        box_corner, level = job_id

        index_sums = cl_util.Buffer(numpy.uint32, 10, pyopencl.mem_flags.READ_WRITE)
        intersecting_counter = cl_util.Buffer(numpy.uint32, 1, pyopencl.mem_flags.READ_WRITE)
        intersecting_list = cl_util.Buffer(pyopencl.cltypes.uchar4,
                                           grid_size**3,
                                           pyopencl.mem_flags.WRITE_ONLY)

        box_step = block_sizes[level][0]
        grid_dimensions = block_sizes[level][1]
        assert all(x <= grid_size for x in grid_dimensions)
        assert all(x % INNER_LOOP_SIDE == 0 for x in grid_dimensions)
        shifted_corner = box_corner + util.Vector.splat(box_step / 2)
        if level < len(block_sizes) - 1:
            distance_threshold = box_step * math.sqrt(3) / 2
        else:
            distance_threshold = 0

        # Enqueue write instead of fill to work around pyopencl bug #168
        fill_ev = index_sums.enqueue_write(numpy.zeros(10, index_sums.dtype))
        fill_ev = intersecting_counter.enqueue_write(numpy.zeros(1, intersecting_counter.dtype), wait_for=[fill_ev])

        global_work_size = [x // INNER_LOOP_SIDE for x in grid_dimensions]
        #print(box_corner, box_step, grid_dimensions, global_work_size)

        yield opencl_manager.k.mass_properties(global_work_size, None,
                                               program_buffer,
                                               shifted_corner.as_float4(), numpy.float32(box_step),
                                               numpy.float32(distance_threshold),
                                               index_sums,
                                               intersecting_counter, intersecting_list,
                                               wait_for=[fill_ev])
        kernel_invocations += 1
        function_evaluations += functools.reduce(operator.mul, grid_dimensions)

        intersecting_count = intersecting_counter.read()[0]
        intersecting_event = intersecting_list.enqueue_read()

        # For all the functions in question, convert `sum f(I)` (where I are indices
        # of occupied cells) to `integral f(X)` over all occupied cells.

        s = block_sizes[level][0]
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

        level = level + 1
        assert level < len(block_sizes) or intersecting_count == 0

        intersecting_event.wait()
        return ((util.Vector(i, j, k) * s + box_corner, level)
                for i, j, k, l in intersecting_list[:intersecting_count])

    cl_util.interleave2(job, [(box.a, 0)])

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
    direct_evaluations = functools.reduce(operator.mul, (box.size() / resolution).applyfunc(math.ceil))
    speedup = 1 - function_evaluations / direct_evaluations
    print("speedup:", speedup)

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
