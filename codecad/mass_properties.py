import collections
import math

import numpy
import pyopencl

from . import util
from .util import cl_util
from . import subdivision
from .compute import program
from .compute import compute

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
        grid_size = 64

    assert shape.dimension() == 3, "3D objects are not supported yet"
    assert resolution > 0, "Non-positive resolution makes no sense"
    assert grid_size > 1, "Grid needs to be at least 2x2x2"
    assert grid_size**5 <= 2**32, "Centroid coordinate sums would overflow"
        # TODO: Increase this limit, use 64bit or figure out something else ...

    program_buffer = pyopencl.Buffer(compute.ctx,
                                     pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=program.make_program(shape))

    box = shape.bounding_box()
    block_sizes = subdivision.calculate_block_sizes(box,
                                                    shape.dimension(),
                                                    resolution,
                                                    grid_size,
                                                    overlap=False)

    block_sizes = [(resolution * cell_size, level_size)
                   for cell_size, level_size in block_sizes]

    helper1 = _Helper(compute.queue, grid_size, program_buffer, block_sizes)
    helper2 = _Helper(compute.queue, grid_size, program_buffer, block_sizes)

    cl_util.interleave([(box.a, 0)], helper1, helper2)
    # now helper1.integrals_* and helper2.integral_* each contain integral for
    # one half of the examined space

    integral_one = helper1.integral_one + helper2.integral_one
    integral_x = helper1.integral_x + helper2.integral_x
    integral_y = helper1.integral_y + helper2.integral_y
    integral_z = helper1.integral_z + helper2.integral_z
    integral_xx = helper1.integral_xx + helper2.integral_xx
    integral_yy = helper1.integral_yy + helper2.integral_yy
    integral_zz = helper1.integral_zz + helper2.integral_zz
    integral_xy = helper1.integral_xy + helper2.integral_xy
    integral_xz = helper1.integral_xz + helper2.integral_xz
    integral_yz = helper1.integral_yz + helper2.integral_yz

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

class _Helper:
    def __init__(self, queue, grid_size, program_buffer, block_sizes):
        self.queue = queue
        self.grid_size = grid_size
        self.program_buffer = program_buffer
        self.block_sizes = block_sizes

        self.integral_one = 0
        self.integral_x = 0
        self.integral_y = 0
        self.integral_z = 0
        self.integral_xx = 0
        self.integral_yy = 0
        self.integral_zz = 0
        self.integral_xy = 0
        self.integral_xz = 0
        self.integral_yz = 0

        self.index_sums = cl_util.Buffer(queue, numpy.uint32, 10, pyopencl.mem_flags.READ_WRITE)

        self.counter = cl_util.Buffer(queue, numpy.uint32, 1, pyopencl.mem_flags.READ_WRITE)
        self.list = cl_util.Buffer(queue, cl_util.Buffer.quad_dtype(numpy.uint8), grid_size * grid_size * grid_size, pyopencl.mem_flags.WRITE_ONLY)

        self.box_corner = None
        self.level = None

    def enqueue(self, box_corner, level):
        box_step = self.block_sizes[level][0]
        grid_dimensions = self.block_sizes[level][1]
        assert all(x <= self.grid_size for x in grid_dimensions)
        self.box_corner = box_corner
        self.level = level
        shifted_corner = box_corner + util.Vector.splat(box_step / 2)
        if level < len(self.block_sizes) - 1:
            distance_threshold = box_step * math.sqrt(3) / 2
        else:
            distance_threshold = 0

        # Enqueue write instead of fill to work around pyopencl bug #168
        fill_ev = self.index_sums.enqueue_write(numpy.zeros(10, self.index_sums.dtype))
        fill_ev = self.counter.enqueue_write(numpy.zeros(1, self.counter.dtype), wait_for=[fill_ev])

        return compute.program.mass_properties(self.queue, grid_dimensions, None,
                                               self.program_buffer,
                                               shifted_corner.as_float4(), numpy.float32(box_step),
                                               numpy.float32(distance_threshold),
                                               self.index_sums.buffer,
                                               self.counter.buffer, self.list.buffer,
                                               wait_for=[fill_ev])

    def process_result(self, event):
        intersecting_count = self.counter.read(wait_for=[event])[0]
        intersecting_indices = self.list.read() # TODO: This read could run in the background

        # For all the functions in question, convert `sum f(I)` (where I are indices
        # of occupied cells) to `integral f(X)` over all occupied cells.

        s = self.block_sizes[self.level][0]
        s2 = s * s
        s3 = s * s2
        b = self.box_corner + util.Vector.splat(s / 2)

        # Order is defined in mass_properties.cl
        sum_xx, sum_xy, sum_xz, sum_x, sum_yy, sum_yz, \
            sum_y, sum_zz, sum_z, n = self.index_sums.read()

        tmp_x = s * sum_x
        tmp_y = s * sum_y
        tmp_z = s * sum_z
        tmp_xx = s2 * sum_xx
        tmp_yy = s2 * sum_yy
        tmp_zz = s2 * sum_zz
        tmp_xy = s2 * sum_xy
        tmp_xz = s2 * sum_xz
        tmp_yz = s2 * sum_yz

        integral_one = s3 * n
        integral_x = s3 * (n * b.x + tmp_x)
        integral_y = s3 * (n * b.y + tmp_y)
        integral_z = s3 * (n * b.z + tmp_z)
        integral_xx = s3 * (n * (b.x * b.x + s2 / 12) + 2 * b.x * tmp_x + tmp_xx)
        integral_yy = s3 * (n * (b.y * b.y + s2 / 12) + 2 * b.y * tmp_y + tmp_yy)
        integral_zz = s3 * (n * (b.z * b.z + s2 / 12) + 2 * b.z * tmp_z + tmp_zz)
        integral_xy = s3 * (n * b.x * b.y + b.x * tmp_y + b.y * tmp_x + tmp_xy)
        integral_xz = s3 * (n * b.x * b.z + b.x * tmp_z + b.z * tmp_x + tmp_xz)
        integral_yz = s3 * (n * b.y * b.z + b.y * tmp_z + b.z * tmp_y + tmp_yz)

        # TODO: Watch out for numerical errors here, convert this to use Kahan sumation
        # Values added to each integral running sum are going to be exponentially
        # decreasing as the grid gets finer, we might run out of space in mantissa
        self.integral_one += integral_one
        self.integral_x += integral_x
        self.integral_y += integral_y
        self.integral_z += integral_z
        self.integral_xx += integral_xx
        self.integral_yy += integral_yy
        self.integral_zz += integral_zz
        self.integral_xy += integral_xy
        self.integral_xz += integral_xz
        self.integral_yz += integral_yz

        level = self.level + 1
        if level == len(self.block_sizes):
            assert intersecting_count == 0
        return ((util.Vector(i, j, k) * s + self.box_corner, level)
                for i, j, k, l in intersecting_indices[:intersecting_count])
