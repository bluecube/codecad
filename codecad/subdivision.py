import math

import pyopencl
import numpy

from . import util
from . import nodes
from . import cl_util
from .cl_util import opencl_manager

cl_util.opencl_manager.add_compile_unit().append_file("subdivision.cl")


class _Helper:
    def __init__(self, queue, grid_size, dimension, program_buffer, block_sizes, origin, resolution, final_blocks):
        self.queue = queue
        self.grid_size = grid_size
        self.dimension = dimension
        self.program_buffer = program_buffer
        self.block_sizes = block_sizes
        self.origin = origin
        self.resolution = resolution
        self.final_blocks = final_blocks

        self.counter = cl_util.Buffer(numpy.uint32, 1, pyopencl.mem_flags.READ_WRITE, queue=queue)
        self.list = cl_util.Buffer(cl_util.Buffer.quad_dtype(numpy.uint8),
                                   grid_size * grid_size * grid_size,
                                   pyopencl.mem_flags.WRITE_ONLY, queue=queue)

        self.level = None

    def enqueue(self, int_box_corner, level):
        int_box_step = self.block_sizes[level][0]
        grid_dimensions = self.block_sizes[level][1]
        assert all(x <= self.grid_size for x in grid_dimensions)
        self.int_box_corner = int_box_corner
        self.level = level

        if self.dimension == 3:
            int_shifted_corner = int_box_corner + util.Vector.splat(int_box_step / 2)
        elif self.dimension == 2:
            int_shifted_corner = int_box_corner + util.Vector(int_box_step / 2, int_box_step / 2)
        else:
            assert False

        box_step = int_box_step * self.resolution
        shifted_corner = int_shifted_corner * self.resolution + self.origin

        distance_threshold = box_step * math.sqrt(self.dimension) / 2

        # Enqueue write instead of fill to work around pyopencl bug #168
        fill_ev = self.counter.enqueue_write(numpy.zeros(1, self.counter.dtype))

        return opencl_manager.k.subdivision_step(grid_dimensions, None,
                                                 self.program_buffer,
                                                 shifted_corner.as_float4(), numpy.float32(box_step),
                                                 numpy.float32(distance_threshold),
                                                 self.counter, self.list,
                                                 wait_for=[fill_ev])

    def process_result(self, event):
        int_box_step = self.block_sizes[self.level][0]
        box_step = int_box_step * self.resolution

        c = self.counter.read(wait_for=[event])
        intersecting_count = c[0]
        intersecting_indices = self.list.read()

        int_intersecting_pos = [util.Vector(i, j, k) * int_box_step + self.int_box_corner
                                for i, j, k, l in intersecting_indices[:intersecting_count]]

        level = self.level + 1
        if level == len(self.block_sizes) - 1:
            next_int_box_step = self.block_sizes[level][0]
            next_box_step = next_int_box_step * self.resolution
            for int_pos in int_intersecting_pos:
                pos = int_pos * self.resolution + self.origin
                self.final_blocks.append((self.block_sizes[level][1],
                                          pos, next_box_step,
                                          int_pos, next_int_box_step))
            return []
        else:
            return ((int_pos, level) for int_pos in int_intersecting_pos)


def calculate_block_sizes(box, dimension, resolution, grid_size, overlap, level_size_multiplier=1):
    # Figure out the layout of grids for processing.
    # There shouldn't ever be more than ~10 levels.

    if grid_size % level_size_multiplier != 0:
        raise ValueError("Grid size must be divisible by level_size_multiplier")

    block_sizes = []
    if dimension == 2:
        level_size = util.Vector(grid_size, grid_size, 1)
        box = box.flattened()
    elif dimension == 3:
        level_size = util.Vector.splat(grid_size)
    else:
        assert False

    box_int_size = (box.size() / resolution).applyfunc(math.ceil)
    box_max_int_size = box_int_size.max()
    cell_size = 1

    while True:
        block_sizes.append((cell_size, level_size))

        overlap_delta = 1 if overlap and len(block_sizes) == 1 else 0

        next_cell_size = cell_size * (grid_size - overlap_delta)

        if next_cell_size >= box_max_int_size:
            break
        cell_size = next_cell_size

    block_sizes[-1] = (cell_size,
                       util.Vector(*(util.clamp(util.round_up_to(math.ceil(x) + overlap_delta, level_size_multiplier), 1, s)
                                   for x, s in zip(box_int_size / cell_size, level_size))))

    block_sizes.reverse()
    return block_sizes


def subdivision(shape, resolution, overlap_edge_samples=True, grid_size=None):
    """
    Subdivides a space around a shape into blocks that are suitable for evaluating
    with OpenCL in one piece, skipping 100% empty space and 100% filled space.

    :returns: Tuple `(program_buffer, max_grid_size, [(grid_size, corner, spacing, int_corner, int_spacing)])`.
        `program_buffer` is a PyOpenCL buffer object that contains the compiled
        instructions for evaluating the model, `grid_size` and `max_grid_size` are
        `Vector`s describing the size of the sampling grid inside each block and
        maximal size through all blocks for allocations.
        `corner` and `spacing` are respectively a `Vector` of block corner
        and spacing between sample points (so that the block has volume
        (grid_size - 1)**3 * resolution, see :param overlap).
        `int_corner` and `int_spacing` are similar to the previous two, but
        in resolution units (smallest step is 1) and relative to the first calculated node.

        For example box from -5, -5, -5 to 5, 5, 5 and resolution 0.1 and grid_size 16
        will contain a corner block  `((16, 16, 16), (-5.05, -5.05, -5.05), 0.1, (0, 0, 0), 1)`.
    :param shape: The shape to be evaluated.
    :param resolution: Intended resolution of leaf blocks.
    :param overlap_edge_samples: If this is set to true, leaf are effectively
        decreased in size by one `resolution` from the right, bottom and rear.
        This means that sample points on the right, bottom and rear sides of the
        block will overlap with sample points of the neighboring blocks.
        .. image:: docs/subdivision_overlap.svg.
    :param grid_size: This value cubed is the size of non top-level blocks
        evaluated during subdivision and also size of the output blocks (except
        when the whole shape fits into a single block.
        If set to None a reasonable default is used.
    """

    if grid_size is None:
        # TODO: Determine default grid size
        grid_size = 128

    assert resolution > 0, "Non-positive resolution makes no sense"
    assert grid_size > 1, "Grid needs to be at least 2x2x2"
    assert grid_size <= 256, "Grid size > 256 would cause overflows in returned index list."

    program_buffer = nodes.make_program_buffer(shape)

    box = shape.bounding_box().expanded_additive(resolution/2)
    dimension = shape.dimension()

    if dimension == 2:
        box = box.flattened()

    block_sizes = calculate_block_sizes(box,
                                        shape.dimension(),
                                        resolution,
                                        grid_size,
                                        overlap_edge_samples)

    if len(block_sizes) == 1:
        return program_buffer, block_sizes[0][1], [(block_sizes[0][1], box.a, resolution, util.Vector(0, 0, 0), 1)]

    final_blocks = []
    helper1 = _Helper(opencl_manager.queue, grid_size, dimension,
                      program_buffer, block_sizes, box.a, resolution,
                      final_blocks)
    helper2 = _Helper(opencl_manager.queue, grid_size, dimension,
                      program_buffer, block_sizes, box.a, resolution,
                      final_blocks)

    cl_util.interleave([(util.Vector(0, 0, 0), 0)], helper1, helper2)

    return program_buffer, block_sizes[-1][1], final_blocks
