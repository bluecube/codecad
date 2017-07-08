import math

import pyopencl
import numpy

from . import util
from .util import cl_util
from .compute import compute
from .compute import program


class _Helper:
    def __init__(self, queue, grid_size, program_buffer, block_sizes):
        self.queue = queue
        self.grid_size = grid_size
        self.program_buffer = program_buffer
        self.block_sizes = block_sizes
        self.counter = cl_util.Buffer(queue, numpy.uint32, 1, pyopencl.mem_flags.READ_WRITE)
        self.list = cl_util.Buffer(queue, cl_util.Buffer.quad_dtype(numpy.uint8), grid_size * grid_size * grid_size, pyopencl.mem_flags.WRITE_ONLY)

        self.level = None

    def enqueue(self, box_corner, level):
        box_step = self.block_sizes[level][0]
        grid_dimensions = self.block_sizes[level][1]
        assert all(x <= self.grid_size for x in grid_dimensions)
        self.box_corner = box_corner
        self.level = level
        shifted_corner = box_corner + util.Vector.splat(box_step / 2)

        # Enqueue write instead of fill to work around pyopencl bug #168
        fill_ev = self.counter.enqueue_write(numpy.zeros(1, self.counter.dtype))

        self.ev = compute.program.subdivision_step(self.queue, grid_dimensions, None,
                                                   self.program_buffer,
                                                   shifted_corner.as_float4(), numpy.float32(box_step),
                                                   numpy.float32(box_step * math.sqrt(3)),
                                                   self.counter.buffer, self.list.buffer,
                                                   wait_for=[fill_ev])

    def process_result(self):
        box_step = self.block_sizes[self.level][0]

        c = self.counter.read(wait_for=[self.ev])
        intersecting_count = c[0]
        intersecting_indices = self.list.read()

        intersecting_pos = [util.Vector(i, j, k) * box_step + self.box_corner
                            for i, j, k, l in intersecting_indices[:intersecting_count]]

        return self.level, intersecting_pos

def _calculate_block_sizes(box, resolution, grid_size, overlap):
    # Figure out the layout of grids for processing.
    # There shouldn't ever be more than ~10 levels.

    box_size = box.size()
    box_max_size = box_size.max()

    block_sizes = []
    current_resolution = resolution
    level_size = (grid_size,) * 3
    while True:
        block_sizes.append((current_resolution, level_size))
        if overlap and len(block_sizes) == 1:
            tmp = current_resolution * (grid_size - 1)
        else:
            tmp = current_resolution * grid_size

        if tmp > box_max_size:
            break
        current_resolution = tmp
    if overlap and len(block_sizes) == 1:
        block_sizes[-1] = (current_resolution, tuple(min(math.ceil(x) + 1, grid_size) for x in box_size / current_resolution))
    else:
        block_sizes[-1] = (current_resolution, tuple(min(math.ceil(x), grid_size) for x in box_size / current_resolution))
    block_sizes.reverse()
    return block_sizes

def subdivision(shape, resolution, overlap_edge_samples=True, grid_size=None):
    """
    Subdivides a space around a shape into blocks that are suitable for evaluating
    with OpenCL in one piece, skipping 100% empty space and 100% filled space.

    :returns: Tuple `(program_buffer, grid_dimensions, [(corner, resolution)])`.
        `program_buffer` is a PyOpenCL buffer object that contains the compiled
        instructions for evaluating the model, `grid_dimensions` is a three item
        tuple describing the size of the sampling grid inside each block.
        `corner` and `resolution` are respectively a `Vector3` of block corner
        and spacing between sample points (so that the block has volume
        (grid_size - 1)**3 * resolution, see :param overlap).
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
    assert grid_size**4 <= 2**32, "Centroid coordinate sums would overflow"

    program_buffer = pyopencl.Buffer(compute.ctx,
                                     pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=program.make_program(shape))

    box = shape.bounding_box().expanded_additive(resolution/2)

    block_sizes = _calculate_block_sizes(box, resolution, grid_size, overlap_edge_samples)

    if len(block_sizes) == 1:
        return program_buffer, block_sizes[0][1], [(box.a, block_sizes[0][0], block_sizes[0][1])]

    helper1 = _Helper(compute.queue, grid_size, program_buffer, block_sizes)
    helper1.enqueue(box.a, 0)
    helper2 = _Helper(compute.queue, grid_size, program_buffer, block_sizes)

    stack = []
    final_blocks = []

    while True:
        stack_was_empty = len(stack) == 0
        if not stack_was_empty:
            helper2.enqueue(*stack.pop())

        level, intersecting_pos = helper1.process_result()
        level += 1
        if level == len(block_sizes) - 1:
            for pos in intersecting_pos:
                final_blocks.append((util.Vector(*pos[:3]),
                                     block_sizes[level][0],
                                     block_sizes[level][1]))
            if stack_was_empty:
                break

        else:
            for pos in intersecting_pos:
                stack.append((util.Vector(*pos[:3]), level))

            if stack_was_empty:
                helper2.enqueue(*stack.pop())

        helper1, helper2 = helper2, helper1 # Swap them for the next iteration

    return program_buffer, block_sizes[-1][1], final_blocks
