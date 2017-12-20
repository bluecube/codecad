import numpy
import pyopencl

from .. import cl_util
from ..cl_util import opencl_manager
from .. import util
from .. import subdivision

opencl_manager.add_compile_unit().append_file("polygon2d.cl")

_link_overflow_mask = 0xfff00000


def _collect_polygon(vertices, links, starting_index, chain):
    """ Follow links starting at starting_index and apend outputs onto chain.
    Returns the overflow spec of the last (invalid) link. """
    i = starting_index

    while not i & _link_overflow_mask:
        chain.append(tuple(vertices[i]))
        last_index = i
        i = links[i]
        links[last_index] = _link_overflow_mask

    return i & _link_overflow_mask


def _step_from_overflow_spec(spec):
    step_direction = -1 if spec & 0x20000000 else 1
    if spec & 0x40000000:
        return util.Vector(0, step_direction)
    else:
        return util.Vector(step_direction, 0)


def polygon(obj, resolution, subdivision_grid_size=None):
    """ Generate polygons representing the boundaries of a 2D shape. """
    obj.check_dimension(required=2)

    program_buffer, grid_size, boxes = subdivision.subdivision(obj,
                                                               resolution,
                                                               grid_size=subdivision_grid_size)

    assert grid_size[0] < 512, "Larger grid size would overflow the index encoding"
    assert grid_size[2] == 1

    grid_size = (grid_size[0], grid_size[1])
    grid_size_triangles = (grid_size[0] - 1, grid_size[1] - 1, 2)

    corners = pyopencl.Buffer(opencl_manager.context,
                              pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.HOST_NO_ACCESS,
                              cl_util.Buffer.quad_dtype(numpy.float32).itemsize * grid_size[0] * grid_size[1])
    vertices = cl_util.Buffer(cl_util.Buffer.dual_dtype(numpy.float32),
                              grid_size_triangles[0] * grid_size_triangles[1] * grid_size_triangles[2],
                              pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.HOST_READ_ONLY)
    links = cl_util.Buffer(numpy.uint32,
                           grid_size_triangles[0] * grid_size_triangles[1] * grid_size_triangles[2],
                           pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.HOST_READ_ONLY)
    starts = cl_util.Buffer(numpy.uint32,
                            grid_size_triangles[0] + grid_size_triangles[1],
                            # Start of chain can happen only on a side and
                            # each chain takes at least two cells (start and end)
                            pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.HOST_READ_ONLY)
    start_counter = cl_util.Buffer(numpy.uint32,
                                   1,
                                   pyopencl.mem_flags.READ_WRITE)

    open_chain_beginnings = {}
    open_chain_ends = {}

    for i, (box_size,
            box_corner, box_resolution,
            int_box_corner, int_box_resolution) in enumerate(boxes):

        if len(boxes) > 1:
            assert box_size[0] == box_size[1]
            int_box_step = int_box_resolution * (box_size[0] - 1)
        else:
            int_box_step = None  # There will be no open chains if we only visit one box

        # TODO: Staggered opencl / python processing the way subdivision does it.
        corners_ev = opencl_manager.k.grid_eval(grid_size, None,
                                                program_buffer,
                                                box_corner.as_float4(), numpy.float32(box_resolution),
                                                corners)
        fill_ev = start_counter.enqueue_write(numpy.zeros(1, start_counter.dtype))
        process_ev = opencl_manager.k.process_polygon(grid_size_triangles, None,
                                                      box_corner.as_float2(), numpy.float32(box_resolution),
                                                      corners,
                                                      vertices, links, starts, start_counter,
                                                      wait_for=[corners_ev, fill_ev])

        # Everything is read into the internal array of clutil.Buffer
        vertices.read(wait_for=[process_ev])
        links.read(wait_for=[process_ev])
        vertices.read(wait_for=[process_ev])
        starts.read(wait_for=[process_ev])
        start_counter.read(wait_for=[process_ev])

        # First handle the open chains
        assert start_counter[0] < len(starts)
        for starting_index in starts[:start_counter[0]]:
            overflow_spec = starting_index & _link_overflow_mask
            starting_index = starting_index & (~_link_overflow_mask)

            # Find existing chain in open chains that can be continued here, or
            # create a new one
            # After the block is done, we are holding either a fresh chain,
            # or an existing one and the chain is registered only in open_chain_beginnings
            beginning_key = (int_box_corner, overflow_spec)
            try:
                chain = open_chain_ends.pop(beginning_key)
            except KeyError:
                chain = []
                assert beginning_key not in open_chain_beginnings

            overflow_spec = _collect_polygon(vertices, links, starting_index, chain)

            end_key = (int_box_corner + _step_from_overflow_spec(overflow_spec) * int_box_step,
                       overflow_spec)
            open_chain_beginnings[beginning_key] = chain, end_key

            # Find any chain following the current one and merge them
            try:
                to_append, to_append_end_key = open_chain_beginnings.pop(end_key)
            except KeyError:
                open_chain_ends[end_key] = chain
            else:
                if to_append is chain:
                    # This would close the chain into a loop, we're done with it
                    del open_chain_beginnings[beginning_key]
                    yield chain
                else:
                    chain.extend(to_append)
                    # Overwrite the reference to `to_append` to point to `chain` instead
                    open_chain_ends[to_append_end_key] = chain

            assert len(open_chain_beginnings) == len(open_chain_ends)

        # Next go through the whole array and find all cells that have valid links
        # Each of these must be a part of a closed chain
        for starting_index in range(len(vertices)):
            if links[starting_index] & _link_overflow_mask:
                continue
            polygon = []
            _collect_polygon(vertices, links, starting_index, polygon)
            yield polygon  # Closed chains can be yielded directly

    assert len(open_chain_beginnings) == 0
    assert len(open_chain_ends) == 0
