import numpy
import mcubes
import pyopencl

from ..util import cl_util
from .. import util
from .. import subdivision
from ..compute import compute

_link_overflow_mask = 0xfff00000

def triangular_mesh(obj, resolution, subdivision_grid_size=None, debug_subdivision_boxes = False):
    """ Generate a triangular mesh representing a surface of 3D shape.
    Yields tuples (vertices, indices). """
    obj.check_dimension(required = 3)

    program_buffer, max_box_size, boxes = subdivision.subdivision(obj,
                                                               resolution,
                                                               grid_size=subdivision_grid_size)

    block = numpy.empty(max_box_size, dtype=numpy.float32)
    block_buffer = pyopencl.Buffer(compute.ctx, pyopencl.mem_flags.WRITE_ONLY, block.nbytes)

    for i, (box_size, box_corner, box_resolution, *_) in enumerate(boxes):
        if debug_subdivision_boxes:
            # Export just an outline of the block instead of displaying its contents
            vertices = [util.Vector(i, j, k).elementwise_mul(box_size) * box_resolution + box_corner
                        for k in range(2) for j in range(2) for i in range(2)]
            trinagles = [[0, 3, 1], [0, 2, 3],
                         [1, 3, 5], [3, 7, 5],
                         [4, 5, 6], [5, 7, 6],
                         [0, 6, 2], [0, 4, 6],
                         [0, 1, 5], [0, 5, 4],
                         [3, 2, 6], [3, 6, 7]]
            yield vertices, triangles
            continue

        print(box_corner, box_resolution, box_size)
        #with util.status_block("{}/{}".format(i + 1, len(boxes))):
        # TODO: Staggered opencl / python processing the way subdivision does it.
        ev = compute.program.grid_eval(compute.queue, box_size, None,
                                       program_buffer,
                                       box_corner.as_float4(), numpy.float32(box_resolution),
                                       block_buffer)
        pyopencl.enqueue_copy(compute.queue, block, block_buffer, wait_for=[ev])

        vertices, triangles = mcubes.marching_cubes(block, 0)

        if len(triangles) == 0:
            continue

        vertices[:, [0, 1]] = vertices[:, [1, 0]]
        vertices[:, 1] *= -1
        vertices *= box_resolution
        vertices += box_corner
        triangles[:, [0, 1]] = triangles[:, [1, 0]]

        yield vertices, triangles

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
    obj.check_dimension(required = 2)

    program_buffer, grid_size, boxes = subdivision.subdivision(obj,
                                                               resolution,
                                                               grid_size=subdivision_grid_size)

    assert grid_size[0] < 512, "Larger grid size would overflow the index encoding"
    assert grid_size[2] == 1

    grid_size = (grid_size[0], grid_size[1])
    grid_size_triangles = (grid_size[0] - 1, grid_size[1] - 1, 2)

    corners = pyopencl.Buffer(compute.ctx,
                              pyopencl.mem_flags.READ_WRITE | pyopencl.mem_flags.HOST_NO_ACCESS,
                              cl_util.Buffer.quad_dtype(numpy.float32).itemsize * grid_size[0] * grid_size[1])
    vertices = cl_util.Buffer(compute.queue,
                              cl_util.Buffer.dual_dtype(numpy.float32),
                              grid_size_triangles[0] * grid_size_triangles[1] * grid_size_triangles[2],
                              pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.HOST_READ_ONLY)
    links = cl_util.Buffer(compute.queue,
                           numpy.uint32,
                           grid_size_triangles[0] * grid_size_triangles[1] * grid_size_triangles[2],
                           pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.HOST_READ_ONLY)
    starts = cl_util.Buffer(compute.queue,
                            numpy.uint32,
                            grid_size_triangles[0] + grid_size_triangles[1],
                                # Start of chain can happen only on a side and
                                # each chain takes at least two cells (start and end)
                            pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.HOST_READ_ONLY)
    start_counter = cl_util.Buffer(compute.queue,
                                   numpy.uint32,
                                   1,
                                   pyopencl.mem_flags.READ_WRITE)

    open_chain_beginnings = {}
    open_chain_ends = {}

    for i, (box_size,
            box_corner, box_resolution,
            int_box_corner, int_box_resolution) in enumerate(boxes):

        print()
        print("XXXX", i, int_box_corner)

        assert box_size[0] == box_size[1]
        int_box_step = int_box_resolution * (box_size[0] - 1)

        # TODO: Staggered opencl / python processing the way subdivision does it.
        corners_ev = compute.program.grid_eval_full(compute.queue, grid_size, None,
                                                    program_buffer,
                                                    box_corner.as_float4(), numpy.float32(box_resolution),
                                                    corners)
        fill_ev = start_counter.enqueue_write(numpy.zeros(1, start_counter.dtype))
        process_ev = compute.program.process_polygon(compute.queue, grid_size_triangles, None,
                                                     box_corner.as_float2(), numpy.float32(box_resolution),
                                                     corners,
                                                     vertices.buffer, links.buffer, starts.buffer, start_counter.buffer,
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
                print("nothing before")
                chain = []
                assert beginning_key not in open_chain_beginnings
            else:
                print("found before")

            overflow_spec = _collect_polygon(vertices, links, starting_index, chain)

            end_key = (int_box_corner + _step_from_overflow_spec(overflow_spec) * int_box_step,
                       overflow_spec)
            open_chain_beginnings[beginning_key] = chain, end_key

            # Find any chain following the current one and merge them
            try:
                to_append, to_append_end_key = open_chain_beginnings.pop(end_key)
            except KeyError:
                print("nothing after", len(chain))
                open_chain_ends[end_key] = chain
            else:
                print("found after")
                if to_append is chain:
                    # This would close the chain into a loop, we're done with it
                    del open_chain_beginnings[beginning_key]
                    yield chain
                    print("yielding result")
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
            yield polygon # Closed chains can be yielded directly

        print("beginnings")
        for k, v in open_chain_beginnings.items():
            print(k[0], bin(k[1]), "-", len(v[0]), "items, id:", hex(id(v[0])), ", key: ", v[1])
        print("ends")
        for k, v in open_chain_ends.items():
            print(k[0], bin(k[1]), "-", len(v), "items, id:", hex(id(v)))

    assert len(open_chain_beginnings) == 0
    assert len(open_chain_ends) == 0
