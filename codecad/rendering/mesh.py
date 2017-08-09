import numpy
import mcubes
import pyopencl

from ..util import cl_util
from .. import util
from .. import subdivision
from ..compute import compute

_invalid_link = numpy.uint32(-1) # We use -1 as invalid flag in the polygon OpenCL code

def triangular_mesh(obj, resolution, subdivision_grid_size=None, debug_subdivision_boxes = False):
    """ Generate a triangular mesh representing a surface of 3D shape.
    Yields tuples (vertices, indices). """
    obj.check_dimension(required = 3)

    program_buffer, grid_size, boxes = subdivision.subdivision(obj,
                                                               resolution,
                                                               grid_size=subdivision_grid_size)

    block = numpy.empty((grid_size[1], grid_size[0], grid_size[2]),
                        dtype=numpy.float32)
    block_buffer = pyopencl.Buffer(compute.ctx, pyopencl.mem_flags.WRITE_ONLY, block.nbytes)

    for i, (box_corner, box_resolution, box_size) in enumerate(boxes):
        box_size = util.Vector(*box_size)

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

        #print(box_corner, box_resolution, box_size)
        #with util.status_block("{}/{}".format(i + 1, len(boxes))):
        # TODO: Staggered opencl / python processing the way subdivision does it.
        ev = compute.program.grid_eval(compute.queue, grid_size, None,
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
    Returns index of the last valid cell. """
    i = starting_index

    while i != _invalid_link:
        chain.append(vertices[i])
        last_index = i
        i = links[i]
        links[last_index] = _invalid_link

    return last_index

def polygon(obj, resolution, subdivision_grid_size=None):
    """ Generate polygons representing the boundaries of a 2D shape. """
    obj.check_dimension(required = 2)

    program_buffer, grid_size, boxes = subdivision.subdivision(obj,
                                                               resolution,
                                                               grid_size=32)
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

    open_chains = {} # Indexed by coordinates of the last node

    for i, (box_corner, box_resolution, box_size) in enumerate(boxes):
        box_size = util.Vector(*box_size)

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
        open_chains = []
        assert start_counter[0] < len(starts)
        for starting_index in starts[:start_counter[0]]:
            polygon = []
            _collect_polygon(vertices, links, starting_index, polygon)
            yield polygon # TODO: Merge open chains

        # Next go through the whole array and find all cells that have valid links
        # Each of these must be a part of a closed chain
        for starting_index in range(len(vertices)):
            if links[starting_index] == _invalid_link:
                continue
            polygon = []
            last_index = _collect_polygon(vertices, links, starting_index, polygon)
            assert last_index == starting_index
            yield polygon # Closed chains can be yielded directly
