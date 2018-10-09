import numpy
import mcubes
import pyopencl

from .. import util
from .. import subdivision
from ..cl_util import opencl_manager


def triangular_mesh(obj, subdivision_grid_size=None, debug_subdivision_boxes=False):
    """ Generate a triangular mesh representing a surface of 3D shape.
    Yields tuples (vertices, indices). """
    obj.check_dimension(required=3)

    program_buffer, max_box_size, boxes = subdivision.subdivision(
        obj, obj.feature_size() / 2, grid_size=subdivision_grid_size
    )

    block = numpy.empty(max_box_size, dtype=numpy.float32)
    block_buffer = pyopencl.Buffer(
        opencl_manager.context, pyopencl.mem_flags.WRITE_ONLY, block.nbytes
    )

    for i, (box_size, box_corner, box_resolution, *_) in enumerate(boxes):
        if debug_subdivision_boxes:
            # Export just an outline of the block instead of displaying its contents
            vertices = [
                util.Vector(i, j, k).elementwise_mul(box_size) * box_resolution
                + box_corner
                for k in range(2)
                for j in range(2)
                for i in range(2)
            ]
            triangles = [
                [0, 3, 1],
                [0, 2, 3],
                [1, 3, 5],
                [3, 7, 5],
                [4, 5, 6],
                [5, 7, 6],
                [0, 6, 2],
                [0, 4, 6],
                [0, 1, 5],
                [0, 5, 4],
                [3, 2, 6],
                [3, 6, 7],
            ]
            yield vertices, triangles
            continue

        # TODO: Staggered opencl / python processing the way subdivision does it.
        ev = opencl_manager.k.grid_eval_pymcubes(
            box_size,
            None,
            program_buffer,
            box_corner.as_float4(),
            numpy.float32(box_resolution),
            block_buffer,
        )
        pyopencl.enqueue_copy(opencl_manager.queue, block, block_buffer, wait_for=[ev])

        vertices, triangles = mcubes.marching_cubes(block, 0)

        if len(triangles) == 0:
            continue

        vertices[:, [0, 1]] = vertices[:, [1, 0]]
        vertices[:, 1] *= -1
        vertices *= box_resolution
        vertices += box_corner
        triangles[:, [0, 1]] = triangles[:, [1, 0]]

        yield vertices, triangles
