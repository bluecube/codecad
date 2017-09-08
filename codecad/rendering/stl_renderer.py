import numpy
import stl.mesh

from . import mesh
from .. import util


def render_stl(obj, filename, resolution):
    with util.status_block("generating mesh"):
        pieces = list(mesh.triangular_mesh(obj, resolution))
    triangle_count = sum(len(piece[1]) for piece in pieces)

    with util.status_block("exporting {} triangles".format(triangle_count)):
        stl_mesh = stl.mesh.Mesh(numpy.zeros(triangle_count, dtype=stl.mesh.Mesh.dtype))
        i = 0
        for vertices, indices in pieces:
            for triangle in indices:
                for j, vertex in enumerate(triangle):
                    v = vertices[vertex]
                    stl_mesh.vectors[i, j] = v
                i += 1

    with util.status_block("saving"):
        stl_mesh.save(filename)
