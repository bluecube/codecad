import numpy
import mcubes
import stl.mesh

from ..compute import grid_eval
from .. import util

def render_stl(obj, filename, resolution):
    obj.check_dimension(required = 3)

    with util.status_block("running"):
        values, corner, step = grid_eval.grid_eval(obj, resolution)

    values = numpy.transpose(values, (1, 2, 0))

    with util.status_block("marching cubes"):
        vertices, triangles = mcubes.marching_cubes(values, 0)

    with util.status_block("exporting {} triangles".format(len(triangles))):
        mesh = stl.mesh.Mesh(numpy.empty(triangles.shape[0], dtype=stl.mesh.Mesh.dtype))
        for i, f in enumerate(triangles):
            mesh.vectors[i][0] = step * vertices[f[1],:]
            mesh.vectors[i][1] = step * vertices[f[0],:]
            mesh.vectors[i][2] = step * vertices[f[2],:]

    with util.status_block("saving"):
        mesh.save(filename)
