import theano
import theano.tensor as T
import numpy
import mcubes
import stl.mesh

from .. import util

def render_stl(obj, filename, resolution):
    obj.check_dimension(required = 3)
    with util.status_block("calculating bounding box"):
        box = obj.bounding_box().expanded_additive(resolution)

    with util.status_block("building expression"):
        distances = obj.distance(util.theano_box_grid(box, resolution))

    with util.status_block("compiling"):
        f = theano.function([], distances)

    with util.status_block("running"):
        values = f()

    with util.status_block("marching cubes"):
        vertices, triangles = mcubes.marching_cubes(values, 0)

    with util.status_block("exporting {} triangles".format(len(triangles))):
        mesh = stl.mesh.Mesh(numpy.empty(triangles.shape[0], dtype=stl.mesh.Mesh.dtype))
        for i, f in enumerate(triangles):
            mesh.vectors[i][0] = resolution * vertices[f[1],:]
            mesh.vectors[i][1] = resolution * vertices[f[0],:]
            mesh.vectors[i][2] = resolution * vertices[f[2],:]

    with util.status_block("saving"):
        mesh.save(filename)
