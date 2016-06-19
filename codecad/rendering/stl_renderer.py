import theano
import theano.tensor as T
import numpy
import mcubes
import stl.mesh

from .. import util
from .. import shapes

def render_stl(obj, filename, resolution):
    box = obj.bounding_box()
    box_size = box.b - box.a

    x = T.tensor3("x")
    y = T.tensor3("y")
    z = T.tensor3("z")

    with util.status_block("compiling"):
        f = theano.function([x, y, z], obj.distance(util.Vector(x, y, z)))

    with util.status_block("running"):
        resolution_vector = util.Vector(resolution, resolution, resolution)

        box_a = box.a - resolution_vector
        box_b = box.b + resolution_vector * 2

        xx = numpy.arange(box_a.x, box_b.x, resolution)
        yy = numpy.arange(box_a.y, box_b.y, resolution)
        zz = numpy.arange(box_a.z, box_b.z, resolution)
        xs, ys, zs = numpy.meshgrid(numpy.arange(box_a.x, box_b.x, resolution),
                                    numpy.arange(box_a.y, box_b.y, resolution),
                                    numpy.arange(box_a.z, box_b.z, resolution))

        values = f(xs, ys, zs)

    with util.status_block("marching cubes"):
        vertices, triangles = mcubes.marching_cubes(values, 0)

    with util.status_block("exporting {} triangles".format(len(triangles))):
        mesh = stl.mesh.Mesh(numpy.empty(triangles.shape[0], dtype=stl.mesh.Mesh.dtype))
        for i, f in enumerate(triangles):
            for j in range(3):
                mesh.vectors[i][2 - j] = resolution * vertices[f[j],:]

    with util.status_block("saving"):
        mesh.save(filename)
