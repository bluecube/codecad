import theano
import theano.tensor as T
import numpy
import mcubes
import stl.mesh

from .. import util
from .. import shapes

class StlRenderer:
    def __init__(self, filename, resolution):
        self.filename = filename
        self.resolution = resolution

    def render(self, shape):
        box = shape.bounding_box()
        box_size = box.b - box.a

        x = T.tensor3("x")
        y = T.tensor3("y")
        z = T.tensor3("z")

        print("compiling...")
        f = theano.function([x, y, z],
                            shape.distance_estimate(util.Vector(x, y, z)),
                            givens=[(shapes.Shape.Epsilon, self.resolution)],
                            on_unused_input = 'ignore') # Epsilon might not be used
        print("running...")

        resolution = util.Vector(self.resolution, self.resolution, self.resolution)

        box_a = box.a - resolution
        box_b = box.b + resolution * 2

        xs, ys, zs = numpy.meshgrid(numpy.arange(box_a.x, box_b.x, self.resolution),
                                    numpy.arange(box_a.y, box_b.y, self.resolution),
                                    numpy.arange(box_a.z, box_b.z, self.resolution))

        values = f(xs, ys, zs)

        print("marching cubes...")

        vertices, triangles = mcubes.marching_cubes(values, 0)

        print("exporting ...")

        mesh = stl.mesh.Mesh(numpy.empty(triangles.shape[0], dtype=stl.mesh.Mesh.dtype))
        for i, f in enumerate(triangles):
            for j in range(3):
                mesh.vectors[i][2 -  j] = vertices[f[j],:]

        mesh.save(self.filename)
