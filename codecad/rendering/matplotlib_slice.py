from .. import util
import theano
import theano.tensor as T
import numpy
import matplotlib
import matplotlib.pyplot as plt

def render_slice(obj,
                 resolution,
                 filename=None # For interface compatibility with other renderers
                 ):
    with util.status_block("calculating bounding box"):
        box = obj.bounding_box().expanded(0.1)

    box_size = box.b - box.a

    x = T.matrix("x")
    y = T.matrix("y")
    z = T.zeros_like(x)

    with util.status_block("building expression"):
        distances = obj.distance(util.Vector(x, y, z))

    dist_range = T.max(abs(distances))

    with util.status_block("compiling"):
        f = theano.function([x, y], (distances, dist_range))

    with util.status_block("running"):
        resolution_vector = util.Vector(resolution, resolution, resolution)

        box_a = box.a - resolution_vector
        box_b = box.b + resolution_vector * 2

        values, values_range = f(*numpy.meshgrid(numpy.arange(box_a.x, box_b.x, resolution),
                                                 numpy.arange(box_a.y, box_b.y, resolution)))

    print("plotting")
    plt.imshow(values,
               cmap=plt.get_cmap("seismic"),
               norm=matplotlib.colors.SymLogNorm(0.1,
                                                 vmin=-values_range,
                                                 vmax=values_range),
               origin="lower",
               interpolation="none",
               aspect="equal",
               extent=(box_a.x, box_b.x, box_a.y, box_b.y))
    plt.show()
