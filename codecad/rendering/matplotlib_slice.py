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

    box = box.expanded(1.1)

    # Flatten the box in the Z axis
    box = util.BoundingBox(util.Vector(box.a.x, box.a.y, 0),
                           util.Vector(box.b.x, box.b.y, 0))

    with util.status_block("building expression"):
        coords = util.theano_box_grid(box, resolution)
        distances = obj.distance(coords)[:,:,0]

    dist_range = T.max(abs(distances))

    with util.status_block("compiling"):
        f = theano.function([], (distances, dist_range))

    with util.status_block("running"):
        values, values_range = f()

    print(values.shape)

    print("plotting")
    plt.imshow(values,
               cmap=plt.get_cmap("seismic"),
               norm=matplotlib.colors.SymLogNorm(0.1,
                                                 vmin=-values_range,
                                                 vmax=values_range),
               origin="lower",
               interpolation="none",
               aspect="equal",
               extent=(box.a.x, box.b.x, box.a.y, box.b.y))
    plt.show()
