from .. import util
from ..compute import grid_eval
import numpy
import matplotlib
import matplotlib.pyplot as plt

def render_slice(obj,
                 resolution,
                 filename=None # For interface compatibility with other renderers
                 ):

    box = obj.bounding_box().expanded(0.1).flattened()

    with util.status_block("running"):
        values, corner, step = grid_eval.grid_eval(obj, resolution, box)

    values = values.reshape((values.shape[0], values.shape[1]))

    values_range = numpy.max(numpy.abs(values))

    with util.status_block("plotting"):
        plt.imshow(values,
                   cmap=plt.get_cmap("seismic"),
                   norm=matplotlib.colors.SymLogNorm(0.1,
                                                     vmin=-values_range,
                                                     vmax=values_range),
                   origin="upper",
                   interpolation="none",
                   aspect="equal",
                   extent=(corner.x, corner.x + (values.shape[1] - 1) * step,
                           corner.y, corner.y + (values.shape[0] - 1) * step))
    plt.show()
