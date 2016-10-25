from .geometry import *
from .misc import *
from .theanomath import (is_theano, maximum, minimum, sqrt, round,
                         sin, cos, arcsin, arctan, switch, radians)

from . import derivatives

import theano.tensor as T

def theano_meshgrid(*dimensions):
    ret = []
    for i, dim in enumerate(dimensions):
        spec = ['x'] * len(dimensions)
        spec[i] = 0
        ret.append(T.arange(dim).dimshuffle(*spec))
    return ret

def theano_box_grid(box, resolution):
    """ Return a vector of 3D tensors that can be broadcast together to cover
    box with a regular grid. """
    size = box.size().applyfunc(lambda x: x // resolution + 1)
    tensors = Vector(*theano_meshgrid(*size))
    tensors = (tensors - (size - Vector.splat(1)) / 2) * resolution
    return tensors
