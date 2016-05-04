""" Shapes and operations that are only safe under certain (not checked) conditions.
Read docstrings of the individual objects. """

import math

from theano import tensor as T

from . import shapes
from . import util

class Repetition(shapes.Shape):
    """ Infinite repetition of an object along X, Y and Z axis.
    If spacing along an axis is not infinite, then the object should be symmetrical
    about a plane perpendicular to that axis going through origin, and its
    bounding box size should be smaller than the spacing along that axis. """
    #TODO: At least minimal checks generating warnings

    def __init__(self, s, x_spacing, y_spacing=float("inf"), z_spacing=float("inf")):
        self.s = s
        self.spacing = util.Vector(x_spacing, y_spacing, z_spacing)

    def _coordinate(self, x, spacing):
        if math.isinf(spacing):
            return x
        else:
            return x - util.round(x / spacing) * spacing

    def distance(self, point):
        p = util.Vector(*(self._coordinate(x, s) for x, s in zip(point, self.spacing)))
        return self.s.distance(p)

    def bounding_box(self):
        b = self.s.bounding_box()
        return util.BoundingBox(util.Vector(*(x if math.isinf(s) else s for x, s in zip(b.a, self.spacing))),
                                util.Vector(*(x if math.isinf(s) else s for x, s in zip(b.b, self.spacing))))
