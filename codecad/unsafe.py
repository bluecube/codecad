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

    def __init__(self, s, spacing):
        """
        spacing - tuple of distances along axes, if value is None, the object is not repeated along this axis
        """
        self.s = s
        self.spacing = util.Vector(*spacing)

    def _coordinate(self, x, spacing):
        if spacing is None:
            return x
        else:
            return x - util.round(x / spacing) * spacing

    def distance(self, point):
        p = util.Vector(*(self._coordinate(x, s) for x, s in zip(point, self.spacing)))
        return self.s.distance(p)

    def bounding_box(self):
        b = self.s.bounding_box()
        return util.BoundingBox(util.Vector(*(x if s is None else s for x, s in zip(b.a, self.spacing))),
                                util.Vector(*(x if s is None else s for x, s in zip(b.b, self.spacing))))
