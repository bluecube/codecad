""" Shapes and operations that are only safe under certain (not checked) conditions.
Read docstrings of the individual objects. """

import math

from . import simple2d
from . import simple3d
from .. import util

class _Repetition:
    """ Infinite repetition of an object along X, Y and Z axis.
    If spacing along an axis is not infinite, then the object should be symmetrical
    about a plane perpendicular to that axis going through origin, and its
    bounding box size should be smaller than the spacing along that axis. """
    #TODO: At least minimal checks generating warnings

    def __init__(self, s, spacing):
        """
        spacing - tuple of distances along axes, if value is 0 or None, the object is not repeated along this axis
        """
        self.check_dimension(s)
        self.s = s
        self.spacing = util.Vector(*(float("inf") if x is None or x == 0 else x for x in spacing))

        if (self.dimension() == 2 and self.spacing[2] != float("inf")):
            raise ValueError("Attempting repetition along Z axis for 2D shape")

    def bounding_box(self):
        b = self.s.bounding_box()
        return util.BoundingBox(util.Vector(*(x if s is None else s for x, s in zip(b.a, self.spacing))),
                                util.Vector(*(x if s is None else s for x, s in zip(b.b, self.spacing))))

    def get_node(self, point, cache):
        return self.s.get_node(cache.make_node("repetition",
                                               self.spacing,
                                               [point]),
                               cache)


class Repetition2D(_Repetition, simple2d.Shape2D):
    pass


class Repetition(_Repetition, simple3d.Shape3D):
    pass
