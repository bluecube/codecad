""" Shapes and operations that are only safe under certain (not checked) conditions.
Read docstrings of the individual objects. """

import math

from .. import util
from . import base
from ..cl_util import opencl_manager

_c_file = opencl_manager.add_compile_unit()
_c_file.append_file("common.h")
_c_file.append_file("unsafe.cl")


class _RepetitionMixin:
    """ Infinite repetition of an object along X, Y and Z axis.
    If spacing along an axis is not infinite, then the object should be symmetrical
    about a plane perpendicular to that axis going through origin, and its
    bounding box size should be smaller than the spacing along that axis. """
    # TODO: At least minimal checks generating warnings

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
        return util.BoundingBox(util.Vector(*(x if s is None else -float("inf") for x, s in zip(b.a, self.spacing))),
                                util.Vector(*(x if s is None else float("inf") for x, s in zip(b.b, self.spacing))))

    def get_node(self, point, cache):
        return self.s.get_node(cache.make_node("repetition",
                                               self.spacing,
                                               [point]),
                               cache)


class Repetition2D(_RepetitionMixin, base.Shape2D):
    pass


class Repetition(_RepetitionMixin, base.Shape3D):
    pass


class _CircularRepetitionMixin:
    """ Repeats object by rotating it around the Z axis n times in regular intervals """
    def __init__(self, s, n):
        self.check_dimension(s)
        self.s = s
        self.n = n

    def bounding_box(self):
        v = util.Vector.splat(self.s.bounding_box().b.x)
        return util.BoundingBox(-v, v)

    def get_node(self, point, cache):
        pi_over_n = math.pi / self.n
        new_point = cache.make_node("circular_repetition_to",
                                    [pi_over_n],
                                    [point])
        sub_node = self.s.get_node(new_point, cache)
        return cache.make_node("circular_repetition_from",
                               [pi_over_n],
                               [sub_node, point])


class CircularRepetition2D(_CircularRepetitionMixin, base.Shape2D):
    pass


class CircularRepetition(_CircularRepetitionMixin, base.Shape3D):
    pass

class Flatten(base.Shape2D):
    """ Converts 3D shape to 2D shape by taking a slice at Z = 0.
    Doesn't correct directions or distances. """
    def __init__(self, s):
        self.check_dimension(s, required=3)
        self.s = s

    def bounding_box(self):
        return self.s.bounding_box()

    def get_node(self, point, cache):
        return self.s.get_node(point, cache)
