import math
import numpy

from .. import util
from . import base
from . import mixins

class Rectangle(base.Shape2D):
    def __init__(self, x = 1, y = None):
        if y is None:
            y = x
        self.half_size = util.Vector(x, y) / 2

    def bounding_box(self):
        return util.BoundingBox(-self.half_size, self.half_size)

    def get_node(self, point, cache):
        return cache.make_node("rectangle", [self.half_size.x, self.half_size.y], [point])

class Circle(base.Shape2D):
    def __init__(self, d = 1, r = None):
        if r is None:
            self.r = d / 2
        else:
            self.r = r

    def bounding_box(self):
        v = util.Vector(self.r, self.r)
        return util.BoundingBox(-v, v)

    def get_node(self, point, cache):
        return cache.make_node("circle", [self.r], [point])


class HalfPlane(base.Shape2D):
    """ Half space y > 0. """
    def bounding_box(self):
        return util.BoundingBox(util.Vector(-float("inf"), 0, -float("inf")),
                                util.Vector.splat(float("inf")));

    def get_node(self, point, cache):
        return cache.make_node("half_space", [], [point])


class Polygon2D(base.Shape2D):
    def __init__(self, points):
        self.points = numpy.asarray(points, dtype=numpy.float32, order="c")
        s = self.points.shape
        if len(s) != 2 or s[1] != 2:
            raise ValueError("points must be a list of (x, y) pairs or array with shape (x, 2)")

        self.box = util.BoundingBox(util.Vector(*numpy.amin(self.points, axis=0)),
                                    util.Vector(*numpy.amax(self.points, axis=0)))

    def bounding_box(self):
        return self.box

    def get_node(self, point, cache):
        return cache.make_node("polygon2d",
                               util.Concatenate([len(self.points)], self.points.flat),
                               [point])


class Union2D(mixins.UnionMixin, base.Shape2D):
    pass


class Intersection2D(mixins.IntersectionMixin, base.Shape2D):
    pass


class Subtraction2D(mixins.SubtractionMixin, base.Shape2D):
    pass


class Offset2D(mixins.OffsetMixin, base.Shape2D):
    pass


class Shell2D(mixins.ShellMixin, base.Shape2D):
    pass


class Transformation2D(mixins.TransformationMixin, base.Shape2D):
    def bounding_box(self):
        b = self.s.bounding_box().flattened()

        if any(math.isinf(x) for x in b.a) or any(math.isinf(x) for x in b.b):
            # Special case for rotating infinite objects.
            inf = util.Vector(float("inf"), float("inf"), float("inf"))
            return util.BoundingBox(-inf, inf)
        else:
            inf = float("inf")
            ret = util.BoundingBox.containing(self.transformation.transform_vector(v) for v in b.vertices())
            return util.BoundingBox(util.Vector(ret.a.x, ret.a.y, -inf),
                                    util.Vector(ret.b.x, ret.b.y, inf))
