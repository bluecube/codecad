import math

from .. import util
from . import base
from . import common
from ..cl_util import opencl_manager

_c_file = opencl_manager.add_compile_unit()
_c_file.append_resource("common.h")
_c_file.append_resource("simple2d.cl")


class Rectangle(base.Shape2D):
    def __init__(self, x=1, y=None):
        if y is None:
            y = x
        self.half_size = util.Vector(x, y) / 2

    def bounding_box(self):
        return util.BoundingBox(-self.half_size, self.half_size)

    def feature_size(self):
        return 2 * min(self.half_size.x, self.half_size.y)

    def get_node(self, point, cache):
        return cache.make_node(
            "rectangle", [self.half_size.x, self.half_size.y], [point]
        )


class Circle(base.Shape2D):
    def __init__(self, d=1, r=None):
        if r is None:
            self.d = d
            self.r = d / 2
        else:
            self.r = r
            self.d = 2 * r

    def bounding_box(self):
        v = util.Vector(self.r, self.r)
        return util.BoundingBox(-v, v)

    def feature_size(self):
        return self.d

    def get_node(self, point, cache):
        return cache.make_node("circle", [self.r], [point])


class HalfPlane(base.Shape2D):
    """ Half space y > 0. """

    def bounding_box(self):
        return util.BoundingBox(
            util.Vector(-float("inf"), 0), util.Vector(float("inf"), float("inf"))
        )

    def feature_size(self):
        return float("inf")

    def get_node(self, point, cache):
        return cache.make_node("half_space", [], [point])


class RegularPolygon2D(base.Shape2D):
    def __init__(self, n, d=1, r=None, side_length=None, across_flats=None):
        self.n = n

        if not util.at_most_one(
            [d != 1, r is not None, side_length is not None, across_flats is not None]
        ):
            raise ValueError(
                "At most one of d, r, side_length and across_flats can be used at the same time"
            )

        self.d = None
        self.r = None
        self.side_length = None
        self.across_flats = None

        if across_flats is not None:
            self.across_flats = across_flats
            if n % 2:  # Odd number of sides
                self.r = across_flats / (math.cos(math.pi / n) + 1)
            else:  # Odd number of sides
                self.r = across_flats / (2 * math.cos(math.pi / n))
            self.d = 2 * self.r

        elif side_length is not None:
            self.side_length = side_length
            self.d = side_length / math.sin(math.pi / n)
            self.r = self.d / 2

        elif r is not None:
            self.r = r
            self.d = 2 * r

        else:
            self.d = d
            self.r = d / 2

        if self.across_flats is None:
            if n % 2:  # Odd number of sides
                self.across_flats = self.r * (math.cos(math.pi / n) + 1)
            else:  # Odd number of sides
                self.across_flats = self.r * (2 * math.cos(math.pi / n))

        if self.side_length is None:
            self.side_length = self.d * math.sin(math.pi / n)

    @staticmethod
    def calculate_n(r, side_length):
        """ Calculate n that would make given radius and side_length work.

        Note that this will typically not be an integer and must be appropriately
        rounded before use. """
        return math.pi / math.asin(side_length / (2 * r))

    def bounding_box(self):
        v = util.Vector(self.r, self.r)
        return util.BoundingBox(-v, v)

    def feature_size(self):
        return self.side_length

    def get_node(self, point, cache):
        return cache.make_node("regular_polygon2d", [math.pi / self.n, self.r], [point])


class Union2D(common.UnionMixin, base.Shape2D):
    pass


class Intersection2D(common.IntersectionMixin, base.Shape2D):
    pass


class Subtraction2D(common.SubtractionMixin, base.Shape2D):
    pass


class Offset2D(common.OffsetMixin, base.Shape2D):
    pass


class Shell2D(common.ShellMixin, base.Shape2D):
    pass


class Transformation2D(common.TransformationMixin, base.Shape2D):
    def bounding_box(self):
        b = self.s.bounding_box().flattened()

        if any(math.isinf(x) for x in b.a) or any(math.isinf(x) for x in b.b):
            # Special case for rotating infinite objects.
            inf = util.Vector(float("inf"), float("inf"))
            return util.BoundingBox(-inf, inf)
        else:
            ret = util.BoundingBox.containing(
                self.transformation.transform_vector(v) for v in b.vertices()
            )
            return util.BoundingBox(
                util.Vector(ret.a.x, ret.a.y), util.Vector(ret.b.x, ret.b.y)
            )


class Mirror2D(common.MirrorMixin, base.Shape2D):
    pass


class Symmetrical2D(common.SymmetricalMixin, base.Shape2D):
    pass
