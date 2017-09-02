import math
import numpy

from .. import util
from . import base
from . import common
from .. import opencl_manager

_c_file = opencl_manager.instance.add_compile_unit()
_c_file.append_file("common.h")
_c_file.append_file("simple2d.cl")

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
    """ 2D simple (without self intersections) polygon.
    First and last point are implicitly connected. """

    def __init__(self, points):
        self.points = numpy.asarray(points, dtype=numpy.float32, order="c")
        s = self.points.shape
        if len(s) != 2 or s[1] != 2:
            raise ValueError("points must be a list of (x, y) pairs or array with shape (x, 2)")
        if s[0] < 3:
            raise ValueError("Polygon must have at least three vertices")

        area = 0
        minimum = util.Vector.splat(float("inf"))
        maximum = -minimum
        previous = util.Vector(*self.points[-1])
        for i, current in enumerate(self.points):
            current = util.Vector(*current)
            direction = current - previous
            direction_abs_squared = direction.abs_squared()
            perpendicular_direction = direction.perpendicular2d()

            area += direction.x * (previous.y + current.y) / 2
                # TODO: Use better precision sum once we have it coded

            minimum = minimum.min(current)
            maximum = maximum.max(current)

            if direction_abs_squared == 0:
                raise ValueError("Zero length segments are not allowed in polygon")

            inner_previous = current
            has_intersection = False
            for j, inner_current in enumerate(self.points[i + 1:]):
                inner_current = util.Vector(*inner_current)
                inner_direction = inner_current - inner_previous

                inner_perpendicular_direction = inner_direction.perpendicular2d()

                direction_cross_product = inner_direction.dot(perpendicular_direction)
                between_starts = (previous - inner_previous)

                consecutive = j == 0 or j == len(points) - 2 # i precedes or follows j
                parallel = abs(direction_cross_product) < 1e-12

                if consecutive:
                    # Consecutive segments always intersect, but we must check that they are
                    # not collinear and in opposite direction
                    if parallel and direction.dot(inner_direction) < 0:
                        raise ValueError("Polygon cannot be self intersecting (anti-parallel consecutive edges)")
                else:
                    if parallel:
                        if abs(between_starts.dot(perpendicular_direction)) < 1e-12:
                            # collinear
                            scaled_direction = direction / direction_abs_squared
                            t1 = between_starts.dot(scaled_direction)
                            t2 = t1 + inner_direction.dot(scaled_direction)

                            t1, t2 = sorted([t1, t2])

                            if t2 >= 0 and t1 <= 1:
                                raise ValueError("Polygon cannot be self intersecting (colinear segments)")

                        else:
                            # parallel
                            pass
                    else:
                        tmp = between_starts / direction_cross_product

                        t1 = tmp.dot(inner_perpendicular_direction)
                        t2 = tmp.dot(perpendicular_direction)

                        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                            raise ValueError("Polygon cannot be self intersecting")

                inner_previous = inner_current

            previous = current

        if area < 0:
            self.points = numpy.flipud(self.points)

        self.box = util.BoundingBox(minimum, maximum)

    def bounding_box(self):
        return self.box

    def get_node(self, point, cache):
        return cache.make_node("polygon2d",
                               util.Concatenate([len(self.points)], self.points.flat),
                               [point])


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
            inf = util.Vector(float("inf"), float("inf"), float("inf"))
            return util.BoundingBox(-inf, inf)
        else:
            inf = float("inf")
            ret = util.BoundingBox.containing(self.transformation.transform_vector(v) for v in b.vertices())
            return util.BoundingBox(util.Vector(ret.a.x, ret.a.y, -inf),
                                    util.Vector(ret.b.x, ret.b.y, inf))
