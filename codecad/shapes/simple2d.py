import math

from .. import util
from . import base
from . import simple3d

class Shape2D(base.ShapeBase):
    """ A base 2D shape. """

    @staticmethod
    def dimension():
        return 2

    def __and__(self, second):
        return Intersection2D([self, second])

    def __add__(self, second):
        return Union2D([self, second])

    def __sub__(self, second):
        return Subtraction2D(self, second)

    def translated(self, x, y = None):
        """ Returns current shape translated by a given offset """
        if isinstance(x, util.Vector):
            if y is not None:
                raise TypeError("If first parameter is Vector, the others must be left unspecified.")
            offset = x.flattened()
        else:
            if y is None:
                raise ValueError("Y coordinate can only be missing if first parameter is a Vector.")
            offset = util.Vector(x, y)
        return Translation2D(self, offset)

    def rotated(self, angle):
        """ Returns current shape rotated by given angles """
        return Rotation2D(self, angle)

    def scaled(self, s):
        """ Returns current shape scaled by given ratio """
        return Scaling2D(self, s)

    def shell(self, inside, outside):
        """ Returns a shell of the current shape"""
        return Shell2D(self, inside, outside)

    def extruded(self, height):
        return simple3d.Extrusion(self, height)

    def revolved(self):
        """ Returns current shape taken as 2D in xy plane and revolved around y axis """
        return simple3d.Revolution(self)


class Rectangle(Shape2D):
    def __init__(self, x = 1, y = None):
        if y is None:
            y = x
        self.half_size = util.Vector(x, y) / 2

    def distance(self, point):
        v = point.flattened().elementwise_abs() - self.half_size
        return util.maximum(v.x, v.y)

    def bounding_box(self):
        return util.BoundingBox(-self.half_size, self.half_size)

    def get_node(self, point, cache):
        return cache.make_node("rectangle", self.half_size, [point])

class Circle(Shape2D):
    def __init__(self, d = 1, r = None):
        if r is None:
            self.r = d / 2
        else:
            self.r = r

    def distance(self, point):
        return abs(point.flattened()) - self.r;

    def bounding_box(self):
        v = util.Vector(self.r, self.r, float("inf"))
        return util.BoundingBox(-v, v)

    def get_node(self, point, cache):
        return cache.make_node("rectangle", [self.r], [point])


class Union2D(base.Union, Shape2D):
    pass


class Intersection2D(base.Intersection, Shape2D):
    pass


class Subtraction2D(base.Subtraction, Shape2D):
    pass


class Translation2D(base.Translation, Shape2D):
    pass


class Scaling2D(base.Scaling, Shape2D):
    pass


class Shell2D(base.Shell, Shape2D):
    pass


class Rotation2D(Shape2D):
    def __init__(self, s, angle):
        self.check_dimension(s)
        self.s = s
        phi = -util.radians(angle)
        self.cos = util.cos(phi)
        self.sin = util.sin(phi)

    def distance(self, point):
        v = util.Vector(point.x * self.cos - point.y * self.sin,
                        point.x * self.sin + point.y * self.cos,
                        0)
        return self.s.distance(v)

    def bounding_box(self):
        b = self.s.bounding_box()
        if any(math.isinf(x) for x in [b.a.x, b.a.y, b.b.x, b.b.y]):
            # Special case for rotating infinite objects.
            # TODO: Make even more special cases for axis aligned rotations and 90 degree
            # rotations.
            inf = util.Vector(float("inf"), float("inf"), float("inf"))
            return util.BoundingBox(-inf, inf)
        else:
            def rotate(point):
                return util.Vector(point.x * self.cos + point.y * self.sin,
                                   -point.x * self.sin + point.y * self.cos,
                                   0)
            return util.BoundingBox.containing(rotate(v) for v in b.vertices())

    def get_node(self, point, cache):
        return self.s.get_node(cache.make_node("rotation2d",
                                               [self.cos, self.sin],
                                               [point]),
                               cache)
