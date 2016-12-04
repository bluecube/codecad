import math

from . import base
from .. import util

class Shape3D(base.ShapeBase):
    """ A base 3D shape. """

    @staticmethod
    def dimension():
        return 3

    def __and__(self, second):
        return Intersection([self, second])

    def __add__(self, second):
        return Union([self, second])

    def __sub__(self, second):
        return Subtraction(self, second)

    def translated(self, x, y = None, z = None):
        """ Returns current shape translated by a given offset """
        if isinstance(x, util.Vector):
            if y is not None or z is not None:
                raise TypeError("If first parameter is Vector, the others must be left unspecified.")
            o = x
        else:
            o = util.Vector(x, y, z)
        return Transformation.make_merged(self,
                                          util.Quaternion.from_degrees(util.Vector(0, 0, 1), 0),
                                          o)

    def rotated(self, vector, angle):
        """ Returns current shape rotated by an angle around the vector """
        return Transformation.make_merged(self,
                                          util.Quaternion.from_degrees(util.Vector(*vector), angle),
                                          util.Vector(0, 0, 0))

    def scaled(self, s):
        """ Returns current shape scaled by given ratio """
        return Transformation.make_merged(self,
                                          util.Quaternion.from_degrees(util.Vector(0, 0, 1), 0, s),
                                          util.Vector(0, 0, 0))

    def shell(self, inside, outside):
        """ Returns a shell of the current shape"""
        return Shell(self, inside, outside)


class Box(Shape3D):
    def __init__(self, x = 1, y = None, z = None):
        if y is None:
            y = x
        if z is None:
            z = x
        self.half_size = util.Vector(x, y, z) / 2

    def distance(self, point):
        v = point.elementwise_abs() - self.half_size
        return v.max()

    def bounding_box(self):
        return util.BoundingBox(-self.half_size, self.half_size)

    def get_node(self, point, cache):
        return cache.make_node("box", self.half_size, [point])


class Sphere(Shape3D):
    def __init__(self, d = 1, r = None):
        if r is None:
            self.r = d / 2
        else:
            self.r = r

    def distance(self, point):
        return abs(point) - self.r;

    def bounding_box(self):
        v = util.Vector(self.r, self.r, self.r)
        return util.BoundingBox(-v, v)

    def get_node(self, point, cache):
        return cache.make_node("sphere", [self.r], [point])

class Cylinder(Shape3D):
    def __init__(self, h = 1, d = 1, r = None):
        self.h = h
        if r is not None:
            self.r = r
        else:
            self.r = d / 2

    def distance(self, point):
        infinite_cylinder = util.sqrt(point.x * point.x + point.y * point.y) - self.r

        if math.isinf(self.h):
            return infinite_cylinder
        else:
            return util.maximum(infinite_cylinder,
                                abs(point.z) - self.h / 2)

    def bounding_box(self):
        v = util.Vector(self.r, self.r, self.h)
        return util.BoundingBox(-v, v)

    def get_node(self, point, cache):
        return cache.make_node("cylinder", [self.r, self.h], [point])


class Union(base.Union, Shape3D):
    pass


class Intersection(base.Intersection, Shape3D):
    pass


class Subtraction(base.Subtraction, Shape3D):
    pass


class Shell(base.Shell, Shape3D):
    pass


class Transformation(base.Transformation, Shape3D):
    def bounding_box(self):
        b = self.s.bounding_box()

        if any(math.isinf(x) for x in b.a) or any(math.isinf(x) for x in b.b):
            # Special case for rotating infinite objects.
            inf = util.Vector(float("inf"), float("inf"), float("inf"))
            return util.BoundingBox(-inf, inf)
        else:
            return util.BoundingBox.containing(self.transform_vector(v) for v in b.vertices())

    def get_node(self, point, cache):
        return self.s.get_node(cache.make_node("rotation3d",
                                               list(self.quat.v) + [self.quat.w],
                                               [point]),
                               cache)

class Extrusion(Shape3D):
    def __init__(self, s, height):
        self.check_dimension(s, required=2)
        self.s = s
        self.h = height

    def distance(self, point):
        return util.maximum(self.s.distance(point.flattened()), abs(point.z) - self.h / 2)

    def bounding_box(self):
        box = self.s.bounding_box()
        return util.BoundingBox(util.Vector(box.a.x, box.a.y, -self.h / 2),
                                util.Vector(box.b.x, box.b.y, self.h / 2))

    def get_node(self, point, cache):
        return cache.make_node("extrusion",
                               [self.h],
                               [self.s.get_node(point, cache)])

class Revolution(Shape3D):
    def __init__(self, s):
        self.check_dimension(s, required=2)
        self.s = s

    def distance(self, point):
        new_point = util.Vector(util.sqrt(point.x * point.x + point.z * point.z),
                                    point.y,
                                    0)
        return self.s.distance(new_point)

    def bounding_box(self):
        box = self.s.bounding_box()
        radius = util.maximum(-box.a.x, box.b.x)
        return util.BoundingBox(util.Vector(-radius, box.a.y, -radius),
                                util.Vector(radius, box.b.y, radius))

    def get_node(self, point, cache):
        return cache.make_node("extrusion",
                               [],
                               [self.s.get_node(point, cache)])
