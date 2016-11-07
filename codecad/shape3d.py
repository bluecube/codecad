import math

from . import util
from . import shape
from . import common

class Shape3D(shape.ShapeBase):
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
            offset = x
        else:
            offset = util.Vector(x, y, z)
        return Translation(self, offset)

    def rotated(self, vector, angle):
        """ Returns current shape rotated by an angle around the vector """
        return Rotation(self, util.Vector(*vector), angle)

    def scaled(self, s):
        """ Returns current shape scaled by given ratio """
        return Scaling(self, s)

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


class Union(common.Union, Shape3D):
    pass


class Intersection(common.Intersection, Shape3D):
    pass


class Subtraction(common.Subtraction, Shape3D):
    pass


class Translation(common.Translation, Shape3D):
    pass


class Scaling(common.Scaling, Shape3D):
    pass


class Shell(common.Shell, Shape3D):
    pass


class Rotation(Shape3D):
    def __init__(self, s, axis, angle):
        self.check_dimension(s)
        self.s = s
        phi = -util.radians(angle) / 2
        self.quat = util.Quaternion(axis.normalized() * util.sin(phi), util.cos(phi))

    def distance(self, point):
        return self.s.distance(self.quat.rotate_vector(point))

    def bounding_box(self):
        b = self.s.bounding_box()
        if any(math.isinf(x) for x in b.a) or any(math.isinf(x) for x in b.b):
            # Special case for rotating infinite objects.
            # TODO: Make even more special cases for axis aligned rotations and 90 degree
            # rotations.
            inf = util.Vector(float("inf"), float("inf"), float("inf"))
            return util.BoundingBox(-inf, inf)
        else:
            inv_quat = self.quat.conjugate()
            return util.BoundingBox.containing(inv_quat.rotate_vector(v) for v in b.vertices())


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
