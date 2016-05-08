import math

import theano
import theano.tensor as T

from . import util

class Shape:
    """ A base 3D or 2D shape. """

    # This is a constant, to be set as given when creating the function
    Epsilon = T.fscalar("Epsilon")

    Outside = Epsilon
    Inside = -Epsilon

    def distance(point):
        """ Returns lower bound on distance between the given point and surface
        of the shape as a Theano expression.
        Must be overridden in subclasses.
        If distance cannot be determined, return Shape.Inside or Shape.Outside. """
        raise NotImplementedError()

    def bounding_box(point):
        """ Returns abox that contains the whole shape.
        Must be overridden by subclasses.
        The closer the wrapping, the faster the render. """
        raise NotImplementedError()

    def render(self, renderer):
        """ Render the shape using the given renderer. """
        renderer.render(self)

    def __or__(self, second):
        """ Returns union of the two shapes """
        return Union(self, second)

    def __add__(self, second):
        """ Returns union of the two shapes """
        return Union(self, second)

    def __and__(self, second):
        """ Returns intersection of the two shapes """
        return Intersection(self, second)

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
        """ Returns current shape rotated by given angles """
        return Rotation(self, util.Vector(*vector), angle)

    def scaled(self, s):
        """ Returns current shape scaled by given ratio """
        return Scaling(self, s)


class Box(Shape):
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


class Sphere(Shape):
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


class Cylinder(Shape):
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


class Union(Shape):
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def distance(self, point):
        return util.minimum(self.s1.distance(point),
                            self.s2.distance(point))

    def bounding_box(self):
        b1 = self.s1.bounding_box()
        b2 = self.s2.bounding_box()
        return util.BoundingBox(b1.a.min(b2.a), b1.b.max(b2.b))


class Intersection(Shape):
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def distance(self, point):
        return util.maximum(self.s1.distance(point),
                            self.s2.distance(point))

    def bounding_box(self):
        b1 = self.s1.bounding_box()
        b2 = self.s2.bounding_box()
        return util.BoundingBox(b1.a.max(b2.a), b1.b.min(b2.b))


class Subtraction(Shape):
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def distance(self, point):
        return util.maximum(self.s1.distance(point),
                            -self.s2.distance(point))

    def bounding_box(self):
        return self.s1.bounding_box()


class Translation(Shape):
    def __init__(self, s, offset):
        self.s = s
        self.offset = offset

    def distance(self, point):
        return self.s.distance(point - self.offset)

    def bounding_box(self):
        b = self.s.bounding_box()
        return util.BoundingBox(b.a + self.offset, b.b + self.offset)


class Rotation(Shape):
    def __init__(self, s, axis, angle):
        self.s = s
        phi = math.radians(angle) / 2
        self.quat = util.Quaternion(axis.normalized() * math.sin(phi), math.cos(phi))

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
            return util.BoundingBox.containing(self.quat.rotate_vector(v) for v in b.vertices())


class Scaling(Shape):
    def __init__(self, s, scale):
        self.s = s
        self.scale = scale

    def distance(self, point):
        return self.s.distance(point / self.scale) * self.scale

    def bounding_box(self):
        b = self.s.bounding_box()
        return util.BoundingBox(b.a * self.scale, b.b * self.scale)


class RoundedUnion(Shape):
    def __init__(self, s1, s2, r):
        self.s1 = s1
        self.s2 = s2
        self.r = r

    @staticmethod
    def rmin(a, b, r):
        return util.switch(abs(a - b) >= r,
                           util.minimum(a, b),
                           b + r * util.sin(math.pi / 4 + util.asin((a - b) / (r * math.sqrt(2)))) - r)

    def distance(self, point):
        return self.rmin(self.s1.distance(point), self.s2.distance(point), self.r)

    def bounding_box(self):
        b1 = self.s1.bounding_box()
        b2 = self.s2.bounding_box()
        return util.BoundingBox(b1.a.min(b2.a), b1.b.max(b2.b))

class Extrude(Shape):
    pass
    #TODO
