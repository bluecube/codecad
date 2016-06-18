import math
import functools

import theano
import theano.tensor as T

from . import util

class Shape:
    """ A base 3D or 2D shape. """

    def distance(point):
        """ Returns distance between the given point and surface of the shape as
        a Theano tensor. Must be overridden in subclasses. """
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
        return Union([self, second])

    def __add__(self, second):
        """ Returns union of the two shapes """
        return Union([self, second])

    def __and__(self, second):
        """ Returns intersection of the two shapes """
        return Intersection([self, second])

    def __sub__(self, second):
        return Subtraction(self, second)

    def __neg__(self):
        return Inversion(self)

    def __inv__(self):
        return Inversion(self)

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

    def shell(self, inside, outside):
        """ Returns a shell of the current shape"""
        return Shell(self, inside, outside)


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
    def __init__(self, shapes, r = None):
        self.shapes = list(shapes)
        self.r = r

    @staticmethod
    def distance2(r, s1, s2, point):
        g1 = util.derivatives.gradient(s1.distance(point), point)
        g2 = util.derivatives.gradient(s2.distance(point), point)
        cos_alpha = -g1.dot(g2)

        d1 = s1.distance(point)
        d2 = s2.distance(point)
        x1 = r - d1
        x2 = r - d2

        dist_to_rounding = r - util.sqrt((2 * cos_alpha * x1 * x2 + x1 * x1 + x2 * x2) / (1 - cos_alpha * cos_alpha))

        cond1 = (cos_alpha * x1 + x2 <= 0)
        cond2 = (cos_alpha * x2 + x1 <= 0)

        return util.switch(cond1 | cond2, util.minimum(d1, d2), dist_to_rounding)

    def distance(self, point):
        if self.r is None:
            return util.minimum(*(s.distance(point) for s in self.shapes))
        else:
            return functools.reduce(lambda a, b: self.distance2(self.r, a, b, point),
                                    self.shapes)

    def bounding_box(self):
        return functools.reduce(lambda a, b: a.union(b),
                                (s.bounding_box() for s in self.shapes))


class Intersection(Shape):
    def __init__(self, shapes, r = 0):
        self.shapes = list(shapes)
        self.r = r
        if r != 0:
            raise NotImplementedError("Rounded intersections are not supported yet") #TODO

    def distance(self, point):
        if self.r == 0:
            return util.maximum(*(s.distance(point) for s in self.shapes))
        else:
            return functools.reduce(lambda a, b: self.rmax(a, b, self.r),
                                    (s.distance(point) for s in self.shapes))

    def bounding_box(self):
        return functools.reduce(lambda a, b: a.intersection(b),
                                (s.bounding_box() for s in self.shapes))


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
            inv_quat = self.quat.conjugate()
            return util.BoundingBox.containing(inv_quat.rotate_vector(v) for v in b.vertices())


class Scaling(Shape):
    def __init__(self, s, scale):
        self.s = s
        self.scale = scale

    def distance(self, point):
        return self.s.distance(point / self.scale) * self.scale

    def bounding_box(self):
        b = self.s.bounding_box()
        return util.BoundingBox(b.a * self.scale, b.b * self.scale)

class Inversion(Shape):
    def __init__(self, s):
        self.s = s

    def distance(self, point):
        return -self.s.distance(point)

    def bounding_box(self):
        inf = util.Vector(float("inf"), float("inf"), float("inf"))
        return util.BoundingBox(-inf, inf)

class Extrude(Shape):
    pass
    #TODO

class Shell(Shape):
    def __init__(self, s, inside, outside):
        self.s = s
        self.inside = inside
        self.outside = outside

    def distance(self, point):
        return abs(self.s.distance(point) - (self.inside - self.outside) / 2) - \
               (self.inside + self.outside) / 2

    def bounding_box(self):
        return self.s.bounding_box().expanded_additive(self.outside)

def bounding_box_to_shape(box):
    return Box(*box.size()).translated(box.midpoint())
