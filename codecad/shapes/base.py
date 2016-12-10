import abc
import functools
import math
import numpy

from .. import util

class ShapeBase(metaclass=abc.ABCMeta):
    """ Abstract base class for 2D and 3D shapes """

    @abc.abstractmethod
    def distance(self, point):
        """ Returns distance between the given point and surface of the shape as
        a Theano tensor. Must be overridden in subclasses. """

    @abc.abstractmethod
    def bounding_box(self, point):
        """ Returns a box that contains the whole shape.
        Must be overridden by subclasses. """

    @abc.abstractmethod
    def __and__(self, second):
        """ Returns intersection of the two shapes """

    @abc.abstractmethod
    def __add__(self, second):
        """ Returns union of the two shapes """

    @abc.abstractmethod
    def __sub__(self, second):
        """ Return difference between the two shapes """

    def __or__(self, second):
        """ Return union of the two shapes """
        return self.__add__(second)

    @staticmethod
    @abc.abstractmethod
    def dimension():
        """ Returns dimension of the shape (2 or 3) """

    def check_dimension(self, *shapes, required = None):
        if required is None:
            required = self.dimension()
        if not len(shapes):
            shapes = [self]
        for shape in shapes:
            if shape.dimension() != required:
                raise TypeError("Shape must be of dimension {}, but is {}".format(required, shape.dimension()))


class Union:
    def __init__(self, shapes, r = None):
        self.shapes = list(shapes)
        self.check_dimension(*self.shapes)
        self.r = r

    @staticmethod
    def distance2(r, s1, s2, point):
        epsilon = min(s1.bounding_box().size().min(),
                      s2.bounding_box().size().min()) / 10000;

        d1 = s1.distance(point)
        d2 = s2.distance(point)
        x1 = r - d1
        x2 = r - d2

        # epsilon * gradient(s1)(point)
        g1 = util.Vector(s1.distance(point + util.Vector(epsilon, 0, 0)) - d1,
                         s1.distance(point + util.Vector(0, epsilon, 0)) - d1,
                         s1.distance(point + util.Vector(0, 0, epsilon)) - d1)

        cos_alpha = abs((s2.distance(point + g1) - d2) / epsilon)

        dist_to_rounding = r - util.sqrt((x1 * x1 + x2 * x2 - 2 * cos_alpha * x1 * x2) / (1 - cos_alpha * cos_alpha))

        cond1 = (cos_alpha * x1 < x2)
        cond2 = (cos_alpha * x2 < x1)

        return util.switch(cond1 & cond2, dist_to_rounding, util.minimum(d1, d2))

    def distance(self, point):
        if self.r is None:
            return util.minimum(*[s.distance(point) for s in self.shapes])
        else:
            return functools.reduce(lambda a, b: self.distance2(self.r, a, b, point),
                                    self.shapes)

    def bounding_box(self):
        return functools.reduce(lambda a, b: a.union(b),
                                (s.bounding_box() for s in self.shapes))


class Intersection:
    def __init__(self, shapes, r = 0):
        self.shapes = list(shapes)
        self.check_dimension(*self.shapes)
        self.r = r
        if r != 0:
            raise NotImplementedError("Rounded intersections are not supported yet") #TODO

    def distance(self, point):
        if self.r == 0:
            return util.maximum(*[s.distance(point) for s in self.shapes])
        else:
            return functools.reduce(lambda a, b: self.rmax(a, b, self.r),
                                    (s.distance(point) for s in self.shapes))

    def bounding_box(self):
        return functools.reduce(lambda a, b: a.intersection(b),
                                (s.bounding_box() for s in self.shapes))


class Subtraction:
    def __init__(self, s1, s2):
        self.check_dimension(s1, s2)
        self.s1 = s1
        self.s2 = s2

    def distance(self, point):
        return util.maximum(self.s1.distance(point),
                            -self.s2.distance(point))

    def bounding_box(self):
        return self.s1.bounding_box()

class Transformation:
    """ Rotation and scaling followed by translation. """
    def __init__(self, s, quaternion, translation):
        self.check_dimension(s)
        self.s = s
        self.quaternion = quaternion
        self.translation = translation

    @classmethod
    def make_merged(cls, s, quaternion, translation):
        t = cls(s, quaternion, translation)
        if isinstance(s, cls):
            t.s = s.s
            t.quaternion = quaternion * s.quaternion
            t.translation = translation + quaternion.rotate_vector(s.translation)

        return t

    def distance(self, point):
        new_point = self.quaternion.inverse().rotate_vector(point - self.translation)
        return self.s.distance(new_point) * self.quaternion.abs_squared()

    def transform_vector(self, v):
        return self.quaternion.rotate_vector(v) + self.translation;

class Shell:
    def __init__(self, s, inside, outside):
        self.check_dimension(s)
        self.s = s
        self.inside = inside
        self.outside = outside

    def distance(self, point):
        return abs(self.s.distance(point) - (self.inside - self.outside) / 2) - \
               (self.inside + self.outside) / 2

    def bounding_box(self):
        return self.s.bounding_box().expanded_additive(self.outside)
