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

    @abc.abstractmethod
    def get_node(self, point, cache):
        """ Return a computation node for this shape. The node should be created
        using cache.make_node to give CSE a chance to work """


class Union:
    def __init__(self, shapes, r = 0):
        self.shapes = list(shapes)
        self.check_dimension(*self.shapes)
        self.r = r

    @staticmethod
    def rmin(a, b, r):
        return util.switch(abs(a - b) >= r,
                           util.minimum(a, b),
                           b + r * util.sin(math.pi / 4 + util.arcsin((a - b) / (r * math.sqrt(2)))) - r)

    def distance(self, point):
        if self.r == 0:
            return util.minimum(*[s.distance(point) for s in self.shapes])
        else:
            return functools.reduce(lambda a, b: self.rmin(a, b, self.r),
                                    (s.distance(point) for s in self.shapes))

    def bounding_box(self):
        return functools.reduce(lambda a, b: a.union(b),
                                (s.bounding_box() for s in self.shapes))

    def get_node(self, point, cache):
        return cache.make_node("union",
                               [self.r],
                               (shape.get_node(point, cache) for shape in self.shapes))


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

    def get_node(self, point, cache):
        return cache.make_node("intersection",
                               [self.r],
                               (shape.get_node(point, cache) for shape in self.shapes))


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

    def get_node(self, point, cache):
        return cache.make_node("subtraction",
                               [],
                               [self.s1.get_node(point, cache), self.s2.get_node(point, cache)])

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

    def get_node(self, point, cache):
        #TODO: Merge transformation nodes
        inverse_quaternion = self.quaternion.inverse()
        inverse_translation = -self.translation
        #TODO: Is the inverse transformation correct?
        new_point = cache.make_node("transformation",
                                    inverse_quaternion.as_list() + list(inverse_translation),
                                    [point],
                                    (inverse_quaternion, self.translation))
        distance = self.s.get_node(new_point, cache)
        return cache.make_node("reverse_transformation",
                               self.quaternion.as_list(),
                               [distance],
                               self.quaternion)


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

    def get_node(self, point, cache):
        return cache.make_node("shell",
                               [self.inside, self.outside],
                               [self.s.get_node(point, cache)])
