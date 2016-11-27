import abc
import functools
import math

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

class Translation:
    def __init__(self, s, offset):
        self.check_dimension(s)
        self.s = s
        self.offset = offset

    def distance(self, point):
        return self.s.distance(point - self.offset)

    def bounding_box(self):
        b = self.s.bounding_box()
        return util.BoundingBox(b.a + self.offset, b.b + self.offset)

    def get_node(self, point, cache):
        return self.s.get_node(cache.make_node("translation",
                                               self.offset,
                                               [point]),
                               cache)


class Scaling:
    def __init__(self, s, scale):
        self.check_dimension(s)
        self.s = s
        self.scale = scale

    def distance(self, point):
        return self.s.distance(point / self.scale) * self.scale

    def bounding_box(self):
        b = self.s.bounding_box()
        return util.BoundingBox(b.a * self.scale, b.b * self.scale)

    def get_node(self, point, cache):
        return self.s.get_node(cache.make_node("scaling",
                                               [self.scale],
                                               [point]),
                               cache)


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
        return cache.make_node("subtraction",
                               [self.inside, self.outside],
                               [self.s.get_node(point, cache)])
