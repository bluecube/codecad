import abc
import functools
import math
import numpy

from .. import util

class ShapeBase(metaclass=abc.ABCMeta):
    """ Abstract base class for 2D and 3D shapes """

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

    @abc.abstractmethod
    def scaled(self, s):
        """ Returns current shape scaled by given ratio """

    @abc.abstractmethod
    def offset(self, d):
        """ Returns current shape offset by given distance (positive increases size) """

    @abc.abstractmethod
    def shell(self, wall_thickness):
        """ Returns a shell of the current shape (centered around the original surface) """


class Union:
    def __init__(self, shapes, r = None):
        self.shapes = list(shapes)
        self.check_dimension(*self.shapes)
        self.r = r

    def bounding_box(self):
        return functools.reduce(lambda a, b: a.union(b),
                                (s.bounding_box() for s in self.shapes))

    def get_node(self, point, cache):
        return cache.make_node("union",
                               [self.r if self.r is not None else -1],
                               (shape.get_node(point, cache) for shape in self.shapes))


class Intersection:
    def __init__(self, shapes, r = 0):
        self.shapes = list(shapes)
        self.check_dimension(*self.shapes)
        self.r = r
        if r != 0:
            raise NotImplementedError("Rounded intersections are not supported yet") #TODO

    def bounding_box(self):
        return functools.reduce(lambda a, b: a.intersection(b),
                                (s.bounding_box() for s in self.shapes))

    def get_node(self, point, cache):
        return cache.make_node("intersection",
                               [self.r if self.r is not None else -1],
                               (shape.get_node(point, cache) for shape in self.shapes))


class Subtraction:
    def __init__(self, s1, s2):
        self.check_dimension(s1, s2)
        self.s1 = s1
        self.s2 = s2

    def bounding_box(self):
        return self.s1.bounding_box()

    def get_node(self, point, cache):
        return cache.make_node("subtraction",
                               [-1],
                               [self.s1.get_node(point, cache), self.s2.get_node(point, cache)])

class Transformation:
    """ Rotation and scaling followed by translation. """
    def __init__(self, s, quaternion, translation):
        self.check_dimension(s)
        self.s = s
        self.transformation = util.Transformation(quaternion, translation)

    @classmethod
    def make_merged(cls, s, quaternion, translation):
        t = cls(s, quaternion, translation)
        if isinstance(s, cls):
            t.s = s.s
            t.transformation = t.transformation * s.transformation

        return t

    def get_node(self, point, cache):
        #TODO: Merge transformation nodes
        inverse_transformation = self.transformation.inverse()
        new_point = cache.make_node("transformation_to",
                                    inverse_transformation.as_list(),
                                    [point],
                                    inverse_transformation)
        distance = self.s.get_node(new_point, cache)
        return cache.make_node("transformation_from",
                               self.transformation.quaternion.as_list(),
                               [distance],
                               self.transformation.quaternion)


class Offset:
    def __init__(self, s, distance):
        self.check_dimension(s)
        self.s = s
        self.distance = distance

    def bounding_box(self):
        return self.s.bounding_box().expanded_additive(self.distance)

    def get_node(self, point, cache):
        return cache.make_node("offset",
                               [self.distance],
                               [self.s.get_node(point, cache)])


class Shell:
    def __init__(self, s, wall_thickness):
        self.check_dimension(s)
        self.s = s
        self.wall_thickness = wall_thickness

    def bounding_box(self):
        return self.s.bounding_box().expanded_additive(self.wall_thickness / 2)

    def get_node(self, point, cache):
        return cache.make_node("shell",
                               [self.wall_thickness / 2],
                               [self.s.get_node(point, cache)])
