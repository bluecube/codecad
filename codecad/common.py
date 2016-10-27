""" Stuff common to 2D and 3D shapes """

import functools
import math

from . import util

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
