import collections
import functools
import math
import theano
import theano.tensor as T
import numpy

def is_theano(x):
    return isinstance(x, theano.Variable)

def maximum(*args):
    if any(is_theano(x) for x in args):
        return functools.reduce(T.maximum, args)
    else:
        return functools.reduce(numpy.maximum, args)

def minimum(*args):
    if any(is_theano(x) for x in args):
        return functools.reduce(T.minimum, args)
    else:
        return functools.reduce(numpy.minimum, args)

def sqrt(x):
    if is_theano(x):
        return T.sqrt(x)
    else:
        return numpy.sqrt(x)

class Vector(collections.namedtuple("Vector", "x y z")):
    __slots__ = ()

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return Vector(self.x / other, self.y / other, self.z / other)

    def __neg__(self):
        return Vector(-self.x, -self.y, -self.z)

    def __pos__(self):
        return self

    def __abs__(self):
        return sqrt(self.dot(self))

    def elementwise_abs(self):
        return Vector(abs(self.x), abs(self.y), abs(self.z))

    def max(self, other=None):
        return self._minmax(other, maximum)

    def min(self, other=None):
        return self._minmax(other, minimum)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def normalized(self):
        return self / abs(self)

    def _minmax(self, other, op):
        if other is None:
            return op(self.x, self.y, self.z)
        else:
            return Vector(op(self.x, other.x),
                          op(self.y, other.y),
                          op(self.z, other.z))


BoundingBox = collections.namedtuple("BoundingBox", "a b")
