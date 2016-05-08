import collections
import functools
import itertools
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

def round(x):
    if is_theano(x):
        return T.round(x)
    else:
        return numpy.round(x)

def sin(x):
    if is_theano(x):
        return T.sin(x)
    else:
        return numpy.sin(x)

def asin(x):
    if is_theano(x):
        return T.arcsin(x)
    else:
        return numpy.arcsin(x)

def switch(cond, true, false):
    if is_theano(cond) or is_theano(true) or is_theano(false):
        return T.switch(cond, true, false)
    else:
        return numpy.switch(cond, true, false)

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

    def cross(self, other):
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)

    def normalized(self):
        return self / abs(self)

    def _minmax(self, other, op):
        if other is None:
            return op(self.x, self.y, self.z)
        else:
            return Vector(op(self.x, other.x),
                          op(self.y, other.y),
                          op(self.z, other.z))


class BoundingBox(collections.namedtuple("BoundingBox", "a b")):
    __slots__ = ()

    def vertices(self):
        for selector in itertools.product((0, 1), repeat=3):
            yield Vector(*(self[x][i] for i, x in enumerate(selector)))

    @classmethod
    def containing(cls, vector_iterable):
        a = Vector(float("inf"), float("inf"), float("inf"))
        b = -a

        for v in vector_iterable:
            a = a.min(v)
            b = b.min(v)

        return cls(a, b)

class Quaternion(collections.namedtuple("Quaternion", "v w")):
    # http://www.cs.ucr.edu/~vbz/resources/quatut.pdf
    __slots__ = ()

    def __mul__(self, other):
        return Quaternion(self.v * other.w + other.v * self.w + self.v.cross(other.v),
                          self.w * other.w - self.v.dot(other.v))

    def conjugate(self):
        return Quaternion(-self.v, self.w)

    def rotate_vector(self, vector):
        q = Quaternion(vector, 0)
        rotated = self * q * self.conjugate()
        return rotated.v
