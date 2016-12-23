from . import theanomath
import collections
import itertools
import numpy

class Vector(collections.namedtuple("Vector", "x y z")):
    __slots__ = ()

    def __new__(cls, x, y, z = 0):
        return super().__new__(cls, x, y, z)

    @classmethod
    def splat(cls, value):
        return cls(value, value, value)

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
        return theanomath.sqrt(self.abs_squared())

    def abs_squared(self):
        return self.dot(self)

    def elementwise_abs(self):
        return Vector(abs(self.x), abs(self.y), abs(self.z))

    def elementwise_mul(self, other):
        return Vector(self.x * other.x, self.y * other.y, self.z * other.z)

    def elementwise_div(self, other):
        return Vector(self.x / other.x, self.y / other.y, self.z / other.z)

    def max(self, other=None):
        return self._minmax(other, theanomath.maximum)

    def min(self, other=None):
        return self._minmax(other, theanomath.minimum)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector(self.y * other.z - self.z * other.y,
                      self.z * other.x - self.x * other.z,
                      self.x * other.y - self.y * other.x)

    def normalized(self):
        return self / abs(self)

    def eval(self, inputs):
        """ Evaluate components from theano expressions to values. """
        return Vector(self.x.eval(inputs), self.y.eval(inputs), self.z.eval(inputs))

    def _minmax(self, other, op):
        if other is None:
            return op(self.x, self.y, self.z)
        else:
            return Vector(op(self.x, other.x),
                          op(self.y, other.y),
                          op(self.z, other.z))

    def applyfunc(self, f):
        return Vector(f(self.x), f(self.y), f(self.z))

    def flattened(self):
        return Vector(self.x, self.y, 0)

    def as_float4(self):
        return numpy.array((self.x, self.y, self.z, 0), dtype=numpy.float32)


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
            b = b.max(v)

        return cls(a, b)

    def intersection(self, other):
        return BoundingBox(self.a.max(other.a), self.b.min(other.b))

    def union(self, other):
        return BoundingBox(self.a.min(other.a), self.b.max(other.b))

    def expanded(self, expansion_factor):
        """ Expand the bounding box by a given factor on each side """
        expansion_vector = self.size() * expansion_factor
        return BoundingBox(self.a - expansion_vector,
                           self.b + expansion_vector)

    def expanded_additive(self, expansion):
        """ Expand the bounding box by a given factor on each side """
        expansion_vector = Vector(expansion, expansion, expansion)
        return BoundingBox(self.a - expansion_vector,
                           self.b + expansion_vector)

    def size(self):
        return self.b - self.a

    def midpoint(self):
        return (self.a + self.b) / 2

    def volume(self):
        size = self.size()
        return size.x * size.y * size.z

    def eval(self, inputs):
        """ Evaluate components from theano expressions to values. """
        return BoundingBox(self.a.eval(inputs), self.b.eval(inputs))

    def flattened(self):
        return BoundingBox(self.a.flattened(), self.b.flattened())

class Quaternion(collections.namedtuple("Quaternion", "v w")):
    # http://www.cs.ucr.edu/~vbz/resources/quatut.pdf
    __slots__ = ()

    @classmethod
    def from_degrees(cls, axis, angle, scale = 1):
        phi = theanomath.radians(angle) / 2
        mul = theanomath.sqrt(scale)
        axis = Vector(*axis)
        return cls(axis.normalized() * theanomath.sin(phi) * mul,
                   theanomath.cos(phi) * mul)

    def __mul__(self, other):
        return Quaternion(self.v * other.w + other.v * self.w + self.v.cross(other.v),
                          self.w * other.w - self.v.dot(other.v))

    def abs_squared(self):
        return self.w * self.w + self.v.abs_squared()

    def inverse(self):
        abs_squared = self.abs_squared()
        return Quaternion(-self.v / abs_squared, self.w / abs_squared)

    def conjugate(self):
        return Quaternion(-self.v, self.w)

    def transform_vector(self, vector):
        return (self.v * self.v.dot(vector) + self.v.cross(vector) * self.w) * 2 + \
                vector * (self.w * self.w - self.v.abs_squared())

    def as_list(self):
        """ Return parameters as a list of floats (for nodes) """
        return list(self.v) + [self.w]


class Transformation(collections.namedtuple("Transformation", "quaternion offset")):
    """ Quaternion and a vector offset """
    __slots__ = ()

    @classmethod
    def from_degrees(cls, axis, angle, scale, offset):
        return cls(Quaternion.from_degrees(axis, angle, scale),
                   Vector(*offset))

    def __mul__(self, other):
        return Transformation(self.quaternion * other.quaternion,
                              self.offset + self.quaternion.transform_vector(other.offset))

    def inverse(self):
        inverse_quaternion = self.quaternion.inverse()
        return Transformation(inverse_quaternion,
                              -inverse_quaternion.transform_vector(self.offset))

    def transform_vector(self, vector):
        return self.quaternion.transform_vector(vector) + self.offset

    def as_list(self):
        """ Return parameters as a list of floats (for nodes) """
        return self.quaternion.as_list() + list(self.offset)
