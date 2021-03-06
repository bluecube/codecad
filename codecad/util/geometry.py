import math
import collections
import itertools
import numpy
import pyopencl.cltypes


class Vector(collections.namedtuple("Vector", "x y z")):
    __slots__ = ()

    def __new__(cls, x, y, z=0):
        return super().__new__(cls, x, y, z)

    @classmethod
    def splat(cls, value):
        return cls(value, value, value)

    @classmethod
    def zero(cls):
        return cls(0, 0, 0)

    @classmethod
    def polar(cls, r, phi, rho=0):
        """ Initialize the vector from polar coordinates in degrees.
        phi is longitude, rho is latitude. (rho = 0 -> z = 0) """
        phi = math.radians(phi)
        rho = math.radians(rho)
        k = math.cos(rho)
        return cls(k * math.cos(phi), k * math.sin(phi), math.sin(rho)) * r

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
        return math.sqrt(self.abs_squared())

    def abs_squared(self):
        return self.dot(self)

    def elementwise_abs(self):
        return Vector(abs(self.x), abs(self.y), abs(self.z))

    def elementwise_mul(self, other):
        return Vector(self.x * other.x, self.y * other.y, self.z * other.z)

    def elementwise_div(self, other):
        return Vector(self.x / other.x, self.y / other.y, self.z / other.z)

    def max(self, other=None):
        return self._minmax(other, max)

    def min(self, other=None):
        return self._minmax(other, min)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def normalized(self):
        return self / abs(self)

    def _minmax(self, other, op):
        if other is None:
            return op(self.x, self.y, self.z)
        else:
            return Vector(op(self.x, other.x), op(self.y, other.y), op(self.z, other.z))

    def applyfunc(self, f):
        return Vector(f(self.x), f(self.y), f(self.z))

    def flattened(self):
        return Vector(self.x, self.y, 0)

    def perpendicular2d(self):
        return Vector(self.y, -self.x, self.z)

    def as_float4(self, w=0):
        return numpy.array((self.x, self.y, self.z, w), dtype=pyopencl.cltypes.float4)

    def as_float2(self):
        return numpy.array((self.x, self.y), dtype=pyopencl.cltypes.float2)

    def as_tuple2(self):
        return (self.x, self.y)

    def as_matrix(self):
        """
        Return a 1x4 numpy array corresponding to this vector.

        Currently this is slow (but robust) and only intended for testing.
        """
        return numpy.array([[self.x], [self.y], [self.z], [1]])

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y, self.z)


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
        a = []
        b = []
        for a1, b1, a2, b2 in zip(self.a, self.b, other.a, other.b):
            a_tmp = max(a1, a2)
            b_tmp = max(a_tmp, min(b1, b2))
            a.append(a_tmp)
            b.append(b_tmp)
        return BoundingBox(Vector(*a), Vector(*b))  # noqa

    def union(self, other):
        return BoundingBox(self.a.min(other.a), self.b.max(other.b))

    def expanded(self, expansion_factor):
        """ Expand the bounding box by a given factor on each side """
        expansion_vector = self.size() * expansion_factor
        return BoundingBox(self.a - expansion_vector, self.b + expansion_vector)

    def expanded_additive(self, expansion):
        """ Expand the bounding box by a given factor on each side """
        expansion_vector = Vector.splat(expansion)
        return BoundingBox(self.a - expansion_vector, self.b + expansion_vector)

    def size(self):
        return self.b - self.a

    def midpoint(self):
        return (self.a + self.b) / 2

    def volume(self):
        size = self.size()
        return size.x * size.y * size.z

    def flattened(self):
        return BoundingBox(self.a.flattened(), self.b.flattened())

    def points(self):
        """ Yield coordinates of all corner points of this box """
        for x in [self.a.x, self.b.x]:
            for y in [self.a.y, self.b.y]:
                for z in [self.a.z, self.b.z]:
                    yield Vector(x, y, z)

    def points2d(self):
        """ Yield coordinates of all corner points of the box, when taken as 2D. """
        for x in [self.a.x, self.b.x]:
            for y in [self.a.y, self.b.y]:
                yield Vector(x, y)


class Quaternion(collections.namedtuple("Quaternion", "v w")):
    # http://www.cs.ucr.edu/~vbz/resources/quatut.pdf
    __slots__ = ()

    @classmethod
    def from_degrees(cls, axis, angle, scale=1):
        phi = math.radians(angle) / 2
        mul = math.sqrt(scale)
        axis = Vector(*axis)
        return cls(axis.normalized() * math.sin(phi) * mul, math.cos(phi) * mul)

    @classmethod
    def zero(cls):
        return cls(Vector.zero(), 1)

    def __mul__(self, other):
        return Quaternion(
            self.v * other.w + other.v * self.w + self.v.cross(other.v),
            self.w * other.w - self.v.dot(other.v),
        )

    def abs_squared(self):
        return self.w * self.w + self.v.abs_squared()

    def inverse(self):
        abs_squared = self.abs_squared()
        return Quaternion(-self.v / abs_squared, self.w / abs_squared)

    def conjugate(self):
        return Quaternion(-self.v, self.w)

    def transform_vector(self, vector):
        return (
            self.v * self.v.dot(vector) + self.v.cross(vector) * self.w
        ) * 2 + vector * (self.w * self.w - self.v.abs_squared())

    def as_list(self):
        """ Return parameters as a list of floats (for nodes) """
        return list(self.v) + [self.w]

    def as_matrix(self):
        """
        Return a 4x4 numpy array that represents the same transformation.

        Currently this is slow (but robust) and only intended for testing.
        """

        ret = numpy.hstack(
            [
                self.transform_vector(Vector(1, 0, 0)).as_matrix(),
                self.transform_vector(Vector(0, 1, 0)).as_matrix(),
                self.transform_vector(Vector(0, 0, 1)).as_matrix(),
                [[0], [0], [0], [1]],
            ]
        )
        return ret


class Transformation(collections.namedtuple("Transformation", "quaternion offset")):
    """ Quaternion and a vector offset """

    __slots__ = ()

    @classmethod
    def from_degrees(cls, axis, angle, scale, offset):
        return cls(Quaternion.from_degrees(axis, angle, scale), Vector(*offset))

    @classmethod
    def zero(cls):
        return cls(Quaternion.zero(), Vector.zero())

    def __mul__(self, other):
        """ Combines two transformations into one,
        order is "second * first" """
        return Transformation(
            self.quaternion * other.quaternion,
            self.offset + self.quaternion.transform_vector(other.offset),
        )

    def inverse(self):
        inverse_quaternion = self.quaternion.inverse()
        return Transformation(
            inverse_quaternion, -inverse_quaternion.transform_vector(self.offset)
        )

    def transform_vector(self, vector):
        return self.quaternion.transform_vector(vector) + self.offset

    def as_list(self):
        """ Return parameters as a list of floats (for nodes) """
        return self.quaternion.as_list() + list(self.offset)

    def as_matrix(self):
        """
        Return a 4x4 numpy matrix that represents the same transformation.
        """
        ret = self.quaternion.as_matrix()
        ret[0, 3] = self.offset.x
        ret[1, 3] = self.offset.y
        ret[2, 3] = self.offset.z
        return ret

    def is_2d(self):
        return (
            self.quaternion.v.x == 0 and self.quaternion.v.y == 0 and self.offset.z == 0
        )
