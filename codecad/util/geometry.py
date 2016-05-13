from . import theanomath
import collections
import itertools

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
        return theanomath.sqrt(self.dot(self))

    def elementwise_abs(self):
        return Vector(abs(self.x), abs(self.y), abs(self.z))

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

    def intersection(self, other):
        return BoundingBox(self.a.max(other.a), self.b.min(other.b))

    def union(self, other):
        return BoundingBox(self.a.min(other.a), self.b.max(other.b))

    def expanded(self, expansion_factor):
        """ Expand the bounding box by a given factor on each side """
        box_size = self.b - self.a
        expansion_vector = box_size * expansion_factor
        return BoundingBox(self.a - expansion_vector,
                           self.b + expansion_vector)

    def volume(self):
        size = self.b - self.a
        return size.x * size.y * size.z

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