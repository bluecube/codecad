import math

from .. import util
from . import base
from . import simple3d

class Shape2D(base.ShapeBase):
    """ A base 2D shape. """

    @staticmethod
    def dimension():
        return 2

    def __and__(self, second):
        return Intersection2D([self, second])

    def __add__(self, second):
        return Union2D([self, second])

    def __sub__(self, second):
        return Subtraction2D(self, second)

    def translated(self, x, y = None):
        """ Returns current shape translated by a given offset """
        if isinstance(x, util.Vector):
            if y is not None:
                raise TypeError("If first parameter is Vector, the others must be left unspecified.")
            o = x
        else:
            if y is None:
                raise ValueError("Y coordinate can only be missing if first parameter is a Vector.")
            o = util.Vector(x, y)
        return Transformation2D.make_merged(self,
                                            util.Quaternion.from_degrees(util.Vector(0, 0, 1), 0),
                                            o)

    def rotated(self, angle):
        """ Returns current shape rotated by given angle """
        return Transformation2D.make_merged(self,
                                            util.Quaternion.from_degrees(util.Vector(0, 0, 1), angle),
                                            util.Vector(0, 0, 0))

    def scaled(self, s):
        """ Returns current shape scaled by given ratio """
        return Transformation2D.make_merged(self,
                                            util.Quaternion.from_degrees(util.Vector(0, 0, 1), 0, s),
                                            util.Vector(0, 0, 0))

    def offset(self, d):
        """ Returns current shape offset by given distance (positive is outside) """
        return Offset2D(self, d)

    def shell(self, wall_thickness):
        """ Returns a shell of the current shape (centered around the original surface) """
        return Shell2D(self, wall_thickness)

    def extruded(self, height):
        return simple3d.Extrusion(self, height)

    def revolved(self):
        """ Returns current shape taken as 2D in xy plane and revolved around y axis """
        return simple3d.Revolution(self)


class Rectangle(Shape2D):
    def __init__(self, x = 1, y = None):
        if y is None:
            y = x
        self.half_size = util.Vector(x, y) / 2

    def bounding_box(self):
        return util.BoundingBox(-self.half_size, self.half_size)

    def get_node(self, point, cache):
        return cache.make_node("rectangle", [self.half_size.x, self.half_size.y], [point])

class Circle(Shape2D):
    def __init__(self, d = 1, r = None):
        if r is None:
            self.r = d / 2
        else:
            self.r = r

    def bounding_box(self):
        v = util.Vector(self.r, self.r)
        return util.BoundingBox(-v, v)

    def get_node(self, point, cache):
        return cache.make_node("circle", [self.r], [point])


class Union2D(base.Union, Shape2D):
    pass


class Intersection2D(base.Intersection, Shape2D):
    pass


class Subtraction2D(base.Subtraction, Shape2D):
    pass


class Offset2D(base.Offset, Shape2D):
    pass


class Shell2D(base.Shell, Shape2D):
    pass


class Transformation2D(base.Transformation, Shape2D):
    def bounding_box(self):
        b = self.s.bounding_box().flattened()

        if any(math.isinf(x) for x in b.a) or any(math.isinf(x) for x in b.b):
            # Special case for rotating infinite objects.
            inf = util.Vector(float("inf"), float("inf"), float("inf"))
            return util.BoundingBox(-inf, inf)
        else:
            inf = float("inf")
            ret = util.BoundingBox.containing(self.transformation.transform_vector(v) for v in b.vertices())
            return util.BoundingBox(util.Vector(ret.a.x, ret.a.y, -inf),
                                    util.Vector(ret.b.x, ret.b.y, inf))
