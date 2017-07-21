import abc
import math

from .. import util
# simple2d and simple3d are imported in functions to break circular dependencies

class ShapeBase(metaclass=abc.ABCMeta):
    """ Abstract base class for 2D and 3D shapes """

    @abc.abstractmethod
    def bounding_box(self):
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


class Shape2D(ShapeBase):
    """ A base 2D shape. """

    @staticmethod
    def dimension():
        return 2

    def __and__(self, second):
        from . import simple2d
        return simple2d.Intersection2D([self, second])

    def __add__(self, second):
        from . import simple2d
        return simple2d.Union2D([self, second])

    def __sub__(self, second):
        from . import simple2d
        return simple2d.Subtraction2D(self, second)

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

        from . import simple2d
        return simple2d.Transformation2D.make_merged(self,
                                                     util.Quaternion.from_degrees(util.Vector(0, 0, 1), 0),
                                                     o)

    def rotated(self, angle, n = 1):
        """ Returns current shape rotated by given angle.

        If n > 1, returns an union of n copies of self, rotated in regular intervals
        up to given angle.
        For example angle = 180, n = 3 makes copies of self rotated by 60, 120
        and 180 degrees. """
        from . import simple2d
        if n == 1:
            return simple2d.Transformation2D.make_merged(self,
                                                         util.Quaternion.from_degrees(util.Vector(0, 0, 1), angle),
                                                         util.Vector(0, 0, 0))
        else:
            angle_step = angle / n
            return simple2d.Union2D([self.rotated((1 + i) * angle_step) for i in range(n)])

    def scaled(self, s):
        """ Returns current shape scaled by given ratio """
        from . import simple2d
        return simple2d.Transformation2D.make_merged(self,
                                                     util.Quaternion.from_degrees(util.Vector(0, 0, 1), 0, s),
                                                     util.Vector(0, 0, 0))

    def offset(self, d):
        """ Returns current shape offset by given distance (positive is outside) """
        from . import simple2d
        return simple2d.Offset2D(self, d)

    def shell(self, wall_thickness):
        """ Returns a shell of the current shape (centered around the original surface) """
        from . import simple2d
        return simple2d.Shell2D(self, wall_thickness)

    def extruded(self, height, symmetrical=True):
        from . import simple3d
        s = simple3d.Extrusion(self, height)
        if symmetrical:
            return s
        else:
            return s.translated(0, 0, height/2)

    def revolved(self):
        """ Returns current shape taken as 2D in xy plane and revolved around y axis """
        from . import simple3d
        return simple3d.Revolution(self)


class Shape3D(ShapeBase):
    """ A base 3D shape. """

    @staticmethod
    def dimension():
        return 3

    def __and__(self, second):
        from . import simple3d
        return simple3d.Intersection([self, second])

    def __add__(self, second):
        from . import simple3d
        return simple3d.Union([self, second])

    def __sub__(self, second):
        from . import simple3d
        return simple3d.Subtraction(self, second)

    def translated(self, x, y = None, z = None):
        """ Returns current shape translated by a given offset """
        if isinstance(x, util.Vector):
            if y is not None or z is not None:
                raise TypeError("If first parameter is Vector, the others must be left unspecified.")
            o = x
        else:
            o = util.Vector(x, y, z)
        from . import simple3d
        return simple3d.Transformation.make_merged(self,
                                                   util.Quaternion.from_degrees(util.Vector(0, 0, 1), 0),
                                                   o)

    def rotated(self, vector, angle, n = 1):
        """ Returns current shape rotated by an angle around the vector.

        If n > 1, returns an union of n copies of self, rotated in regular intervals
        up to given angle.
        For example angle = 180, n = 3 makes copies of self rotated by 60, 120
        and 180 degrees."""
        from . import simple3d
        if n == 1:
            return simple3d.Transformation.make_merged(self,
                                                       util.Quaternion.from_degrees(util.Vector(*vector), angle),
                                                       util.Vector(0, 0, 0))
        else:
            angle_step = angle / n
            return Union([self.rotated(vector, (1 + i) * angle_step) for i in range(n)])

    def scaled(self, s):
        """ Returns current shape scaled by given ratio """
        from . import simple3d
        return simple3d.Transformation.make_merged(self,
                                                   util.Quaternion.from_degrees(util.Vector(0, 0, 1), 0, s),
                                                   util.Vector(0, 0, 0))

    def offset(self, d):
        """ Returns current shape offset by given distance (positive increases size) """
        from . import simple3d
        return simple3d.Offset(self, d)

    def shell(self, wall_thickness):
        """ Returns a shell of the current shape (centered around the original surface) """
        from . import simple3d
        return simple3d.Shell(self, wall_thickness)
