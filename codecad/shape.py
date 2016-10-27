import abc

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
