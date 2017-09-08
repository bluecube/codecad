import functools

from .. import util
from .. import opencl_manager
opencl_manager.instance.add_compile_unit().append_file("common.cl")


class UnionMixin:
    def __init__(self, shapes, r=-1):
        self.shapes = list(shapes)
        self.check_dimension(*self.shapes)
        self.r = r

    def bounding_box(self):
        return functools.reduce(lambda a, b: a.union(b),
                                (s.bounding_box() for s in self.shapes))

    def get_node(self, point, cache):
        return cache.make_node("union",
                               [self.r],
                               (shape.get_node(point, cache) for shape in self.shapes))


class IntersectionMixin:
    def __init__(self, shapes, r=-1):
        self.shapes = list(shapes)
        self.check_dimension(*self.shapes)
        self.r = r

    def bounding_box(self):
        return functools.reduce(lambda a, b: a.intersection(b),
                                (s.bounding_box() for s in self.shapes))

    def get_node(self, point, cache):
        return cache.make_node("intersection",
                               [self.r],
                               (shape.get_node(point, cache) for shape in self.shapes))


class SubtractionMixin:
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


class TransformationMixin:
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
        # TODO: Merge transformation nodes
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


class OffsetMixin:
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


class ShellMixin:
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
