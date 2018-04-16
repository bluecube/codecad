import functools

from .. import util
from .. import cl_util

_c_file = cl_util.opencl_manager.add_compile_unit()
_c_file.append_file("common.h")
_c_file.append_file("common.cl")


class UnionMixin:
    def __init__(self, shapes, r=-1):
        self.shapes = list(shapes)
        self.check_dimension(*self.shapes)
        self.r = r

    def bounding_box(self):
        return functools.reduce(lambda a, b: a.union(b),
                                (s.bounding_box() for s in self.shapes))

    def feature_size(self):
        return min(s.feature_size() for s in self.shapes)

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

    def feature_size(self):
        return min(s.feature_size() for s in self.shapes)

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

    def feature_size(self):
        return min(self.s1.feature_size(), self.s2.feature_size())

    def get_node(self, point, cache):
        return cache.make_node("subtraction",
                               [-1],
                               [self.s1.get_node(point, cache), self.s2.get_node(point, cache)])


class TransformationMixin:
    def __init__(self, s, quaternion, translation):
        self.check_dimension(s)
        self.s = s
        self.transformation = util.Transformation(quaternion, translation)

    def feature_size(self):
        return self.s.feature_size() * self.transformation.quaternion.abs_squared()

    def get_node(self, point, cache):
        inverse_transformation = self.transformation.inverse()
        if point.name == "transformation_to" or point.name == "initial_transformation_to":
            # Merge transformation_to nodes
            inverse_transformation = inverse_transformation * point.extra_data
            dependencies = point.dependencies
            node_name = point.name
        else:
            dependencies = [point]
            node_name = "transformation_to"

        new_point = cache.make_node(node_name,
                                    inverse_transformation.as_list(),
                                    dependencies,
                                    inverse_transformation)

        inner_result = self.s.get_node(new_point, cache)

        quat = self.transformation.quaternion
        if inner_result.name == "transformation_from":
            # Merge transformation_from nodes
            quat = quat * inner_result.extra_data
            dependencies = inner_result.dependencies
        else:
            dependencies = [inner_result]

        return cache.make_node("transformation_from",
                               quat.as_list(),
                               dependencies,
                               quat)


class MirrorMixin:
    def __init__(self, s):
        self.check_dimension(s)
        self.s = s

    def bounding_box(self):
        box = self.s.bounding_box()
        return util.BoundingBox(util.Vector(-box.b.x, box.a.y, box.a.z),
                                util.Vector(-box.a.x, box.b.y, box.b.z))

    def feature_size(self):
        return self.s.feature_size()

    def get_node(self, point, cache):
        mirrored_point = cache.make_node("mirror", [], [point])
        mirrored_result = self.s.get_node(mirrored_point, cache)
        return cache.make_node("mirror", [], [mirrored_result])


class OffsetMixin:
    def __init__(self, s, distance):
        self.check_dimension(s)
        self.s = s
        self.distance = distance

    def bounding_box(self):
        box = self.s.bounding_box().expanded_additive(self.distance)
        if self.dimension() == 2:
            return box.flattened()
        else:
            return box

    def feature_size(self):
        return max(0, self.s.feature_size() + self.distance * 2)

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
        box = self.s.bounding_box().expanded_additive(self.wall_thickness / 2)
        if self.dimension() == 2:
            return box.flattened()
        else:
            return box

    def feature_size(self):
        # If there were any smaller features in the original shape,
        # they will be swallowed by the wall
        return self.wall_thickness

    def get_node(self, point, cache):
        return cache.make_node("shell",
                               [self.wall_thickness / 2],
                               [self.s.get_node(point, cache)])
