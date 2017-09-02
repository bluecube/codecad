import math

from .. import util
from . import base
from . import common
from .. import opencl_manager

_c_file = opencl_manager.instance.add_compile_unit()
_c_file.append_file("common.h")
_c_file.append_file("simple3d.cl")

class Sphere(base.Shape3D):
    def __init__(self, d = 1, r = None):
        if r is None:
            self.r = d / 2
        else:
            self.r = r

    def bounding_box(self):
        v = util.Vector(self.r, self.r, self.r)
        return util.BoundingBox(-v, v)

    def get_node(self, point, cache):
        return cache.make_node("sphere", [self.r], [point])


class HalfSpace(base.Shape3D):
    """ Half space y > 0. """
    def bounding_box(self):
        return util.BoundingBox(util.Vector(-float("inf"), 0, -float("inf")),
                                util.Vector.splat(float("inf")));

    def get_node(self, point, cache):
        return cache.make_node("half_space", [], [point])


class Union(common.UnionMixin, base.Shape3D):
    pass


class Intersection(common.IntersectionMixin, base.Shape3D):
    pass


class Subtraction(common.SubtractionMixin, base.Shape3D):
    pass


class Offset(common.OffsetMixin, base.Shape3D):
    pass


class Shell(common.ShellMixin, base.Shape3D):
    pass


class Transformation(common.TransformationMixin, base.Shape3D):
    def bounding_box(self):
        b = self.s.bounding_box()

        if any(math.isinf(x) for x in b.a) or any(math.isinf(x) for x in b.b):
            # Special case for rotating infinite objects.
            inf = util.Vector(float("inf"), float("inf"), float("inf"))
            return util.BoundingBox(-inf, inf)
        else:
            return util.BoundingBox.containing(self.transformation.transform_vector(v) for v in b.vertices())


class Extrusion(base.Shape3D):
    def __init__(self, s, height):
        self.check_dimension(s, required=2)
        self.s = s
        self.h = height

    def bounding_box(self):
        box = self.s.bounding_box()
        return util.BoundingBox(util.Vector(box.a.x, box.a.y, -self.h / 2),
                                util.Vector(box.b.x, box.b.y, self.h / 2))

    def get_node(self, point, cache):
        sub_node = self.s.get_node(point, cache)
        if math.isinf(self.h):
            return sub_node
        else:
            return cache.make_node("extrusion", [self.h / 2], [point, sub_node])

class Revolution(base.Shape3D):
    def __init__(self, s):
        self.check_dimension(s, required=2)
        self.s = s

    def bounding_box(self):
        box = self.s.bounding_box()
        radius = max(-box.a.x, box.b.x)
        return util.BoundingBox(util.Vector(-radius, box.a.y, -radius),
                                util.Vector(radius, box.b.y, radius))

    def get_node(self, point, cache):
        new_point = cache.make_node("revolution_to",
                                    [],
                                    [point])
        sub_node = self.s.get_node(new_point, cache)
        return cache.make_node("revolution_from",
                               [],
                               [point, sub_node])
