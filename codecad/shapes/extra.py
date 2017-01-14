import math

from .. import util
from . import simple2d

class InvoluteCurve(simple2d.Shape2D):
    def bounding_box(self):
        pi = math.pi
        return util.BoundingBox(util.Vector(-3 * pi / 2, -2 * pi),
                                util.Vector(pi / 2, pi))

    def get_node(self, point, cache):
        return cache.make_node("involute", [], [point])
