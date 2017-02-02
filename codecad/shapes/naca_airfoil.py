""" Modelling NACA 4 digit airfoils as signed distance functions.
Formulas were adapted from https://en.wikipedia.org/wiki/NACA_airfoil """

from .. import util
from . import simple2d

class NacaAirfoil(simple2d.Shape2D):
    def __init__(self, thickness_or_code, max_camber = None, max_camber_position = None):
        if max_camber is None:
            # It's a 4 digit code

            code = int(thickness_or_code)

            if code <= 0 or code > 9999:
                raise ValueError("Only 4 digit NACA airfoils are allowed. The code must be between 1 and 9999")

            self.thickness = (code % 100) / 100;
            self.max_camber = ((code // 1000) % 10) / 100
            self.max_camber_position = ((code // 100) % 10) / 10

        else:
            self.thickness = thickness_or_code
            self.max_camber = max_camber
            self.max_camber_position = max_camber_position

    def bounding_box(self):
        return util.BoundingBox(util.Vector(0, -self.thickness / 2, -float("inf")),
                                util.Vector(1, self.thickness / 2, float("inf")))

    def get_node(self, point, cache):
        return cache.make_node("naca_airfoil",
                               [self.thickness, self.max_camber, self.max_camber_position],
                               [point])
