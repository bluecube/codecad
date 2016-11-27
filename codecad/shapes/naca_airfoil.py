""" Modelling NACA 4 digit airfoils as signed distance functions.
Formulas were adapted from https://en.wikipedia.org/wiki/NACA_airfoil """

from .. import util
from . import simple2d

def _naca_center(x, p):
    """ Y coordinate of camber line.
    p is position of max camber.
    """
    xx = util.switch(x <= p, x / p, (1 - x) / (1 - p))
    return xx * (2 - xx)

def _naca_thickness(x):
    "Thickness of NACA airfoil at given x."
    return 5 * (0.2969 * util.sqrt(x)
           - 0.1260 * x
           - 0.3516 * x * x
           + 0.2843 * x * x * x
           #- 0.1015 * x * x * x * x)
           - 0.1036 * x * x * x * x) # Closed trailing edge

def _naca_airfoil_half(t, bottom, max_camber, max_camber_position, thickness):
    """ Parametric model of one half of naca airfoil.
    Calculates x and y with given parameter t.
    If bottom is True, returns bottom half of the airfoil, else returns top side. """

    sign = -1 if top else 1

    if max_camber == 0:
        return (t, sign * thickness * _naca_thickness(t))
    else:
        phi = _naca_phi(x, max_camber_position, max_camber)
        tt = thickness * _naca_thickness(t)
        center = max_camber * _naca_center(t, max_camber_position)
        return (sign * util.sin(phi) * tt + t,
                sign * util.cos(phi) * tt + center)


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

    def distance(self, point):
        if self.max_camber != 0:
            raise NotImplementedError("Only symmetrical airfoils are supported now")

        no_camber_expr = abs(point.y) - self.thickness * _naca_thickness(point.x)

        return no_camber_expr
        #return util.derivatives.fixup_derivatives(no_camber_expr, point)

    def bounding_box(self):
        return util.BoundingBox(util.Vector(0, -self.thickness / 2, -float("inf")),
                                util.Vector(1, self.thickness / 2, float("inf")))

    def get_node(self, point, cache):
        return cache.make_node("naca_airfoil",
                               [self.thickness, self.max_camber, self.max_camber_position],
                               [point])
