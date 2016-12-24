""" User facing constructors for basic shapes.

Names from this module may optionally be imported as "from codecad.shapes import *".

Basic shape interface is composed of functions defined here and of methods on
Shape objects (transformations, shell, extrude ...)"""

from . import simple2d as _s2
from . import simple3d as _s3

from . import unsafe
from . import naca_airfoil

def rectangle(x = 1, y = None):
    if y is None:
        y = x
    return _s2.Rectangle(x, y)

def circle(d = 1, r = None):
    if r is not None:
        d = 2 * r
    return _s2.Circle(d)

def box(x = 1, y = None, z = None):
    if (y is None) != (z is None):
        raise ValueError("y and z must either both be None, or both be number")
    if y is None:
        y = x
        z = x
    return rectangle(x, y).extruded(z)

def sphere(d = 1, r = None):
    if r is not None:
        d = 2 * r
    return _s3.Sphere(d)

def cylinder(h = 1, d = 1, r = None):
    return circle(d = d, r = r).extruded(h)

def _group_op_helper(shapes, name, op2, op3, r):
    """ Check that shapes is not empty and that dimensions match """
    shapes = list(shapes)
    l = len(shapes)
    if l == 0:
        raise ValueError(name + " of empty set objects doesn't make much sense, does it?")
    elif l == 1:
        return shapes[0]
    else:
        dim = shapes[0].dimension()
        if any(shape.dimension() != dim for shape in shapes):
            raise ValueError(name + " needs shapes of identical dimensions")

        if dim == 2:
            return op2(shapes, r = r)
        if dim == 3:
            return op3(shapes, r = r)

def union(shapes, r = 0):
    return _group_op_helper(shapes, "Union", _s2.Union2D, _s3.Union, r)

def intersection(shapes, r = 0):
    return _group_op_helper(shapes, "Intersection", _s2.Intersection2D, _s3.Intersection, r)
