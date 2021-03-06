""" User facing constructors for basic shapes.

Names from this module may optionally be imported as "from codecad.shapes import *".

Basic shape interface is composed of functions defined here and of methods on
Shape objects (transformations, shell, extrude ...)"""

import math

from . import simple2d as _s2
from . import simple3d as _s3
from . import polygons2d as _polygons2d

from . import unsafe
from . import gears
from . import airfoils


def rectangle(x=1, y=None):
    if y is None:
        y = x
    return _s2.Rectangle(x, y)


def circle(d=1, r=None):
    return _s2.Circle(d, r)


def half_plane():
    return _s2.HalfPlane()


def regular_polygon2d(n, d=1, r=None, side_length=None, across_flats=None):
    return _s2.RegularPolygon2D(n, d, r, side_length, across_flats)


def polygon2d(points):
    return _polygons2d.Polygon2D(points)


def polygon2d_builder(origin_x, origin_y):
    return _polygons2d.Polygon2D.build(origin_x, origin_y)


def capsule(x1, y1, x2, y2, width):
    """ Use zero thickness rectangle trick to model a 2D capsule between two points """
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    return (
        rectangle(length, 0)
        .offset(width / 2)
        .rotated(math.degrees(angle))
        .translated((x1 + x2) / 2, (y1 + y2) / 2)
    )


def box(x=1, y=None, z=None):
    if (y is None) != (z is None):
        raise ValueError("y and z must either both be None, or both be number")
    if y is None:
        y = x
        z = x
    return rectangle(x, y).extruded(z)


def sphere(d=1, r=None):
    if r is not None:
        d = 2 * r
    return _s3.Sphere(d)


def cylinder(h=1, d=1, r=None, symmetrical=True):
    return circle(d=d, r=r).extruded(h, symmetrical)


def half_space():
    return _s3.HalfSpace()


def _group_op_helper(shapes, name, op2, op3, r):
    """ Check that shapes is not empty and that dimensions match """
    shapes = list(shapes)
    if len(shapes) == 0:
        raise ValueError(
            name + " of empty set objects doesn't make much sense, does it?"
        )
    elif len(shapes) == 1:
        return shapes[0]
    else:
        dim = shapes[0].dimension()
        if any(shape.dimension() != dim for shape in shapes):
            raise ValueError(name + " needs shapes of identical dimensions")

        if dim == 2:
            return op2(shapes, r=r)
        else:
            return op3(shapes, r=r)


def union(shapes, r=-1):
    return _group_op_helper(shapes, "Union", _s2.Union2D, _s3.Union, r)


def intersection(shapes, r=-1):
    return _group_op_helper(
        shapes, "Intersection", _s2.Intersection2D, _s3.Intersection, r
    )


# pylama:ignore=W0611
