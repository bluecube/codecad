import math
import sys

import numpy

from .. import util
from . import base
from .. import math as ccmath
from ..cl_util import opencl_manager

_c_file = opencl_manager.add_compile_unit()
_c_file.append_resource("common.h")
_c_file.append_resource("polygons2d.cl")


class Polygon2DBuilder:
    """ Builder class for creating list of points to be interpreted as a 2D polygon. """

    def __init__(self, close_callback, x, y):
        """ Initialize the builder, starting at coordinates x, y.
        Callback will be called and its return value returned when closing the builder
        (by default it should be a Polygon2D constructor). """
        self._close_callback = close_callback

        self.points = [(x, y)]
        """ List of points currently in the builder. This is the only state of the
        geometry and can be modified at will. """

    def close(self):
        """ Finalize the polygon.
        Typically this connects the last added point to the starting point and
        returns the polygon shape. """
        return self._close_callback(self.points)

    def symmetrical_x(self, center_x):
        """ Finalize the polygon by mirroring the existing points in X and close it.

        Adds all existing points again in a reverse order with flipped X coordinate. """
        for i in reversed(range(len(self.points))):
            p = self.points[i]
            self.points.append((2 * center_x - p[0], p[1]))

        return self

    def symmetrical_y(self, center_y):
        """ Finalize the polygon by mirroring the existing points in Y and close it.

        Adds all existing points again in a reverse order with flipped Y coordinate. """
        for i in reversed(range(len(self.points))):
            p = self.points[i]
            self.points.append((p[0], 2 * center_y - p[1]))

        return self

    def block(self, modifier=lambda x: x):
        """ Start a new polygon builder from the last added point that after closing
        continues the current block.

        modifier is a function that given a list of points of the closed internal block
        returns an iterable of points to be added.

        With the default no-op modifier this is useful for delimiting parts that
        should be symmetrical. """

        # Captures self, allows returning to the original builder.
        def new_callback(new_points):
            self.points.extend(modifier(new_points))
            return self

        return self.__class__(new_callback, *self.points.pop())

    def reversed_block(self):
        """ Replace the last segment with a part that gets built from its end point
        towards its start point using the polygon builder api.

        Close the internal builder to resume processing the outer polygon """
        return self.block(reversed)

    def xy(self, x, y):
        self.points.append((x, y))
        return self

    def x(self, x):
        self.points.append((x, self.points[-1][1]))
        return self

    def y(self, y):
        self.points.append((self.points[-1][0], y))
        return self

    def dxdy(self, dx, dy):
        self.points.append((self.points[-1][0] + dx, self.points[-1][1] + dy))
        return self

    def dx(self, dx):
        return self.dxdy(dx, 0)

    def dy(self, dy):
        return self.dxdy(0, dy)

    def angle(self, angle, distance):
        """ Continue the polygon by moving in direction of the absolute angle (in degrees) """
        return self.dxdy(ccmath.cos(angle) * distance, ccmath.sin(angle) * distance)

    def angle_dx(self, angle, dx):
        """ Continue the polygon by moving in the direction of the absolute angle (in degrees)
        for dx units in X direction. """
        return self.dxdy(dx, dx * ccmath.tan(angle))

    def angle_dy(self, angle, dy):
        """ Continue the polygon by moving in the direction of the absolute angle (in degrees)
        for dy units in Y direction. """
        return self.dxdy(dy / ccmath.tan(angle), dy)

    def tangent_point(self, center_x, center_y, radius):
        """ Continue the polygon to a tangent point on a circle with given center and radius.

        Sign of the radius distinguishes the tangent points. Positive radius has the circle center
        on the right side of the line, negative radius on the left side. """
        cx = center_x - self.points[-1][0]
        cy = center_y - self.points[-1][1]
        l2 = cx * cx + cy * cy
        s = 1 - radius * radius / l2
        t = radius * math.sqrt(s / l2)

        return self.dxdy(cx * s - cy * t, cx * t + cy * s)

    def print(self, file=sys.stdout):
        """ Print the points to given file object and return self. Debugging tool. """
        print(str(self.points), file=file)
        return self


class Polygon2D(base.Shape2D):
    """ 2D simple (without self intersections) polygon.
    First and last point are implicitly connected. """

    def __init__(self, points):
        self.points = numpy.asarray(
            [util.types.wrap_vector_like(p).as_tuple2() for p in points],
            dtype=numpy.float32,
            order="c",
        )
        if self.points.shape[0] < 3:
            raise ValueError("Polygon must have at least three vertices")

        area = util.KahanSummation()
        minimum = util.Vector.splat(float("inf"))
        maximum = -minimum
        feature_size = float("inf")
        previous = util.Vector(*self.points[-1])
        for i, current in enumerate(self.points):
            current = util.Vector(*current)
            direction = current - previous
            direction_abs_squared = direction.abs_squared()
            perpendicular_direction = direction.perpendicular2d()

            area += direction.x * (previous.y + current.y) / 2

            minimum = minimum.min(current)
            maximum = maximum.max(current)

            if direction_abs_squared == 0:
                raise ValueError("Zero length segments are not allowed in polygon")

            inner_previous = current
            for j, inner_current in enumerate(self.points[i + 1 :]):
                inner_current = util.Vector(*inner_current)
                inner_direction = inner_current - inner_previous

                inner_perpendicular_direction = inner_direction.perpendicular2d()

                direction_cross_product = inner_direction.dot(perpendicular_direction)
                between_starts = previous - inner_previous

                feature_size = min(feature_size, abs(between_starts))

                consecutive = j in (0, len(points) - 2)  # i precedes or follows j
                parallel = abs(direction_cross_product) < 1e-12

                if consecutive:
                    # Consecutive segments always intersect, but we must check that they are
                    # not collinear and in opposite direction
                    if parallel and direction.dot(inner_direction) < 0:
                        raise ValueError(
                            "Polygon cannot be self intersecting (anti-parallel consecutive edges)"
                        )
                else:
                    if parallel:
                        if abs(between_starts.dot(perpendicular_direction)) < 1e-12:
                            # collinear
                            scaled_direction = direction / direction_abs_squared
                            t1 = between_starts.dot(scaled_direction)
                            t2 = t1 + inner_direction.dot(scaled_direction)

                            t1, t2 = sorted([t1, t2])

                            if t2 >= 0 and t1 <= 1:
                                raise ValueError(
                                    "Polygon cannot be self intersecting (colinear segments)"
                                )

                        else:
                            # parallel
                            pass
                    else:
                        tmp = between_starts / direction_cross_product

                        t1 = tmp.dot(inner_perpendicular_direction)
                        t2 = tmp.dot(perpendicular_direction)

                        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                            raise ValueError("Polygon cannot be self intersecting")

                inner_previous = inner_current

            previous = current

        if area.result < 0:
            self.points = numpy.flipud(self.points)

        self.box = util.BoundingBox(minimum, maximum)
        self._feature_size = feature_size

    def bounding_box(self):
        return self.box

    def feature_size(self):
        return self._feature_size

    def get_node(self, point, cache):
        return cache.make_node(
            "polygon2d", util.Concatenate([len(self.points)], self.points.flat), [point]
        )

    @classmethod
    def build(cls, origin_x, origin_y):
        """ Return a polygon builder class to help constructing the polygon as a series of steps """
        return Polygon2DBuilder(cls, origin_x, origin_y)
