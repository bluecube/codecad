#!/usr/bin/env python3

""" A cube with axis directions engraved in the surface """

import codecad
import math


def line(x0, y0, x1, y1, t):
    dx = x1 - x0
    dy = y1 - y0
    midx = (x0 + x1) / 2
    midy = (y0 + y1) / 2
    l = math.hypot(dx, dy)
    a = math.degrees(math.atan2(dy, dx))
    return codecad.shapes.rectangle(l, t).rotated(a).translated(midx, midy)


t = 2
l = 40
depth = 1
a = 50
# arrow = line(0, 0, l, 0, t) + line(l, 0, l - 10, 10, t) + line(l, 0, l - 10, -10, t)
x = line(-5, -10, 5, 10, t) + line(-5, 10, 5, -10, t)
y = line(-5, 10, 0, 0, t) + line(5, 10, 0, 0, t) + line(0, 0, 0, -10, t)
z = line(-5, 10, 5, 10, t) + line(5, 10, -5, -10, t) + line(-5, -10, 5, -10, t)
plus = line(-5, 0, 5, 0, t) + line(0, -5, 0, 5, t)
minus = line(-5, 0, 5, 0, t)
# arrows = arrow + arrow.rotated(90) + codecad.Circle(4 * t)

# def labeled_arrows(x_label, y_label):
#     return arrows + \
#            x_label.translated(l - 5, -25) + \
#            y_label.translated(-20, l - 10)


def cut(sym1, sym2):
    flat = sym1.translated(-7.5, 0) + sym2.translated(7.5, 0)
    extruded = flat.extruded(depth + 1).translated(0, 0, (a - depth + 1) / 2)
    return extruded


cube = codecad.shapes.box(a)

axes_cube = cube - (
    cut(plus, z)
    + cut(minus, y).rotated((1, 0, 0), 90)
    + cut(plus, x).rotated((1, 0, 0), 90).rotated((0, 0, 1), 90)
    + cut(plus, y).rotated((1, 0, 0), 90).rotated((0, 0, 1), 180)
    + cut(minus, x).rotated((1, 0, 0), 90).rotated((0, 0, 1), 270)
    + cut(minus, z).rotated((0, 1, 0), 180)
)


if __name__ == "__main__":
    codecad.commandline_render(axes_cube)
