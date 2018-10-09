#!/usr/bin/env python3
""" Generator of cubical statuettes and a wireframe cube """

import codecad
from codecad.shapes import *
import math

base_diameter = 50
base_height = 5
base_knob_diameter = 15
base_knob_height = 5
base_chamfer = 1.2
cube_side = 50


def tetrahedron_cube():
    d = 0.08
    bars = []
    spheres = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            bars.append(cylinder(h=1, d=d).translated(i / 2, j / 2, 0))
            for k in [-1, 1]:
                spheres.append(sphere(d=d).translated(i / 2, j / 2, k / 2))
    bars = union(bars)
    spheres = union(spheres)

    outer = bars + bars.rotated_x(90) + bars.rotated_y(90)

    inner_bar = cylinder(h=math.sqrt(2), d=d).rotated_x(45).translated_x(0.5)
    x_inner = inner_bar + inner_bar.rotated_y(180)
    y_inner = x_inner.rotated_x(90).rotated_z(90)
    z_inner = x_inner.rotated_x(90).rotated_y(90)
    inner = union([x_inner, y_inner, z_inner])

    not_rotated = outer + inner + spheres

    return not_rotated


def cube_with_base(unit_cube):
    """ Take a unit cube and turn it into a statue thingie, standing on one corner """
    m = cube_side / 2

    prepared_cube = (
        unit_cube.scaled(cube_side)
        .rotated_x(45)
        .rotated_y(math.degrees(math.acos(math.sqrt(2 / 3))))
        .translated(0, 0, cube_side * math.sqrt(3) / 2 + base_height)
    )

    bottom = rectangle(base_diameter, base_height).translated(0, base_height / 2)
    chamfer = (
        rectangle(2 * base_chamfer, 2 * base_chamfer)
        .rotated(-45)
        .translated(base_diameter / 2, base_height)
    )
    bottom = bottom - chamfer

    knob = rectangle(base_knob_diameter, base_height + base_knob_height).translated_y(
        (base_height + base_knob_height) / 2
    )
    knob_chamfer = (
        rectangle(2 * base_chamfer, 2 * base_chamfer)
        .rotated(-45)
        .translated(base_knob_diameter / 2, base_height + base_knob_height)
    )
    knob = knob - knob_chamfer

    base = (bottom + knob).revolved().rotated_x(90)

    return prepared_cube + base


o = cube_with_base(tetrahedron_cube())

if __name__ == "__main__":
    codecad.commandline_render(o)
