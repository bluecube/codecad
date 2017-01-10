#!/usr/bin/env python3

import codecad
from codecad.shapes import *
import math

base_diameter = 50
base_height = 5
base_knob_diameter = 15
base_knob_height = 5
base_chamfer = 1.2
cube_side = 50
cube_bar_diameter = 4

def base():
    bottom = rectangle(base_diameter, base_height).translated(0, base_height / 2)
    chamfer = rectangle(2 * base_chamfer, 2 * base_chamfer).rotated(-45).translated(base_diameter / 2, base_height)
    bottom = bottom - chamfer

    knob = rectangle(base_knob_diameter, base_height + base_knob_height).translated(0, (base_height + base_knob_height) / 2)
    knob_chamfer = rectangle(2 * base_chamfer, 2 * base_chamfer).rotated(-45).translated(base_knob_diameter / 2, base_height + base_knob_height)
    knob = knob - knob_chamfer

    return (bottom + knob).revolved().rotated((1, 0, 0), 90)

def cube():
    m = cube_side / 2
    bars = []
    spheres = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            bars.append(cylinder(h=cube_side, d=cube_bar_diameter).translated(i * m, j * m, 0))
            for k in [-1, 1]:
                spheres.append(sphere(d=cube_bar_diameter).translated(i * m, j * m, k * m))
    bars = union(bars)
    spheres = union(spheres)

    outer =  bars + bars.rotated((1, 0, 0), 90) + bars.rotated((0, 1, 0), 90)

    inner_bar = cylinder(h = math.sqrt(2) * cube_side, d = cube_bar_diameter).rotated((1, 0, 0), 45).translated(m, 0, 0)
    x_inner = inner_bar + inner_bar.rotated((0, 1, 0), 180)
    y_inner = x_inner.rotated((1, 0, 0), 90).rotated((0, 0, 1), 90)
    z_inner = x_inner.rotated((1, 0, 0), 90).rotated((0, 1, 0), 90)
    inner = union([x_inner, y_inner, z_inner])

    not_rotated = outer + inner + spheres

    rotated = not_rotated.rotated((1, 0, 0), 45).rotated((0, 1, 0), math.degrees(math.acos(math.sqrt(2/3))))


    return rotated.translated(0, 0, m * math.sqrt(3) + base_height)

o = base() + cube()

if __name__ == "__main__":
    codecad.commandline_render(o, 0.5)
