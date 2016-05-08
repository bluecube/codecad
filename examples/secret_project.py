#!/usr/bin/env python3

import codecad
import os

def mesh(outline, spacing, diameter, rounding = None):
    if rounding is None:
        rounding = diameter

    cylinder = codecad.Cylinder(d=diameter, h=float("inf"))
    z_grid = codecad.unsafe.Repetition(cylinder, (spacing, spacing, None))
    x_grid = z_grid.rotated((0, 1, 0), 90)
    y_grid = z_grid.rotated((1, 0, 0), 90)

    grid = codecad.Union([x_grid, y_grid, z_grid], rounding)

    return outline & grid


outline = codecad.Box(35e-3)

o = mesh(outline, 15e-3, 1e-3)

o.render(codecad.rendering.RayCaster(os.path.splitext(__file__)[0] + ".png",
         mode=codecad.rendering.RayCaster.dot,
         resolution=0.1e-3))
#o.render(codecad.rendering.StlRenderer(os.path.splitext(__file__)[0] + ".stl", 0.5))
