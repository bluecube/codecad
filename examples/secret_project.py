#!/usr/bin/env python3

import codecad
import os
import math

def mesh(outline, spacing, diameter, rounding = None):
    if rounding is None:
        rounding = diameter

    cylinder = codecad.Cylinder(d=diameter, h=float("inf"))

    grids = []

    z_grid = codecad.unsafe.Repetition(cylinder, (spacing, spacing, None))
    grids.append(z_grid)
    grids.append(z_grid.rotated((0, 1, 0), 90))
    grids.append(z_grid.rotated((1, 0, 0), 90))

    xz_grid1 = codecad.unsafe.Repetition(cylinder.rotated((0, 1, 0), 45),
                                         (2 * spacing, 2 * spacing, None))
    xz_grid = xz_grid1 + xz_grid1.translated(spacing, spacing, 0)
    grids.append(xz_grid)
    grids.append(xz_grid.rotated((0, 1, 0), 90))

    xy_grid = xz_grid.rotated((1, 0, 0), 90)
    grids.append(xy_grid)
    grids.append(xy_grid.rotated((0, 0, 1), 90))

    yz_grid = xz_grid.rotated((0, 0, 1), 90)
    grids.append(yz_grid)
    grids.append(yz_grid.rotated((1, 0, 0), 90))

    return outline & codecad.Union(grids, rounding)


outline = codecad.Box(35e-3)

o = mesh(outline,
         spacing  = 15e-3,
         diameter = 1e-3,
         rounding = 0)

o.render(codecad.rendering.RayCaster(os.path.splitext(__file__)[0] + ".png",
         mode=codecad.rendering.RayCaster.dot,
         resolution=0.1e-3))
#o.render(codecad.rendering.StlRenderer(os.path.splitext(__file__)[0] + ".stl", 0.5e-3))
