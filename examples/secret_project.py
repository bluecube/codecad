#!/usr/bin/env python3

import codecad
import os
import math

def mesh(outline, spacing, diameter, rounding = None):
    if rounding is None:
        rounding = 2 * diameter

    cylinder = codecad.Cylinder(d=diameter, h=float("inf"))

    grid_list = []

    z_grid = codecad.unsafe.Repetition(cylinder, (spacing, spacing, None))
    grid_list.append(z_grid)
    grid_list.append(z_grid.rotated((0, 1, 0), 90))
    grid_list.append(z_grid.rotated((1, 0, 0), 90))

    xz_grid1 = codecad.unsafe.Repetition(cylinder,
                                         (math.sqrt(2) * spacing, 2 * spacing, None)
                                         ).rotated((0, 1, 0), 45)
    xz_grid = xz_grid1 + xz_grid1.translated(spacing, spacing, 0)
    grid_list.append(xz_grid)
    grid_list.append(xz_grid.rotated((0, 1, 0), 90))

    xy_grid = xz_grid.rotated((1, 0, 0), 90)
    grid_list.append(xy_grid)
    grid_list.append(xy_grid.rotated((0, 0, 1), 90))

    yz_grid = xz_grid.rotated((0, 0, 1), 90)
    grid_list.append(yz_grid)
    grid_list.append(yz_grid.rotated((1, 0, 0), 90))


    grids = codecad.Union(grid_list)

    if rounding > 0:
        spheres = codecad.unsafe.Repetition(codecad.Sphere(rounding), (spacing, spacing, spacing))
        grids = codecad.Union([grids, spheres], rounding)

    return outline & grids


limit = codecad.Box(float("inf"), float("inf"), 100)
#airfoil = codecad.naca_airfoil.NacaAirfoil("0024").scaled(100)
airfoil = codecad.Cylinder(float("inf"), 100)

shell = (airfoil.shell(0.5, 0.5) & limit).rotated((1, 0, 0), 90)

m = mesh((airfoil & limit).rotated((1, 0, 0), 90),
         spacing  = 15,
         diameter = 1)

o = codecad.Union([m, shell], 2)

#o.render(codecad.rendering.RayCaster(os.path.splitext(__file__)[0] + ".png",
#         resolution=0.1))
o.render(codecad.rendering.StlRenderer(os.path.splitext(__file__)[0] + ".stl", 0.3))
