#!/usr/bin/env python3

import codecad
import math

def mesh(outline, spacing, diameter, rounding = None):
    if rounding is None:
        rounding = 2 * diameter

    cylinder = codecad.shapes.cylinder(d=diameter, h=float("inf"))

    grid_list = []

    z_grid = codecad.shapes.unsafe.Repetition(cylinder, (spacing, spacing, None))
    grid_list.append(z_grid)
    grid_list.append(z_grid.rotated((0, 1, 0), 90))
    grid_list.append(z_grid.rotated((1, 0, 0), 90))

    xz_grid1 = codecad.shapes.unsafe.Repetition(cylinder,
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


    grids = codecad.shapes.union(grid_list)

    if rounding > 0:
        spheres = codecad.shapes.unsafe.Repetition(codecad.shapes.sphere(rounding), (spacing, spacing, spacing))
        grids = codecad.shapes.union([grids, spheres], rounding)

    return outline & grids


limit = codecad.shapes.box(float("inf"), float("inf"), 100)
#airfoil = codecad.naca_airfoil.NacaAirfoil("0024").scaled(100)
airfoil = codecad.shapes.cylinder(float("inf"), 100)

shell = (airfoil.shell(0.5, 0.5) & limit).rotated((1, 0, 0), 90)

m = mesh((airfoil & limit).rotated((1, 0, 0), 90),
         spacing  = 15,
         diameter = 1)

o = codecad.shapes.union([m, shell], 2)

if __name__ == "__main__":
    codecad.commandline_render(o, 0.3)
