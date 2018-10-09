#!/usr/bin/env python3

import codecad
import math


def mesh(spacing, diameter, rounding=None):
    if rounding is None:
        rounding = 2 * diameter

    cylinder = codecad.shapes.cylinder(d=diameter, h=float("inf"))

    grid_list = []

    z_grid = codecad.shapes.unsafe.Repetition(cylinder, (spacing, spacing, None))
    grid_list.append(z_grid)
    grid_list.append(z_grid.rotated((0, 1, 0), 90))
    grid_list.append(z_grid.rotated((1, 0, 0), 90))

    xz_grid1 = codecad.shapes.unsafe.Repetition(
        cylinder, (math.sqrt(2) * spacing, 2 * spacing, None)
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
        spheres = codecad.shapes.unsafe.Repetition(
            codecad.shapes.sphere(rounding), (spacing, spacing, spacing)
        )
        grids = codecad.shapes.union([grids, spheres], rounding)

    return grids


r = 20
h = 30
wall = 0.5
spacing = 10
rod_size = 0.8

base = (
    codecad.shapes.cylinder(h=(h - r) * 2, r=r)
    + codecad.shapes.sphere(r=r).translated(0, 0, h - r)
).offset(-wall / 2)
mask = codecad.shapes.box(4 * r, 4 * r, h).translated(0, 0, h / 2)

unmasked_inside = mesh(spacing, rod_size) & base
unmasked_skin = base.shell(wall)

o = (unmasked_inside + unmasked_skin) & mask

if __name__ == "__main__":
    codecad.commandline_render(o)
