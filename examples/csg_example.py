#!/usr/bin/env python3

""" Replicating the shape from wikipedia's CSG page.
https://en.wikipedia.org/wiki/Constructive_solid_geometry#/media/File:Csg_tree.png """

import codecad

sphere = codecad.shapes.sphere(130)
cube = codecad.shapes.box(100)
cylinder = codecad.shapes.cylinder(d=40, h=200)

holes = cylinder + \
        cylinder.rotated_x(90) + \
        cylinder.rotated_y(90)

o = sphere & cube - holes

if __name__ == "__main__":
    codecad.commandline_render(o)
