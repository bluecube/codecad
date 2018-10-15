#!/usr/bin/env python3

import os
import mesh

import codecad

chord = 200
span = 80
wall = 1
spacing = 20
rod_size = 2
path = os.path.join(os.path.dirname(__file__), "s1210-il.dat")

airfoil = codecad.shapes.airfoils.load_selig(path).scaled(chord)
base = airfoil.extruded(span, symmetrical=False)

inside = mesh.mesh(spacing, rod_size, -1) & base
skin = airfoil.shell(wall).extruded(span, symmetrical=False)
endcap = airfoil.extruded(wall, symmetrical=False)

o = (inside + skin + endcap).rotated_x(90)

if __name__ == "__main__":
    codecad.commandline_render(o)
