#!/usr/bin/env python3

import codecad
import os
import secret_project

chord = 100
span = 50
wall = 0.5
spacing = 10
rod_size = 2
path = os.path.join(os.path.dirname(__file__), "s1210-il.dat")

airfoil = codecad.shapes.airfoils.load_selig(path)
base = airfoil.scaled(chord).extruded(2 * span)
mask = codecad.shapes.box(2 * chord, 2 * chord, 2 * span).translated(0, 0, span)

unmasked_inside = secret_project.mesh(spacing, rod_size, -1) & base
unmasked_skin = base.offset(-wall / 2).shell(wall)

o = (unmasked_inside + unmasked_skin) & mask

if __name__ == "__main__":
    codecad.commandline_render(o, 0.1)
