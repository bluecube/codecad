#!/usr/bin/env python3

import codecad
import os

o = codecad.shapes.airfoils.load_selig(os.path.join(os.path.dirname(__file__), "s1210-il.dat"))
o = o.extruded(1).rotated((1, 0, 0), 90)

if __name__ == "__main__":
    codecad.commandline_render(o, 0.05)
