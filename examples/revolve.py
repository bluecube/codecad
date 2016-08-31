#!/usr/bin/env python3

import codecad

rectangle = codecad.Box(5, 100, float("inf")).translated(50, 0, 0)
circle = codecad.Cylinder(r = 20, h = float("inf")).translated(50, 50, 0)

hole = codecad.Box(20, 20, float("inf")).translated(50, 50, 0)
notch = codecad.Box(15, 10, float("inf")).translated(55, 0, 0)

shape = (rectangle + circle + notch) - hole

cutout = codecad.Box(200, float("inf"), 200).translated(100, 0, 100).rotated((0, 1, 0), -45)

o = shape.revolved() - cutout

o_display = o.rotated((-1, 1, 1), 60)

if __name__ == "__main__":
    codecad.commandline_render(o_display, 0.5)
