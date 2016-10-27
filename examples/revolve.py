#!/usr/bin/env python3

import codecad

rectangle = codecad.Rectangle(5, 100).translated(50, 0)
circle = codecad.Circle(r = 20).translated(50, 50)

hole = codecad.Rectangle(20).translated(50, 50)
notch = codecad.Rectangle(15, 10).translated(55, 0)

shape = (rectangle + circle + notch) - hole

cutout = codecad.Box(200, float("inf"), 200).translated(100, 0, 100).rotated((0, 1, 0), -45)

o = shape.revolved() - cutout

o_display = o.rotated((-1, 1, 1), 60)

if __name__ == "__main__":
    codecad.commandline_render(o_display, 0.5)
