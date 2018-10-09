#!/usr/bin/env python3

import codecad

rectangle = codecad.shapes.rectangle(5, 100).translated(50, 0)
circle = codecad.shapes.circle(r=20).translated(50, 50)

hole = codecad.shapes.rectangle(20).translated(50, 50)
notch = codecad.shapes.rectangle(15, 10).translated(55, 0)

shape = (rectangle + circle + notch) - hole

cutout = (
    codecad.shapes.box(200, float("inf"), 200)
    .translated(100, 0, 100)
    .rotated((0, 1, 0), -45)
)

o = shape.revolved() - cutout

o_display = o.rotated((0, 1, 0), 30).rotated((1, 0, 0), 45)

if __name__ == "__main__":
    codecad.commandline_render(o_display)
