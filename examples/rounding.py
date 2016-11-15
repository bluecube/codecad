#!/usr/bin/env python3
import codecad

b1 = codecad.shapes.box(50, 100, 100)
b2 = codecad.shapes.box(20, 80, 50).rotated((0, 0, 1), 45).translated(20, 0, 0)

o1 = codecad.shapes.union([b1, b2])
o2 = codecad.shapes.union([b1, b2], 20)

o = o1.translated(-50, 0, 0) + o2.translated(50, 0, 0)

if __name__ == "__main__":
    codecad.commandline_render(o, 1, default_renderer="slice")
