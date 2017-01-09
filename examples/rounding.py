#!/usr/bin/env python3
import codecad

b1 = codecad.shapes.rectangle(50, 100)
b2 = codecad.shapes.rectangle(20, 80).rotated(45).translated(20, 0)

o1 = codecad.shapes.union([b1, b2])
o2 = codecad.shapes.union([b1, b2], 20)

o = o1.translated(-50, 0) + o2.translated(50, 0)

if __name__ == "__main__":
    codecad.commandline_render(o, 0.5, default_renderer="slice")
