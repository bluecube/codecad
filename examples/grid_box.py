#!/usr/bin/env python3

import codecad

boxes = codecad.shapes.unsafe.Repetition(codecad.shapes.box(1), (2, 2, 2))

o = (boxes.translated(-0.5, -0.5, -0.5) + boxes.translated(0.5, 0.5, 0.5)) & codecad.shapes.box(9.9)

print(o.bounding_box())

if __name__ == "__main__":
    codecad.commandline_render(o, 0.1)
