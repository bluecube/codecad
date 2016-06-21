#!/usr/bin/env python3
import codecad

b1 = codecad.Box(50, 100, 100)
b2 = codecad.Box(20, 80, 50).rotated((0, 0, 1), 45).translated(20, 0, 0)

o = codecad.Union([b1, b2], 20)
#o = codecad.Union([b1, b2])

if __name__ == "__main__":
    codecad.commandline_render(o, 1, default_renderer="slice")
