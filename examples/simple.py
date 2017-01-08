#!/usr/bin/env python3

import codecad

o = codecad.shapes.sphere(100)

if __name__ == "__main__":
    codecad.commandline_render(o, 1)
