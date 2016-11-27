#!/usr/bin/env python3

raise Exception("This example is currently broken")

import codecad

cube = codecad.shapes.box(100)

rotating = cube.rotated((0, 0, 1), 36 * codecad.animation.time)
bouncing = rotating.translated(0,
                               0,
                               codecad.util.sin(codecad.util.radians(72 * codecad.animation.time)) * 100)

if __name__ == "__main__":
    codecad.commandline_render(bouncing, 1, default_renderer = "gif")
