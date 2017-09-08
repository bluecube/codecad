#!/usr/bin/env python3

import codecad
from axes import axes_cube as cube

duration = 10
animation_progress = codecad.animation.time / duration

rotating = cube.rotated((0, 0, 1), 360 * animation_progress)
bouncing = rotating.translated(0,
                               0,
                               codecad.util.sin(codecad.util.radians(720 * animation_progress)) * 100)

if __name__ == "__main__":
    codecad.commandline_render(bouncing, 1, default_renderer="gif", duration=duration)
