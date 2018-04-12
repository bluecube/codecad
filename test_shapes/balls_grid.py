#!/usr/bin/env python3
import math
import codecad

n = 3
d = 1
spacing = 2 * d
volume = n**3 * math.pi * d**3 / 6

o = codecad.shapes.sphere(d)
o = codecad.shapes.unsafe.Repetition(o, codecad.util.Vector.splat(spacing))
o &= codecad.shapes.box(n * spacing)

if __name__ == "__main__":
    #codecad.commandline_render(o)
    import logging
    logging.basicConfig()
    mp = codecad.mass_properties(o)
    print(mp)
    assert abs(mp.volume - volume) < mp.volume_error
