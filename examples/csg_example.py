#!/usr/bin/env python3

import codecad

sphere = codecad.Sphere(1.3)
cube = codecad.Box()
cylinder = codecad.Cylinder(d = 0.4, h=2)

holes = cylinder.scaled(1.5) + \
        cylinder.rotated((1, 0, 0), 90) + \
        cylinder.rotated((0, 1, 0), 90).translated(0, 0.2, 0)

o = sphere & cube - holes

if __name__ == "__main__":
    codecad.commandline_render(o, 0.05)
