#!/usr/bin/env python3

import codecad
import os

sphere = codecad.Sphere(1.3)
cube = codecad.Box()
cylinder = codecad.Cylinder(d = 0.4, h=2)

holes = cylinder.scaled(1.5) + \
        cylinder.rotated(1, 0, 0, 90) + \
        cylinder.rotated(0, 1, 0, 90).translated(0, 0.2, 0)

o = sphere & cube - holes

#o.render(codecad.rendering.RayCaster(os.path.splitext(__file__)[0] + ".png"))
o.render(codecad.rendering.StlRenderer(os.path.splitext(__file__)[0] + ".stl", 0.05))
