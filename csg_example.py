#!/usr/bin/env python3

import codecad

sphere = codecad.Sphere(1.3)
cube = codecad.Box()
cylinder = codecad.Cylinder(d = 0.4, h=2)

holes = cylinder + cylinder.rotated(1, 0, 0, 90) + cylinder.rotated(0, 0, 1, 90)

o = sphere & cube - holes

#o.render(codecad.rendering.RayCaster("/tmp/picture.png"))
o.render(codecad.rendering.StlRenderer("/tmp/picture.stl", 0.01))
