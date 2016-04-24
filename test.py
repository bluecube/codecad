#!/usr/bin/env python3

import codecad

sphere = codecad.Sphere(1.55)
cube = codecad.Box()
cylinder = codecad.Cylinder(d = 0.75, h=2)

o = sphere & cube - cylinder

o.render(codecad.rendering.RayCaster("/tmp/picture.png"))
