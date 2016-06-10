#!/usr/bin/env python3

import codecad
import os
import itertools

airfoil_codes = [
    #2412,
    "0012",
    "0015",
    #2415,
    #4415,
    #2520
    ]

limit = codecad.Box(float("inf"), float("inf"), 0.1)

airfoils = [codecad.naca_airfoil.NacaAirfoil(code) for code in airfoil_codes]
airfoil_shapes = codecad.Union(airfoil.translated(0, i, 0)
                               for i, airfoil in enumerate(airfoils))

o = ((airfoil_shapes & limit)).rotated((1, 0, 0), 90)

o.render(codecad.rendering.RayCaster(os.path.splitext(__file__)[0] + ".png"))
#o.render(codecad.rendering.StlRenderer(os.path.splitext(__file__)[0] + ".stl", 0.02))
