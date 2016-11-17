#!/usr/bin/env python3

import codecad

airfoil_codes = [
    #2412,
    "0012",
    "0015",
    #2415,
    #4415,
    #2520
    ]

airfoils = [codecad.shapes.naca_airfoil.NacaAirfoil(code) for code in airfoil_codes]
airfoil_shapes = codecad.shapes.union(airfoil.translated(0, i)
                                      for i, airfoil in enumerate(airfoils))

o = airfoil_shapes.extruded(0.1).rotated((1, 0, 0), 90)

if __name__ == "__main__":
    codecad.commandline_render(o, 0.05)
