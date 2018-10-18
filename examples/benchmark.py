#!/usr/bin/env python3

import codecad
import menger_sponge
import cube_thingie
import planetary
import airfoil

o = codecad.assembly(
    "benchmark_assembly",
    [
        planetary.Planetary(11, 60, 13, 41, 18, 53)
        .make_overview()
        .make_part("sliced_gearbox")
        .translated_x(-50),
        cube_thingie.cube_with_base(menger_sponge.sponge(6))
        .make_part("menger_sponge_statue")
        .translated_x(50),
        airfoil.o.make_part("meshed_airfoil").translated(-120, 0, 100),
    ],
)

if __name__ == "__main__":
    codecad.commandline_render(o)
