#!/usr/bin/env python3

""" Showing off gear generator and assemblies.

This example is easily be 3D printable, all parts should just snap together
with skateboard bearings. """

import codecad
from codecad.shapes import *

gear1 = codecad.shapes.gears.InvoluteGear(13, 2, clearance=1, backlash=0.1)
gear2 = codecad.shapes.gears.InvoluteGear(24, 2, clearance=1, backlash=0.1)

gear_distance = gear1.pitch_diameter / 2 + gear2.pitch_diameter / 2

base = rectangle(gear_distance + 20, 20) \
       .offset(20) \
       .extruded(10, symmetrical=False)
bearing_hole = (cylinder(d=22, h=14) + cylinder(d=20, h=16)).translated_z(10)
base -= bearing_hole.translated_x(-gear_distance / 2)
base -= bearing_hole.translated_x(gear_distance / 2)

bearing_knob = (cylinder(d=8, h=16) + cylinder(d=10, h=2)).translated_z(10)
gear1 = (gear1.extruded(10, symmetrical=False) + bearing_knob)
gear2 = (gear2.extruded(10, symmetrical=False) + bearing_knob)

assembly = codecad.Assembly()
assembly.add(base.make_part("base"))
assembly.add(gear1.make_part("gear1").rotated_x(180).translated(-gear_distance / 2, 0, 21))
assembly.add(gear2.make_part("gear2").rotated_x(180).translated(gear_distance / 2, 0, 21))

if __name__ == "__main__":
    codecad.commandline_render(assembly.shape(), 0.1)
