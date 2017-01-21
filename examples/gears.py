#!/usr/bin/env python3

import codecad
import codecad.shapes as s

n1 = 30
n2 = 15
module = 2
backlash = 0.5
clearance = 1
gear_thickness = 5
bearing_id = 8
bearing_od = 22
bearing_w = 7
bearing_clearance = 0.5
base_offset = 4
base_floor = 0.5

# Generate the gear shapes
g1, g2, center_distance = s.gears.involute_gear_pair(n1, n2,
                                                     module=module,
                                                     backlash=backlash,
                                                     clearance=clearance)

# Prepate the knobs on the gears
knob_h = gear_thickness + bearing_w
knob_notch_h = gear_thickness + bearing_clearance
knob = codecad.shapes.cylinder(d=bearing_id, h=knob_h).translated(0, 0, knob_h / 2) + \
       codecad.shapes.cylinder(d=(bearing_id + bearing_od) / 2, h=knob_notch_h).translated(0, 0, knob_notch_h / 2)

# Extrude gears and add knobs
g1 = g1.extruded(gear_thickness).translated(0, 0, gear_thickness / 2) + knob
g2 = g2.extruded(gear_thickness).translated(0, 0, gear_thickness / 2) + knob

# Prepare bearing holes for the base
bearing_hole = s.cylinder(d=bearing_od, h=2*bearing_w) \
                .translated(0, 0, bearing_w + bearing_clearance) +\
               s.cylinder(d=(bearing_id + bearing_od)/2, h=2*bearing_clearance) \
                .translated(0, 0, bearing_clearance)

# Make base
base_h = base_floor + bearing_clearance + bearing_w
base = s.rectangle(center_distance + bearing_od, bearing_od) \
        .offset(base_offset) \
        .extruded(base_h) \
        .translated(0, 0, base_h / 2) - \
       bearing_hole.translated(-center_distance/2, 0, base_floor) - \
       bearing_hole.translated(+center_distance/2, 0, base_floor)


# Measure sizes for packing to output
g1_size = g1.bounding_box().size().x + 2
g2_size = g2.bounding_box().size().x + 2
base_size = base.bounding_box().size().y + 2


# Pack it
o = g1.translated(-g2_size / 2, -base_size / 2, 0) + \
    g2.translated(g1_size / 2,  -base_size / 2, 0) + \
    base.translated(0, g1_size / 2, 0)

# Rotate for display
o = o.rotated((1, 0, 0), 90)

if __name__ == "__main__":
    codecad.commandline_render(o, 0.5)
