#!/usr/bin/env python3

import codecad
import codecad.shapes as s
import fractions
import math

line_width = 0.45
layer_height = 0.2

module = 1.5

min_wall_thickness = 2 * line_width
wall_thickness = 3
min_floor_thickness = layer_height * math.ceil(0.6 / layer_height)

n_bevel1 = 11
n_bevel2 = 40
n_sun = 11
n_planet1 = 26
n_planet2 = 12
n_ring = 49

print("Transmission ratio: 1 / {}".format((n_ring / n_planet2) * (n_planet1 / n_sun) * (n_bevel2 / n_bevel1)))

bevel_thickness = 5
ring_thickness = 20
sun_thickness = 10

tip_clearance = 0.5
backlash = 0.2

# General clearance between moving parts. Corresponds to thickness of M4 washer
# So it can be used as a spacer
clearance = 0.8

planet_count = 3

total_od = 100

# 624 bearing
bearing_id = 4
bearing_od = 13
bearing_thickness = 5
bearing_shoulder_size = 1.5
bearing_clearance = 0

#M3 screw
screw_diameter = 3

# For the planetary gearbox to mesh properly
assert(n_sun + n_planet1 + n_planet2 == n_ring)

# Sharing wear
assert(fractions.gcd(n_bevel1, n_bevel2) == 1)
assert(fractions.gcd(n_sun, n_planet1) == 1)
assert(fractions.gcd(n_planet2, n_ring) == 1)

# For assembly of planet2 / ring interface
assert(module * n_planet2 - 2 * 0.5 * module - 2 * tip_clearance > bearing_od + 2 * min_wall_thickness)

# To fit the largest diameter into the intended max housing diameter
assert(module * (n_sun + 2 * n_planet1) + module <= total_od - min_wall_thickness - tip_clearance);

bearing_shoulder_thickness = min_floor_thickness
bearing_wall_thickness = bearing_thickness + bearing_shoulder_thickness
inner_carrier_half_length = ((n_bevel1 + 3) * (module + bevel_thickness / n_bevel1) / 2) + clearance
outer_carrier_length = sun_thickness + ring_thickness + 3 * clearance + bearing_wall_thickness
sun_od = module * n_sun + module * 2 * 1.5
outer_carrier_od = (n_sun + n_planet1) * module + bearing_od + 2 * min_wall_thickness
planet_radius = (n_sun + n_planet1) * module / 2
outer_carrier_screw_radius = wall_thickness + screw_diameter / 2 - outer_carrier_od / 2

bearing_hole = s.cylinder(d = bearing_od + 2 * bearing_clearance,
                          h = 2 * bearing_thickness, symmetrical=False) + \
               s.cylinder(d = bearing_od - 2 * bearing_shoulder_size,
                          h = float("inf"))
screw_holes = s.cylinder(h = float("inf"),
                         d = screw_diameter).translated(outer_carrier_screw_radius, 0, 0).rotated((0, 0, 1), 360, planet_count)

def make_planet():
    # Gears are profile shifted by 0.5 module (smaller gear has larger addendum)
    h = sun_thickness
    gear1 = s.gears.involute_gear(tooth_count = n_planet1,
                                  module = module,
                                  addendum_modules = 0.5,
                                  dedendum_modules = 1.5,
                                  backlash = backlash / 2,
                                  clearance = tip_clearance).extruded(h, symmetrical = False)

    h += clearance
    h += ring_thickness
    gear2 = s.gears.involute_gear(tooth_count = n_planet2,
                                  module = module,
                                  addendum_modules = 1.5,
                                  dedendum_modules = 0.5,
                                  backlash = backlash / 2,
                                  clearance = tip_clearance).extruded(h, symmetrical = False)

    h += clearance
    spacer = s.cylinder(h, bearing_id + 2 * bearing_shoulder_size, symmetrical = False)

    hole = s.cylinder(float("inf"), bearing_id)

    return s.union([gear1, gear2, spacer]) - hole

def make_sun():
    h = bearing_wall_thickness + clearance
    spacer1 = s.cylinder(h, sun_od, symmetrical=False)

    h += sun_thickness + 2 * clearance
    gear = s.gears.involute_gear(tooth_count = n_sun,
                                 module = module,
                                 addendum_modules = 1.5,
                                 dedendum_modules = 0.5,
                                 backlash = backlash / 2,
                                 clearance = tip_clearance).extruded(h, symmetrical = False)

    h += ring_thickness + clearance
    spacer2 = s.cylinder(h, bearing_id + 2 * bearing_shoulder_size, symmetrical=False)

    hole = s.cylinder(float("inf"), bearing_id)

    return s.union([spacer1, spacer2, gear]) - hole

def make_carrier_inner():
    side = s.cylinder(h = inner_carrier_half_length, d = total_od, symmetrical = False) -\
           s.cylinder(h = float("inf"), d = total_od - 2 * wall_thickness)
    cap = s.cylinder(h = bearing_wall_thickness, d = total_od, symmetrical = False)

    sun_hole = s.cylinder(d = sun_od + clearance * 2, h = float("inf"))
    cap -= sun_hole

    moved_bearing_hole = bearing_hole.rotated((1, 0, 0), 180).translated(0, 0, bearing_thickness)
    cap -= moved_bearing_hole.translated(planet_radius, 0, 0).rotated((0, 0, 1), 360, planet_count)

    cap -= screw_holes

    return side + cap

def make_carrier_outer():
    side = s.cylinder(h = outer_carrier_length, d = outer_carrier_od, symmetrical = False)
    cap = s.cylinder(h = bearing_wall_thickness, d = outer_carrier_od, symmetrical = False)

    sun_hole = s.cylinder(h = float("inf"), d = outer_carrier_od - 2 * wall_thickness)
    side -= sun_hole

    moved_bearing_hole = bearing_hole.translated(0, 0, bearing_shoulder_thickness)
    cap -= moved_bearing_hole
    cap -= moved_bearing_hole.translated(planet_radius, 0, 0).rotated((0, 0, 1), 360, planet_count)

    planet_hole = s.cylinder(2 * sun_thickness + 2 * clearance,
                             module * n_planet1 + 2 * (module * 0.5 + clearance)) +\
                  s.cylinder(float("inf"),
                              module * n_planet2 + 2 * (module * 1.5 + clearance))
    side -= planet_hole.translated(planet_radius, 0, outer_carrier_length).rotated((0, 0, 1), 360, planet_count)

    screw_tube = s.cylinder(h = outer_carrier_length,
                            d = screw_diameter + 2 * wall_thickness,
                            symmetrical=False)

    body = side + cap + screw_tube.translated(outer_carrier_screw_radius, 0, 0).rotated((0, 0, 1), 360, planet_count)
        #TODO: Rounded unions

    body -= screw_holes

    return body

def make_ring():
    h = bearing_wall_thickness
    cap = s.cylinder(h = h, d = total_od, symmetrical = False)
    cap -= bearing_hole.translated(0, 0, bearing_shoulder_thickness)

    h += bearing_wall_thickness + clearance
    outer = (s.circle(total_od) -
             s.circle(outer_carrier_od + 2 * clearance)).extruded(h, symmetrical=False)

    h += ring_thickness + clearance
    gear = (s.circle(total_od) - \
            s.gears.involute_gear(tooth_count = n_ring,
                         module = module,
                         addendum_modules = 1.5,
                         dedendum_modules = 0.5,
                         backlash = backlash / 2,
                         clearance = tip_clearance,
                         internal=True)).extruded(h, symmetrical = False)

    h += clearance + sun_thickness
    inner = (s.circle(total_od) - \
             s.circle(total_od - 2 * min_wall_thickness)).extruded(h, symmetrical=False)

    return cap + outer + gear + inner

planet_moved = make_planet().translated(planet_radius, 0, inner_carrier_half_length + clearance)

o = s.union([make_carrier_inner().rotated((1, 0, 0), 180).translated(0, 0, inner_carrier_half_length),
             make_carrier_outer().rotated((1, 0, 0), 180).translated(0, 0, outer_carrier_length + inner_carrier_half_length),
             make_sun().translated(0, 0, inner_carrier_half_length - bearing_wall_thickness - clearance),
             make_ring().rotated((1, 0, 0), 180).translated(0, 0, outer_carrier_length + inner_carrier_half_length + bearing_wall_thickness + clearance)
             ]
            + [planet_moved.rotated((0, 0, 1), 360 * i / planet_count) for i in range(planet_count)]
            )

o = (o & s.half_space()).rotated((1, 0, 0), 30)

#o = (make_carrier_inner()
#     & s.half_space()
#     ).rotated((1, 0, 0), 30)

if __name__ == "__main__":
    codecad.commandline_render(o, 0.1)

