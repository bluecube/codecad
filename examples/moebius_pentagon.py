#!/usr/bin/env python3
""" 3D printable pentagon revolved around an axis with a twist.
Only has one face and one edge. """

import codecad
from codecad.shapes import *
import math

minor_r = 15
major_r = 25
pin_d = 1.75
pin_h = 10
second_pin_angle = 170  # Not 180 degrees to have only one way to assemble

mp = regular_polygon2d(5, r=minor_r).revolved(r=major_r, twist=360/5)

pin_hole = cylinder(d=1.1 * pin_d, h=1.2 * pin_h).rotated_x(90).translated_x(major_r)
mp -= pin_hole
mp -= pin_hole.rotated_y(second_pin_angle)

half1 = (mp & half_space()).rotated_x(90).make_part("half1")
half2 = (mp - half_space()).rotated_x(-90).make_part("half2")

pin = cylinder(d=pin_d, h=pin_h, symmetrical=False).make_part("pin")
pin = pin.translated_z(-pin_h/2).translated_x(major_r)

asm = codecad.assembly("moebius_pentagon",
                       [half1,
                        half2.rotated_x(180),
                        pin,
                        pin.rotated_z(second_pin_angle)])

if __name__ == "__main__":
    codecad.commandline_render(asm, 0.5)
