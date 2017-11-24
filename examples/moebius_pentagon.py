#!/usr/bin/env python3

import codecad
from codecad.shapes import *
import math

mp = regular_polygon2d(5, d=30).revolved(r=25, twist=360/5)
half1 = (mp & half_space()).rotated_x(90).make_part("half1")
half2 = (mp - half_space()).rotated_x(-90).make_part("half2")
asm = codecad.Assembly([half1, half2.rotated_x(180)])

if __name__ == "__main__":
    codecad.commandline_render(asm, 0.5)
