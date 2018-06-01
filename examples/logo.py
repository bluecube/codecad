#!/usr/bin/env python3
""" CodeCad logo! """

import codecad
from codecad.shapes import *

def c_generator(width, stroke):
    r = rectangle(width, 1 - width + stroke / 2)
    c = (r
         + circle(d=width).translated_y((1 - width) / 2)
         + circle(d=width).translated_y(-(1 - width) / 2))
    c = c.shell(stroke)
    c -= rectangle(width, 1 - width).translated_x(width / 2)
    return c

c = c_generator(0.6, 0.2) \
    .scaled(0.6) \
    .extruded(0.12) \
    .rotated_x(90) \
    .translated_y(-0.5)

logo = box() + c + c.rotated_z(90)
logo = logo \
    .scaled(100) \
    .rotated_z(-45) \
    .rotated_x(15)

if __name__ == "__main__":
    codecad.commandline_render(logo)
