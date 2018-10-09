#!/usr/bin/env python3
""" CodeCad logo! """

import codecad
from codecad.shapes import *

width = 0.6
stroke = 0.2

# Build the C shape  1 unit high
c = (
    rectangle(width, 1 - width + stroke / 2)
    + circle(d=width).translated_y((1 - width) / 2)
    + circle(d=width).translated_y(-(1 - width) / 2)
)
c = c.shell(stroke)
c -= rectangle(width, 1 - width).translated_x(width / 2)

# Scale it to 1 unit sized cube and prepare for adding
c = c.scaled(0.6).extruded(0.12).rotated_x(90).translated_y(-0.5)

# Put the logo together
logo = box() + c + c.rotated_z(90)
logo = logo.scaled(100).rotated_z(-45).rotated_x(15)

if __name__ == "__main__":
    codecad.commandline_render(logo)
