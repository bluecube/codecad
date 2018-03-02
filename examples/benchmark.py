#!/usr/bin/env python3

import codecad
import menger_sponge
import cube_thingie
import arm
import secret_project

o = arm.Joint(11, 60, 13, 41, 18, 53).make_overview().translated_x(-50) + \
    cube_thingie.cube_with_base(menger_sponge.sponge(6)).translated_x(50) + \
    secret_project.o.rotated_x(-90).scaled(2).translated_z(50).translated_y(50)

if __name__ == "__main__":
    codecad.commandline_render(o, 0.2)
