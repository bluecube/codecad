#!/usr/bin/env python3

import codecad
from codecad.shapes import *

def sponge(iteration):
    if iteration < 0:
        raise ValueError("Iteration must be positive or zero")

    outline = box()

    if iteration == 0:
        return outline

    bar = box(1/3, 1/3, float("inf"))
    cross = bar + \
            bar.rotated_x(90) + \
            bar.rotated_y(90)

    scales = ((1/3)**i for i in range(iteration))

    negative = union(codecad.shapes.unsafe.Repetition(cross.scaled(s), (s, s, s))
                     for s in scales)

    return outline - negative


o = sponge(6) & half_space().rotated_x(-80).rotated_z(45)

if __name__ == "__main__":
    codecad.commandline_render(o.rotated_z(-30).rotated_x(30), 0.001)
