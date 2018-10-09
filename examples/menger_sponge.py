#!/usr/bin/env python3
""" Generates a menger spong up to given number of itreations. """

import codecad
from codecad.shapes import *

import cube_thingie


def sponge(iteration):
    if iteration < 0:
        raise ValueError("Iteration must be positive or zero")

    outline = box()

    if iteration == 0:
        return outline

    bar = box(1 / 3, 1 / 3, float("inf"))
    cross = bar + bar.rotated_x(90) + bar.rotated_y(90)

    scales = ((1 / 3) ** i for i in range(iteration))

    negative = union(
        codecad.shapes.unsafe.Repetition(cross.scaled(s), (s, s, s)) for s in scales
    )

    return outline - negative


o = cube_thingie.cube_with_base(sponge(4))

if __name__ == "__main__":
    codecad.commandline_render(o)
