""" Contains shapes for other tests to use. """
import pytest
from codecad.shapes import *

import test_polygons2d


def bin_counter(n):
    """ Generate a 2D shape that contains binary representation of n """
    blip = circle(d=0.75)
    blips = []
    bit_length = 0
    while n > 0:
        if n & 1:
            blips.append(blip.translated_y(0.5 + bit_length))
        n //= 2
        bit_length += 1
    bit_length = max(bit_length, 1)

    base = rectangle(1, bit_length).translated_y(bit_length / 2) + circle(
        d=0.2
    ).translated_x(0.5)

    if len(blips):
        return base - union(blips)
    else:
        return base


nonconvex = polygon2d([(0, 0), (4, 6), (4, -2), (-4, -2), (-4, 6)])
csg_thing = (
    cylinder(h=5, d=2, symmetrical=False).rotated((1, 2, 3), 15) & sphere(d=3)
) + box(2).translated(.5, 0, -.5)
mirror_2d = rectangle(1, 4).translated_x(-0.5) + circle(r=1).translated_y(-1)
mirror_2d = union(
    [
        mirror_2d,
        mirror_2d.mirrored_x().translated_x(5),
        mirror_2d.mirrored_y().translated_y(5),
    ]
)

mirror_3d = box(1, 4, 4).translated_x(-0.5) + sphere(r=1).translated(0, -1, -1)
mirror_3d = union(
    [
        mirror_3d,
        mirror_3d.mirrored_x().translated_x(5),
        mirror_3d.mirrored_y().translated_y(5),
        mirror_3d.mirrored_z().translated_z(5),
    ]
)

shapes_2d = {
    "rectangle": rectangle(2, 4),
    "circle": circle(4),
    "nonconvex_offset_outside": nonconvex.offset(2),
    "nonconvex_offset_inside1": nonconvex.offset(-0.9),
    "nonconvex_offset_inside2": nonconvex.offset(
        -1.1
    ),  # This one separates into two volumes
    "nonconvex_shell1": nonconvex.shell(1),
    "nonconvex_shell2": nonconvex.shell(2.5),  # Has two holes
    "gear": gears.InvoluteGear(20, 0.5),
    "mirror_2d": mirror_2d,
    "bin_counter_11": bin_counter(11),
    "regular_polygon3": regular_polygon2d(3),
    "symmetrical_xy": circle(d=2).translated(2, 1.75).symmetrical_x().symmetrical_y(),
    "rotated_pattern_2d": circle(1, 1).translated_x(2).rotated(270, 3),
}
shapes_2d.update(
    ("polygon2d_" + k, polygon2d(v)) for k, v in test_polygons2d.valid_polygon2d.items()
)
params_2d = [pytest.param(v, id=k) for k, v in sorted(shapes_2d.items())]

shapes_3d = {
    "sphere": sphere(4),
    "box": box(2, 3, 5),
    "drunk_box": box(2, 3, 5).rotated((7, 11, 13), 17),
    "translated_cylinder": cylinder(d=3, h=5).translated(0, 1, -1),
    "csg_thing": csg_thing,
    "torus": circle(d=4).translated_x(3).revolved(),
    "empty_intersection": sphere().translated_x(-2) & sphere().translated_x(2),
    "nested_transformations": (box().translated_z(-2) + sphere().translated_x(2))
    .rotated_y(45)
    .rotated_x(45),
    "mirror_3d": mirror_3d,
    "revolved_pentagon": regular_polygon2d(5).revolved(2),
    "extreme_twisted_revolve": rectangle(1, 0.1).revolved(2, 19 * 180),
    "symmetrical_xyz": sphere(d=2)
    .translated(2, 2, 2)
    .symmetrical_x()
    .symmetrical_y()
    .symmetrical_z(),
    "rotated_pattern_3d": box(1, 1, 1).translated_x(2).rotated((1, -1, 0), 270, 3),
}
params_3d = [pytest.param(v, id=k) for k, v in sorted(shapes_3d.items())]
