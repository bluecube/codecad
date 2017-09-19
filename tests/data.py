""" Contains shapes for other tests to use. """
import pytest
from codecad.shapes import *

import test_simple2d  # For polygons

nonconvex = polygon2d([(0, 0), (4, 6), (4, -2), (-4, -2), (-4, 6)])
csg_thing = cylinder(h=5, d=2, symmetrical=False).rotated((1, 2, 3), 15) & \
            sphere(d=3) + \
            box(2).translated(2, 0, 0)

shapes_2d = {"rectangle": rectangle(2, 4),
             "circle": circle(4),
             "nonconvex_offset_outside": nonconvex.offset(2),
             "nonconvex_offset_inside1": nonconvex.offset(-0.9),
             "nonconvex_offset_inside2": nonconvex.offset(-1.1),  # This one separates into two volumes
             "nonconvex_shell1": nonconvex.shell(1),
             "nonconvex_shell2": nonconvex.shell(2.5),  # Has two holes
             "gear": gears.InvoluteGear(20, 0.5),
             }
shapes_2d.update(("polygon2d_" + k, polygon2d(v)) for k, v in test_simple2d.valid_polygon_cases.items())
params_2d = [pytest.param(v, id=k) for k, v in sorted(shapes_2d.items())]

shapes_3d = {"sphere": sphere(4),
             "box": box(2, 3, 5),
             "drunk_box": box(2, 3, 5).rotated((7, 11, 13), 17),
             "translated_cylinder": cylinder(d=3, h=5).translated(0, 1, -1),
             "csg_thing": csg_thing,
             "torus": circle(d=4).translated(3, 0).revolved(),
             "empty_intersection": sphere().translated(-2, 0, 0) & sphere().translated(2, 0, 0)}
params_3d = [pytest.param(v, id=k) for k, v in sorted(shapes_3d.items())]
