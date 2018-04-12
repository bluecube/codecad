import codecad
import math
import pytest
from pytest import approx


def test_cube_bbox():
    cube = codecad.shapes.box().rotated((0, 0, 1), 45)
    assert cube.bounding_box().size() == approx(codecad.util.Vector(math.sqrt(2), math.sqrt(2), 1))


@pytest.mark.xfail()
def test_repeated_rotation_bbox_tightness():
    cube = codecad.shapes.box()
    original_bbox = cube.bounding_box()

    for i in range(12):
        cube = cube.rotated_y(45)

    assert cube.bounding_box() == approx(original_bbox)
