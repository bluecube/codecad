import codecad
import math
from pytest import approx


def test_cube_bbox():
    cube = codecad.shapes.box().rotated((0, 0, 1), 45)
    assert cube.bounding_box().size() == approx(codecad.util.Vector(math.sqrt(2), math.sqrt(2), 1))
