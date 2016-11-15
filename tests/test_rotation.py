import codecad
import math

def test_cube_bbox():
    cube = codecad.shapes.box().rotated((0, 0, 1), 45)
    assert codecad.util.check_close(cube.bounding_box().size(),
                                    codecad.util.Vector(math.sqrt(2), math.sqrt(2), 1),
                                    0.01)
