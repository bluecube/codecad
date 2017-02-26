import codecad
import math
from pytest import approx

def check_shape(shape, volume, centroid = None):
    print(repr(shape))
    result = codecad.volume_and_centroid(shape, .01)

    assert result.volume == approx(volume, rel=0.01)

    if centroid is not None:
        assert result.centroid == approx(codecad.util.Vector(*centroid), rel=0.01)

def test_shapes():
    cube = codecad.shapes.box()
    cube_volume = cube.bounding_box().volume()
    check_shape(cube, cube_volume, (0, 0, 0))

    sphere = codecad.shapes.sphere()
    sphere_volume = sphere.bounding_box().volume() * math.pi / 6
    check_shape(sphere, sphere_volume, (0, 0, 0))

    translation = codecad.util.Vector(4, 3, -5)
    translated_sphere = sphere.translated(*translation)
    check_shape(translated_sphere, sphere_volume, translation)

    union = cube + translated_sphere
    union_volume = cube_volume + sphere_volume
    union_centroid = translation * sphere_volume / union_volume
    check_shape(union, union_volume, union_centroid)

    rotation = union.rotated((0, 1, 1), 30)
    print(rotation.bounding_box())
    check_shape(rotation, union_volume)
