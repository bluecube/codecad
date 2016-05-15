import codecad
import math

def check_close(a, b):
    return abs(a - b) < 0.001

def check_shape(shape, volume, centroid):
    print(repr(shape))
    result = codecad.volume_and_centroid(shape, 0.1)

    assert check_close(result.volume, volume)
    assert all(check_close(x, y) for x, y in zip(result.centroid, centroid))

def test_shapes():
    box = codecad.Box()
    box_volume = box.bounding_box().volume()
    check_shape(box, box_volume, (0, 0, 0))

    sphere = codecad.Sphere()
    sphere_volume = box.bounding_box().volume() * math.pi / 6
    check_shape(box, sphere_volume, (0, 0, 0))

    translation = codecad.util.Vector(4, 3, -5)
    translated_sphere = sphere.translated(*translation)
    check_shape(translated_sphere, sphere_volume, translation)

    union = cube + translated_sphere
    union_volume = cube_volume + sphere_volume
    union_centroid = translation * spere_volume / union_volume
    check_shape(union, union_volume, union_centroid)
