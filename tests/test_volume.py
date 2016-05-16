import codecad
import math

def check_close(a, b):
    return abs(a - b) < 0.005

def check_shape(shape, volume, centroid):
    print(repr(shape))
    result = codecad.volume_and_centroid(shape, 0.01)

    assert check_close(result.volume, volume)
    assert check_close(result.centroid, codecad.util.Vector(*centroid))

def test_shapes():
    cube = codecad.Box()
    cube_volume = cube.bounding_box().volume()
    check_shape(cube, cube_volume, (0, 0, 0))

    sphere = codecad.Sphere()
    sphere_volume = sphere.bounding_box().volume() * math.pi / 6
    check_shape(sphere, sphere_volume, (0, 0, 0))

    translation = codecad.util.Vector(4, 3, -5)
    translated_sphere = sphere.translated(*translation)
    check_shape(translated_sphere, sphere_volume, translation)

    union = cube + translated_sphere
    union_volume = cube_volume + sphere_volume
    union_centroid = translation * sphere_volume / union_volume
    check_shape(union, union_volume, union_centroid)
