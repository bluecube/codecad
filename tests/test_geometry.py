import codecad

from codecad.util.geometry import Vector, Quaternion, Transformation

def check_vector_close(v1, v2):
    avg_length = (abs(v1) + abs(v2)) / 2
    if avg_length < 1e-6:
        avg_length = 1e-6
    assert abs(v1 - v2) < avg_length * 1e-6

def c(transformation, v, target):
    v = codecad.util.Vector(*v)
    target = codecad.util.Vector(*target)

    transformed = transformation.transform_vector(v)
    check_vector_close(transformed, target)

    inv_transformation = transformation.inverse()
    check_vector_close(v, inv_transformation.transform_vector(transformed))

    zero_transformation = inv_transformation * transformation
    check_vector_close(v, zero_transformation.transform_vector(v))

def test_quaternions():
    c(Quaternion.from_degrees((0, 0, 1), 90, 1), (1, 1, 1), (-1, 1, 1))
    c(Quaternion.from_degrees((0, 0, 1), 90, 2), (1, 1, 1), (-2, 2, 2))

def test_transformations():
    c(Transformation.from_degrees((0, 0, 1), 90, 1, (5, 5, 5)), (1, 1, 1), (4, 6, 6))
    c(Transformation.from_degrees((0, 0, 1), 90, 2, (1, 0, 1)), (1, 1, 1), (-1, 2, 3))
