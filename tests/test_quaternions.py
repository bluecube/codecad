import codecad

def check_vector_close(v1, v2):
    avg_length = (abs(v1) + abs(v2)) / 2
    assert abs(v1 - v2) < avg_length / 1000

def c(axis, angle, scale, v, target):
    quat = codecad.util.Quaternion.from_degrees(codecad.util.Vector(*axis), angle, scale)

    v = codecad.util.Vector(*v)
    target = codecad.util.Vector(*target)

    transformed = quat.rotate_vector(v)

    check_vector_close(transformed, target)

    inv_quat = quat.inverse()

    check_vector_close(v, inv_quat.rotate_vector(transformed))


def test_quaternions():
    c((0, 0, 1), 90, 1, (1, 1, 1), (-1, 1, 1))
    c((0, 0, 1), 90, 2, (1, 1, 1), (-2, 2, 2))
