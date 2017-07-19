import pytest
from pytest import approx

import codecad
from codecad.util.geometry import Vector, Quaternion, Transformation

@pytest.mark.parametrize("transformation, v, target", [
    (Quaternion.from_degrees((0, 0, 1), 90, 1), (1, 1, 1), (-1, 1, 1)),
    (Quaternion.from_degrees((0, 0, 1), 90, 2), (1, 1, 1), (-2, 2, 2)),
    (Transformation.from_degrees((0, 0, 1), 90, 1, (5, 5, 5)), (1, 1, 1), (4, 6, 6)),
    (Transformation.from_degrees((0, 0, 1), 90, 2, (1, 0, 1)), (1, 1, 1), (-1, 2, 3)),
    ])
def test_transformation(transformation, v, target):
    v = codecad.util.Vector(*v)
    target = codecad.util.Vector(*target)

    transformed = transformation.transform_vector(v)
    assert transformed == approx(target)

    inv_transformation = transformation.inverse()
    assert inv_transformation.transform_vector(transformed) == approx(v)

    zero_transformation = inv_transformation * transformation
    assert zero_transformation.transform_vector(v) == approx(v)
