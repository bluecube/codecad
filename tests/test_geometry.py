import math

import pytest
from pytest import approx

import codecad
from codecad.util.geometry import Vector, BoundingBox, Quaternion, Transformation

quaternions_to_test = [
    (Quaternion.from_degrees((0, 0, 1), 90, 1), (1, 1, 1), (-1, 1, 1)),
    (Quaternion.from_degrees((0, 0, 1), 90, 2), (1, 1, 1), (-2, 2, 2)),
    (Quaternion.from_degrees((0, 1, 0), -45, 2) * Quaternion.from_degrees((0, 0, 1), 45, 0.5),
        (1, 0, 0), (0.5, math.sqrt(0.5), 0.5)),
    ]
quaternions_to_test = [(t, codecad.util.Vector(*a), codecad.util.Vector(*b)) for (t, a, b) in quaternions_to_test]

transformations_to_test = [
    (Transformation.zero(), (9, 8, 7), (9, 8, 7)),
    (Transformation.from_degrees((0, 0, 1), 0, 1, (0, 0, 0)), (2, 3, 4), (2, 3, 4)),
    (Transformation.from_degrees((0, 0, 1), 90, 1, (5, 5, 5)), (1, 1, 1), (4, 6, 6)),
    (Transformation.from_degrees((0, 0, 1), 90, 2, (1, 0, 1)), (1, 1, 1), (-1, 2, 3)),
    (Transformation.from_degrees((1, 0, 0), 0, 1, (1, 0, 0)) * Transformation.from_degrees((0, 0, 1), 90, 1, (0, 0, 0)),
        (0, 0, 0), (1, 0, 0)),
    (Transformation.from_degrees((0, 0, 1), 90, 1, (0, 0, 0)) * Transformation.from_degrees((1, 0, 0), 0, 1, (1, 0, 0)),
        (0, 0, 0), (0, 1, 0))
]
transformations_to_test = [(t, codecad.util.Vector(*a), codecad.util.Vector(*b)) for (t, a, b) in transformations_to_test]

all_to_test = quaternions_to_test + transformations_to_test


@pytest.mark.parametrize("transformation, v, target", all_to_test)
def test_vector_transformation(transformation, v, target):
    assert transformation.transform_vector(v) == approx(target)


@pytest.mark.parametrize("transformation, v, target", all_to_test)
def test_vector_transformation_inverse(transformation, v, target):
    assert transformation.inverse().transform_vector(target) == approx(v)


@pytest.mark.parametrize("transformation, v, target", all_to_test)
def test_vector_transformation_inverse_times_original(transformation, v, target):
    assert (transformation.inverse() * transformation).transform_vector(v) == approx(v)


@pytest.mark.parametrize("transformation, v, target", all_to_test)
def test_vector_transformatio_matrix(transformation, v, target):
    matrix_transformed = transformation.as_matrix() * v.as_matrix()
    assert codecad.util.Vector(*matrix_transformed.A1[:3]) == approx(target)


@pytest.mark.parametrize("transformation, v, target", all_to_test)
def test_vector_transformation_times_zero(transformation, v, target):
    assert (transformation * transformation.zero()).transform_vector(v) == approx(target)
    assert (transformation.zero() * transformation).transform_vector(v) == approx(target)


def test_empty_bounding_box_intersection_zero_volume():
    box1 = BoundingBox(Vector(-2, -2, -2), Vector(-1, -1, -1))
    box2 = BoundingBox(Vector(1, 1, 1), Vector(2, 2, 2))

    assert box1.intersection(box2).volume() == 0
