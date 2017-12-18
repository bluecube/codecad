import math
import pytest

import codecad

import data


def test_revolve_bounding_box():
    box = codecad.shapes.rectangle(2, 4).translated_y(2).revolved(10, twist=0).bounding_box()
    for a, expected in zip(box.a, codecad.util.Vector(-11, 0, -11)):
        assert a == pytest.approx(expected)
    for b, expected in zip(box.b, codecad.util.Vector(11, 4, 11)):
        assert b == pytest.approx(expected)


def test_twisted_revolve_bounding_box():
    box = codecad.shapes.rectangle(2, 4).translated_y(2).revolved(10, twist=180).bounding_box()
    r = math.hypot(1, 4)
    v = codecad.util.Vector(10 + r, r, 10 + r)
    for a, b, expected in zip(box.a, box.b, v):
        assert a == pytest.approx(-expected)
        assert b == pytest.approx(expected)
