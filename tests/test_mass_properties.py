""" Very simple smoke tests for shape evaluation and mass properties """
import math

import pytest
from pytest import approx

import codecad
import codecad.util
import codecad.mass_properties

# Shape, volume, centroid
test_data = [
    (codecad.shapes.box(1),
     1, codecad.util.Vector(0, 0, 0)),
    (codecad.shapes.cylinder(h=2, r=4, symmetrical=False),
     math.pi * 32, codecad.util.Vector(0, 0, 1)),
    (codecad.shapes.sphere(d=2),
     4 * math.pi / 3, codecad.util.Vector(0, 0, 0)),
    (codecad.shapes.box(2).translated(-15, 0, 0) + codecad.shapes.box(2).translated(15, 0, 0),
     16, codecad.util.Vector(0, 0, 0)),
    (codecad.shapes.sphere(r=2) - codecad.shapes.half_space(),
     2 * math.pi * 2**3 / 3, codecad.util.Vector(0, -6 / 8, 0)),
    (codecad.shapes.sphere(d=2).translated(10, 11, 7),
     4 * math.pi / 3, codecad.util.Vector(10, 11, 7)),
    ((codecad.shapes.sphere(r=2) - codecad.shapes.half_space()).translated(2, 0, 0).rotated((1, 0, 0), 90),
     2 * math.pi * 2**3 / 3, codecad.util.Vector(2, 0, -6 / 8)),
    ]

@pytest.mark.parametrize("shape, volume, centroid", test_data)
def test_volume_and_centroid(shape, volume, centroid):
    result = codecad.mass_properties.mass_properties(shape, .05)

    assert result.volume == approx(volume, abs=1e-2)

    if centroid is not None:
        assert result.centroid == approx(centroid, abs=1e-2)
