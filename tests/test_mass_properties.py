import math
import numpy

import pytest
from pytest import approx

import codecad
import codecad.util

drunk_box_matrix = codecad.util.Quaternion.from_degrees((7, 11, 13), 17).as_matrix()[
    :3, :3
]


# Names added so that test failures have clear identification of the shape
@pytest.mark.parametrize(
    "shape, volume, centroid, inertia_tensor",
    [
        pytest.param(
            codecad.shapes.box(1),
            1,
            codecad.util.Vector(0, 0, 0),
            numpy.identity(3) * 2 / 12,
            id="unit_box",
        ),
        pytest.param(
            codecad.shapes.cylinder(h=2, r=4, symmetrical=False),
            math.pi * 32,
            codecad.util.Vector(0, 0, 1),
            numpy.diag([(3 * 4 ** 2 + 2 ** 2) / 6, (3 * 4 ** 2 + 2 ** 2) / 6, 4 ** 2])
            * math.pi
            * 16,
            id="cylinder",
        ),
        pytest.param(
            codecad.shapes.sphere(d=2),
            4 * math.pi / 3,
            codecad.util.Vector(0, 0, 0),
            numpy.identity(3) * (4 * math.pi / 3) * 2 / 5,
            id="sphere",
        ),
        pytest.param(
            codecad.shapes.box(2).translated(-15, 0, 0)
            + codecad.shapes.box(2).translated(15, 0, 0),
            16,
            codecad.util.Vector(0, 0, 0),
            None,
            id="two_boxes",
        ),
        pytest.param(
            codecad.shapes.sphere(r=2) - codecad.shapes.half_space(),
            2 * math.pi * 2 ** 3 / 3,
            codecad.util.Vector(0, -6 / 8, 0),
            None,
            id="hemisphere",
        ),
        pytest.param(
            codecad.shapes.sphere(d=2).translated(10, 11, 7),
            4 * math.pi / 3,
            codecad.util.Vector(10, 11, 7),
            None,
            id="translated_sphere",
        ),
        pytest.param(
            (codecad.shapes.sphere(r=2) - codecad.shapes.half_space())
            .translated(2, 0, 0)
            .rotated((1, 0, 0), 90),
            2 * math.pi * 2 ** 3 / 3,
            codecad.util.Vector(2, 0, -6 / 8),
            None,
            id="translated_and_rotated_hemisphere",
        ),
        pytest.param(
            codecad.shapes.box(4).translated(0, 0, 2)
            + codecad.shapes.box(2, 2, 9).translated(0, 0, -3.5),
            96,
            codecad.util.Vector(0, 0, 0),
            numpy.diag([1120, 1120, 192]),
            id="not_hammer",
        ),
        pytest.param(
            codecad.shapes.box(2, 3, 5).rotated((7, 11, 13), 17),
            2 * 3 * 5,
            codecad.util.Vector(0, 0, 0),
            drunk_box_matrix
            * (
                numpy.diag([3 ** 2 + 5 ** 2, 2 ** 2 + 5 ** 2, 2 ** 2 + 3 ** 2])
                * 2
                * 3
                * 5
                / 12
            )
            * drunk_box_matrix.T,
            id="drunk_box",
        ),
    ],
)
def test_mass_properties(shape, volume, centroid, inertia_tensor):
    precision = 2e-3
    result = codecad.mass_properties(
        shape, 10 * precision
    )  # Just experimentally selected value

    assert result.volume == approx(volume, abs=1e-4, rel=precision)
    assert result.centroid == approx(centroid, abs=1e-4, rel=precision)

    if inertia_tensor is not None:
        assert numpy.allclose(result.inertia_tensor, inertia_tensor, rtol=precision)
