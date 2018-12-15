import math

import pytest
import numpy

import codecad.shapes.polygons2d

valid_polygon2d = {
    "triangle": [(0, 0), (3, 0), (3, 2)],
    "non_convex": [(0, 0), (3, 0), (3, 1), (2, 2), (3, 3), (0, 3)],
    "collinear_consecutive_edges": [(0, 0), (2, 0), (4, 0), (4, 3)],
    "collinear_non_consecutive_edges": [
        (0, 0),
        (3, 0),
        (3, 1),
        (2, 2),
        (3, 3),
        (3, 4),
        (0, 4),
    ],
    "square": [(0, 0), (5, 0), (5, 5), (0, 5)],
    "parallel_same_direction_edges": [(0, 0), (6, -1), (5, 5), (5, 0), (0, 5)],
}

invalid_polygon2d = {
    "edge_crossing": [(0, 0), (3, 0), (0, 3), (3, 3)],
    "point_crossing": [(0, 0), (3, 0), (2.5, 2.5), (0, 3), (3, 3), (2.5, 2.5)],
    "repeated_point_on_collinear_edges": [
        (0, 0),
        (3, 0),
        (3, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (0, 4),
    ],
    "collinear_edge_crossing": [(0, 0), (3, 0), (3, 3), (2, 2), (3, 1), (3, 4), (0, 4)],
    "repeated_point": [(0, 0), (4, 0), (4, 3), (0, 0), (1, 3), (0, 3)],
    "point_on_edge": [(0, 0), (4, 0), (4, 3), (2, 0), (0, 3)],
    "shared_edge": [(0, 0), (3, 0), (5, 0), (3, 0), (3, 3)],
    "shared_edge_part": [(0, 0), (5, 0), (3, 0), (3, 3)],
    "duplicate_point": [(0, 0), (2, 0), (2, 0), (4, 3)],
    "duplicate_point_at_start": [(0, 0), (3, 0), (3, 2), (0, 0)],
}

symmetry_builder_test_data = [
    (10, 0), (12, 1), (13, 3), (11, 2),
    (7, 2), (5, 3), (6, 1), (8, 0)
    ]

@pytest.mark.parametrize(
    "points",
    [pytest.param(v, id=k) for k, v in sorted(valid_polygon2d.items())]
    + [
        pytest.param(list(reversed(v)), id="reversed_" + k)
        for k, v in sorted(valid_polygon2d.items())
    ],
)
def test_valid_polygon_construction(points):
    codecad.shapes.polygons2d.Polygon2D(
        points
    )  # Just check that it constructs without complaining


@pytest.mark.parametrize(
    "points",
    [pytest.param(v, id=k) for k, v in sorted(invalid_polygon2d.items())]
    + [
        pytest.param(list(reversed(v)), id="reversed_" + k)
        for k, v in sorted(invalid_polygon2d.items())
    ],
)
def test_invalid_polygon_construction(points):
    with pytest.raises(ValueError):
        codecad.shapes.polygons2d.Polygon2D(points)


@pytest.mark.parametrize(
    "points",
    [pytest.param(v, id=k) for k, v in sorted(valid_polygon2d.items())]
    + [
        pytest.param(list(reversed(v)), id="reversed_" + k)
        for k, v in sorted(valid_polygon2d.items())
    ],
)
def test_valid_polygon_builder_abs_construction(points):
    codecad.shapes.polygons2d.Polygon2D(
        points
    )  # Just check that it constructs without complaining


@pytest.mark.parametrize(
    "points", [pytest.param(v, id=k) for k, v in sorted(valid_polygon2d.items())]
)
def test_polygon_builder_absolute(points):
    p1 = codecad.shapes.polygons2d.Polygon2D(points)

    builder = codecad.shapes.polygon2d_builder(points[0][0], points[0][1])
    for x, y in points[1:]:
        builder = builder.xy(x, y)
    p2 = builder.close()

    assert p1.points.tolist() == p2.points.tolist()


@pytest.mark.parametrize(
    "name, args, point",
    [
        ("xy", (15, 5), (15, 5)),
        ("x", (15,), (15, 1)),
        ("y", (5,), (1, 5)),
        ("dxdy", (5, 1), (6, 2)),
        ("dx", (10,), (11, 1)),
        ("dy", (-2,), (1, -1)),
        # Values copied from previous runs, looking plausible
        ("angle", (-120, 8), (-3, -5.92820323027551)),
        ("angle_dx", (-30, 3), (4, -0.7320508075688772)),
        ("angle_dy", (-30, -3), (6.196152422706632, -2)),
        ("tangent_point", (6, -4, 5), (6, 1)),
        ("tangent_point", (6, -4, -5), (1, -4)),
        # Negative arguments for angles
        ("angle", (135, -math.sqrt(2)), (2, 0)),
        ("angle_dx", (-45, 1), (2, 0)),
        ("angle_dx", (135, 1), (2, 0)),
        ("angle_dy", (-45, -1), (2, 0)),
        ("angle_dy", (135, -1), (2, 0)),
    ],
)
def test_polygon_builder_step(name, args, point):
    builder_before = codecad.shapes.polygon2d_builder(1, 1)
    builder_after = getattr(builder_before, name)(*args)

    assert callable(builder_after.close), "The returned object must still be a polygon builder to support chaining"
    assert builder_after.points[-1] == pytest.approx(point)


def test_polygon_builder_symmetrical_x():
    points = symmetry_builder_test_data

    builder = codecad.shapes.polygon2d_builder(points[0][0], points[0][1])
    for x, y in points[1:len(points) // 2]:
        builder = builder.xy(x, y)
    builder = builder.symmetrical_x((points[0][0] + points[-1][0]) / 2)

    assert callable(builder.close), "The returned object must still be a polygon builder to support chaining"
    assert builder.points == points


def test_polygon_builder_symmetrical_y():
    points = [(y, x) for (x, y) in symmetry_builder_test_data]

    builder = codecad.shapes.polygon2d_builder(points[0][0], points[0][1])
    for x, y in points[1:len(points) // 2]:
        builder = builder.xy(x, y)
    builder = builder.symmetrical_y((points[0][1] + points[-1][1]) / 2)

    assert callable(builder.close), "The returned object must still be a polygon builder to support chaining"
    assert builder.points == points


def test_polygon_builder_nop_block():
    points = symmetry_builder_test_data
    builder = codecad.shapes.polygon2d_builder(points[0][0], points[0][1])

    for x, y in points[1:len(points) // 2]:
        builder = builder.xy(x, y)
    builder = builder.block()
    for x, y in points[len(points) // 2:]:
        builder = builder.xy(x, y)
    builder = builder.close()

    assert builder.points == points


def test_polygon_builder_block_partial_symmetry():
    builder = codecad.shapes.polygon2d_builder(0, 0) \
        .xy(9, 0) \
        .block() \
        .xy(8, 1) \
        .symmetrical_x(10) \
        .close() \
        .xy(10, -1)

    assert builder.points == [(0, 0), (9, 0), (8, 1), (12, 1), (11, 0), (10, -1)]

def test_polygon_builder_reversed_block():
    builder = codecad.shapes.polygon2d_builder(0, 0) \
        .xy(10, 10) \
        .reversed_block() \
        .dx(-2) \
        .close() \
        .xy(10, 5)

    assert builder.points == [(0, 0), (8, 10), (10, 10), (10, 5)]

def test_polygon_builder_custom_block():
    builder = codecad.shapes.polygon2d_builder(0, 0) \
        .block(lambda points: [(2 * x, 2 * y) for (x, y) in points]) \
        .xy(1, 2) \
        .xy(3, 3) \
        .close()

    assert builder.points == [(0, 0), (2, 4), (6, 6)]
