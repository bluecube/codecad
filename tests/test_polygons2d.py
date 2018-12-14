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

    builder = codecad.shapes.polygons2d.Polygon2D.build(points[0][0], points[0][1])
    for x, y in points[1:]:
        builder = builder.xy(x, y)
    p2 = builder.close()

    assert p1.points.tolist() == p2.points.tolist()


@pytest.mark.parametrize(
    "name, args, point",
    [
        ("xy", [15, 5], [15, 5]),
        ("x", [15], [15, 10]),
        ("y", [5], [10, 5]),
        ("dxdy", [5, 1], [15, 11]),
        ("dx", [10], [20, 10]),
        ("dy", [-2], [10, 8]),
        # Values copied from previous runs, looking plausible
        ("angle", [-120, 8], [6, 3.0717966556549072]),
        ("angle_dx", [-30, 3], [13, 8.267949104309082]),
        ("angle_dy", [-30, -3], [15.196152687072754, 7]),
        # Negative arguments for angles
        ("angle", [135, -math.sqrt(2)], [11, 9]),
        ("angle_dx", [-45, 1], [11, 9]),
        ("angle_dx", [135, 1], [11, 9]),
        ("angle_dy", [-45, -1], [11, 9]),
        ("angle_dy", [135, -1], [11, 9]),
    ],
)
def test_polygon_builder_step(name, args, point):
    builder = codecad.shapes.polygons2d.Polygon2D.build(0, 0).xy(10, 10)
    builder = getattr(builder, name)(*args)
    polygon = builder.xy(5, 0).close()

    # We only check a single point so that pytest.approx works
    assert polygon.points[2].tolist() == pytest.approx(point)
