import pytest

import codecad.shapes.simple2d

valid_polygon_cases = {"triangle": [(0, 0), (3, 0), (3, 2)],
                       "non_convex": [(0, 0), (3, 0), (3, 1), (2, 2), (3, 3), (0, 3)],
                       "collinear_consecutive_edges": [(0, 0), (2, 0), (4, 0), (4, 3)],
                       "collinear_non_consecutive_edges": [(0, 0), (3, 0), (3, 1), (2, 2), (3, 3), (3, 4), (0, 4)],
                       "square": [(0, 0), (5, 0), (5, 5), (0, 5)],
                       "parallel_same_direction_edges": [(0, 0), (6, -1), (5, 5), (5, 0), (0, 5)],
                       }

invalid_polygon_cases = {"edge_crossing": [(0, 0), (3, 0), (0, 3), (3, 3)],
                         "point_crossing": [(0, 0), (3, 0), (2.5, 2.5), (0, 3), (3, 3), (2.5, 2.5)],
                         "repeated_point_on_collinear_edges": [(0, 0), (3, 0), (3, 2), (2, 1), (2, 3), (3, 2), (3, 4), (0, 4)],
                         "collinear_edge_crossing": [(0, 0), (3, 0), (3, 3), (2, 2), (3, 1), (3, 4), (0, 4)],
                         "repeated_point": [(0, 0), (4, 0), (4, 3), (0, 0), (1, 3), (0, 3)],
                         "point_on_edge": [(0, 0), (4, 0), (4, 3), (2, 0), (0, 3)],
                         "shared_edge": [(0, 0), (3, 0), (5, 0), (3, 0), (3, 3)],
                         "shared_edge_part": [(0, 0), (5, 0), (3, 0), (3, 3)],
                         "duplicate_point": [(0, 0), (2, 0), (2, 0), (4, 3)],
                         "duplicate_point_at_start": [(0, 0), (3, 0), (3, 2), (0, 0)]}

@pytest.mark.parametrize("points", [pytest.param(v, id=k) for k, v in sorted(valid_polygon_cases.items())] +
                                   [pytest.param(list(reversed(v)), id="reversed_" + k) for k, v in sorted(valid_polygon_cases.items())])
def test_valid_polygon_construction(points):
    codecad.shapes.simple2d.Polygon2D(points) # Just check that it constructs without complaining

@pytest.mark.parametrize("points", [pytest.param(v, id=k) for k, v in sorted(invalid_polygon_cases.items())] +
                                   [pytest.param(list(reversed(v)), id="reversed_" + k) for k, v in sorted(invalid_polygon_cases.items())])
def test_invalid_polygon_construction(points):
    with pytest.raises(ValueError):
        codecad.shapes.simple2d.Polygon2D(points)
