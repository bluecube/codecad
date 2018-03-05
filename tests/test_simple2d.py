import pytest

import codecad.shapes.simple2d

import data


@pytest.mark.parametrize("points", [pytest.param(v, id=k) for k, v in sorted(data.valid_polygon2d.items())] +
                                   [pytest.param(list(reversed(v)), id="reversed_" + k) for k, v in sorted(data.valid_polygon2d.items())])
def test_valid_polygon_construction(points):
    codecad.shapes.simple2d.Polygon2D(points)  # Just check that it constructs without complaining


@pytest.mark.parametrize("points", [pytest.param(v, id=k) for k, v in sorted(data.invalid_polygon2d.items())] +
                                   [pytest.param(list(reversed(v)), id="reversed_" + k) for k, v in sorted(data.invalid_polygon2d.items())])
def test_invalid_polygon_construction(points):
    with pytest.raises(ValueError):
        codecad.shapes.simple2d.Polygon2D(points)


@pytest.mark.parametrize("shape", data.params_2d)
def test_z_bounding_box_size(shape):
    box = shape.bounding_box()
    assert box.a.z == 0
    assert box.b.z == 0


def test_regular_polygon2d_hexagon():
    hexagon = codecad.shapes.regular_polygon2d(6, r=1)
    assert hexagon.d == 2
    assert hexagon.side_length == pytest.approx(1)
    assert codecad.shapes.simple2d.RegularPolygon2D.calculate_n(1, 1) == pytest.approx(6)


def test_mirrored_bounding_box():
    shape = codecad.shapes.rectangle(1, 2).translated(0.5, 1)
    assert shape.bounding_box() == codecad.util.BoundingBox(codecad.util.Vector(0, 0),
                                                            codecad.util.Vector(1, 2))
    mx = shape.mirrored_x()
    assert mx.bounding_box() == pytest.approx(codecad.util.BoundingBox(codecad.util.Vector(-1, 0),
                                                                       codecad.util.Vector(0, 2)))
    my = shape.mirrored_y()
    assert my.bounding_box() == pytest.approx(codecad.util.BoundingBox(codecad.util.Vector(0, -2),
                                                                       codecad.util.Vector(1, 0)))
