import math

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
    assert hexagon.across_flats == pytest.approx(math.sqrt(3))
    assert hexagon.side_length == pytest.approx(1)
    assert codecad.shapes.simple2d.RegularPolygon2D.calculate_n(1, 1) == pytest.approx(6)

    hexagon2 = codecad.shapes.regular_polygon2d(6, d=hexagon.d)
    assert hexagon2.r == pytest.approx(hexagon.r)
    assert hexagon2.d == pytest.approx(hexagon.d)
    assert hexagon2.across_flats == pytest.approx(hexagon.across_flats)
    assert hexagon2.side_length == pytest.approx(hexagon.side_length)

    hexagon3 = codecad.shapes.regular_polygon2d(6, across_flats=hexagon.across_flats)
    assert hexagon3.r == pytest.approx(hexagon.r)
    assert hexagon3.d == pytest.approx(hexagon.d)
    assert hexagon3.across_flats == pytest.approx(hexagon.across_flats)
    assert hexagon3.side_length == pytest.approx(hexagon.side_length)

    hexagon4 = codecad.shapes.regular_polygon2d(6, side_length=hexagon.side_length)
    assert hexagon4.r == pytest.approx(hexagon.r)
    assert hexagon4.d == pytest.approx(hexagon.d)
    assert hexagon4.across_flats == pytest.approx(hexagon.across_flats)
    assert hexagon4.side_length == pytest.approx(hexagon.side_length)


def test_regular_polygon2d_triangle_across_flats():
    t = codecad.shapes.regular_polygon2d(3, side_length=1)
    assert t.r == pytest.approx(1 / math.sqrt(3))
    assert t.across_flats == pytest.approx(math.sqrt(3) / 2)


def test_regular_polygon2d_invalid_constructor():
    with pytest.raises(ValueError):
        codecad.shapes.regular_polygon2d(4, r=1, d=3)

    with pytest.raises(ValueError):
        codecad.shapes.regular_polygon2d(5, d=2, across_flats=5)

    with pytest.raises(ValueError):
        codecad.shapes.regular_polygon2d(6, side_length=8, across_flats=5)


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


def test_square_feature_size():
    assert codecad.shapes.rectangle(1, 1).feature_size() == \
        pytest.approx(codecad.shapes.regular_polygon2d(4, side_length=1).feature_size()) == \
        1


@pytest.mark.parametrize("n", range(3, 8))
def test_regular_polygon_feature_size(n):
    points = [(math.cos(math.pi * 2 * (i + 0.5) / n), math.sin(math.pi * 2 * (i + 0.5) / n))
              for i in range(n)]

    assert codecad.shapes.regular_polygon2d(n, r=1).feature_size() == \
        pytest.approx(codecad.shapes.polygon2d(points).feature_size())
