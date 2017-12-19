import pytest

import tools
import data

import codecad


@pytest.mark.parametrize("shape", data.params_3d)
def test_identity_equal(shape):
    tools.assert_shapes_equal(shape, shape)


def test_not_equal():
    """ Check that two obviously non-equal shapes evaluate as such. """
    s1 = codecad.shapes.sphere()
    s2 = codecad.shapes.box()
    with pytest.raises(AssertionError):
        tools.assert_shapes_equal(s1, s2)


def test_not_equal_volume_only():
    """ Check that two non-equal shapes evaluate as such;
    this time the difference won't be visible on the render. """
    s1 = codecad.shapes.sphere(r=1)
    s2 = s1 - codecad.shapes.box(1)
    with pytest.raises(AssertionError):
        tools.assert_shapes_equal(s1, s2)
