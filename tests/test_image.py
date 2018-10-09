import os.path

import pytest

import codecad.rendering.image

import data
import tools

names_and_shapes = (
    pytest.param((k, v), id=k)
    for k, v in sorted(data.shapes_2d.items()) + sorted(data.shapes_3d.items())
)


@pytest.mark.parametrize("name_and_shape", names_and_shapes)
def test_image_png(name_and_shape):
    shape_name, shape = name_and_shape
    with tools.FileBaselineCompare(
        os.path.join(os.path.dirname(__file__), "baseline"),
        "rendered_{}.png".format(shape_name),
    ) as compare:
        codecad.rendering.image.render_image(shape, compare.tested_filename)

        tested_fp, baseline_fp = compare.tested_file_ready()

        tools.assert_images_equal(tested_fp, baseline_fp)
        # "test_image_png_error_{}.png".format(shape_name)
