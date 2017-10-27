import os.path
import tempfile

import pytest
import PIL
import numpy

import codecad.rendering.image

import data
import tools


def image_compare(tested_file_obj, baseline_file_obj):

    assert distance

    return True


@pytest.mark.xfail(reason="Visualisation is being changed all the time, "
                   "no need to update baselines until it stabilises a bit.")
@pytest.mark.parametrize("name_and_shape",
                         [pytest.param((k, v), id=k) for k, v in sorted(data.shapes_3d.items())])
def test_raycaster_png(name_and_shape):
    shape_name, shape = name_and_shape
    with tools.FileBaselineCompare(os.path.join(os.path.dirname(__file__), "baseline"),
                                   "rendered_{}.png".format(shape_name)) as compare:
        codecad.rendering.image.render_image(shape, compare.tested_filename)

        tested_fp, baseline_fp = compare.tested_file_ready()

        tested_array = numpy.array(PIL.Image.open(tested_fp), dtype=numpy.float32) / 255
        baseline_array = numpy.array(PIL.Image.open(baseline_fp), dtype=numpy.float32) / 255

        assert tested_array.shape == baseline_array.shape

        error_array = tested_array - baseline_array

        mean_squared_error = numpy.mean(error_array * error_array)

        if mean_squared_error > 1e-3:
            error_image_array = 255 * error_array / numpy.max(error_array)
            error_image = PIL.Image.fromarray(error_image_array.astype(numpy.uint8))
            error_image.save("rendered_{}_error.png".format(shape_name))
            raise ValueError("Mean squared error is too big.")
