import contextlib
import tempfile
import os
import shutil
import io

import pytest
import PIL
import numpy

import codecad


class FileBaselineCompare:
    """ Context manager for comparing a file output of a tested function with
    known baseline. Handles lookup of baseline file. If a baseline file is not found,
    skips the test and moves the generated tested file into its place.

    Usage:

        with FileBaselineCompare(directory_with_baseline_files, filename_of_this_instance) as compare:
            generate_tested_data(compare.tested_filename)
            tested_fp, baseline_fp = compare.tested_file_ready()
            check_that_data_are_close_enough(tested_fp, baseline_fp)

    """

    def __init__(self, baseline_directory, filename):
        self.filename = filename

        self._stack = contextlib.ExitStack()
        self._d = None

        self.tested_filename = None
        self.baseline_filename = os.path.join(baseline_directory, filename)

    def __enter__(self):
        self._stack.__enter__()
        self._d = self._stack.enter_context(tempfile.TemporaryDirectory())

        self.tested_filename = os.path.join(self._d, self.filename)
        os.makedirs(os.path.dirname(self.tested_filename), exist_ok=True)

        return self

    def __exit__(self, *exc_info):
        return self._stack.__exit__(*exc_info)

    def tested_file_ready(self):
        if not os.path.exists(self.baseline_filename):
            os.makedirs(os.path.dirname(self.baseline_filename), exist_ok=True)
            shutil.move(self.tested_filename, self.baseline_filename)
            pytest.skip("Baseline file was missing, will use the current result next time")
        else:
            baseline_fp = self._stack.enter_context(open(self.baseline_filename, "rb"))
            tested_fp = self._stack.enter_context(open(self.tested_filename, "rb"))

            return tested_fp, baseline_fp


def assert_images_equal(tested_fp, baseline_fp, error_filename=None):
    tested_array = numpy.array(PIL.Image.open(tested_fp), dtype=numpy.float32) / 255
    baseline_array = numpy.array(PIL.Image.open(baseline_fp), dtype=numpy.float32) / 255

    assert tested_array.shape == baseline_array.shape

    error_array = tested_array - baseline_array

    mean_squared_error = numpy.mean(error_array * error_array)

    if mean_squared_error > 1e-3:
        if error_filename is not None:
            error_image_array = 255 * error_array / numpy.max(error_array)
            error_image = PIL.Image.fromarray(error_image_array.astype(numpy.uint8))
            error_image.save(error_filename)
        raise AssertionError("Mean squared error is too big.")


def assert_shapes_equal(shape, expected, resolution=0.1):
    """ Checks that two shapes are equal, raises AssertionError if they are not.

    For now this does a bounding box chec, very rough check using volume of
    symmetric difference of the two shapes and then a comparison of low resolution renders.
    This is certainly not optimal (too slow and too imprecise), but that's what we've got. """
    # TODO: Rely only on the volume of symmetric difference once mass properties are more reliable

    box = shape.bounding_box()
    expected_box = expected.bounding_box()
    intersection_box = box.intersection(expected_box)
    assert box.volume() == pytest.approx(intersection_box.volume())
    assert expected_box.volume() == pytest.approx(intersection_box.volume())

    if expected_box.volume() == 0:
        return

    difference = (shape - expected) | (expected - shape)
    mp = codecad.mass_properties(difference, box.volume() * 1e-3)
    assert mp.volume <= mp.volume_error

    shape_render = io.BytesIO()
    codecad.rendering.image.render_PIL_image(shape, size=(800, 400)).save(shape_render, format="png")
    shape_render.seek(0)
    expected_render = io.BytesIO()
    codecad.rendering.image.render_PIL_image(expected, size=(800, 400)).save(expected_render, format="png")
    expected_render.seek(0)
    assert_images_equal(shape_render, expected_render)
