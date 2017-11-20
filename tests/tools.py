import contextlib
import tempfile
import os
import functools
import shutil
import inspect
import collections

import pytest


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
