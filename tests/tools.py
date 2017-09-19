import contextlib
import tempfile
import os
import functools
import shutil
import inspect
import collections

import pytest


class FileBaselineCompare:
    """ Decorator factory. The wrapped function must return a filename, which is
    then checked against a baseline file using `comparison_func(tested_file_obj, baseline_file_obj)`.
    If file is not present in baseline directory this function skips the test
    and stores the current result as a baseline."""

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
