import contextlib
import time
import sys
import math


@contextlib.contextmanager
def status_block(title):
    print(title, end="...")
    sys.stdout.flush()
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(" {:0.2f} s".format(elapsed))


class Concatenate:
    def __init__(self, *iterables):
        self._iterables = iterables

    def __iter__(self):
        for iterable in self._iterables:
            yield from iterable


def at_most_one(iterable):
    it = iter(iterable)
    any(it)
    return not any(it)


def safe_div(a, b, zero_over_zero=0):
    if b == 0:
        if a == 0:
            return zero_over_zero
        else:
            return math.copysign(float("inf"), a)
    else:
        return a / b
