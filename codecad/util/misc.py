import contextlib
import time
import sys


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
    """ Return true if an iterable has at most one truthy value """
    it = iter(iterable)
    any(it)
    return not any(it)
