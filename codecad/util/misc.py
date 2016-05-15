import contextlib
import time
import sys

#TODO: Performance: Parallelization friendly version of reduce? Probably won't help, though.

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
