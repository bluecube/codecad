import decimal

import codecad
import pytest


def test_kahan_summation():
    eps = 1.0
    while 1.0 + eps != 1.0:
        eps /= 2

    s = 0.0
    s += 1.0
    s += eps
    s -= eps
    assert s != 1.0, "Sanity check, if this doesn't fail, then we're testing a wrong thing"

    s = codecad.util.KahanSummation()
    s += 1.0
    s += eps
    s -= eps
    assert s.result == 1.0
