import codecad
import hypothesis
import pytest


def test_kahan_summation():
    eps = 1.0
    while 1.0 + eps != 1.0:
        eps /= 2

    s = 0.0
    s += 1.0
    s += eps
    s -= eps
    assert (
        s != 1.0
    ), "Sanity check, if this doesn't fail, then we're testing a wrong thing"

    s = codecad.util.KahanSummation()
    s += 1.0
    s += eps
    s -= eps
    assert s.result == 1.0


@hypothesis.given(hypothesis.strategies.lists(hypothesis.strategies.booleans()))
def test_at_most_one(l):
    assert codecad.util.at_most_one(l) == (sum(l) <= 1)
