import decimal

import hypothesis
import pytest

import codecad


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


class NumberWrapper:
    def __init__(self, v):
        self.v = v

    def __float__(self):
        return float(self.v)


@hypothesis.given(
    hypothesis.strategies.lists(
        hypothesis.strategies.integers()
        | hypothesis.strategies.floats()
        | hypothesis.strategies.decimals()
        | hypothesis.strategies.fractions()
        | hypothesis.strategies.floats().map(NumberWrapper)
        | hypothesis.strategies.floats().map(str),
        min_size=2,
        max_size=3,
    )
)
def test_wrap_vector_like1(value):
    codecad.util.types.wrap_vector_like(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        (codecad.util.Vector(1, 2, 3), codecad.util.Vector(1, 2, 3)),
        ((1, 2, 3), codecad.util.Vector(1, 2, 3)),
        ((4, 5), codecad.util.Vector(4, 5, 0)),
        (
            (decimal.Decimal("1e5"), NumberWrapper(7), 2),
            codecad.util.Vector(decimal.Decimal("1e5"), 7.0, 2),
        ),
        (range(2), codecad.util.Vector(0, 1, 0)),
        (range(3), codecad.util.Vector(0, 1, 2)),
    ],
)
def test_wrap_vector_like2(value, expected):
    assert codecad.util.wrap_vector_like(value) == expected


@pytest.mark.parametrize("value", [[], (1, 2, 3, 4), (4,), 6, "X"])
def test_wrap_vector_like_fail(value):
    with pytest.raises((TypeError, ValueError)):
        codecad.util.wrap_vector_like(value)


def test_wrap_vector_like2d():
    assert codecad.util.wrap_vector_like(
        (1, 2), max_dimension=2
    ) == codecad.util.Vector(1, 2)


def test_wrap_vector_like2d_fail():
    with pytest.raises((TypeError, ValueError)):
        codecad.util.wrap_vector_like((1, 2, 3), max_dimension=2)
