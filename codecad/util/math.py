import math


class KahanSummation:
    """ Helper class that allows summing many floating point numbers with better precision
    Algorithm adapted from wikipedia. """

    def __init__(self):
        self.result = 0
        self.correction = 0

    def __iadd__(self, x):
        y = x - self.correction
        tmp = self.result + y
        self.correction = (tmp - self.result) - y
        self.result = tmp
        return self

    def __isub__(self, x):
        self += -x
        return self


def round_up_to(x, y):
    """ Round x away from zero to a nearest multiple of y """
    return ((x + y - 1) // y) * y


def round_up_to_power_of_2(x):
    return 2 ** math.ceil(math.log2(x))


def clamp(v, lower, upper):
    return max(lower, min(v, upper))
