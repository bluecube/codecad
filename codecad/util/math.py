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
