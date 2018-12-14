""" Trig functions working in degrees rather than radians. """

import math


def cos(deg):
    return math.cos(math.radians(deg))


def sin(deg):
    return math.sin(math.radians(deg))


def tan(deg):
    return math.tan(math.radians(deg))


def acos(v):
    return math.degrees(math.acos(v))


def asin(v):
    return math.degrees(math.asin(v))


def atan(v):
    return math.degrees(math.atan(v))


def atan2(y, x):
    return math.degrees(math.atan2(y, x))
