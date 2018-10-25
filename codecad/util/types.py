import numbers
from . import geometry


def wrap_number_like(value):
    """ Try to use the value as a number.

    Instances of numbers.Number are passed through unchanged,
    float is returned if a number supports conversion to float, otherwise an exception
    is raised. """
    if isinstance(value, numbers.Number):
        return value
    else:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise TypeError(
                "Value must be instance of numbers.Number or support conversion to float to be number-like"
            )


def wrap_vector_like(value, max_dimension=3):
    """ Try to use value as a vector.
    Vector-like is either instance of Vector or its subclass, or an iterable with
    between two and max_dimension elements. """

    if isinstance(value, geometry.Vector):
        return value

    try:
        it = iter(value)
    except TypeError:
        raise TypeError("Value must be iterable to be vector-like")

    wrapped = (wrap_number_like(x) for x in it)

    try:
        x = next(wrapped)
        y = next(wrapped)
    except StopIteration:
        raise TypeError("Value must have at least two items to be vector-like")

    z = 0
    if max_dimension == 3:
        try:
            z = next(wrapped)
        except StopIteration:
            pass

    try:
        next(it)
        # We don't want to try converting the item after the last one to not
        # hide the "too long" message with potentional "not a number" message
    except StopIteration:
        pass
    else:
        raise TypeError("Value must have at most three items to be vector-like")

    return geometry.Vector(x, y, z)
