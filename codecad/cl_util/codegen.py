import inspect
import os.path
import sys
import pkg_resources


def format_c_string_literal(s):
    """ Returns a representation of a string suitable to be used in C99, including quotes. """

    def inner(s):
        yield '"'
        for c in s:
            o = ord(c)
            if c in '\\"':
                yield "\\" + c
            elif o in range(0x20, 0x7F):
                yield c
            elif o < 0x80:
                yield "\\{:03o}".format(o)
            elif o in range(0x80, 0xA0) or o in range(0xD800, 0xE000):
                raise ValueError(
                    "Codepoints between 0x80 and 0xA0 and between 0xD800 and 0xE000 "
                    "can't be properly escaped in C strings, AFAIK"
                )
            else:
                yield "\\U{:08x}".format(o)
        yield '"'

    return "".join(inner(s))


def _frame_from_stacklevel(stacklevel):
    """ Returns frameinfo stacklevel + 1 above the current frame one or None, if inspect.currentframe fails. """
    frame = inspect.currentframe()
    if frame is None:
        return None

    for _i in range(stacklevel + 1):
        frame = frame.f_back
        if frame is None:
            return None

    return frame


def string_with_origin(string, stacklevel=1, include_origin=True):
    """ Yield a string, optionally also yield a C #line directive before.
    Ignores newlines at the beginning of the string (useful for keeping alignment).

    stacklevel determines how many stack levels above this function we look when
    determining location of the origin. The default value is correct when calling
    this function directly with a string literal (its call site will be marked as
    the origin). """

    # Trim initial newlines
    trimmed = string.lstrip("\n")

    if include_origin:
        frame = _frame_from_stacklevel(stacklevel)
        if frame is not None:
            frameinfo = inspect.getframeinfo(frame, context=0)
            lines = trimmed.count("\n")
            line = frameinfo.lineno - lines

            yield "#line {} {}".format(
                line, format_c_string_literal(frameinfo.filename)
            )

    yield trimmed


def resource_with_origin(resource_name, stacklevel=1, include_origin=True):
    """ Read content of a resource usning pkg_resources and yield it,
    optionally also yields a corresponding prepend a #line directive before.
    Module name is determined from caller, see string_with_origin for description of stacklevel. """

    frame = _frame_from_stacklevel(stacklevel)
    if frame is None:
        raise Exception("We can't handle inspect.currentframe() failing! (TODO?)")

    module_name = frame.f_globals["__name__"]

    # Read the resource before yieldin line number so that there is no output at all
    # in case the resource fails.
    trimmed = pkg_resources.resource_string(module_name, resource_name).decode("utf8")

    if include_origin:
        path = os.path.join(
            os.path.dirname(sys.modules[module_name].__file__), resource_name
        )
        yield "#line 1 {}".format(format_c_string_literal(path))

    yield trimmed
