import string
import inspect
import os.path

printable = string.digits + string.ascii_letters + string.punctuation + " "


def format_c_string_literal(s):
    """ Returns a representation of a string suitable to be used in C99, including quotes. """

    def inner(s):
        yield '"'
        for c in s:
            o = ord(c)
            if c in '\\"':
                yield '\\' + c
            elif o in range(0x20, 0x7F):
                yield c
            elif o < 0x80:
                yield '\\{:03o}'.format(o)
            elif o in range(0x80, 0xA0) or o in range(0xD800, 0xE000):
                raise ValueError("Codepoints between 0x80 and 0xA0 and between 0xD800 and 0xE000 "
                                 "can't be properly escaped in C strings, AFAIK")
            else:
                yield '\\U{:08x}'.format(o)
        yield '"'

    return "".join(inner(s))


def _frameinfo_from_stacklevel(stacklevel):
    """ Returns frameinfo stacklevel + 1 above the current frame one or None, if inspect.currentframe fails. """
    frame = inspect.currentframe()
    if frame is None:
        return None

    for i in range(stacklevel + 1):
        frame = frame.f_back

    return inspect.getframeinfo(frame, context=0)


def string_with_origin(string, stacklevel=1):
    """ Prepend a C #line directive in front of a string, for generating C files.

    stacklevel determines how many stack levels above this function we look when
    determining location of the origin. The default value is correct when calling
    this function directly with a string literal (its call site will be marked as
    the origin). """

    frameinfo = _frameinfo_from_stacklevel(stacklevel)
    if frameinfo is None:
        return string

    lines = string.count("\n")
    line = frameinfo.lineno - lines

    # Trim initial newlines
    l = len(string)
    string = string.lstrip("\n")
    line += l - len(string)

    return '#line {} {}\n'.format(line, format_c_string_literal(frameinfo.filename)) + string


def file_with_origin(path, stacklevel=1):
    """ Read content of a file and prepend a #line directive to it.

    Unless `path` is absolute, filename is taken as relative to caller filename's directory.
    See `string_with_origin` for description of stacklevel functionality.

    In the output the path is made relative to cwd. """

    frameinfo = _frameinfo_from_stacklevel(stacklevel)
    if frameinfo is None:
        raise Exception("We can't handle inspect.currentframe() failing! (TODO?)")

    if os.path.isabs(path):
        abs_path = path
    else:
        abs_path = os.path.join(os.path.dirname(frameinfo.filename), path)
    rel_path = os.path.relpath(abs_path)

    with open(abs_path, "r") as fp:
        return '#line 1 {}\n'.format(format_c_string_literal(rel_path)) + fp.read()
