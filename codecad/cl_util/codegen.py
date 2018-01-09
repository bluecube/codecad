import string

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
