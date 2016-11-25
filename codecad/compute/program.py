import inspect
import os

def _(string):
    frame = inspect.stack()[1]
    lines = string.count("\n")
    return '#line {} "{}"\n'.format(frame[2] - lines, frame[1]) + string

def collect_files():
    root = os.path.dirname(os.path.dirname(__file__))
    sources = []
    for path, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".cl"):
                sources.append(os.path.join(path, filename))

    sources.sort()

    for source in sources:
        yield '#line 1 "{}"'.format(source)
        with open(source, "r") as fp:
            yield fp.read()

def collect_program():
    print("\n".join(collect_files()))
    return _(r'''abc
    def
    ghi''')
