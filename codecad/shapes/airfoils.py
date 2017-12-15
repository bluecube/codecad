from . import simple2d


def _load_selig(fileobj):
    header = fileobj.readline()

    l = list()
    return simple2d.Polygon2D([[float(x) for x in l.split()] for l in fileobj])


def load_selig(path, fileobj=None):

    if fileobj is not None:
        return _load_selig(fileobj)
    else:
        with open(path, "r") as fp:
            return _load_selig(fp)
