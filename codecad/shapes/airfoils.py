from . import simple2d


def _load_selig(fileobj):
    header = fileobj.readline()
    points = [tuple(float(x) for x in l.split()) for l in fileobj]
    if points[0] == points[-1]:
        points = points[:-1]
    return simple2d.Polygon2D(points)


def load_selig(path, fileobj=None):
    if fileobj is not None:
        return _load_selig(fileobj)
    else:
        with open(path, "r") as fp:
            return _load_selig(fp)
