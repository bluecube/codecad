import gzip
import collections
import os.path
import numpy

from . import render_params
from .. import util

def _interp(a, b):
    return -a / (b - a)

def _html_color(c):
    return "#" + hex(c.x << 16 | c.y << 8 | c.z)[2:]

class _PolylineSegment:
    def __init__(self):
        self._points = collections.deque()
        self._prev = None
        self._next = None

        self._count = 0
        self._first = self
        self._last = self
            # _first and _last point to the first and last segment
            # in chain, _count contains total number of points in chain.
            # These are only valid on end segments or on segments that were just joined
            # TODO: These links create circular dependencies and will mess up
            #       garbage collection, use weakref here

    def join(self, other):
        """ Join two segments and return false or return true and
        do nothing if they were already connected """
        assert(self._next is None)
        assert(other._prev is None)

        if self._first is other._first:
            return True

        self._next = other
        other._prev = self

        count = self._count + other._count

        self._first._last = other._last
        self._last = other._last
        other._last._first = self._first
        other._first = self._first

        self._first._count = count
        self._count = count
        other._last._count = count
        other._count = count

        return False

    def append(self, point):
        assert(self._next is None)
        self._points.append(point)

        count = self._count + 1
        self._first._count = count
        self._last._count = count

    def appendleft(self, point):
        assert(self._prev is None)
        self._points.appendleft(point)

        count = self._count + 1
        self._first._count = count
        self._last._count = count

    def numpyfy(self):
        seg = self._first
        x = numpy.empty(self._count)
        y = numpy.empty(self._count)

        i = 0
        while seg is not None:
            for point in seg._points:
                x[i] = point[0]
                y[i] = point[1]
                i += 1
            seg = seg._next

        return util.Vector(x, y)

def render_svg(obj, resolution, filename):
    with util.status_block("calculating bounding box"):
        box = obj.bounding_box().expanded_additive(resolution).flattened()

    with util.status_block("building expression"):
        coords = util.theano_box_grid(box, resolution)
        distances = obj.distance(coords)[:,:,0]
        inside = distances <= 0

        classification = inside[ :-1,  :-1] * 1 + \
                         inside[ :-1, 1:  ] * 2 + \
                         inside[1:  , 1:  ] * 4 + \
                         inside[1:  ,  :-1] * 8

    with util.status_block("compiling"):
        f = theano.function([], (distances, classification))

    with util.status_block("running"):
        values, cells = f()

    # In this algorithm polygons are represented either by a deque.

    # Because our bounding box is larger than the scene (or at least we assume that),
    # No geometry can start to the left or to the top of the first cells. This
    # means that top_polygons can be initialized to empty dict before first row and
    # left_polygon to None before first column

    top_polygons = {} # mapping of polyline ends coming from the top -- column number -> polygon

    output_polygons = []

    with util.status_block("marching squares"):
        for j, row in enumerate(cells):
            left_polygon = None # Polygon that will continue to the right on the current cell
            bottom_polygons = {}

            for i, c in enumerate(row):
                # Cell classifications:

                #    0       1       2       3       4       5       6       7
                # 1-----2 X-----2 1-----X X-----X 1-----2 X-----2 1-----X X-----X
                # |     | |/    | |    \| |-----| |     | |/    | |  |  | |     |
                # |     | |     | |     | |     | |    /| |    /| |  |  | |\    |
                # 8-----4 8-----4 8-----4 8-----4 8-----X 8-----X 8-----X 8-----X

                #    8       9       10      11      12      13      14      15
                # 1-----2 X-----2 1-----X X-----X 1-----2 X-----2 1-----X X-----X
                # |     | |  |  | |    \| |     | |     | |    \| |/    | |     |
                # |\    | |  |  | |\    | |    /| |-----| |     | |     | |     |
                # X-----4 X-----4 X-----4 X-----4 X-----X X-----X X-----X X-----X

                if c == 0 or c == 15:
                    continue

                # Classifications that require right and bottom edge coordinates:
                # r: 2 3 4 5         10 11 12 13
                # b:     4 5 6 7 8 9 10 11
                if c <= 5 or c >= 10:
                    r = (j + _interp(values[j, i + 1], values[j + 1, i + 1]),
                         i + 1)

                if c >= 4 and c <= 11:
                    b = (j + 1,
                         i + _interp(values[j + 1, i], values[j + 1, i + 1]))

                if c == 1:
                    if left_polygon.join(top_polygons[i]):
                        output_polygons.append(left_polygon.numpyfy())
                    left_polygon = None
                elif c == 2:
                    left_polygon = top_polygons[i]
                    left_polygon.append(r)
                elif c == 3:
                    left_polygon.append(r)
                elif c == 4:
                    left_polygon = _PolylineSegment()
                    bottom_polygons[i] = left_polygon
                    left_polygon.append(r)
                    left_polygon.append(b)
                elif c == 5:
                    if left_polygon.join(top_polygons[i]):
                        output_polygons.append(left_polygon.numpyfy())
                    left_polygon = _PolylineSegment()
                    bottom_polygons[i] = left_polygon
                    left_polygon.append(r)
                    left_polygon.append(b)
                elif c == 6:
                    p = top_polygons[i]
                    p.append(b)
                    bottom_polygons[i] = p
                elif c == 7:
                    left_polygon.append(b)
                    bottom_polygons[i] = left_polygon
                    left_polygon = None
                elif c == 8:
                    left_polygon.appendleft(b)
                    bottom_polygons[i] = left_polygon
                    left_polygon = None
                elif c == 9:
                    p = top_polygons[i]
                    p.appendleft(b)
                    bottom_polygons[i] = p
                elif c == 10:
                    left_polygon.appendleft(b)
                    bottom_polygons[i] = left_polygon
                    left_polygon = top_polygons[i]
                    left_polygon.append(r)
                elif c == 11:
                    left_polygon = _PolylineSegment()
                    bottom_polygons[i] = left_polygon
                    left_polygon.append(b)
                    left_polygon.append(r)
                elif c == 12:
                    left_polygon.appendleft(r)
                elif c == 13:
                    left_polygon = top_polygons[i]
                    left_polygon.appendleft(r)
                elif c == 14:
                    if top_polygons[i].join(left_polygon):
                        output_polygons.append(left_polygon.numpyfy())
                    left_polygon = None

            assert(left_polygon is None)
            top_polygons = bottom_polygons
        assert(len(top_polygons) == 0)

    if filename.endswith(".svgz"):
        open_fun = gzip.open
    else:
        open_fun = open

    with util.status_block("saving"), \
         open_fun(filename, "w") as fp:
        box_size = box.size()

        fp.write('<svg xmlns="http://www.w3.org/2000/svg" ')
        fp.write('width="{}mm" height="{}mm" '.format(box_size.x, box_size.y))
        fp.write('viewBox="{} {} {} {}">'.format(0, -values.shape[1], values.shape[0], values.shape[1]))

        color = _html_color(render_params.surface + render_params.ambient)

        fp.write('<path d="')
        for polygon in output_polygons:
            it = iter(zip(polygon.x, polygon.y))
            x, y = next(it)
            fp.write("M{},{}".format(x, -y))
            for x, y in it:
                fp.write("L{},{}".format(x, -y))
            fp.write("L{},{}".format(polygon.x[0], -polygon.y[0]))
        fp.write('" style="stroke:black;stroke-width:1px;fill:{}"/>'.format(color))

        fp.write('</svg>')
