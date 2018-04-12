import gzip
import collections
import os.path
import numpy

from . import polygon2d


def render_svg(obj, filename):
    polygons = polygon2d.polygon(obj)

    with open(filename, "w") as fp:
        box = obj.bounding_box()
        box_size = box.size()

        fp.write('<svg xmlns="http://www.w3.org/2000/svg" ')
        fp.write('width="{}mm" height="{}mm" '.format(box_size.x, box_size.y))
        fp.write('viewBox="{} {} {} {}">'.format(box.a.x, -box.b.y, box_size.x, box_size.y))
        fp.write('<style type="text/css">')
        fp.write('path{')
        fp.write('stroke:#000;')
        fp.write('stroke-width:1px;')
        fp.write('vector-effect:non-scaling-stroke;')
        fp.write('fill:#BBF23C{};')
        fp.write('}')
        fp.write('</style>')

        fp.write('<path d="')
        for polygon in polygons:
            it = reversed(polygon)
            x, y = next(it)
            fp.write("M{},{}".format(x, -y))
            for x, y in it:
                fp.write("L{},{}".format(x, -y))
            fp.write("L{},{}".format(polygon[-1][0], -polygon[-1][1]))
        fp.write('"/>')

        fp.write('</svg>')
