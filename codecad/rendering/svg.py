import gzip
import collections
import os.path
import numpy

from . import render_params
from . import mesh

def _interp(a, b):
    return -a / (b - a)

def _html_color(c):
    return "#" + hex(c.x << 16 | c.y << 8 | c.z)[2:]

def render_svg(obj, resolution, filename):
    polygons = mesh.polygon(obj, resolution)

    with open(filename, "w") as fp:
        box = obj.bounding_box()
        box_size = box.size()

        fp.write('<svg xmlns="http://www.w3.org/2000/svg" ')
        fp.write('width="{}mm" height="{}mm" '.format(box_size.x, box_size.y))
        fp.write('viewBox="{} {} {} {}">'.format(box.a.x, -box.b.y, box_size.x, box_size.y))

        color = _html_color(render_params.surface)

        fp.write('<path d="')
        for polygon in polygons:
            it = reversed(polygon)
            x, y = next(it)
            fp.write("M{},{}".format(x, -y))
            for x, y in it:
                fp.write("L{},{}".format(x, -y))
            fp.write("L{},{}".format(polygon[-1][0], -polygon[-1][1]))
        fp.write('" style="stroke:black;stroke-width:1px;fill:{}"/>'.format(color))

        fp.write('</svg>')
