from . import util
import collections
import theano
import theano.tensor as T
import numpy

VolumeAndCentroid = collections.namedtuple("VolumeAndCentroid", "volume centroid")

def volume_and_centroid(shape, resolution):
    x = T.tensor3("x")
    y = T.tensor3("y")
    z = T.tensor3("z")

    inside = shape.distance(util.Vector(x, y, z)) < 0
    cell_count = inside.sum()

    with util.status_block("compiling"):
        f = theano.function([x, y, z],
                            cell_count,
                            on_unused_input = 'ignore') # Epsilon might not be used

    box = shape.bounding_box()
    box_a = box.a + util.Vector(resolution, resolution, resolution) / 2

    xs = numpy.arange(box_a.x, box.b.x, resolution)
    ys = numpy.arange(box_a.y, box.b.y, resolution)
    zs = numpy.arange(box_a.z, box.b.z, resolution)

    with util.status_block("running"):
        cell_count = f(*numpy.meshgrid(xs, ys, zs))

    return VolumeAndCentroid(cell_count * resolution * resolution * resolution,
                             float("nan"))

