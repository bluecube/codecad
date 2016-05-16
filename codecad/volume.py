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
    inside_count = inside.sum()
    volume = inside_count * (resolution * resolution * resolution)
    centroid_x = (inside * x).sum() / inside_count
    centroid_y = (inside * y).sum() / inside_count
    centroid_z = (inside * z).sum() / inside_count

    with util.status_block("compiling"):
        f = theano.function([x, y, z], (volume, centroid_x, centroid_y, centroid_z))

    box = shape.bounding_box()
    box_a = box.a + util.Vector(resolution, resolution, resolution) / 2

    xs = numpy.arange(box_a.x, box.b.x, resolution)
    ys = numpy.arange(box_a.y, box.b.y, resolution)
    zs = numpy.arange(box_a.z, box.b.z, resolution)

    with util.status_block("running"):
        result = [float(x) for x in f(*numpy.meshgrid(xs, ys, zs))]

    return VolumeAndCentroid(result[0], util.Vector(*result[1:]))

