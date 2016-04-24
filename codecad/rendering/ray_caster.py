import PIL
import theano
import theano.tensor as T
import numpy
import math

from .. import util
from .. import shapes

class RayCaster:
    def __init__(self, filename, size = (800, 600), view_angle = math.radians(90)):
        self.filename = filename
        self.size = size
        self.view_angle = 90

    @staticmethod
    def get_trace_function(obj, epsilon):
        """ Returns a trace function that takes vectors of origin x, y, z and
        normalized direction x, y, z and returns distance to intersection."""

        ox = T.vector("ox")
        oy = T.vector("oy")
        oz = T.vector("oz")

        dx = T.vector("dx")
        dy = T.vector("dy")
        dz = T.vector("dz")

        def trace_func(previous, ox, oy, oz, dx, dy, dz):
            o = util.Vector(ox, oy, oz)
            d = util.Vector(dx, dy, dz)

            distance = obj.distance_estimate(o + d * previous)

            return previous + distance, \
                   theano.scan_module.until(T.all(T.or_(distance < epsilon / 2,
                                                        T.isinf(distance))))
        trace_expr, _ = theano.scan(trace_func,
                                    outputs_info=T.zeros_like(dx),
                                    non_sequences=[ox, oy, oz, dx, dy, dz],
                                    n_steps = 100)

        print("compiling...")
        f = theano.function([ox, oy, oz, dx, dy, dz],
                            trace_expr[-1],
                            givens=[(shapes.Shape.Epsilon, epsilon)],
                            on_unused_input = 'ignore') # Epsilon might not be used
        print("compiled")
        return f


    def render(self, obj):
        box = obj.bounding_box()
        box_size = box.b - box.a

        xs, ys = numpy.meshgrid(numpy.arange(self.size[0]), numpy.arange(self.size[1]))
        xs = xs.flatten()
        ys = ys.flatten()

        epsilon = min(box_size.x, box_size.y, box_size.z) / 100;

        focal_length = self.size[0] / math.tan(self.view_angle / 2)
        distance = 1.2 * max(box_size.x * focal_length / self.size[0],
                             box_size.z * focal_length / self.size[1])

        origin = (box.a + box.b) / 2 - util.Vector(0, distance + box_size.y / 2, 0)

        directions = util.Vector(self.size[0] / 2 - xs,
                                 focal_length,
                                 ys - self.size[1] / 2)
        directions = directions.normalized()

        trace_f = self.get_trace_function(obj, epsilon)

        pixels = trace_f(numpy.full_like(directions.x, origin.x),
                         numpy.full_like(directions.x, origin.y),
                         numpy.full_like(directions.x, origin.z),
                         directions.x,
                         numpy.full_like(directions.x, directions.y),
                         directions.z)

        print(pixels)

        img = PIL.Image.new("L", self.size)
        img.putdata(numpy.minimum(pixels, 255))
        img.save(self.filename)
