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

        ox = T.vector("ox")
        oy = T.vector("oy")
        oz = T.vector("oz")

        dx = T.vector("dx")
        dy = T.vector("dy")
        dz = T.vector("dz")

        def trace_func(previous, _, ox, oy, oz, dx, dy, dz):
            o = util.Vector(ox, oy, oz)
            d = util.Vector(dx, dy, dz)

            distance = obj.distance_estimate(o + d * previous)

            return [previous + 0.8 * distance, distance], \
                   theano.scan_module.until(T.all(T.or_(distance < epsilon / 2,
                                                        T.isinf(distance))))
        d1, d2 = theano.scan(trace_func,
                             outputs_info=[T.zeros_like(dx),T.zeros_like(dx)],
                             non_sequences=[ox, oy, oz, dx, dy, dz],
                             n_steps = 100)[0]

        d1 = d1[-1]
        d2 = d2[-1]

        min_distance = T.min(d1)
        max_distance = T.ptp(T.switch(d2 < epsilon, d1, 0.0))
        colors = 255 - 255 * T.clip(0.8 * (d1 - min_distance) / (max_distance - min_distance), 0.0, 1.0)

        print("compiling...")
        f = theano.function([ox, oy, oz, dx, dy, dz],
                            colors,
                            givens=[(shapes.Shape.Epsilon, epsilon)],
                            on_unused_input = 'ignore') # Epsilon might not be used
        print("running...")

        pixels = f(numpy.full_like(directions.x, origin.x),
                   numpy.full_like(directions.y, origin.y),
                   numpy.full_like(directions.z, origin.z),
                   directions.x,
                   numpy.full_like(directions.x, directions.y),
                   directions.z)

        print("saving...")

        img = PIL.Image.new("L", self.size)
        img.putdata(pixels)
        img.save(self.filename)
