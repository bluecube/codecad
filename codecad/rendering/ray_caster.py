import PIL
import theano
import theano.tensor as T
import numpy
import math

from .. import util
from .. import shapes
from . import render_params

def render_picture(obj, filename, size = (800, 600),
                   view_angle = math.radians(90),
                   mode="nice",
                   resolution=None):
    mode_func = {"nice": nice_mode,
                 "dot": dot_mode,
                 "distance": distance_mode}[mode]

    box = obj.bounding_box()
    box_size = box.b - box.a

    xs, ys = numpy.meshgrid(numpy.arange(size[0]), numpy.arange(size[1]))

    if resolution is None:
        epsilon = min(box_size.x, box_size.y, box_size.z) / 1000;
    else:
        epsilon = resolution

    focal_length = size[0] / math.tan(view_angle / 2)
    distance = focal_length * max(box_size.x / size[0],
                                  box_size.z / size[1])

    origin = box.midpoint() - util.Vector(0, distance + box_size.y / 2, 0)

    directions = util.Vector(xs - size[0] / 2,
                             focal_length,
                             size[1] / 2 - ys)
    directions = directions.normalized()

    ox = T.matrix("ox")
    oy = T.matrix("oy")
    oz = T.matrix("oz")

    dx = T.matrix("dx")
    dy = T.matrix("dy")
    dz = T.matrix("dz")

    def _trace_func(previous, _, ox, oy, oz, dx, dy, dz):
        o = util.Vector(ox, oy, oz)
        d = util.Vector(dx, dy, dz)

        distance = obj.distance(o + d * previous)

        return [previous + 0.8 * distance, distance], \
               theano.scan_module.until(T.all(T.or_(distance < epsilon / 2,
                                                    T.isinf(distance))))

    distance, final_value = theano.scan(_trace_func,
                                        outputs_info=[T.zeros_like(dx),T.zeros_like(dx)],
                                        non_sequences=[ox, oy, oz, dx, dy, dz],
                                        n_steps = 100)[0]

    distance = distance[-1]
    final_value = final_value[-1]

    r, g, b = mode_func(obj,
                        distance, final_value, epsilon,
                        util.Vector(ox, oy, oz),
                        util.Vector(dx, dy, dz))

    colors = T.clip(T.stack((r, g, b), 2), 0, 255).astype("uint8")

    with util.status_block("compiling"):
        f = theano.function([ox, oy, oz, dx, dy, dz], colors)

    with util.status_block("running"):
        pixels = f(numpy.full_like(directions.x, origin.x),
                   numpy.full_like(directions.y, origin.y),
                   numpy.full_like(directions.z, origin.z),
                   directions.x,
                   numpy.full_like(directions.x, directions.y),
                   directions.z)

    with util.status_block("saving"):
        img = PIL.Image.fromarray(pixels)
        img.save(filename)

def nice_mode(obj, distances, final_values, epsilon, origins, directions):
    intersections = origins + directions * distances

    dot = (obj.distance(intersections + render_params.light * epsilon) - final_values) / epsilon
    intensities = T.clip(-dot, 0, 1)

    return [T.switch(final_values < epsilon, surface * intensities + ambient, bg)
            for surface, bg, ambient in zip(render_params.surface, render_params.background, render_params.ambient)]

def dot_mode(obj, distances, final_values, epsilon, origins, directions):
    intersections = origins + directions * distances

    normals = util.Vector(obj.distance(intersections + util.Vector(epsilon, 0, 0)) - final_values,
                          obj.distance(intersections + util.Vector(0, epsilon, 0)) - final_values,
                          obj.distance(intersections + util.Vector(0, 0, epsilon)) - final_values)
    normals = normals.normalized()

    return [T.switch(final_values < epsilon, -255 * normals.dot(directions), 128)] * 3

def distance_mode(obj, distances, final_values, epsilon, origins, directions):
    min_distances = T.min(distances)
    max_distances = T.max(T.switch(final_values < epsilon, distances, 0.0))
    return [255 * T.clip(1 - 0.8 * (distances - min_distances) / (max_distances - min_distances),
                        0.0, 1.0)] * 3
