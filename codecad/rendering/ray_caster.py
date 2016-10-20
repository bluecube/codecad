import PIL
import theano
import theano.tensor as T
import numpy
import math

from .. import util
from .. import shapes
from .. import animation
from . import render_params

def make_func(obj, size, epsilon):
    with util.status_block("building expression"):
        # Variables used as function parameters
        ox = T.scalar("ox")
        oy = T.scalar("oy")
        oz = T.scalar("oz")

        dx = T.scalar("dx")
        dy = T.scalar("dy")
        dz = T.scalar("dz")

        upx = T.scalar("upx")
        upy = T.scalar("upy")
        upz = T.scalar("upz")

        fl = T.scalar("fl")

        tau = animation.tau

        # Preparing ray vectors for tracing
        size = (size[1], size[0])
            # We need to swap size components to be compatible with the
            # height * width convention of numpy matrices

        direction = util.Vector(dx, dy, dz).normalized()

        up = util.Vector(upx, upy, upz)
        up = up - direction * up.dot(direction)
        up = up.normalized()

        right = direction.cross(up)

        film_x = T.tile(T.arange(size[1]), (size[0], 1)) - (size[1] - 1) / 2
        film_y = T.tile(T.arange(size[0]), (size[1], 1)).T - (size[0] - 1) / 2

        rays = direction * fl + (right * film_x - up * film_y)
        rays = rays.normalized()

        origins = util.Vector(T.tile(ox, size), T.tile(oy, size), T.tile(oz, size))

        # Actual tracing
        def trace_func(previous, _, ox, oy, oz, dx, dy, dz):
            o = util.Vector(ox, oy, oz)
            d = util.Vector(dx, dy, dz)

            distance = obj.distance(o + d * previous)

            return [previous + 0.8 * distance, distance], \
                   theano.scan_module.until(T.all(T.or_(distance < epsilon / 2,
                                                        T.isinf(distance))))

        distances, final_values = theano.scan(trace_func,
                                              outputs_info=[T.zeros(size), T.zeros(size)],
                                              non_sequences=[origins.x, origins.y, origins.z,
                                                             rays.x, rays.y, rays.z],
                                              n_steps = 100)[0]

        distances = distances[-1]
        final_values = final_values[-1]

        # Shading
        intersections = origins + rays * distances

        dot = (obj.distance(intersections + render_params.light * epsilon) - final_values) / epsilon
            # Dot product of a gradient and a vector is the same as a directional
            # derivative along the vector

        intensities = T.clip(-dot, 0, 1)

        rgb = [T.switch(final_values < epsilon, surface * intensities + ambient, bg)
               for surface, bg, ambient in zip(render_params.surface,
                                               render_params.background,
                                               render_params.ambient)]

        colors = T.clip(T.stack(rgb, 2), 0, 255).astype("uint8")

    with util.status_block("compiling"):
        f = theano.function([ox, oy, oz,
                             dx, dy, dz,
                             upx, upy, upz,
                             fl,
                             tau],
                            colors,
                            on_unused_input = 'ignore')

    def render_frame(origin, direction, up, focal_length, tau = 0):
        return f(origin.x, origin.y, origin.z,
                 direction.x, direction.y, direction.z,
                 up.x, up.y, up.z,
                 focal_length,
                 tau)

    return render_frame


def render_picture(obj, filename, size = (800, 600),
                   view_angle = 90,
                   resolution=None):

    with util.status_block("calculating bounding box"):
        box = obj.bounding_box()
    box_size = box.size()

    if resolution is None:
        epsilon = min(box_size.x, box_size.y, box_size.z) / 1000;
    else:
        epsilon = resolution

    focal_length = size[0] / math.tan(math.radians(view_angle) / 2)
    distance = focal_length * max(box_size.x / size[0],
                                  box_size.z / size[1])

    origin = box.midpoint() - util.Vector(0, distance + box_size.y / 2, 0)
    direction = util.Vector(0, 1, 0)
    up = util.Vector(0, 0, 1)

    f = make_func(obj, size, epsilon)

    with util.status_block("running"):
        pixels = f(origin, direction, up, focal_length)

    with util.status_block("saving"):
        img = PIL.Image.fromarray(pixels)
        img.save(filename)
