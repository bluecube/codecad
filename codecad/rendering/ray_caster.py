import PIL
import theano
import theano.tensor as T
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

        time = animation.time

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
                             time],
                            colors,
                            on_unused_input = 'ignore')

    def render_frame(origin, direction, up, focal_length, time = 0):
        return f(origin.x, origin.y, origin.z,
                 direction.x, direction.y, direction.z,
                 up.x, up.y, up.z,
                 focal_length,
                 time)

    return render_frame

def get_camera_params(box, size, view_angle):
    box_size = box.size()

    size_diagonal = math.hypot(*size)

    if view_angle is None:
        focal_length = size_diagonal # Normal lens by default
    else:
        focal_length = size_diagonal / (2 * math.tan(math.radians(view_angle) / 2))

    distance = focal_length * max(box_size.x / size[0],
                                  box_size.z / size[1])
    distance *= 1.2 # 20% margin around the object

    origin = box.midpoint() - util.Vector(0, distance + box_size.y / 2, 0)
    direction = util.Vector(0, 1, 0)
    up = util.Vector(0, 0, 1)

    return (origin, direction, up, focal_length)

def render_picture(obj, filename, size = (800, 600),
                   view_angle = None,
                   resolution=None):

    with util.status_block("calculating bounding box"):
        box = obj.bounding_box()

    if resolution is None:
        epsilon = min(box_size.x, box_size.y, box_size.z) / 1000;
    else:
        epsilon = resolution

    f = make_func(obj, size, epsilon)

    camera_params = get_camera_params(box, size, view_angle)

    with util.status_block("rendering"):
        pixels = f(*camera_params)

    with util.status_block("saving"):
        img = PIL.Image.fromarray(pixels)
        img.save(filename)

def render_gif(obj, filename, size = (640, 480),
               view_angle = None,
               duration = 5,
               fps = 20,
               loop = True,
               resolution=None):

    with util.status_block("calculating bounding box"):
        box = obj.bounding_box().eval({animation.time: 0})

    if resolution is None:
        epsilon = min(box_size.x, box_size.y, box_size.z) / 1000;
    else:
        epsilon = resolution

    f = make_func(obj, size, epsilon)

    camera_params = get_camera_params(box, size, view_angle)

    frame_duration = int(1000 / fps) # Frame duration in milliseconds
    count = round(1000 * duration / frame_duration)

    frames = []
    for i in range(count):
        time = (i * frame_duration) / 1000
        with util.status_block("rendering frame {}/{}".format(i + 1, count)):
            pixels = f(*camera_params, time = time)
            frame = PIL.Image.fromarray(pixels)
            frames.append(frame)

    with util.status_block("saving"):
        argv = {}
        if loop:
            argv["loop"] = 0

        frames[0].save(filename,
                       append_images = frames[1:],
                       save_all = True,
                       duration = frame_duration,
                       **argv)
