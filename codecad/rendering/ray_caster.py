import PIL
import math
import pyopencl
import numpy

from .. import util
from .. import animation
from . import render_params

from ..compute import compute, program

def _zero_if_inf(x):
    if math.isinf(x):
        return 0
    else:
        return x

def _render_frame(obj,
                  origin, direction, up, focal_length,
                  size, epsilon):

    box = obj.bounding_box()
    obj.check_dimension(required = 3)

    forward = direction.normalized()
    up = up - forward * up.dot(forward)
    up = up.normalized()
    right = forward.cross(up)
    forward = forward * focal_length

    print("origin", origin)
    print("forward", forward)

    origin_to_midpoint = abs(origin - box.midpoint())
    box_radius = abs(box.size()) / 2
    min_distance = max(0, origin_to_midpoint - box_radius)
    max_distance = origin_to_midpoint + box_radius

    output = numpy.empty([size[1], size[0], 3], dtype=numpy.uint8)

    mf = pyopencl.mem_flags
    program_buffer = pyopencl.Buffer(compute.ctx,
                                     mf.READ_ONLY | mf.COPY_HOST_PTR,
                                     hostbuf=program.make_program(obj))
    output_buffer = pyopencl.Buffer(compute.ctx,
                                    mf.WRITE_ONLY,
                                    output.nbytes)

    compute.program.ray_caster(compute.queue, size, None,
                               program_buffer,
                               origin.as_float4(), forward.as_float4(), up.as_float4(), right.as_float4(),
                               render_params.surface.as_float4(), render_params.background.as_float4(),
                               render_params.light.as_float4(), numpy.float32(render_params.ambient),
                               numpy.float32(epsilon), numpy.uint32(100), numpy.float32(min_distance), numpy.float32(max_distance),
                               output_buffer)

    pyopencl.enqueue_copy(compute.queue, output, output_buffer)

    return output

def get_camera_params(box, size, view_angle):
    box_size = box.size()

    size_diagonal = math.hypot(*size)

    if view_angle is None:
        focal_length = size_diagonal # Normal lens by default
    else:
        focal_length = size_diagonal / (2 * math.tan(math.radians(view_angle) / 2))

    distance = focal_length * max(_zero_if_inf(box_size.x) / size[0],
                                  _zero_if_inf(box_size.z) / size[1])

    print(box_size, distance)

    if distance == 0:
        distance = 1

    distance *= 1.2 # 20% margin around the object

    origin = box.midpoint() - util.Vector(0, distance + _zero_if_inf(box_size.y) / 2, 0)
    direction = util.Vector(0, 1, 0)
    up = util.Vector(0, 0, 1)

    return (origin, direction, up, focal_length)

def render_picture(obj, filename, size = (800, 600),
                   view_angle = None,
                   resolution=None):

    with util.status_block("calculating bounding box"):
        box = obj.bounding_box()

    box_size = box.size()

    if resolution is None:
        epsilon = min(1, box_size.x, box_size.y, box_size.z) / 10000;
    else:
        epsilon = resolution / 10

    camera_params = get_camera_params(box, size, view_angle)

    with util.status_block("rendering"):
        pixels = _render_frame(obj, size=size, epsilon=epsilon, *camera_params)

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
        epsilon = min(box_size.x, box_size.y, box_size.z) / 10000;
    else:
        epsilon = resolution / 10

    camera_params = get_camera_params(box, size, view_angle)

    frame_duration = int(1000 / fps) # Frame duration in milliseconds
    count = round(1000 * duration / frame_duration)

    frames = []
    for i in range(count):
        time = (i * frame_duration) / 1000
        with util.status_block("rendering frame {}/{}".format(i + 1, count)):
            pixels = _render_frame(obj, size=size, epsilon=epsilon, *camera_params)
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
