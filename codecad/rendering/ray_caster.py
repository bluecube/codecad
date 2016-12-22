import PIL
import math
import pyopencl
import numpy

from .. import util
from .. import animation
from . import render_params

from ..compute import compute, program

def render_frame(obj,
                 origin, direction, up, focal_length,
                 size, epsilon):

    obj.check_dimension(required = 3)

    direction = direction.normalized()
    up = up - direction * up.dot(direction)
    up = up.normalized()
    right = direction.cross(up)
    direction = direction * focal_length

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
                               origin.as_float3(), direction.as_float3(), up.as_float3(), right.as_float3(),
                               render_params.surface.as_float3(), render_params.background.as_float3(),
                               render_params.light.as_float3(), numpy.float32(render_params.ambient),
                               numpy.float32(epsilon), numpy.uint32(100),
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

    camera_params = get_camera_params(box, size, view_angle)

    with util.status_block("rendering"):
        pixels = render_frame(obj, size=size, epsilon=epsilon, *camera_params)

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

    camera_params = get_camera_params(box, size, view_angle)

    frame_duration = int(1000 / fps) # Frame duration in milliseconds
    count = round(1000 * duration / frame_duration)

    frames = []
    for i in range(count):
        time = (i * frame_duration) / 1000
        with util.status_block("rendering frame {}/{}".format(i + 1, count)):
            pixels = render_frame(obj, size=size, epsilon=epsilon, *camera_params)
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
