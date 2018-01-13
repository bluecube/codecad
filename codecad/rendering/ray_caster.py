import math

import pyopencl
import numpy
import flags

from .. import util
from .. import cl_util
from ..cl_util import opencl_manager
from .. import nodes


class RenderOptions(flags.Flags):
    false_color = ()


_c_file = opencl_manager.add_compile_unit()
for flag in RenderOptions:
    _c_file.append_define("RENDER_OPTIONS_{}".format(flag.to_simple_str().upper()), int(flag))
_c_file.append_file("ray_caster.cl")


def _zero_if_inf(x):
    if math.isinf(x):
        return 0
    else:
        return x


def render(obj,
           origin, direction, up, focal_length,
           size, options=RenderOptions.no_flags):

    box = obj.bounding_box()
    obj.check_dimension(required=3)

    forward = direction.normalized()
    up = up - forward * up.dot(forward)
    up = up.normalized()
    right = forward.cross(up)
    forward = forward * focal_length

    pixel_tolerance = 0.5 / focal_length  # Tangents of half pixel angle

    origin_to_midpoint = abs(origin - box.midpoint())
    box_radius = abs(box.size()) / 2
    min_distance = max(0, origin_to_midpoint - box_radius)
    max_distance = origin_to_midpoint + box_radius

    mf = pyopencl.mem_flags
    program_buffer = nodes.make_program_buffer(obj)
    output_buffer = cl_util.Buffer(numpy.uint8, [size[0], size[1], 3], mf.WRITE_ONLY)

    assert_buffer = cl_util.AssertBuffer()

    ev = opencl_manager.k.ray_caster(size, None,
                                     program_buffer,
                                     origin.as_float4(), forward.as_float4(), up.as_float4(), right.as_float4(),
                                     numpy.float32(pixel_tolerance), numpy.float32(box_radius),
                                     numpy.float32(min_distance), numpy.float32(max_distance),
                                     numpy.float32(box.a.z - box.size().z / 20),
                                     numpy.uint32(options),
                                     output_buffer,
                                     assert_buffer)

    assert_buffer.check(wait_for=[ev])

    output_buffer.read(wait_for=[ev])

    print("Render took", (ev.profile.end - ev.profile.start) / 1e9)

    if options & RenderOptions.false_color:
        for i, name in enumerate(["Steps taken", "Residual * 1000"]):
            channel = output_buffer[:, :, i]
            print("{}: min: {}, max: {}, mean: {}".format(name,
                                                          channel.min(),
                                                          channel.max(),
                                                          channel.mean()))

    return output_buffer.array.transpose((1, 0, 2))


def get_camera_params(box, size, view_angle):
    box_size = box.size()

    size_diagonal = math.hypot(*size)

    if view_angle is None:
        focal_length = size_diagonal  # Normal lens by default
    else:
        focal_length = size_diagonal / (2 * math.tan(math.radians(view_angle) / 2))

    distance = focal_length * max(_zero_if_inf(box_size.x) / size[0],
                                  _zero_if_inf(box_size.z) / size[1])

    if distance == 0:
        distance = 1

    distance *= 1.2  # 20% margin around the object

    origin = box.midpoint() - util.Vector(0, distance + _zero_if_inf(box_size.y) / 2, 0)
    direction = util.Vector(0, 1, 0)
    up = util.Vector(0, 0, 1)

    return (origin, direction, up, focal_length)
