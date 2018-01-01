import numpy
import pyopencl

from .. import util
from .. import cl_util
from ..cl_util import opencl_manager
from .. import nodes

opencl_manager.add_compile_unit().append_file("bitmap.cl")


def render(obj, size):
    obj.check_dimension(required=2)

    box = obj.bounding_box().flattened()
    box_size = box.size()
    resolution = util.Vector(size[0], size[1], 1)
    # The final 1 is to avoid division by zero problems.

    step_size = box_size.elementwise_div(resolution).max()
    origin = box.midpoint() - resolution * step_size / 2

    shape = (size[0], size[1], 3)
    program_buffer = nodes.make_program_buffer(obj)
    output = cl_util.Buffer(numpy.uint8,
                            shape,
                            pyopencl.mem_flags.WRITE_ONLY)

    ev = opencl_manager.k.bitmap(size, None,
                                 program_buffer,
                                 origin.as_float4(), numpy.float32(step_size),
                                 output)
    return output.read(wait_for=[ev]).reshape(shape).transpose((1, 0, 2))
