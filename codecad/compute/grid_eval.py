import math
import numpy
import pyopencl
from . import compute, program
from .. import util

def grid_eval(shape, resolution, bounding_box = None):
    """ Return numpy array of values. """

    assert resolution > 0

    if bounding_box is None:
        box = shape.bounding_box().expanded_additive(resolution)
    else:
        box = bounding_box

    box_size = box.size()

    grid_dimensions = [math.ceil(s / resolution) + 1 for s in box_size]

    new_box_size = util.Vector(*(resolution * (k - 1) for k in grid_dimensions))

    corner = box.midpoint() - new_box_size / 2

    mf = pyopencl.mem_flags
    program_buffer = pyopencl.Buffer(compute.ctx,
                                     mf.READ_ONLY | mf.COPY_HOST_PTR,
                                     hostbuf=program.make_program(shape))
    output = numpy.empty((grid_dimensions[1], grid_dimensions[0], grid_dimensions[2]),
                         dtype=numpy.float32)

    output_buffer = pyopencl.Buffer(compute.ctx,
                                    mf.WRITE_ONLY,
                                    output.nbytes)
    compute.program.grid_eval(compute.queue, grid_dimensions, None,
                              program_buffer,
                              corner.as_float4(), numpy.float32(resolution),
                              output_buffer)
    pyopencl.enqueue_copy(compute.queue, output, output_buffer)

    return output, corner, resolution
