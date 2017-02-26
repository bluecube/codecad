from . import util
from .compute import compute, program
import collections
import numpy
import math
import pyopencl

VolumeAndCentroid = collections.namedtuple("VolumeAndCentroid", "volume centroid")

def volume_and_centroid(shape, resolution):
    """ Return iterator over pairs (bounding box, numpy array of samples).
    Blocks that don't have any chance of crossing the boundary are skipped. """

    assert resolution > 0
    box = shape.bounding_box().expanded_additive(resolution)
    grid_dimensions = [math.ceil(s / resolution) + 1 for s in box.size()]

    #TODO: Since we're only doing a single pass, we can't have too dense sampling not to overflow the counters
    cell_count = grid_dimensions[0] * grid_dimensions[1] * grid_dimensions[2]
    print(grid_dimensions, cell_count)
    assert cell_count * grid_dimensions[0] <= 2**32
    assert cell_count * grid_dimensions[0] <= 2**32
    assert cell_count * grid_dimensions[0] <= 2**32

    corner = box.a + util.Vector.splat(resolution / 2)

    mf = pyopencl.mem_flags
    program_buffer = pyopencl.Buffer(compute.ctx,
                                     mf.READ_ONLY | mf.COPY_HOST_PTR,
                                     hostbuf=program.make_program(shape))
    counters = numpy.zeros(4, dtype=numpy.uint32)
    counters_buffer = pyopencl.Buffer(compute.ctx,
                                      mf.READ_WRITE,
                                      counters.nbytes)

    ev = pyopencl.enqueue_copy(compute.queue,
                               counters_buffer, counters,
                               is_blocking=False)
    ev = compute.program.volume(compute.queue, grid_dimensions, None,
                                program_buffer,
                                corner.as_float4(), numpy.float32(resolution),
                                counters_buffer,
                                wait_for=[ev])
    pyopencl.enqueue_copy(compute.queue,
                          counters, counters_buffer,
                          wait_for=[ev])

    count = counters[0]
    coords_sum = util.Vector(*(counters[i + 1] for i in range(3)))

    return VolumeAndCentroid(count * resolution * resolution * resolution,
                             coords_sum * (resolution / count) + corner)

