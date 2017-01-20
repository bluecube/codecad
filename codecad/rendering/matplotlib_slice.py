from .. import util
from ..compute import compute, program
import math
import numpy
import pyopencl
import matplotlib
import matplotlib.pyplot as plt

def render_slice(obj,
                 resolution,
                 filename=None # For interface compatibility with other renderers
                 ):

    assert resolution > 0

    box = obj.bounding_box().expanded(0.1).flattened()
    box_size = box.size()

    grid_dimensions = [math.ceil(s / resolution) + 1 for s in box_size]
    new_box_size = util.Vector(*(resolution * (k - 1) for k in grid_dimensions)).flattened()

    grid_dimensions[2] = 3

    corner = box.midpoint() - new_box_size / 2

    mf = pyopencl.mem_flags
    program_buffer = pyopencl.Buffer(compute.ctx,
                                     mf.READ_ONLY | mf.COPY_HOST_PTR,
                                     hostbuf=program.make_program(obj))
    values = numpy.empty((grid_dimensions[1], grid_dimensions[0], grid_dimensions[2]), dtype=numpy.float32)

    output_buffer = pyopencl.Buffer(compute.ctx,
                                    mf.WRITE_ONLY,
                                    values.nbytes)
    with util.status_block("running"):
        compute.program.matplotlib_slice(compute.queue, (grid_dimensions[0], grid_dimensions[1]), None,
                                         program_buffer,
                                         corner.as_float4(), numpy.float32(resolution),
                                         output_buffer)
        pyopencl.enqueue_copy(compute.queue, values, output_buffer)


    distances = values[:,:,0]
    distance_range = numpy.max(numpy.abs(distances))

    with util.status_block("plotting"):
        plt.imshow(distances,
                   cmap=plt.get_cmap("seismic"),
                   norm=matplotlib.colors.SymLogNorm(0.1,
                                                     vmin=-distance_range,
                                                     vmax=distance_range),
                   origin="lower",
                   interpolation="none",
                   aspect="equal",
                   extent=(corner.x, corner.x + (values.shape[1] - 1) * resolution,
                           corner.y, corner.y + (values.shape[0] - 1) * resolution))
        plt.colorbar()

        quiver_thinning = 10
        quiver_x = numpy.arange(0, grid_dimensions[0], quiver_thinning) * resolution + corner.x
        quiver_y = numpy.arange(0, grid_dimensions[1], quiver_thinning) * resolution + corner.y
        plt.quiver(quiver_x,
                   quiver_y,
                   values[::quiver_thinning,::quiver_thinning,1],
                   values[::quiver_thinning,::quiver_thinning,2])
    plt.show()
