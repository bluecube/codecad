import math
import numpy
import pyopencl
import matplotlib
import matplotlib.pyplot as plt

from .. import util
from ..cl_util import opencl_manager
from .. import nodes

opencl_manager.add_compile_unit().append_file("matplotlib_slice.cl")


def render_slice(obj,
                 filename=None  # For interface compatibility with other renderers
                 ):

    resolution = obj.feature_size()

    box = obj.bounding_box().expanded(0.1).flattened()
    box_size = box.size()

    grid_dimensions = [math.ceil(s / resolution) + 1 for s in box_size]
    new_box_size = util.Vector(*(resolution * (k - 1) for k in grid_dimensions)).flattened()

    grid_dimensions[2] = 3

    corner = box.midpoint() - new_box_size / 2

    mf = pyopencl.mem_flags
    program_buffer = nodes.make_program_buffer(obj)

    values = numpy.empty((grid_dimensions[1], grid_dimensions[0], grid_dimensions[2]), dtype=numpy.float32)

    output_buffer = pyopencl.Buffer(opencl_manager.context,
                                    mf.WRITE_ONLY,
                                    values.nbytes)
    with util.status_block("running"):
        ev = opencl_manager.k.matplotlib_slice((grid_dimensions[0], grid_dimensions[1]), None,
                                               program_buffer,
                                               corner.as_float4(), numpy.float32(resolution),
                                               output_buffer)
        pyopencl.enqueue_copy(opencl_manager.queue, values, output_buffer, wait_for=[ev])

    distances = values[:, :, 0]
    distance_range = numpy.max(numpy.abs(distances))

    with util.status_block("plotting"):
        common_args = {"norm": matplotlib.colors.SymLogNorm(0.1,
                                                            vmin=-distance_range,
                                                            vmax=distance_range),
                       "origin": "lower",
                       "aspect": "equal",
                       "extent": (corner.x, corner.x + (values.shape[1] - 1) * resolution,
                                  corner.y, corner.y + (values.shape[0] - 1) * resolution)}

        plt.imshow(distances,
                   cmap=plt.get_cmap("RdBu"),
                   interpolation="none",
                   **common_args)
        plt.colorbar()
        plt.contour(distances,
                    colors="black",
                    **common_args)

        quiver_thinning = 10
        quiver_x = numpy.arange(0, grid_dimensions[0], quiver_thinning) * resolution + corner.x
        quiver_y = numpy.arange(0, grid_dimensions[1], quiver_thinning) * resolution + corner.y
        plt.quiver(quiver_x,
                   quiver_y,
                   values[::quiver_thinning, ::quiver_thinning, 1],
                   values[::quiver_thinning, ::quiver_thinning, 2],
                   color="white",
                   angles="xy",
                   scale_units="xy",
                   scale=200 / new_box_size.max())
    plt.show()
