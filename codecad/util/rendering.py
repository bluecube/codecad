from . import geometry
from . import misc
import theano
import theano.tensor as T
import numpy

def shape_apply(fun, shape, max_resolution, split_factor=8):
    """ Call fun on progressively smaller and smaller blocks of shape.
    Expanding blocks continues until either the blocks are smaller than
    max_resolution, or fun returns False for a block.
    Blocks are expanded into (approximately) split_factor**3 smaller blocks.

    Parameters passed to the function:
        - bounding box,
        - 2*2*2 array of values in box corners
        - is_large -- true if the box is larger than max_resolution in any dimension
        - is_intersecting -- true if the box intersects the surface.

    If the function returns True, the box will be split again. """

    x = T.tensor3("x")
    y = T.tensor3("y")
    z = T.tensor3("z")

    with misc.status_block("compiling"):
        evaluate = theano.function([x, y, z],
                                   shape.distance(geometry.Vector(x, y, z)),
                                   on_unused_input = 'ignore') # Epsilon might not be used

    stack = [shape.bounding_box().expanded(0.05)]

    with misc.status_block("eavaluating"):
        while len(stack):
            box = stack.pop()
            box_size = box.b - box.a

            step_size = box_size.min() / split_factor

            xs = numpy.arange(box.a.x, box.b.x + step_size, step_size)
            ys = numpy.arange(box.a.y, box.b.y + step_size, step_size)
            zs = numpy.arange(box.a.z, box.b.z + step_size, step_size)

            values = evaluate(*numpy.meshgrid(xs, ys, zs))

            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    for k in range(len(zs) - 1):
                        new_box = geometry.BoundingBox(geometry.Vector(xs[i], ys[j], zs[k]),
                                                       geometry.Vector(xs[i + 1], ys[j + 1], zs[k + 1]))

                        box_size = new_box.size()
                        is_large = box_size.min() > max_resolution

                        half_box_diagonal = abs(box_size) / 2

                        vertex_values = values[i:i+2,j:j+2,k:k+2]

                        is_intersecting = any(x < half_box_diagonal for x in vertex_values.flat) and \
                                          any(x > -half_box_diagonal for x in vertex_values.flat)

                        expand = fun(new_box,
                                     vertex_values,
                                     is_large,
                                     is_intersecting)

                        if expand:
                            stack.append(new_box)
