from . import geometry
from .. import shapes
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
        - bool flag determining if the box is completely inside or outside of the shape
        - bool flag determining if the box will be expanded again.
    If the function returns true (and the last parameter was true), the box will be recursively
        expanded later (again calling the function). """

    x = T.tensor3("x")
    y = T.tensor3("y")
    z = T.tensor3("z")

    print("compiling...")
    evaluate = theano.function([x, y, z],
                               shape.distance(geometry.Vector(x, y, z)),
                               givens=[(shapes.Shape.Epsilon, max_resolution)],
                               on_unused_input = 'ignore') # Epsilon might not be used
    print("done")

    stack = [shape.bounding_box().expanded(0.05)]

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
                    expand = (new_box.b - new_box.a).min() > max_resolution
                    expand = expand and fun(new_box, values[i:i+2,j:j+2,k:k+2], expand)

                    if expand:
                        stack.append(new_box)
