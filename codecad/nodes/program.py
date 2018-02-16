import numpy
import pyopencl
import pyopencl.cltypes

from .. import util
from . import scheduler, node
from ..cl_util import opencl_manager


class NodeCache:
    def __init__(self):
        self._cache = {}

    def make_node(self, name, params, dependencies, extra_data=None):
        n = node.Node(name, params, dependencies, extra_data)

        # TODO: Maybe try caching subsets of dependencies of >2-ary nodes?
        try:
            cached = self._cache[n]
        except KeyError:
            self._cache[n] = n
            return n
        else:
            assert cached is not n
            return cached


def get_shape_nodes(shape):
    cache = NodeCache()
    zero_transform = util.Transformation.zero()
    point = cache.make_node("initial_transformation_to",
                            zero_transform.as_list(),
                            (),
                            zero_transform)
    return_node = cache.make_node("_return",
                                  (),
                                  (shape.get_node(point, cache),))

    return return_node


def get_opcode(n):
    opcode = node.Node._node_types[n.name][2]
    if n.name == "_load":
        assert len(n.dependencies) == 1
        assert n.dependencies[0].name == "_store"
        secondaryRegister = n.dependencies[0].register
    elif n.name == "_store":
        secondaryRegister = n.register
    elif len(n.dependencies) >= 2:
        secondaryRegister = n.dependencies[1].register
    else:
        secondaryRegister = 0

    return opcode, secondaryRegister


def _make_program_pieces(shape):
    nodes = get_shape_nodes(shape)
    scheduler.calculate_node_refcounts(nodes)
    registers_needed, schedule = scheduler.randomized_scheduler(nodes)

    assert registers_needed <= opencl_manager.max_register_count

    for n in schedule:
        assert len(n.dependencies) <= 2

        opcode, secondaryRegister = get_opcode(n)
        instruction = opcode * opencl_manager.max_register_count + secondaryRegister

        assert int(numpy.float32(instruction)) == instruction

        yield instruction
        yield from n.params


def make_program(shape):
    """ Returns numpy array containing the eval instructions for eval """
    return numpy.fromiter(_make_program_pieces(shape), pyopencl.cltypes.float)


def make_program_buffer(shape):
    return pyopencl.Buffer(opencl_manager.context,
                           pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
                           hostbuf=make_program(shape))
