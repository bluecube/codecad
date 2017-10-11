import struct

import pyopencl

from .. import util
from . import scheduler, node
from .. import opencl_manager


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


def _make_program_pieces(shape):
    nodes = get_shape_nodes(shape)
    scheduler.calculate_node_refcounts(nodes)
    registers_needed, schedule = scheduler.randomized_scheduler(nodes)

    assert registers_needed <= opencl_manager.instance.max_register_count

    parameter_encoder = struct.Struct("f")  # TODO Endian
    instruction_encoder = struct.Struct("BBBB")

    for n in schedule:
        assert len(n.dependencies) <= 2
        yield instruction_encoder.pack(node.Node._node_types[n.name][2],
                                       n.register,
                                       n.dependencies[0].register if len(n.dependencies) > 0 else 0,
                                       n.dependencies[1].register if len(n.dependencies) > 1 else 0)
        for param in n.params:
            yield parameter_encoder.pack(param)


def make_program(shape):
    return b"".join(_make_program_pieces(shape))


def make_program_buffer(shape):
    return pyopencl.Buffer(opencl_manager.instance.context,
                           pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
                           hostbuf=make_program(shape))
