import struct

import pyopencl

from . import scheduler, node
from .. import opencl_manager

class NodeCache:
    def __init__(self):
        self._cache = {}

    def make_node(self, name, params, dependencies, extra_data = None):
        n = node.Node(name, params, dependencies, extra_data)

        #TODO: Maybe try caching subsets of dependencies of >2-ary nodes?
        try:
            cached = self._cache[n]
            assert cached is not n
            n.disconnect() # Don't let node increase dependencies' refcount
                           # when we're not using it
            return cached
        except KeyError:
            pass
        self._cache[n] = n
        return n

def get_shape_nodes(shape):
    cache = NodeCache() # TODO: Figure out how to share cache between shapes?
    point = cache.make_node("_point", (), ())
    return shape.get_node(point, cache)

def _make_program_pieces(shape):
    registers_needed, schedule = scheduler.randomized_scheduler(get_shape_nodes(shape))

    assert schedule[0].name == "_point"
    assert schedule[0].register == 0
    assert schedule[-1].register == 0

    parameter_encoder = struct.Struct("f") #TODO Endian
    instruction_encoder = struct.Struct("BBBB")

    for n in schedule[1:]:
        assert len(n.dependencies) > 0
        assert len(n.dependencies) <= 2
        yield instruction_encoder.pack(node.Node._type_map[n.name],
                                       n.register,
                                       n.dependencies[0].register,
                                       n.dependencies[1].register if len(n.dependencies) > 1 else 0)
        for param in n.params:
            yield parameter_encoder.pack(param)

    yield instruction_encoder.pack(0, 0, 0, 0)

def make_program(shape):
    return b"".join(_make_program_pieces(shape))

def make_program_buffer(shape):
    return pyopencl.Buffer(opencl_manager.instance.context,
                           pyopencl.mem_flags.READ_ONLY | pyopencl.mem_flags.COPY_HOST_PTR,
                           hostbuf=make_program(shape))
