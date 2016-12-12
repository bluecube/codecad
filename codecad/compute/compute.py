import struct

from . import nodes
from . import scheduler

def _make_program_pieces(shape):
    node = nodes.get_shape_nodes(shape)
    registers_needed, schedule = scheduler.randomized_scheduler(node)

    assert schedule[0].name == "_point"
    assert schedule[0].register == 0

    parameter_encoder = struct.Struct("f") #TODO Endian
    instruction_encoder = struct.Struct("BBBB")

    for n in schedule[1:]:
        assert len(n.dependencies) > 0
        assert len(n.dependencies) <= 2
        yield instruction_encoder.pack(nodes.Node._type_map[n.name],
                                       n.register,
                                       n.dependencies[0].register,
                                       n.dependencies[1].register if len(n.dependencies) > 1 else 0)
        for param in n.params:
            yield parameter_encoder.pack(param)

    yield instruction_encoder.pack(0, 0, 0, 0)

def make_program(shape):
    return b"".join(_make_program_pieces(shape))
