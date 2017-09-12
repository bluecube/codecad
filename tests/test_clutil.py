import pytest
import pyopencl.cltypes
import numpy

import codecad

codecad.opencl_manager.instance.add_compile_unit().append_file("test_clutil.cl")


def _tuple_from_xyz(xyz):
    return (xyz["x"], xyz["y"], xyz["z"])


@pytest.mark.parametrize("size", [4, (4), (4, 4), (4, 4, 4)])
def test_buffer_indexing(size):
    b = codecad.util.cl_util.Buffer(codecad.opencl_manager.instance.queue,
                                    pyopencl.cltypes.uint3,
                                    size,
                                    pyopencl.mem_flags.WRITE_ONLY)
    ev = codecad.opencl_manager.instance.k.indexing_identity(b.size, None, b.buffer)
    b.read(wait_for=[ev])

    for coords in b.array:
        seed = _tuple_from_xyz(coords)[:len(b.size)]
        assert _tuple_from_xyz(b[seed])[:len(b.size)] == seed
