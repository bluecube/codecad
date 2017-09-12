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


@pytest.mark.parametrize("size, nitems", [((4), 4), ((4, 4), 16), ((4, 4, 4), 64)])
@pytest.mark.parametrize("item_type, item_size", [(pyopencl.cltypes.uint3, 16),
                                                  (pyopencl.cltypes.uchar, 1),
                                                  (pyopencl.cltypes.double16, 128)])
def test_buffer_alloc_size(size, nitems, item_type, item_size):
    b = codecad.util.cl_util.Buffer(codecad.opencl_manager.instance.queue,
                                    item_type,
                                    size,
                                    pyopencl.mem_flags.WRITE_ONLY)

    b.create_host_side_array()

    assert b.array.nbytes == nitems * item_size


def test_buffer_read_write():
    b = codecad.util.cl_util.Buffer(codecad.opencl_manager.instance.queue,
                                    pyopencl.cltypes.ulong,
                                    1,
                                    pyopencl.mem_flags.READ_WRITE)

    b.create_host_side_array()
    b[0] = 42
    ev = b.enqueue_write()
    ev = codecad.opencl_manager.instance.k.one_item_double((1,), None,
                                                           b.buffer,
                                                           wait_for=[ev])
    b.read(wait_for=[ev])

    assert b[0] == 84


def test_buffer_read_write():
    b = codecad.util.cl_util.Buffer(codecad.opencl_manager.instance.queue,
                                    pyopencl.cltypes.ulong,
                                    1,
                                    pyopencl.mem_flags.READ_WRITE)

    b.create_host_side_array()
    b[0] = 42
    ev = b.enqueue_write()
    ev = codecad.opencl_manager.instance.k.one_item_double((1,), None,
                                                           b.buffer,
                                                           wait_for=[ev])
    b.read(wait_for=[ev])

    assert b[0] == 84
