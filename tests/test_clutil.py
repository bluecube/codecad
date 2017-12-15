import pytest
import pyopencl.cltypes
import numpy
import os.path

import codecad

codecad.cl_util.opencl_manager.add_compile_unit().append_file("test_clutil.cl")


def _tuple_from_xyz(xyz):
    return (xyz["x"], xyz["y"], xyz["z"])


@pytest.mark.parametrize("size", [4, (4), (4, 4), (4, 4, 4)])
def test_buffer_indexing(size):
    b = codecad.cl_util.Buffer(codecad.cl_util.opencl_manager.queue,
                               pyopencl.cltypes.uint3,
                               size,
                               pyopencl.mem_flags.WRITE_ONLY)
    ev = codecad.cl_util.opencl_manager.k.indexing_identity(b.size, None, b.buffer)
    b.read(wait_for=[ev])

    for coords in b.array:
        seed = _tuple_from_xyz(coords)[:len(b.size)]
        assert _tuple_from_xyz(b[seed])[:len(b.size)] == seed


@pytest.mark.parametrize("size, nitems", [((4), 4), ((4, 4), 16), ((4, 4, 4), 64)])
@pytest.mark.parametrize("item_type, item_size", [(pyopencl.cltypes.uint3, 16),
                                                  (pyopencl.cltypes.uchar, 1),
                                                  (pyopencl.cltypes.double16, 128)])
def test_buffer_alloc_size(size, nitems, item_type, item_size):
    b = codecad.cl_util.Buffer(codecad.cl_util.opencl_manager.queue,
                               item_type,
                               size,
                               pyopencl.mem_flags.WRITE_ONLY)

    b.create_host_side_array()

    assert b.nitems == nitems
    assert b.nbytes == nitems * item_size
    assert b.array.nbytes == nitems * item_size


def test_buffer_read_write():
    b = codecad.cl_util.Buffer(codecad.cl_util.opencl_manager.queue,
                               pyopencl.cltypes.ulong,
                               1,
                               pyopencl.mem_flags.READ_WRITE)

    b.create_host_side_array()
    b[0] = 42
    ev = b.enqueue_write()
    ev = codecad.cl_util.opencl_manager.k.one_item_double((1,), None,
                                                          b.buffer,
                                                          wait_for=[ev])
    b.read(wait_for=[ev])

    assert b[0] == 84


def test_buffer_read_write():
    b = codecad.cl_util.Buffer(codecad.cl_util.opencl_manager.queue,
                               pyopencl.cltypes.ulong,
                               1,
                               pyopencl.mem_flags.READ_WRITE)

    b.create_host_side_array()
    b[0] = 42
    ev = b.enqueue_write()
    ev = codecad.cl_util.opencl_manager.k.one_item_double((1,), None,
                                                          b.buffer,
                                                          wait_for=[ev])
    b.read(wait_for=[ev])

    assert b[0] == 84


def test_buffer_read_write_map():
    b = codecad.cl_util.Buffer(codecad.cl_util.opencl_manager.queue,
                               pyopencl.cltypes.ulong,
                               1,
                               pyopencl.mem_flags.READ_WRITE)

    with b.map(pyopencl.map_flags.WRITE_INVALIDATE_REGION, wait_for=None) as mapped:
        mapped[0] = 31

    ev = codecad.cl_util.opencl_manager.k.one_item_double((1,), None,
                                                          b.buffer)

    with b.map(pyopencl.map_flags.READ, wait_for=[ev]) as result:
        assert result[0] == 62


def test_buffer_write_map():
    b = codecad.cl_util.Buffer(codecad.cl_util.opencl_manager.queue,
                               pyopencl.cltypes.uchar8,
                               1,
                               pyopencl.mem_flags.READ_WRITE)

    v = pyopencl.cltypes.filled_uchar8(17)

    with b.map(pyopencl.map_flags.WRITE_INVALIDATE_REGION, wait_for=None) as mapped:
        mapped[0] = v
    b.read()
    assert b[0] == v


def test_buffer_read_map():
    b = codecad.cl_util.Buffer(codecad.cl_util.opencl_manager.queue,
                               pyopencl.cltypes.uchar8,
                               1,
                               pyopencl.mem_flags.READ_WRITE)

    v = pyopencl.cltypes.filled_uchar8(71)

    b.create_host_side_array()
    b[0] = v
    ev = b.enqueue_write()

    with b.map(pyopencl.map_flags.READ, wait_for=[ev]) as mapped:
        assert mapped[0] == v


@pytest.mark.parametrize("item_type, item_size", [(pyopencl.cltypes.uint3, 16),
                                                  (pyopencl.cltypes.uchar, 1),
                                                  (pyopencl.cltypes.double16, 128)])
def test_buffer_map_size(item_type, item_size):
    b = codecad.cl_util.Buffer(codecad.cl_util.opencl_manager.queue,
                               item_type,
                               (2, 2, 2),
                               pyopencl.mem_flags.WRITE_ONLY)

    with b.map(pyopencl.map_flags.WRITE_INVALIDATE_REGION, wait_for=None) as mapped:
        assert mapped.nbytes == b.nbytes
        assert mapped.dtype == item_type


def test_assert_pass():
    assertBuffer = codecad.cl_util.AssertBuffer(codecad.cl_util.opencl_manager.queue)

    # Checking before first use should always pass
    assertBuffer.check()

    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(100), assertBuffer.buffer)

    # The kernel shouldn't have logged an assert
    assertBuffer.check(wait_for=[ev])


def test_assert_fail():
    assertBuffer = codecad.cl_util.AssertBuffer(codecad.cl_util.opencl_manager.queue)
    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(5), assertBuffer.buffer)

    with pytest.raises(codecad.cl_util.cl_assert.OpenClAssertionError) as exc_info:
        assertBuffer.check(wait_for=[ev])

    assert exc_info.value.count == 1
    assert os.path.basename(exc_info.value.filename) == "test_clutil.cl"
    assert exc_info.value.global_id == [5, 5, 0, 0]


def test_assert_chaining_pass():
    assertBuffer = codecad.cl_util.AssertBuffer(codecad.cl_util.opencl_manager.queue)
    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(101), assertBuffer.buffer)
    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(102), assertBuffer.buffer,
                                                        wait_for=[ev])

    # The kernel shouldn't have logged an assert
    assertBuffer.check(wait_for=[ev])


def test_assert_chaining_multiple_fail():
    assertBuffer = codecad.cl_util.AssertBuffer(codecad.cl_util.opencl_manager.queue)
    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(1), assertBuffer.buffer)
    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(2), assertBuffer.buffer,
                                                        wait_for=[ev])

    with pytest.raises(codecad.cl_util.cl_assert.OpenClAssertionError) as exc_info:
        assertBuffer.check(wait_for=[ev])

    assert exc_info.value.count == 2
    assert os.path.basename(exc_info.value.filename) == "test_clutil.cl"
    assert exc_info.value.global_id == [1, 1, 0, 0]
