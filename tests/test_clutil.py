import pytest
import pyopencl.cltypes
import numpy
import os.path

import codecad

codecad.cl_util.opencl_manager.add_compile_unit().append_file("test_clutil.cl")


def _tuple_from_xyz(xyz):
    return (xyz["x"], xyz["y"], xyz["z"])


@pytest.mark.parametrize("size", [4, (4,), (4, 4), (4, 4, 4)])
def test_buffer_indexing(size):
    b = codecad.cl_util.Buffer(pyopencl.cltypes.uint3,
                               size,
                               pyopencl.mem_flags.WRITE_ONLY)
    ev = codecad.cl_util.opencl_manager.k.indexing_identity(b.shape, None, b)
    b.read(wait_for=[ev])

    for index in numpy.ndindex(b.shape):
        assert tuple(b[index])[:len(b.shape)] == index
        assert tuple(b.array[index])[:len(b.shape)] == index


@pytest.mark.parametrize("size, nitems", [((4,), 4), ((4, 4), 16), ((4, 4, 4), 64)])
@pytest.mark.parametrize("item_type, item_size", [(pyopencl.cltypes.uint3, 16),
                                                  (pyopencl.cltypes.uchar, 1),
                                                  (pyopencl.cltypes.double16, 128)])
def test_buffer_alloc_size(size, nitems, item_type, item_size):
    b = codecad.cl_util.Buffer(item_type,
                               size,
                               pyopencl.mem_flags.WRITE_ONLY)

    b.create_host_side_array()

    assert b.nitems == nitems
    assert b.size == nitems * item_size
    assert b.array.nbytes == b.size


def test_buffer_read_write():
    b = codecad.cl_util.Buffer(pyopencl.cltypes.ulong,
                               1,
                               pyopencl.mem_flags.READ_WRITE)

    b.create_host_side_array()
    b[0] = 42
    ev = b.enqueue_write()
    ev = codecad.cl_util.opencl_manager.k.one_item_double((1,), None,
                                                          b, wait_for=[ev])
    b.read(wait_for=[ev])

    assert b[0] == 84


def test_buffer_read_write():
    b = codecad.cl_util.Buffer(pyopencl.cltypes.ulong,
                               1,
                               pyopencl.mem_flags.READ_WRITE)

    b.create_host_side_array()
    b[0] = 42
    ev = b.enqueue_write()
    ev = codecad.cl_util.opencl_manager.k.one_item_double((1,), None,
                                                          b, wait_for=[ev])
    b.read(wait_for=[ev])

    assert b[0] == 84


def test_buffer_read_write_map():
    b = codecad.cl_util.Buffer(pyopencl.cltypes.ulong,
                               1,
                               pyopencl.mem_flags.READ_WRITE)

    with b.map(pyopencl.map_flags.WRITE_INVALIDATE_REGION, wait_for=None) as mapped:
        mapped[0] = 31

    ev = codecad.cl_util.opencl_manager.k.one_item_double((1,), None, b)

    with b.map(pyopencl.map_flags.READ, wait_for=[ev]) as result:
        assert result[0] == 62


def test_buffer_write_map():
    b = codecad.cl_util.Buffer(pyopencl.cltypes.uchar8,
                               1,
                               pyopencl.mem_flags.READ_WRITE)

    v = pyopencl.cltypes.filled_uchar8(17)

    with b.map(pyopencl.map_flags.WRITE_INVALIDATE_REGION, wait_for=None) as mapped:
        mapped[0] = v
    b.read()
    assert b[0] == v


def test_buffer_read_map():
    b = codecad.cl_util.Buffer(pyopencl.cltypes.uchar8,
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
    b = codecad.cl_util.Buffer(item_type,
                               (2, 2, 2),
                               pyopencl.mem_flags.WRITE_ONLY)

    with b.map(pyopencl.map_flags.WRITE_INVALIDATE_REGION, wait_for=None) as mapped:
        assert mapped.nbytes == b.size
        assert mapped.dtype == item_type


def test_assert_pass():
    assertBuffer = codecad.cl_util.AssertBuffer()

    # Checking before first use should always pass
    assertBuffer.check()

    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(100), assertBuffer)

    # The kernel shouldn't have logged an assert
    assertBuffer.check(wait_for=[ev])


def test_assert_fail():
    assertBuffer = codecad.cl_util.AssertBuffer()
    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(5), assertBuffer)

    with pytest.raises(codecad.cl_util.cl_assert.OpenClAssertionError) as exc_info:
        assertBuffer.check(wait_for=[ev])

    assert exc_info.value.count == 1
    assert os.path.basename(exc_info.value.filename) == "test_clutil.cl"
    assert exc_info.value.global_id == [5, 5, 0, 0]


def test_assert_chaining_pass():
    assertBuffer = codecad.cl_util.AssertBuffer()
    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(101), assertBuffer)
    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(102), assertBuffer,
                                                        wait_for=[ev])

    # The kernel shouldn't have logged an assert
    assertBuffer.check(wait_for=[ev])


def test_assert_chaining_multiple_fail():
    assertBuffer = codecad.cl_util.AssertBuffer()
    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(1), assertBuffer)
    ev = codecad.cl_util.opencl_manager.k.assert_tester((10, 10), None,
                                                        numpy.uint32(2), assertBuffer,
                                                        wait_for=[ev])

    with pytest.raises(codecad.cl_util.cl_assert.OpenClAssertionError) as exc_info:
        assertBuffer.check(wait_for=[ev])

    assert exc_info.value.count == 2
    assert os.path.basename(exc_info.value.filename) == "test_clutil.cl"
    assert exc_info.value.global_id == [1, 1, 0, 0]


def test_interleave():
    job_log = []
    job_counter = 0

    class MockEvent:
        def __init__(self, job):
            self.job = job

        def wait(self):
            nonlocal job_log
            job_log.append((self.job, "wait"))

    def job_func(job):
        nonlocal job_log, job_counter
        job_log.append((job, 1))

        yield MockEvent(job)

        job_log.append((job, 2))

        if job < 10:
            return [2 * job, 2 * job + 1]

    codecad.cl_util.interleave2(job_func, [2, 3])

    checked = set()
    working_jobs = set()
    tics_working = 0
    max_working_jobs = 0

    for job, step in job_log:
        # print(job, step, working_jobs)

        if len(working_jobs):
            tics_working += 1
        max_working_jobs = max(max_working_jobs, len(working_jobs))

        if step == 1:
            working_jobs.add(job)
        elif step == "wait":
            working_jobs.remove(job)
        elif step == 2:
            assert (job, "wait") in checked, "The event must be waited on before step 2"

        if job > 3:
            assert (job // 2, 2) in checked, "Parent task must be finished before running a child"

        checked.add((job, step))

    assert (18, 2) in checked
    assert (19, 2) in checked

    assert max_working_jobs == 2
    assert tics_working >= len(checked) - 4
    # 1 tick inactive during startup, 1 tick during finshing, 2 ticks because of
    # a stall when the jobs start running out
