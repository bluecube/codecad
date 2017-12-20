import numpy
import pyopencl

from .opencl_manager import instance as opencl_manager
from . import cl_buffer


class OpenClAssertionError(Exception):
    def __init__(self, count, filename, line, global_id, expression):
        self.count = count
        self.filename = filename
        self.line = line
        self.global_id = global_id
        self.expression = expression

    def __str__(self):
        string = "{}:{}: [{}, {}, {}, {}]: Assertion".format(self.filename, self.line,
                                                             self.global_id[0],
                                                             self.global_id[1],
                                                             self.global_id[2],
                                                             self.global_id[3])
        if self.expression is not None:
            string += " `{}'".format(self.expression)

        string += " failed."

        if self.count > 1:
            string += " (+{} others)".format(self.count - 1)

        return string


class AssertBuffer(cl_buffer.Buffer):
    """ Buffer subclass for processing asserts. """
    ASSERT_BUFFER_SIZE = 1024

    def __init__(self, queue=None):
        dtype_list = [("assert_count", numpy.uint32),
                      ("global_id", numpy.uint32, 4),
                      ("line", numpy.uint32)]
        dtype_list_size = numpy.dtype(dtype_list).itemsize
        dtype_list.append(("text", "a{}".format(self.ASSERT_BUFFER_SIZE - dtype_list_size)))

        super().__init__(dtype_list, 1, pyopencl.mem_flags.READ_WRITE, queue=queue)
        assert self.size == self.ASSERT_BUFFER_SIZE

        self.reset()

    def reset(self):
        """ Reset the buffer for next use.
        Returns the event. """
        return self.enqueue_write(numpy.zeros(1, dtype=numpy.uint32))

    def check(self, wait_for=None):
        """ Check if there were any assertion failures reported in this buffer and
        possibly raise an exception. """
        # TODO: This operation is blocking
        with self.map(pyopencl.map_flags.READ, wait_for=wait_for) as mapped:
            if mapped["assert_count"][0] == 0:
                return

            split = mapped["text"][0].split(b"\0", 2)
            if len(split) == 1:
                # Filename is not complete
                filename = split[0].decode("ascii") + "..."
                expr = None
            elif len(split) == 2:
                # expression is not complete
                filename = split[0].decode("ascii")
                expr = split[1].decode("ascii") + "..."
            else:
                # We have room to spare
                filename = split[0].decode("ascii")
                expr = split[1].decode("ascii")

            raise OpenClAssertionError(mapped["assert_count"][0],
                                       filename,
                                       mapped["line"][0],
                                       list(mapped["global_id"][0]),
                                       expr)


opencl_manager.common_header.append("#define ASSERT_BUFFER_SIZE {}".format(AssertBuffer.ASSERT_BUFFER_SIZE))
if __debug__:
    opencl_manager.common_header.append("#define DEBUG 1")
opencl_manager.common_header.append_file("assert.h")

opencl_manager.add_compile_unit().append_file("assert.cl")
