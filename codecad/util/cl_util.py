import contextlib

import numpy
import pyopencl


class Buffer:
    """ A simple wrapper around pyopencl buffer that handles the corresponding
    numpy arrays and transfers. """

    @staticmethod
    def dual_dtype(scalar):
        return numpy.dtype([(name, scalar) for name in 'xy'])

    @staticmethod
    def quad_dtype(scalar):
        return numpy.dtype([(name, scalar) for name in 'xyzw'])

    def __init__(self, queue, dtype, size, mem_flags):
        self.queue = queue
        self.dtype = numpy.dtype(dtype)
        self.nitems = 1

        try:
            for s in size:
                self.nitems *= s
            self.size = size
        except TypeError:
            self.nitems = size
            self.size = (size,)

        self.nbytes = self.nitems * self.dtype.itemsize
        self.buffer = pyopencl.Buffer(self.queue.context,
                                      mem_flags | pyopencl.mem_flags.ALLOC_HOST_PTR,
                                      self.nbytes)
        self.array = None

    def create_host_side_array(self):
        """ Create numpy array of appropriate size and dtype, assign it to buffer's
        internal `array` field and return it """
        self.array = numpy.empty(self.nitems, dtype=self.dtype)

    def _process_array(self, array):
        if array is None:
            if self.array is None:
                self.create_host_side_array()
            return self.array
        else:
            return array

    def read(self, out=None, wait_for=None):
        """ Read contents of the buffer either into `self.array`, or to `out`.
        `wait_for` can be either None or list of pyopencl.Event. """

        array = self._process_array(out)
        if array.nbytes < self.nbytes:
            raise RuntimeError("Not enough space to store contents of the buffer")

        pyopencl.enqueue_copy(self.queue, array, self.buffer,
                              wait_for=wait_for, is_blocking=True)

        return array

    def enqueue_write(self, a=None, wait_for=None):
        """ Write contents of the buffer either from `self.array`, or from `a`.
        Returns event of the copy operation.
        `wait_for` can be either None or list of opencl.Event. """

        array = self._process_array(a)
        if array.nbytes > self.nbytes:
            raise RuntimeError("Not enough space to store contents in the buffer")

        return pyopencl.enqueue_copy(self.queue, self.buffer, array,
                                     wait_for=wait_for, is_blocking=False)

    @contextlib.contextmanager
    def map(self, map_flags, wait_for=None):
        """ Context manager that maps the buffer data as a numpy array.
        `wait_for` can be either None or list of opencl.Event. """

        array, event = pyopencl.enqueue_map_buffer(self.queue, self.buffer, map_flags,
                                                   0, (self.nitems,), self.dtype,
                                                   wait_for=wait_for, is_blocking=True)
        with array.base:
            yield array.view(self.dtype)

    def _index(self, key):
        try:
            iter(key)
        except TypeError:
            key = (key,)

        if len(key) != len(self.size):
            raise ValueError("Wrong number of indices for buffer")

        index = 0
        for k, s in zip(reversed(key), reversed(self.size)):
            index = index * s + k
        return index

    def __getitem__(self, key):
        return self.array[self._index(key)]

    def __setitem__(self, key, value):
        self.array[self._index(key)] = value

    def __len__(self):
        return self.nitems


class _InterleavingHelperWrapper:
    def __init__(self, helper):
        self._helper = helper
        self._event = None

    def enqueue(self, *args, **kwargs):
        self._event = self._helper.enqueue(*args, **kwargs)

    def process_result(self):
        return self._helper.process_result(self._event)


def interleave(initial_jobs, helper1, helper2):
    assert len(initial_jobs), "There must be at least one job to start"

    stack = list(initial_jobs)

    wrapped_helper1 = _InterleavingHelperWrapper(helper1)
    wrapped_helper2 = _InterleavingHelperWrapper(helper2)

    wrapped_helper1.enqueue(*stack.pop())

    while True:
        stack_was_empty = not len(stack)

        if not stack_was_empty:
            wrapped_helper2.enqueue(*stack.pop())

        stack.extend(wrapped_helper1.process_result())

        if stack_was_empty:
            if len(stack):
                wrapped_helper2.enqueue(*stack.pop())
            else:
                break

        # Swap them for the next iteration
        wrapped_helper1, wrapped_helper2 = wrapped_helper2, wrapped_helper1


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


class AssertBuffer(Buffer):
    """ Buffer subclass for processing asserts. """
    ASSERT_BUFFER_SIZE = 1024

    def __init__(self, queue):
        dtype_list = [("assert_count", numpy.uint32),
                      ("global_id", numpy.uint32, 4),
                      ("line", numpy.uint32)]
        dtype_list_size = numpy.dtype(dtype_list).itemsize
        dtype_list.append(("text", "a{}".format(self.ASSERT_BUFFER_SIZE - dtype_list_size)))

        super().__init__(queue, dtype_list, 1, pyopencl.mem_flags.READ_WRITE)
        assert self.nbytes == self.ASSERT_BUFFER_SIZE

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
                                       mapped["global_id"][0],
                                       expr)
