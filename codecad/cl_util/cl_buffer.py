import contextlib

import numpy
import pyopencl

from . import opencl_manager


class Buffer(pyopencl.Buffer):
    """ A simple wrapper around pyopencl buffer that handles the corresponding
    numpy arrays and transfers. """

    @staticmethod
    def dual_dtype(scalar):
        return numpy.dtype([(name, scalar) for name in 'xy'])

    @staticmethod
    def quad_dtype(scalar):
        return numpy.dtype([(name, scalar) for name in 'xyzw'])

    def __init__(self, dtype, shape, mem_flags, queue=None):
        self.queue = queue if queue is not None else opencl_manager.instance.queue
        self.dtype = numpy.dtype(dtype)
        self.nitems = 1

        try:
            for s in shape:
                self.nitems *= s
            self.shape = shape
        except TypeError:
            self.nitems = shape
            self.shape = (shape,)

        self.array = None

        super().__init__(self.queue.context,
                         mem_flags | pyopencl.mem_flags.ALLOC_HOST_PTR,
                         self.nitems * self.dtype.itemsize)

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
        if array.nbytes < self.size:
            raise RuntimeError("Not enough space to store contents of the buffer")

        pyopencl.enqueue_copy(self.queue, array, self,
                              wait_for=wait_for, is_blocking=True)

        return array

    def enqueue_write(self, a=None, wait_for=None):
        """ Write contents of the buffer either from `self.array`, or from `a`.
        Returns event of the copy operation.
        `wait_for` can be either None or list of opencl.Event. """

        array = self._process_array(a)
        if array.nbytes > self.size:
            raise RuntimeError("Not enough space to store contents in the buffer")

        return pyopencl.enqueue_copy(self.queue, self, array,
                                     wait_for=wait_for, is_blocking=False)

    @contextlib.contextmanager
    def map(self, map_flags, wait_for=None):
        """ Context manager that maps the buffer data as a numpy array.
        `wait_for` can be either None or list of opencl.Event. """

        array, event = pyopencl.enqueue_map_buffer(self.queue, self, map_flags,
                                                   0, (self.nitems,), self.dtype,
                                                   wait_for=wait_for, is_blocking=True)
        with array.base:
            yield array.view(self.dtype)

    def _index(self, key):
        try:
            iter(key)
        except TypeError:
            key = (key,)

        if len(key) != len(self.shape):
            raise ValueError("Wrong number of indices for buffer")

        index = 0
        for k, s in zip(reversed(key), reversed(self.shape)):
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
