import numpy
import pyopencl

class Buffer:
    """ A simple wrapper around pyopencl buffer that handles the corresponding
    numpy arrays and transfers. """

    @staticmethod
    def quad_dtype(scalar):
        return numpy.dtype([(name, scalar) for name in 'xyzw'])

    def __init__(self, queue, dtype, size, mem_flags):
        self.queue = queue
        self.dtype = numpy.dtype(dtype)
        self.size = size
        self.nbytes = size * self.dtype.itemsize
        self.buffer = pyopencl.Buffer(self.queue.context,
                                      mem_flags | pyopencl.mem_flags.ALLOC_HOST_PTR,
                                      self.nbytes)
        self.array = None

    def _process_array(self, array):
        if array is None:
            if self.array is None:
                self.array = numpy.empty((self.size,), dtype=self.dtype)
            return self.array
        else:
            return array

    def read(self, out=None, wait_for=None):
        """ Read contents of the buffer either into `self.array`, or to `out`.
        `wait_for` can be either None or list of opencl.Event. """

        array = self._process_array(out)
        if array.nbytes < self.nbytes:
            raise RuntimeError("Not enough space to store contents of the buffer")

        pyopencl.enqueue_copy(self.queue, array, self.buffer,
                              wait_for=wait_for, is_blocking=True)

        return array

    def enqueue_write(self, a=None, wait_for=None):
        """ Write contents of the buffer either into `self.array`, or to `out`.
        `wait_for` can be either None or list of opencl.Event. """

        array = self._process_array(a)
        if array.nbytes > self.nbytes:
            raise RuntimeError("Not enough space to store contents in the buffer")

        return pyopencl.enqueue_copy(self.queue, self.buffer, array,
                                     wait_for=wait_for, is_blocking=False)

    # This is broken now
    #@contextlib.contextmanager
    #def map(self, map_flags, wait_for=None):
    #    """ Context manager that maps the buffer data as a numpy array.
    #    `wait_for` can be either None or list of opencl.Event. """
    #
    #    print("map", self.dtype);
    #
    #    array, event = pyopencl.enqueue_map_buffer(self.queue, self.buffer, map_flags,
    #                                               0, (self.size,), self.dtype,
    #                                               wait_for=wait_for, is_blocking=True)
    #    with array.base:
    #        print("array", array.dtype);
    #        yield array
