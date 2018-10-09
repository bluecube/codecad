from .cl_buffer import *
from .cl_assert import *
from .codegen import *
from .opencl_manager import instance as opencl_manager
from . import parallel_sum


def interleave2(job_func, initial_jobs):
    """ Runs two instances of job_func in (pseudo)parallel, switching between them every time
    one yields a pyopencl.Event (or anything with `.wait()` method, really).
    The function is resumed after the event is waited for.
    This serves mainly to hide bus transfer times when running multiple individual
    kernels.

    initial_jobs is an iterable of job specifications that need to be processed,
    job_func gets run with each of them.
    job_func may return a list of job specifications that get processed too. """

    job1 = None
    job2 = None
    event1 = None
    event2 = None

    stack = list(initial_jobs)

    class _NoEvent:
        """ Helper class to simplify handling freshly created job functions """

        def wait(self):
            pass

    no_event = _NoEvent()

    while True:
        if job1 is None:
            if len(stack):
                # Job1 finished, but we have enough assignments to run it again
                job1 = job_func(stack.pop())
                event1 = no_event

            elif job2 is None:
                # Nothing more to do
                return

            else:
                # Job1 finished, we don't have any more assignments, but we might
                # have some once job2 gives us a result

                # Swap jobs for next iteration
                job1, job2 = job2, job1
                event1, event2 = event2, event1

                continue

        # At this point we are certain that job1 has something to do

        event1.wait()
        try:
            event1 = job1.send(None)
        except StopIteration as s:
            if s.value is not None:
                stack.extend(s.value)
            job1 = None
        else:
            # Swap jobs for next iteration
            job1, job2 = job2, job1
            event1, event2 = event2, event1


# Collecting utility files that don't belong anywhere else:
opencl_manager.common_header.append_resource("util.h")
opencl_manager.common_header.append_resource("indexing.h")
opencl_manager.add_compile_unit().append_resource("util.cl")
