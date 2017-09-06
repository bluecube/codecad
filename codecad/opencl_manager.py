import inspect
import itertools
import functools
import warnings
import os.path

import pyopencl

from . import util

_default_compiler_options = ["-Werror",
                             "-cl-single-precision-constant",
                             "-cl-denorms-are-zero",
                             "-cl-no-signed-zeros",
                             "-cl-fast-relaxed-math"]

class CompileUnit:
    def __init__(self, options = _default_compiler_options):
        self.pieces = []
        self.options = _default_compiler_options

    # Workaround, see OpenCLManager._build_program()
    #def compile(self, context, extra_headers):
    #    program = pyopencl.Program(context, self.code(extra_headers))
    #    return program.compile(self.options)

    def code(self, extra_headers = []):
        return "\n".join(itertools.chain(extra_headers, self.pieces))

    def append_file(self, filename):
        frame = inspect.stack()[1]

        abs_filename = os.path.join(os.path.dirname(frame[1]), filename)

        self.pieces.append('#line 1 "{}"'.format(abs_filename)) # TODO: Escape filename
        with open(abs_filename, "r") as fp:
            self.pieces.append(fp.read())

    def append(self, code):
        frame = inspect.currentframe().f_back
        frameinfo = inspect.getframeinfo(frame, context=0)
        lines = code.count("\n")
        line = frameinfo.lineno - lines

        while code.startswith("\n"):
            code = code[1:]
            line += 1

        self.pieces.append('#line {} "{}"'.format(line, frameinfo.filename)) # TODO: Escape filename
        self.pieces.append(code)

class _Kernels:
    def __init__(self, manager):
        self.manager = manager

    def __getattr__(self, name):
        kernel = getattr(self.manager.get_program(), name)

        @functools.wraps(kernel)
        def ret(*args, **kwargs):
            return kernel(self.manager.queue, *args, **kwargs)

        return ret

class OpenCLManager:
    def __init__(self):
        self.context = pyopencl.create_some_context()

        for dev in self.context.devices:
            print("Device", dev.name)

        self.queue = pyopencl.CommandQueue(self.context,
                                           properties=pyopencl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)

        self._compile_units = []
        self.common_header = CompileUnit()

        self._program = None

        self.k = _Kernels(self)

    def add_compile_unit(self, *args, **kwargs):
        ret = CompileUnit(*args, **kwargs)
        self._compile_units.append(ret)
        return ret

    def _build_program(self):
        code = "\n".join(itertools.chain(self.common_header.pieces,
                                       itertools.chain.from_iterable(cu.pieces for cu in self._compile_units)))
        program = pyopencl.Program(self.context, code)
        with warnings.catch_warnings():
            # Intel OpenCL generates non empty output and that causes warnings
            # from pyopencl. We just silence them.
            warnings.simplefilter("ignore")
            return program.build(options=_default_compiler_options)

        # Working around bug in pyopencl.Program.compile in pyopencl 2017.1.1
        #compiled_units = [unit.compile(self.context, self.common_header.pieces)
        #                  for unit in self._compile_units]
        # return pyopencl.link_program(self.context, compiled_units)

    def get_program(self):
        if self._program is None:
            with util.status_block("compiling"):
                self._program = self._build_program()
        return self._program

instance = OpenCLManager()

# Collecting files that don't belong anywhere else
instance.common_header.append_file("indexing.h")
