import itertools
import functools
import warnings
import re

import pyopencl

from . import codegen

from .. import util

DEFAULT_COMPILER_OPTIONS = [
    "-Werror",
    "-cl-single-precision-constant",
    "-cl-denorms-are-zero",
    "-cl-no-signed-zeros",
    "-cl-fast-relaxed-math",
]


class CompileUnit:
    def __init__(self, options=DEFAULT_COMPILER_OPTIONS):
        self.options = options
        self.include_origin = True
        self.clear()

    def clear(self):
        self.pieces = []

    # Workaround, see OpenCLManager._build_program()
    # def compile(self, context, extra_headers):
    #     program = pyopencl.Program(context, self.code(extra_headers))
    #     return program.compile(self.options)

    def code(self, extra_headers=[]):
        return "\n".join(itertools.chain(extra_headers, self.pieces)) + "\n"

    def append_resource(self, resource_name, stacklevel=1):
        """ Append content of a file to this compile unit.
        Unless `filename` is absolute, filename is taken as relative to caller filename's directory."""

        self.pieces.extend(
            codegen.resource_with_origin(
                resource_name,
                stacklevel=stacklevel + 1,
                include_origin=self.include_origin,
            )
        )

    def append_define(self, name, value, stacklevel=1):
        self.append("#define {} {}".format(name, value), stacklevel=stacklevel + 1)

    def append_flags(self, flags, prefix=None):
        """ Given a subclass of flags.Flags, generate defines for its values.
        If prefix is not specified, it is inferred from flags class name. """
        if prefix is None:
            prefix = re.sub("(?!^)([A-Z]+)", r"_\1", flags.__name__).upper() + "_"
        elif prefix[-1] != "_":
            prefix = prefix + "_"

        for flag in flags:
            self.append_define(prefix + flag.to_simple_str().upper(), int(flag))

    def append(self, code, stacklevel=1):
        """ Append a string to the compile unit. Ignores leading newlines. """
        self.pieces.extend(
            codegen.string_with_origin(
                code, stacklevel=stacklevel + 1, include_origin=self.include_origin
            )
        )


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

        for dev in self.context.devices:  # noqa
            print("Device", dev.name)

        self.queue = pyopencl.CommandQueue(
            self.context,
            properties=pyopencl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
            | pyopencl.command_queue_properties.PROFILING_ENABLE,
        )

        self._compile_units = []
        self.common_header = CompileUnit()

        self._program = None

        self.k = _Kernels(self)

        self.max_register_count = 0
        # TODO: Having max register count here is a bit of an abstraction leak
        # We should move it somewhere else once node rematerialization is implemented

    def add_compile_unit(self, *args, **kwargs):
        ret = CompileUnit(*args, **kwargs)
        self._compile_units.append(ret)
        return ret

    def _build_program(self):
        code = "\n".join(
            itertools.chain(
                self.common_header.pieces,
                itertools.chain.from_iterable(cu.pieces for cu in self._compile_units),
            )
        )
        program = pyopencl.Program(self.context, code)
        with warnings.catch_warnings():
            # Intel OpenCL generates non empty output and that causes warnings
            # from pyopencl. We just silence them.
            warnings.simplefilter("ignore")
            return program.build(options=DEFAULT_COMPILER_OPTIONS)

        # Working around bug in pyopencl.Program.compile in pyopencl
        # (https://lists.tiker.net/pipermail/pyopencl/2015-September/001986.html)
        # TODO: Fix this in PyOpenCL
        # compiled_units = [unit.compile(self.context, self.common_header.pieces)
        #                   for unit in self._compile_units]
        # return pyopencl.link_program(self.context, compiled_units)

    def get_program(self):
        if self._program is None:
            with util.status_block("compiling"):
                self._program = self._build_program()
        return self._program


instance = OpenCLManager()
