import struct
import warnings

import pyopencl
import numpy

from . import codegen
from .. import util

ctx = pyopencl.create_some_context()

for dev in ctx.devices:
    print("Device", dev.name)

queue = pyopencl.CommandQueue(ctx)

with warnings.catch_warnings(), \
     util.status_block("compiling"):
    # We compile with Werror, so there is no need for python warnings here
    warnings.simplefilter("ignore")
    program = pyopencl.Program(ctx, codegen.collect_program()).build(
        options=["-Werror",
                 "-cl-single-precision-constant",
                 "-cl-denorms-are-zero",
                 "-cl-no-signed-zeros",
                 "-cl-fast-relaxed-math"])
