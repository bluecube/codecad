import struct
import warnings

import pyopencl

from . import codegen

ctx = pyopencl.create_some_context()

for dev in ctx.devices:
    print("Device", dev.name)

queue = pyopencl.CommandQueue(ctx)

#mf = cl.mem_flags
#a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
#b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

with warnings.catch_warnings():
    # We compile with Werror, so there is no need for python warnings here
    warnings.simplefilter("ignore")
    program = pyopencl.Program(ctx, codegen.collect_program()).build(
        options=["-Werror",
                 "-cl-single-precision-constant",
                 "-cl-denorms-are-zero",
                 "-cl-no-signed-zeros",
                 "-cl-fast-relaxed-math"])

#res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
#prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

#res_np = np.empty_like(a_np)
#cl.enqueue_copy(queue, res_np, res_g)

# Check on CPU with Numpy:
#print(res_np - (a_np + b_np))
#print(np.linalg.norm(res_np - (a_np + b_np)))
