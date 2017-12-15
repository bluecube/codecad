from . cl_buffer import *
from . cl_assert import *
from . opencl_manager import instance as opencl_manager

# Collecting utility files that don't belong anywhere else:
opencl_manager.common_header.append_file("util.h")
opencl_manager.common_header.append_file("indexing.h")
opencl_manager.add_compile_unit().append_file("util.cl")
