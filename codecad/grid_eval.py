from .cl_util import opencl_manager

opencl_manager.add_compile_unit().append_file("grid_eval.cl")
