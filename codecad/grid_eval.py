from . import opencl_manager

opencl_manager.instance.add_compile_unit().append_file("grid_eval.cl")
