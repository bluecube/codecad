from . opencl_manager import instance as opencl_manager


def generate_sum_helper(h_file, c_file, type_name,
                        op="a + b", name=None):
    """ Generates header and implementation of a sum helper function.

    h_file is where the declaration goes, c_file where the definition goes.
    The resulting function is called sum_helper_{name} and sums items of type
    type_name. If name is None, it gets set from type_name.
    op must be a valid C expression that reduces two items `a` and `b` of type
    type_name into a value of type type_name.

    See inside of this method (or the generated code) for sum_helper docs. """

    if name is None:
        name = type_name

    h_file.append("""
/** Sum get_local_size(0) values of type {type_name} in parallel with operation `{op}`.
Input values are passed into the function as the first parameter.
Sum of the group is returned if get_local_id(0) == 0, otherwise this function
returns garbage.

Local buffer will be used for temporary values and must have at least
get_local_size(0) / 2 items available.

Barriers are used inside, so all threads in a workgroup must go through this
function if any one goes through it.

The algorithm works by recursively summing pairs, so it should work fairly well
with floating point precision. */""".format(**locals()))

    for f, end in [(h_file, ";"), (c_file, "\n{")]:
        f.append("""
{type_name} sum_helper_{name}({type_name} value, __local {type_name}* buffer){end}""".format(**locals()))

    c_file.append("""

    size_t n = get_local_size(0);

    while (n > 1)
    {{
        size_t nextN = (n + 1) / 2;

        if (get_local_id(0) < n && get_local_id(0) >= nextN)
            buffer[get_local_id(0) - nextN] = value;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (get_local_id(0) < n / 2)
        {{
            {type_name} a = value;
            {type_name} b = buffer[get_local_id(0)];
            value = ({op});
        }}

        n = nextN;
    }}

    return value;
}}""".format(**locals()))


parallel_sum_c = opencl_manager.add_compile_unit()
parallel_sum_c.append_file("parallel_sum.cl")
opencl_manager.common_header.append_file("parallel_sum.h")

generate_sum_helper(opencl_manager.common_header, parallel_sum_c,
                    type_name="float")
