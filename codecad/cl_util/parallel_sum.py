def generate_sum_helper(h_file, c_file, type_name,
                        op="a + b", default_return = "NAN", name=None):
    """ Generates header and implementation of a sum helper function.

    h_file is where the declaration goes, c_file where the definition goes.
    The resulting function is called sum_helper_{name} and sums items of type
    type_name. If name is None, it gets set from type_name.
    op must be a valid C expression that reduces two items `a` and `b` of type
    type_name into a value of type type_name.
    default_return specifies what gets returned from the helper for threads that
    don't have the final sum.

    See inside of this method (or the generated code) for sum_helper docs. """

    if name is None:
        name = type_name

    h_file.append("""
/** Sum get_local_size(0) values of type {type_name} in parallel with operation `{op}`.
Input values are passed into the function as the first parameter.
Sum of the group is returned if get_local_id(0) == 0, otherwise this function
returns {default_return}.

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
    if (get_local_id(0) >= get_local_size(0) / 2)
        buffer[get_local_id(0) - get_local_size(0) / 2] = value;

    for (size_t i = get_local_size(0) / 2; i > 1; i /= 2)
    {{
        barrier(CLK_LOCAL_MEM_FENCE);

        if (get_local_id(0) >= i)
            continue;

        {type_name} a = value;
        {type_name} b = buffer[get_local_id(0)];
        value = ({op});

        if (get_local_id(0) >= i / 2)
            buffer[get_local_id(0) - i / 2] = value;
    }}

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0)
    {{
        {type_name} a = value;
        {type_name} b = buffer[0];
        return ({op});
    }}
    else
        return {default_return};
}}""".format(**locals()))
