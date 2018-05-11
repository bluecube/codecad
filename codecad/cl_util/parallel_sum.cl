/*void indexing_prefix_sum_helper(uint value, __local uint* buffer)
{
    buffer[get_local_id(0)] = value;

    barrier(CLK_LOCAL_MEM_FENCE);

    size_t n = get_local_size(0) / 2;

    while (n > 1)
    {
        if (get_local_id(0) < n)
            buffer[get_local_id(0)] += buffer[get_local_id(0) + n];

        barrier(CLK_LOCAL_MEM_FENCE);

        if (get_local_id(0) < n / 2)
            value += buffer[get_local_id(0)];

        n = nextN;
    }

    return value;
}*/

// vim: filetype=c
