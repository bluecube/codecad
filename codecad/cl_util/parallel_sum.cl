uint indexing_prefix_sum_helper(uint value, __local uint* buffer)
{

    // TODO: Require only half buffer space, in a similar way to sum_helper.

    value = sum_helper_uint(value, buffer);
    buffer[get_local_id(0)] = value;

    size_t i = 2;
    while (i < get_local_size(0))
    {
        if (get_local_id(0) < i - 1)
        {
            size_t x = i - get_local_id(0) - 1;
            size_t secondIndex = 3 * (((size_t)-1) >> clz(x) >> 1) + 2 - x;
            //printf("%d, %d (of %d): x = %lu, ... = %lu\n", (int)i, (int)get_local_id(0), (int)get_local_size(0), (long)x, (long)secondIndex);
            buffer[i + get_local_id(0)] += buffer[secondIndex];
        }
        i *= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    return buffer[get_local_id(0)];
}

// vim: filetype=c
