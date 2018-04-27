/*void indexing_prefix_sum_helper(float uint, __local uint* buffer)
{
    size_t n = get_local_size(0);

    while (n > 1)
    {
        size_t nextN = (n + 1) / 2;

        if (get_local_id(0) < n && get_local_id(0) >= nextN)
            buffer[get_local_id(0) - nextN] = value;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (get_local_id(0) < n / 2)
            value += buffer[get_local_id(0)];

        n = nextN;
    }

    return value;
}*/

// vim: filetype=c
