__kernel void one_item_double(__global ulong* value)
{
    *value = *value * 2;
}

__kernel void indexing_identity(__global uint3* output)
{
    output[INDEX3_GG] = (uint3)(get_global_id(0), get_global_id(1), get_global_id(2));
}

__kernel void assert_tester(uint failingCoord,
                            __global AssertBuffer* restrict assertBuffer)
{
    assert(assertBuffer,
           get_global_id(0) != failingCoord || get_global_id(1) != failingCoord);
}

// vim: filetype=c
