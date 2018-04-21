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

__kernel void sum_tester(__global float* inBuffer,
                         __global float* outBuffer)
{
    __local float buffer[64];

    float sum = sum_helper_float(inBuffer[get_global_id(0)], buffer);
    if (get_local_id(0) == 0)
        outBuffer[get_group_id(0)] = sum;
}

// vim: filetype=c
