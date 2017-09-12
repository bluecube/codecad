__kernel void indexing_identity(__global uint3* output)
{
    output[INDEX3_GG] = (uint3)(get_global_id(0), get_global_id(1), get_global_id(2));
}
// vim: filetype=c
