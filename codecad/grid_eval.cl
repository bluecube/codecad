/** Version of grid_eval that follows the weird pymcubes/numpy indexing */
__kernel void grid_eval_pymcubes(__constant float* scene,
                                 float4 boxCorner, float boxStep,
                                 __global float* output)
{
    uint3 coords = (uint3)(get_global_id(0),
                           get_global_id(1),
                           get_global_id(2));
    uint3 size = (uint3)(get_global_size(0),
                         get_global_size(1),
                         get_global_size(2));

    float3 point = as_float3(boxCorner) + boxStep * convert_float3(coords);
    float value = evaluate(scene, point).w;

    // We need to flip Y coordinates, because coordinate system weirdness with mcubes
    // TODO: Get rid of this weirdness
    size_t index = coords.z + (coords.x + (size.y - coords.y - 1) * size.x) * size.z;

    output[index] = value;
}

__kernel void grid_eval(__constant float* scene,
                        float4 boxCorner, float boxStep,
                        __global float4* output)
{
    uint3 coords = (uint3)(get_global_id(0),
                           get_global_id(1),
                           get_global_id(2));

    float3 point = as_float3(boxCorner) + boxStep * convert_float3(coords);

    output[INDEX3_GG] = evaluate(scene, point);
}
// vim: filetype=c
