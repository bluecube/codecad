__kernel void grid_eval(__constant float* scene,
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

    output[coords.z + (coords.x + coords.y * size.x) * size.z] = evaluate(scene, point).w;
}

// vim: filetype=c
