__kernel void matplotlib_slice(__constant float* scene,
                               float4 boxCorner, float boxStep,
                               __global float* output)
{
    uint3 coords = (uint3)(get_global_id(0),
                           get_global_id(1),
                           0);
    uint w = get_global_size(0);
    uint h = get_global_size(1);

    float3 point = as_float3(boxCorner) + boxStep * convert_float3(coords);
    float4 value = evaluate(scene, point);

    size_t index = (coords.x + coords.y * w) * 3;

    output[index + 0] = value.w;
    output[index + 1] = value.x;
    output[index + 2] = value.y;
}

// vim: filetype=c
