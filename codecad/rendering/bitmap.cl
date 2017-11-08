__kernel void bitmap(__constant float* restrict scene,
                     float4 origin, float stepSize,
                     __global uchar* restrict output)
{
    float3 point = origin.xyz +
                   stepSize * (float3)(get_global_id(0),
                                       get_global_size(1) - get_global_id(1) - 1,
                                       0);
    float evalResult = evaluate(scene, point).w;
    float3 color = mix((float3)(125, 179, 0), // Inside color
                       (float3)(230, 230, 241), // Background color,
                       step(0, evalResult)); // TODO: Antialiasing

    size_t index = INDEX2_GG * 3;
    output[index + 0] = clamp(color.x, 0.0f, 255.0f);
    output[index + 1] = clamp(color.y, 0.0f, 255.0f);
    output[index + 2] = clamp(color.z, 0.0f, 255.0f);
}
