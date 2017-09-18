static float estimate_component(__constant float* scene, float3 point,
                                float3 direction, float epsilon)
{
    return evaluate(scene, point - epsilon * direction).w;
}

__kernel void estimate_direction(__constant float* scene,
                                 float4 boxCorner, float boxStep,
                                 float epsilon,
                                 __global float3* output)
{
    uint3 coords = (uint3)(get_global_id(0),
                           get_global_id(1),
                           get_global_id(2));
    float3 point = as_float3(boxCorner) + boxStep * convert_float3(coords);

    float center = evaluate(scene, point).w;
    float3 vPlus = (float3)(estimate_component(scene, point, (float3)(1, 0, 0), epsilon),
                            estimate_component(scene, point, (float3)(0, 1, 0), epsilon),
                            estimate_component(scene, point, (float3)(0, 0, 1), epsilon));
    float3 vMinus = (float3)(estimate_component(scene, point, (float3)(1, 0, 0), -epsilon),
                             estimate_component(scene, point, (float3)(0, 1, 0), -epsilon),
                             estimate_component(scene, point, (float3)(0, 0, 1), -epsilon));
    output[2 * INDEX3_GG + 0] = center - vPlus;
    output[2 * INDEX3_GG + 1] = vMinus - center;
}

__kernel void grid_eval_twice(__constant float* scene,
                              float4 boxCorner, float boxStep,
                              __global float4* output)
{
    uint3 coords = (uint3)(get_global_id(0),
                           get_global_id(1),
                           get_global_id(2));
    float3 point = as_float3(boxCorner) + boxStep * convert_float3(coords);

    float4 firstResult = evaluate(scene, point);
    output[2 * INDEX3_GG + 0] = firstResult;

    point -= firstResult.xyz * firstResult.w;
    output[2 * INDEX3_GG + 1] = evaluate(scene, point);
}

__kernel void actual_distance_to_surface(float boxStep,
                                         __global float4* input,
                                         __global float* output)
{
    // This kernel is going to be super slow, but we don't mind too much because
    // it's only ever used on small buffers as part of the tests.

    uint3 coords = (uint3)(get_global_id(0),
                           get_global_id(1),
                           get_global_id(2));
    float3 point = boxStep * convert_float3(coords);
    float sign0 = sign(input[2 * INDEX3_GG + 0].w);

    float nearestSquared = MAXFLOAT;
    for (uint z = 0; z < get_global_size(2); ++z)
        for (uint y = 0; y < get_global_size(1); ++y)
            for (uint x = 0; x < get_global_size(0); ++x)
            {
                if (sign(input[2 * INDEX3_G(x, y, z) + 0].w) == sign0)
                    continue;

                float3 toPoint = point - boxStep * (float3)(x, y, z);
                float distanceSquared = dot(toPoint, toPoint);
                if (distanceSquared < nearestSquared)
                    nearestSquared = distanceSquared;
            }

    output[INDEX3_GG] = sqrt(nearestSquared);
}

// vim: filetype=c
