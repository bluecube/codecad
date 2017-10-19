static float light_contribution(__constant float* restrict scene,
                                float3 point, float3 normal, float3 toLight,
                                float epsilon, uint maxSteps, float maxDistance)
{
    float surfaceToLightDotProduct = dot(normal, toLight);

    if (surfaceToLightDotProduct <= 0)
        return 0;

    float lightVisibility = 1;
    float distance = epsilon;
    for (uint i = 0; i < maxSteps; ++i)
    {
        float evalResult = evaluate(scene, point + distance * toLight).w;
        lightVisibility = min(lightVisibility, evalResult / distance);

        if (evalResult < epsilon)
            break;

        distance += evalResult;

        if (distance > maxDistance)
            break;
    }

    return lightVisibility * surfaceToLightDotProduct;
}

__kernel void ray_caster(__constant float* restrict scene,
                         float4 origin, float4 forward, float4 up, float4 right,
                         float4 surfaceColor, float4 backgroundColor,
                         float4 light, float ambient,
                         float epsilon, uint maxSteps, float minDistance, float maxDistance,
                         uint renderOptions,
                         __global uchar* restrict output)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t w = get_global_size(0);
    size_t h = get_global_size(1);
    size_t index = (x + y * w) * 3;

    float filmx = x - (w - 1) / 2.0f;
    float filmy = y - (h - 1) / 2.0f;

    float3 direction = normalize(as_float3(forward) +
                                 as_float3(right) * filmx -
                                 as_float3(up) * filmy);

    float distance = minDistance;
    float4 evalResult;
    bool hit;
    uint stepCount = 0;
    for (stepCount = 0; stepCount < maxSteps; ++stepCount)
    {
        evalResult = evaluate(scene, origin.xyz + distance * direction);
        hit = evalResult.w < epsilon;

        if (hit)
            break;

        distance += evalResult.w;

        if (distance > maxDistance)
            break;
    }

    float4 color;

    if (renderOptions & RENDER_OPTIONS_FALSE_COLOR)
    {
        color = (float4)(stepCount, 0, 0, 0);
    }
    else if (hit)
    {
        float lightness = ambient;
        float3 normal = evalResult.xyz;

        float3 point = origin.xyz + direction * (distance - 2 * epsilon);// + normal * hitResult.w;

        lightness += light_contribution(scene, point, normal, -light.xyz,
                                        epsilon, maxSteps, maxDistance);

        color = lightness * surfaceColor;
    }
    else
        color = backgroundColor;

    output[index + 0] = clamp(color.x, 0.0f, 255.0f);
    output[index + 1] = clamp(color.y, 0.0f, 255.0f);
    output[index + 2] = clamp(color.z, 0.0f, 255.0f);
}

// vim: filetype=c
