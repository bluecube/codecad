// Cast a ray through scene.
// Returns estimated non-occlusion (0 if the ray hit the geometry,
// 1 if it went directly away from the scene, INFINITY if there never was any scene;
// this is tan(angle of a maximal cone centered at the ray that doesn't intesect any geometry)
bool cast_ray(__constant float* scene,
              float3 origin, float3 direction,
              float epsilon, uint maxSteps, float minDistance, float maxDistance,
              float4* evalResult, float* distance)
{
    //float occlusion = INFINITY;
    *distance = minDistance;
    for (size_t i = 0; i < maxSteps; ++i)
    {
        if (*distance > maxDistance)
            break;
        float3 point = origin + (*distance) * direction;
        *evalResult = evaluate(scene, point);

        if (evalResult->w < epsilon)
            return true;
            //return evalResult;
        else
            *distance += evalResult->w;
            //occlusion = min(occlusion, evalResult.w / distance);
    }

    return false;
    //return occlusion;
}

__kernel void ray_caster(__constant float* scene,
                         float4 origin, float4 forward, float4 up, float4 right,
                         float4 surfaceColor, float4 backgroundColor,
                         float4 light, float ambient,
                         float epsilon, uint maxSteps, float minDistance, float maxDistance,
                         __global uchar* output)
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

    float4 lastResult;
    float distance;
    float4 color;

    if (cast_ray(scene, as_float3(origin), as_float3(direction),
                 epsilon, maxSteps, minDistance, maxDistance,
                 &lastResult, &distance))
    {
        float lightness = ambient;
        float3 gradient = as_float3(lastResult);
            // Theoretically the shapes should output unit length gradient in the first
            // place, but we might have approximate ones that don't.
            // TODO: Normalize gradient, if it causes problems
        lightness += fmax(-dot(gradient, as_float3(light)), 0);
        color = lightness * surfaceColor;
    }
    else
        color = backgroundColor;

    output[index + 0] = clamp(color.x, 0.0f, 255.0f);
    output[index + 1] = clamp(color.y, 0.0f, 255.0f);
    output[index + 2] = clamp(color.z, 0.0f, 255.0f);
}

// vim: filetype=c
