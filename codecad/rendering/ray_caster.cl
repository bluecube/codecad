// Cast a ray through scene.
// Returns estimated non-occlusion (0 if the ray hit the geometry,
// 1 if it went directly away from the scene, INFINITY if there never was any scene;
// this is tan(angle of a maximal cone centered at the ray that doesn't intesect any geometry)
bool cast_ray(__constant float* scene,
              float3 origin, float3 direction,
              float epsilon, uint maxSteps,
              float4* evalResult, float* distance)
{
    //float occlusion = INFINITY;
    *distance = 0;
    for (size_t i = 0; i < maxSteps; ++i)
    {
        float3 point = origin + (*distance) * direction;
        *evalResult = evaluate(scene, point);

        if (evalResult->w < epsilon)
            return true;
            //return evalResult;
        else if (evalResult->w == INFINITY)
            break;
        //else
            //occlusion = min(occlusion, evalResult.w / distance);
    }

    *distance = INFINITY;
    return false;
    //return occlusion;
}

__kernel void ray_caster(__constant float* scene,
                         float4 origin, float4 forward, float4 up, float4 right,
                         float4 surfaceColor, float4 backgroundColor,
                         float4 light, float ambient,
                         float epsilon, uint maxSteps,
                         __global uchar* output)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t w = get_global_size(0);
    size_t h = get_global_size(0);
    size_t index = (x + y * w) * 3;

    output[index + 0] = 0;
    output[index + 1] = 127;
    output[index + 2] = (uchar)(255 * (y / (float)h));
    return;

    float filmx = x - (w - 1) / 2.0f;
    float filmy = y - (h - 1) / 2.0f;

    float3 direction = normalize(as_float3(forward) +
                                 as_float3(right) * filmx -
                                 as_float3(up) * filmy);

    float4 lastResult;
    float distance;
    float4 color;

    if (cast_ray(scene, as_float3(origin), as_float3(direction), epsilon, maxSteps, &lastResult, &distance))
    {
        float lightness = ambient;
        float3 gradient = (float3)(lastResult.x, lastResult.y, lastResult.z);
            // Theoretically the shapes should output unit length gradient in the first
            // place, but we might have approximate ones that don't.
            // TODO: Normalize gradient, if it causes problems
        lightness += fmax(dot(gradient, as_float3(light)), 0);
        color = lightness * surfaceColor;
    }
    else
        color = backgroundColor;

    output[index + 0] = color.x;
    output[index + 1] = color.y;
    output[index + 2] = color.z;
}

// vim: filetype=c
