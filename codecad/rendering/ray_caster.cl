// Enhanced Sphere Tracing by Keinert et al. is a gold mine!

// Some constants that don't depend on the geometry
#define OVER_RELAXATION_CONSTANT 0.5f
#define PRIMARY_RAY_MAX_STEPS 1000
#define LIGHT_RAY_MAX_STEPS 100
#define LIGHT_MIN_INFLUENCE (1.0f/512.0f)

static float light_contribution(__constant float* restrict scene,
                                float3 point, float3 normal, float3 toLight,
                                float minDistance, float maxDistance)
{
    float surfaceToLightDotProduct = dot(normal, toLight);

    if (surfaceToLightDotProduct <= 0)
        return 0;

    float lightVisibility = 1;
    float distance = minDistance;
    for (uint i = 0; i < LIGHT_RAY_MAX_STEPS; ++i)
    {
        float evalResult = evaluate(scene, point + distance * toLight).w;
        lightVisibility = min(lightVisibility, evalResult / distance);

        if (lightVisibility < LIGHT_MIN_INFLUENCE)
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
                         float pixelTolerance, float minDistance, float maxDistance,
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
    float fallbackDistance = minDistance;
    float4 evalResult;
    bool hit;
    uint stepCount = 0;
    for (stepCount = 0; stepCount < PRIMARY_RAY_MAX_STEPS; ++stepCount)
    {
        evalResult = evaluate(scene, origin.xyz + distance * direction);

        if (distance - fallbackDistance > evalResult.w)
        {
            // over relaxation was too optimistic, we need to throw away this
            // result and go back to fallback.
            distance = fallbackDistance;
            continue;
        }

        hit = evalResult.w < pixelTolerance * distance;

        if (hit)
        {
            // Approximate the object by a plane and move all the way to it
            // (but never backwards).
            distance += evalResult.w * clamp(1.0f / dot(evalResult.xyz, -direction),
                                             0.0f, 2.0f);
            break;
        }

        fallbackDistance = distance + evalResult.w;

        // Control over-relaxation by the distance field direction.
        // This gives about 20% speedup over `overRelaxation = 0`.
        // Setting `overRelaxation` to any constant > 0 actually makes it slower on
        // on complex models.
        // Surprisingly this linear model works better than actually approximating
        // the surface with a plane and using distance to the plane to drive
        // the step length
        float overRelaxation = OVER_RELAXATION_CONSTANT * \
                               min(1.0f, 1.0f + dot(direction, evalResult.xyz));
        distance = distance + evalResult.w * (1 + overRelaxation);

        if (distance > maxDistance)
            break;
    }

    float4 color;

    if (renderOptions & RENDER_OPTIONS_FALSE_COLOR)
    {
        float residual;
        if (hit)
            residual = fabs(evaluate(scene, origin.xyz + direction * distance).w);
        else
            residual = 0;

        color = (float4)(stepCount, 1000 * residual, 0, 0);
    }
    else if (hit)
    {
        float3 point = origin.xyz + direction * distance;
        float3 normal = evalResult.xyz;

        float lightness = ambient;
        lightness += light_contribution(scene, point, normal, -light.xyz, 1e-3, maxDistance);

        color = lightness * surfaceColor;
    }
    else
        color = backgroundColor;

    output[index + 0] = clamp(color.x, 0.0f, 255.0f);
    output[index + 1] = clamp(color.y, 0.0f, 255.0f);
    output[index + 2] = clamp(color.z, 0.0f, 255.0f);
}

// vim: filetype=c
