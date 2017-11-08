// Enhanced Sphere Tracing by Keinert et al. is a gold mine!

// Some constants that don't depend on the geometry
#define OVER_RELAXATION_CONSTANT 0.5f
#define PRIMARY_RAY_MAX_STEPS 1000
#define LIGHT_RAY_MAX_STEPS 100
#define AMBIENT_OCCLUSION_STEPS 4
#define LIGHT_MIN_INFLUENCE (1.0f/128.0f)

#define LIGHT_DIRECTION normalize((float3)(1, 2, -1))
#define SECONDARY_LIGHT_DIRECTION normalize((float3)(-1, 1, 0))

static float overRelaxationStepLength(float3 direction, float4 evalResult)
{
    // Control over-relaxation by the distance field direction.
    // This gives about 20% speedup over `overRelaxation = 0`.
    // Setting `overRelaxation` to any constant > 0 actually makes it slower on
    // on complex models.
    // Surprisingly this linear model works better than actually approximating
    // the surface with a plane and using distance to the plane to drive
    // the step length
    float overRelaxation = OVER_RELAXATION_CONSTANT * \
                           min(1.0f, 1.0f + dot(direction, evalResult.xyz));
    return evalResult.w * (1 + overRelaxation);
}

static float2 light_contribution_no_trace(float3 point, float3 normal, float3 toLight, float3 toCamera)
{
    float3 halfwayVector = normalize(toLight + toCamera);

    float diffuseIntensity = max(0.0f, dot(normal, toLight));
    float specularIntensity = max(0.0f, dot(normal, halfwayVector));
    specularIntensity *= specularIntensity;
    specularIntensity *= specularIntensity;
    specularIntensity *= specularIntensity;

    return (float2)(diffuseIntensity, specularIntensity);
}

static float2 light_contribution(__constant float* restrict scene,
                                float3 point, float3 normal, float3 toLight, float3 toCamera,
                                float minDistance, float maxDistance,
                                uint renderOptions)
{
    float2 resultBeforeTrace = light_contribution_no_trace(point, normal, toLight, toCamera);

    if (resultBeforeTrace.x <= 0 && resultBeforeTrace.y <= 0)
        return 0;

    float threshold = LIGHT_MIN_INFLUENCE / max(resultBeforeTrace.x, resultBeforeTrace.y);

    float lightVisibility = 1;

    float distance = minDistance;
    float fallbackDistance = distance;
    uint stepCount;
    for (stepCount = 0; stepCount < LIGHT_RAY_MAX_STEPS; ++stepCount)
    {
        float4 evalResult = evaluate(scene, point + distance * toLight);

        lightVisibility = min(lightVisibility, evalResult.w / distance);

        if (lightVisibility < threshold)
            break;

        if (distance - fallbackDistance > evalResult.w)
        {
            // over relaxation was too optimistic, we need to throw away this
            // result and go back to fallback.

            // This check comes _after_ lightVisibility calculation, because we
            // are approximating minimum of `evalResult.w / distance` over the
            // whole path, so adding an extra option to decrease it might only
            // improve quality of the approximation.
            // Also calculating it eagerly gives an extra option to abort the ray
            // early and that saves about 0.07 evaluation per pixel (on arm.py example).

            distance = fallbackDistance;
            continue;
        }


        fallbackDistance = distance + evalResult.w;
        distance = distance + overRelaxationStepLength(toLight, evalResult);

        if (distance > maxDistance)
            break;
    }

    if (renderOptions & RENDER_OPTIONS_FALSE_COLOR)
        return (float2)(stepCount, 0);
    else
        return lightVisibility * resultBeforeTrace;
}

static float ambient_occlusion(__constant float* restrict scene,
                               float3 point, float3 normal, float distanceStep)
{
    float3 bentNormal = normal;
    float occlusion = 0.0;
    float scale = 1.0;
    float distance = distanceStep;
    for (uint i = 0; i < AMBIENT_OCCLUSION_STEPS; ++i)
    {
        float4 evalResult = evaluate(scene, point + distance * normal);
        occlusion += scale * (distance - evalResult.w);
        scale /= 2;
        bentNormal += (float3)(0, 0, 0.25);
        distance += distanceStep;
    }
    return clamp(1 - occlusion * 0.5 / (1 - scale), 0.0f, 1.0f);
}

static float3 map_color(float ambient, float diffuse, float specular)
{
    // hue is fixed to 78°
    float saturation = 0.75 * smoothstep(0.0f, 0.25f, diffuse);
    float value = 0.1 + 0.8 * mix(diffuse, ambient, 0.3f);

    float chroma = value * saturation;
    float X = chroma * 0.7; // Corresponds to hue of 78°
    float m = value - chroma;

    float3 color = 255 * ((float3)(X, chroma, 0) + m);

    return color + specular * 128;
}

__kernel void ray_caster(__constant float* restrict scene,
                         float4 origin, float4 forward, float4 up, float4 right,
                         float pixelTolerance, float boxRadius,
                         float minDistance, float maxDistance, float floorZ,
                         uint renderOptions,
                         __global uchar* restrict output)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t w = get_global_size(0);
    size_t h = get_global_size(1);

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
        distance = distance + overRelaxationStepLength(direction, evalResult);

        if (distance > maxDistance)
        {
            distance = INFINITY;
            break;
        }
    }

    float3 color;
    float localEpsilon = max(1e-4f, 2 * fabs(evalResult.w));

    if (renderOptions & RENDER_OPTIONS_FALSE_COLOR)
    {
        float3 point = origin.xyz + direction * distance;
        float3 normal = evalResult.xyz;

        float residual;
        if (hit)
            residual = fabs(evaluate(scene, point).w);
        else
            residual = 0;

        float steps = stepCount;
        steps += light_contribution(scene, point, normal, -LIGHT_DIRECTION, -direction,
                                    localEpsilon, maxDistance,
                                    renderOptions).s0;
        steps += AMBIENT_OCCLUSION_STEPS;
        color = (float3)(steps, 1000 * residual, 0);
    }
    else if (hit)
    {
        float3 point = origin.xyz + direction * distance;
        float3 normal = evalResult.xyz;

        float ambient = ambient_occlusion(scene, point, normal, boxRadius / 100);
        float2 light = 0.8 * light_contribution(scene, point, normal, -LIGHT_DIRECTION, -direction,
                                                localEpsilon, maxDistance,
                                                renderOptions);
        light += 0.2 * light_contribution_no_trace(point, normal, -SECONDARY_LIGHT_DIRECTION, -direction);
        color = map_color(ambient, light.x, light.y);
    }
    else
        color = (float3)(230, 230, 241);

    float floorDistance = (floorZ - origin.z) / direction.z;
    if (floorDistance > 0 && floorDistance < distance)
    {
        float3 floorPoint = origin.xyz + direction * floorDistance;
        float floorDistance = evaluate(scene, floorPoint).w;
        float shadow = clamp(2 * floorDistance / boxRadius, 0.0f, 1.0f);
        shadow = 1 - shadow;
        shadow *= shadow;
        shadow = 1 - shadow;
        color = mix((float3)(0, 0, 0), color, 0.4 + 0.6 * shadow);
    }

    size_t index = INDEX2_GG * 3;
    output[index + 0] = clamp(color.x, 0.0f, 255.0f);
    output[index + 1] = clamp(color.y, 0.0f, 255.0f);
    output[index + 2] = clamp(color.z, 0.0f, 255.0f);
}

// vim: filetype=c
