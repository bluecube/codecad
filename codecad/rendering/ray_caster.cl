// Cast a ray through scene.
// Returns estimated tan(apex angle of unobstructed cone) (0 if the ray hit the geometry,
// 1 if it went directly away from the scene, INFINITY if there never was any scene).
// Tracing starts at minDistance and ends at maxDistance.
// nearest outputs the evaluation result at a point where the cone got maximally
// obstructed, nearestDistance returns distance along the ray of this event
static float cast_ray(__constant float* scene,
                      float3 origin, float3 direction,
                      float epsilon, uint maxSteps, float minDistance, float maxDistance,
                      float4* nearestResult, float* nearestDistance)
{
    float tanApexAngle = 1; //INFINITY;
    float distance = minDistance;
    nearestResult->w = INFINITY;
    for (size_t i = 0; i < maxSteps; ++i)
    {
        float3 point = origin + distance * direction;
        float4 evalResult = evaluate(scene, point);

        tanApexAngle = min(tanApexAngle, evalResult.w / distance);

        if (evalResult.w < nearestResult->w)
        {
            *nearestResult = evalResult;
            *nearestDistance = distance;
        }

        if (evalResult.w < epsilon)
            return 0;
        else
        {
            distance += evalResult.w;

            if (distance > maxDistance)
                break;
        }
    }

    return tanApexAngle;
}

static float light_contribution(__constant float* scene,
                                float3 point, float3 normal, float3 toLight,
                                float epsilon, uint maxSteps, float maxDistance)
{
    float surfaceToLightDotProduct = dot(normal, toLight);

    if (surfaceToLightDotProduct <= 0)
        return 0;

    float4 hitResult;
    float hitDistance;
    float lightVisibility = cast_ray(scene, point, toLight,
                                     epsilon, maxSteps, epsilon, maxDistance,
                                     &hitResult, &hitDistance);

    return lightVisibility * surfaceToLightDotProduct;
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

    float4 hitResult;
    float hitDistance;
    float4 color;

    if (cast_ray(scene, origin.xyz, direction,
                 epsilon, maxSteps, minDistance, maxDistance,
                 &hitResult, &hitDistance) == 0)
    {
        float lightness = ambient;
        float3 normal = hitResult.xyz;

        float3 point = origin.xyz + direction * (hitDistance - 2 * epsilon);// + normal * hitResult.w;

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
