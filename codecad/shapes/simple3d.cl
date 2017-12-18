float4 sphere_op(float r, float4 coords) {
    float absCoords = length(coords.xyz);
    float dist = absCoords - r;

    if (absCoords == 0)
        return (float4)(1, 0, 0, dist);
    else {
        float4 ret = coords / absCoords;
        ret.w = dist;
        return ret;
    }
}

float4 half_space_op(float4 coords) {
    return (float4)(0, -1, 0, -coords.y);
}

float4 extrusion_op(float halfH, float4 input, float4 coords) {
    return perpendicular_intersection(slab_z(halfH, coords),
                                      input);
}

float4 revolution_to_op(float4 coord) {
    float x = hypot(coord.x, coord.z);
    return (float4)(x, coord.y, 0, 0);
}

float4 revolution_from_op(float4 flat, float4 coords) {
    float length = hypot(coords.x, coords.z);
    float multiplier;
    if (length == 0) {
        coords.x = 1;
        multiplier = flat.x;
    }
    else
        multiplier = flat.x / length;

    return (float4)(coords.x * multiplier, flat.y, coords.z * multiplier, flat.w);
}


float4 twist_revolution_to_op(float r, float twist, float4 coord)
{
    float alpha = fmod(atan2(coord.z, coord.x) + M_PI_F,  M_2PI_F);
    float beta = twist * alpha / M_2PI_F;
    float axisDistance = length(coord.xz);
    float2 inPlaneCoordinates = (float2)(axisDistance - r, coord.y);

    float2 inPlaneCoordinatesRotated = rotated2d(inPlaneCoordinates, -beta);
    return (float4)(inPlaneCoordinatesRotated, 0, 0);
}

float4 twist_revolution_from_op(float minorR, float r, float twist, float4 inPlaneResult, float4 coord)
{
    // This only calculates lower bound to the distance using twist's Lipschitz constant.
    // To avoid infinite Lipschitz constant at the center point and going to
    // infinity in X and Z, we separate a torus that contains all
    // of the shapes geometry and use it as a distance bound for points outside.
    // For the inside we calculate Lipschitz constant radius and use that.
    // We're padding the wrapper torus by 0.05 * r to avoid zero distances on
    // the boundary between the two approximations. The wrapped torus must not
    // intersect the axis.
    // TODO: Once bounding volumes are implemented, get rid of the wrapper here
    // and use an explicit wrapper torus instead
    // TODO: Is it worth passing a pre-computed Lipschitz multiplier?
    float axisDistance = length(coord.xz);
    float2 inPlaneCoordinates = (float2)(axisDistance - r, coord.y);
    float inPlaneCenterDistance = length(inPlaneCoordinates);
    float wrapperDistance = inPlaneCenterDistance - minorR;
    float wrapperPadding = 0.05 * r;
    float finalDistanceBound;
    float2 inPlaneDirection;
    if (axisDistance == 0) //TODO: Epsilon?
        return (float4)(1, 0, 0, r - minorR);
    else if (wrapperDistance > wrapperPadding)
    {
        finalDistanceBound = wrapperDistance;
        inPlaneDirection = inPlaneCoordinates / inPlaneCenterDistance;
        // Not dividing by zero, because we're outside of the wrapping torus.
    }
    else
    {
        float alpha = fmod(atan2(coord.z, coord.x) + M_PI_F,  M_2PI_F);
        float beta = twist * alpha / M_2PI_F;

        float lipschitzMultiplier = (r - minorR) * 2 * sin(min(M_PI_F, M_PI_2_F * M_PI_2_F / fabs(twist))) / minorR;

        finalDistanceBound = inPlaneResult.w * min(1.0f, lipschitzMultiplier);

        inPlaneDirection = rotated2d(inPlaneResult.xy, beta);
    }
    float multiplier = inPlaneDirection.x / axisDistance;
    return (float4)(coord.x * multiplier,
                    inPlaneDirection.y,
                    coord.z * multiplier,
                    finalDistanceBound);
}

// vim: filetype=c
