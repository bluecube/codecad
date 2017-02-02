uchar rectangle_op(__constant float* params, float4* output, float4 coords, float4 unused) {
    float halfW = params[0];
    float halfH = params[1];
    *output = perpendicular_intersection(slab_x(halfW, coords),
                                         slab_y(halfH, coords));
    return 2;
}

uchar circle_op(__constant float* params, float4* output, float4 coords, float4 unused) {
    float r = params[0];
    float2 flat = (float2)(coords.x, coords.y);
    float absFlat = length(flat);
    if (absFlat == 0) // TODO: Check this
        flat = (float2)(1, 0);
    else
        flat /= absFlat;
    *output = (float4)(flat.x, flat.y, 0, absFlat - r);
    return 1;
}

uchar polygon2d_op(__constant float* params, float4* output, float4 coords, float4 unused) {
    uint pointCount = *(params++);

    //printf("%d\n", pointCount);
    float2 query = (float2)(coords.x, coords.y);

    float2 nearestNormal;
    float nearestDistanceSquared = INFINITY;
    float outside = 1;

    float2 previousPoint = vload2((pointCount - 1), params);
    for (uint i = 0; i < pointCount; ++i)
    {
        float2 currentPoint = vload2(i, params);
        //printf("%d: %v2f -> %v2f\n", i, previousPoint, currentPoint);

        float2 direction = currentPoint - previousPoint;
        float2 toQuery = query - previousPoint;

        float t = dot(direction, toQuery) / dot(direction, direction);

        float2 candidate;
        if (t < 0)
            candidate = previousPoint;
        else if (t > 1)
            candidate = currentPoint;
        else
            candidate = previousPoint + t * direction;

        float2 candidateNormal = query - candidate;
        float currentDistanceSquared = dot(candidateNormal, candidateNormal);

        if (currentDistanceSquared < nearestDistanceSquared)
        {
            nearestDistanceSquared = currentDistanceSquared;
            nearestNormal = candidateNormal;
        }

        if (((previousPoint.y < coords.y) != (currentPoint.y < coords.y)) &&
            ((direction.y > 0) == (direction.x * toQuery.y > direction.y * toQuery.x)))
            outside = -outside;

        previousPoint = currentPoint;
    }

    float distance = outside * sqrt(nearestDistanceSquared);
    *output = (float4)(nearestNormal.x / distance, nearestNormal.y / distance, 0, distance);

    return 2 * pointCount + 1;
}

// vim: filetype=c
