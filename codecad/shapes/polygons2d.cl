float4 polygon2d_op(__constant float* restrict* restrict params, float4 coords) {
    uint pointCount = **params;

    ++*params;

    float2 query = (float2)(coords.x, coords.y);

    float2 nearestNormal;
    float nearestDistanceSquared = INFINITY;
    bool nearestIsVertex;
    float outside = 1;

    float2 currentPoint = vload2((pointCount - 1), *params);
    float2 previousPoint;
    for (uint i = 0; i < pointCount; ++i)
    {
        previousPoint = currentPoint;
        currentPoint = vload2(i, *params);

        float2 direction = currentPoint - previousPoint;
        float2 toQuery = query - previousPoint;
        float2 segmentNormal = (float2)(-direction.y, direction.x);

        if (((previousPoint.y < coords.y) != (currentPoint.y < coords.y)) &&
            (direction.y * dot(segmentNormal, toQuery) > 0))
            outside = -outside;

        float t = dot(direction, toQuery) / dot(direction, direction);

        if (t > 1)
            continue;

        float2 candidateNormal;
        float candidateDistanceSquared;
        bool candidateIsVertex;

        if (t >= 0)
        {
            float2 toCandidate = toQuery - t * direction;
            candidateDistanceSquared = dot(toCandidate, toCandidate);
            candidateNormal = segmentNormal;
            candidateIsVertex = false;
        }
        else
        {
            candidateNormal = query - previousPoint;
            candidateDistanceSquared = dot(candidateNormal, candidateNormal);
            candidateIsVertex = candidateDistanceSquared > FLT_EPSILON;
                // If query is too close to the vertex, we can't use it for normal

            if (!candidateIsVertex)
                candidateNormal = segmentNormal;
        }


        if (candidateDistanceSquared < nearestDistanceSquared)
        {
            nearestDistanceSquared = candidateDistanceSquared;
            nearestNormal = candidateNormal;
            nearestIsVertex = candidateIsVertex;
        }
    }

    float distance = outside * sqrt(nearestDistanceSquared);
    float2 normal;
    if (nearestIsVertex)
        normal = nearestNormal / distance;
    else
        normal = normalize(nearestNormal);

    *params += 2 * pointCount;

    return (float4)(normal.x, normal.y, 0, distance);
}

// vim: filetype=c
