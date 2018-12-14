float4 rectangle_op(float halfW, float halfH, float4 coords) {
    return perpendicular_intersection(slab_x(halfW, coords),
                                      slab_y(halfH, coords));
}

float4 circle_op(float r, float4 coords) {
    float2 flat = (float2)(coords.x, coords.y);
    float absFlat = length(flat);
    if (absFlat == 0) // TODO: Check this
        flat = (float2)(1, 0);
    else
        flat /= absFlat;
    return (float4)(flat.x, flat.y, 0, absFlat - r);
}

float4 regular_polygon2d_op(float piOverN, float r, float4 coords) {
    float len = hypot(coords.x, coords.y);
    float alpha = atan2(coords.y, coords.x) + 2 * M_PI + piOverN;
    int side = floor(alpha / (2 * piOverN));
    float modAlpha = alpha - side * 2 * piOverN - piOverN;

    float c;
    float s = sincos(modAlpha, &c);

    if (fabs(s * len) > r * sin(piOverN))
    {
        float c2;
        float2 nearest;
        nearest.y = sincos(side * 2 * piOverN + sign(s) * piOverN, &c2);
        nearest.x = c2;
        nearest *= r;
        float2 direction = coords.xy - nearest;
        float dist = length(direction);
        if (dist > 0) // TODO: Check this
        {
            direction /= dist;
            return (float4)(direction.x, direction.y, 0, dist);
        }
    }

    float c2;
    float2 direction;
    direction.y = sincos(side * 2 * piOverN, &c2);
    direction.x = c2;
    return (float4)(direction.x, direction.y, 0, len * c - r * cos(piOverN));
}

// vim: filetype=c
