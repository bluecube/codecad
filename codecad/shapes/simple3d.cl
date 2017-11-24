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

float4 revolution_to_op(float r, float twist, float4 coord) {
    float alpha = fmod(atan2pi(coord.z, coord.x) + 2, 2) / 2; // Angle along the main diameter, 0 to 1
    float beta = alpha * twist; // Local twist angle
    float dist = hypot(coord.x, coord.z);

    float2 flatCoord = rotated2d((float2)(dist - r, coord.y), -beta);
    return (float4)(flatCoord.x, flatCoord.y, 0, 0);
}

float4 revolution_from_op(float twist, float4 flatResult, float4 coord) {
    float alpha = fmod(atan2pi(coord.z, coord.x) + 2, 2) / 2; // Angle along the main diameter, 0 to 1
    float beta = alpha * twist; // Local twist angle
    float dist = hypot(coord.x, coord.z);

    float2 unrotatedFlatResult = rotated2d(flatResult.xy, beta);

    float multiplier = unrotatedFlatResult.x;
    if (dist == 0)
        coord.x = 1;
    else
        multiplier /= dist;

    return (float4)(coord.x * multiplier,
                    unrotatedFlatResult.y,
                    coord.z * multiplier,
                    flatResult.w);
}

// vim: filetype=c
