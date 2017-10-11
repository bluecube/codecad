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

float4 extrusion_op(float halfH, float4 coords, float4 input) {
    return perpendicular_intersection(slab_z(halfH, coords),
                                      input);
}

float4 revolution_to_op(float4 coord) {
    float x = hypot(coord.x, coord.z);
    return (float4)(x, coord.y, 0, 0);
}

float4 revolution_from_op(float4 coords, float4 flat) {
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

// vim: filetype=c
