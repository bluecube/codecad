uchar sphere_op(__constant float* params, float4* output, float4 coords, float4 unused) {
    float r = params[0];
    float absCoords = length(as_float3(coords));
    if (absCoords == 0)
        *output = (float4)(1, 0, 0, 0);
    else
        *output = coords / absCoords;
    output->w = absCoords - r;
    return 1;
}

uchar half_space_op(__constant float* params, float4* output, float4 coords, float4 unused) {
    *output = (float4)(0, -1, 0, -coords.y);
}

uchar extrusion_op(__constant float* params, float4* output, float4 coords, float4 input) {
    float halfH = params[0];
    *output = perpendicular_intersection(slab_z(halfH, coords),
                                         input);
    return 1;
}

uchar revolution_to_op(__constant float* params, float4* output, float4 coord, float4 unused) {
    float x = hypot(coord.x, coord.z);
    *output = (float4)(x, coord.y, 0, 0);
    return 0;
}

uchar revolution_from_op(__constant float* params, float4* output, float4 coords, float4 flat) {
    float length = hypot(coords.x, coords.z);
    float multiplier;
    if (length == 0) {
        coords.x = 1;
        multiplier = flat.x;
    }
    else
        multiplier = flat.x / length;

    *output = (float4)(coords.x * multiplier, flat.y, coords.z * multiplier, flat.w);
    return 0;
}

// vim: filetype=c
