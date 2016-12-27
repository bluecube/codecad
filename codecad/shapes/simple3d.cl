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

uchar extrusion_op(__constant float* params, float4* output, float4 coords, float4 input) {
    float halfH = params[0];
    *output = perpendicular_intersection(slab_z(halfH, coords),
                                         input);
    return 1;
}

uchar revolution_op(__constant float* params, float4* output, float4 coord, float4 unused) {
    *output = (float4)(hypot(coord.x, coord.z), coord.y, 0, 0);
    return 0;
}

// vim: filetype=c
