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
    flat /= absFlat;
    *output = (float4)(flat.x, flat.y, 0, absFlat - r);
    return 1;
}

// vim: filetype=c
