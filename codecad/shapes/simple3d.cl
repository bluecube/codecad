void sphere_op(float r, float4 coords, float4* output) {
    float absCoords = length(as_float3(coords));
    if (absCoords == 0)
        *output = (float4)(1, 0, 0, 0);
    else
        *output = coords / absCoords;
    output->w = absCoords - r;
}

void half_space_op(float4 coords, float4* output) {
    *output = (float4)(0, -1, 0, -coords.y);
}

void extrusion_op(float halfH, float4 coords, float4 input, float4* output) {
    *output = perpendicular_intersection(slab_z(halfH, coords),
                                         input);
}

void revolution_to_op(float4 coord, float4* output) {
    float x = hypot(coord.x, coord.z);
    *output = (float4)(x, coord.y, 0, 0);
}

void revolution_from_op(float4 coords, float4 flat, float4* output) {
    float length = hypot(coords.x, coords.z);
    float multiplier;
    if (length == 0) {
        coords.x = 1;
        multiplier = flat.x;
    }
    else
        multiplier = flat.x / length;

    *output = (float4)(coords.x * multiplier, flat.y, coords.z * multiplier, flat.w);
}

// vim: filetype=c
