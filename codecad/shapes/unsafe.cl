float4 repetition_op(float ox, float oy, float oz, float4 coords) {
    return (float4)(remainder(coords.x, ox),
                    remainder(coords.y, oy),
                    remainder(coords.z, oz),
                    0);
}

float4 circular_repetition_to_op(float piOverN, float4 coords) {
    float len = length(coords.xy);
    float alpha = atan2(coords.y, coords.x) + 2 * M_PI + piOverN;
    int side = floor(alpha / (2 * piOverN));
    float modAlpha = alpha - side * 2 * piOverN - piOverN;

    return (float4)(len * sincos2(modAlpha), coords.z, 0);
}

float4 circular_repetition_from_op(float piOverN, float4 distance, float4 coords) {
    float alpha = atan2(coords.y, coords.x) + 2 * M_PI + piOverN;
    int side = floor(alpha / (2 * piOverN));

    return (float4)(rotated2d(distance.xy, side * 2 * piOverN),
                    distance.z, distance.w);
}

// vim: filetype=c
