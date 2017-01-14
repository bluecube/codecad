// Calculate distance and gradient for first loop of involute of circle.
// Assumes that the point is not inside the base circle.
float2 involute_base(float2 coords) {
    float len = length(coords);
    float phi = fmod(atan2(coords.y, coords.x) + 2 * M_PI + acos(1 / len), 2 * M_PI); // Curve parameter

    float distance = sqrt(len * len - 1) - phi;

    return (float2)(distance, phi);
}

float4 involute_output(float2 toCurve) {
    float cosVal;
    float sinVal = sincos(toCurve.s1, &cosVal);
    return (float4)(sinVal, cosVal, 0, toCurve.s0);
}

uchar involute_op(__constant float* params, float4* output, float4 coords, float4 unused) {
    float2 flatCoords = (float2)(coords.x, coords.y);
    if (dot(flatCoords, flatCoords) <= 1) {
        // coords are inside the base circle
        if (coords.y > 0) {
            // for the top half this means that curve start is nearest
            float2 v = (float2)(1, 0) - flatCoords;
            float d = length(v);
            *output = (float4)(v.x / d, v.y / d, 0, -d);
        }
        else {
            // lower half of base circle is nearest to the cutoff line
            *output = (float4)(1, 0, 0, coords.x - 1);
        }
    }
    else if (coords.y > 0) {
        // outside the base circle, top half is plain single loop involute,
        // but there is still a chance that the start point is nearest

        float2 toCurve = involute_base(flatCoords);
        if (toCurve.s0 < 0) {
            float2 v = (float2)(1, 0) - flatCoords;
            float d = length(v);
            if (toCurve.s0 > -d)
                *output = involute_output(toCurve);
            else
                *output = (float4)(v.x / d, v.y / d, 0, -d);
        }
        else
            *output = involute_output(toCurve);

    }
    else if (coords.x > 1 && coords.y < -2 * M_PI) {
        // All the way to botom right -- a corner of the involute and the dividing line
        float2 v = (float2)(1, -2 * M_PI) - flatCoords;
        float d = length(v);
        *output = (float4)(v.x / d, v.y / d, 0, d);
    }
    else {
        // Lower half without base circle and .
        // Need to check cutoff line as well.
        float2 toCurve = involute_base(flatCoords);
        if ((toCurve.s0 > coords.x - 1) != coords.x < 1)
            // We're closer to the cutoff line than to the curve
            *output = (float4)(1, 0, 0, coords.x - 1);
        else
            *output = involute_output(toCurve);
    }
    return 0;
}

// vim: filetype=c
