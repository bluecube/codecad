float4 involute_gear_op(float toothCount, float pressureAngle, float4 coords) {
    float baseRadius = cos(pressureAngle);
    float toothAngle = M_PI / toothCount;

    // Angle from the start of the tooth profile on base circle to the tip of the tooth
    float halfToothBaseAngle = toothAngle / 2 + tan(pressureAngle) - pressureAngle;

    float len = hypot(coords.x, coords.y);
    float alpha = atan2(coords.y, coords.x);

    float wrappedAlpha = fmod(alpha + 2 * M_PI, 2 * toothAngle);
    float involuteAlpha = halfToothBaseAngle - fabs(wrappedAlpha - toothAngle);

    if (len < baseRadius) {
        float2 normal = (float2)(coords.y / len, -coords.x / len);
        if (wrappedAlpha > toothAngle)
            normal = -normal;
        float angularDistance = fabs(wrappedAlpha - toothAngle) - halfToothBaseAngle;
        return (float4)(normal.x, normal.y, 0, angularDistance * len);
    }
    else {
        float phi = involuteAlpha + acos(baseRadius / len); // Curve parameter

        // TODO: Outside the tooth tip the normals and distances are a little too sharp.
        // In that area the nearest point is the tooth tip, not the tooth profile itself.

        float normalAngle;
        if (wrappedAlpha < toothAngle)
            normalAngle = M_PI - phi - (alpha - involuteAlpha);
        else
            normalAngle = phi - (alpha - involuteAlpha);

        float2 normal;
        float normaly;
        normal.x = sincos(normalAngle, &normaly);
        normal.y = normaly;

        float distance = sqrt(len * len - baseRadius * baseRadius) - baseRadius * phi;

        return (float4)(normal.x, normal.y, 0, distance);
    }
}

// vim: filetype=c
