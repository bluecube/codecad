float naca_airfoil_yt(float t) {
    return 5 * (0.2969f * sqrt(t)
           - 0.1260f * t
           - 0.3516f * t * t
           + 0.2843f * t * t * t
           //- 0.1015f * t * t * t * t);
           - 0.1036f * t * t * t * t); // Closed trailing edge
}

float naca_airfoil_camber(float t, float maxCamber, float maxCamberPosition) {
    float tt = (t <= maxCamberPosition) ?
               t / maxCamberPosition :
               (1 - t) / (1 - maxCamberPosition);
    return maxCamber * tt * (2 - tt);
}

float naca_airfoil_camber_diff(float t, float maxCamber, float maxCamberPosition) {
    float pp = (t <= maxCamberPosition) ?
               maxCamberPosition :
               (1 - maxCamberPosition);
    return 2 * maxCamber * (maxCamberPosition - t) / (pp * pp);
}

float2 naca_airfoil_point(float t, bool lower, float thickness, float maxCamber, float maxCamberPosition) {
    float yt = naca_airfoil_yt(1 - fabs(t - 1));

    if (lower)
        yt = -yt;

    float camber = naca_airfoil_camber(t, maxCamber, maxCamberPosition);
    float camberDiff = naca_airfoil_camber_diff(t, maxCamber, maxCamberPosition);

    // float phi = atan(camberDiff);
    // float cosPhi;
    // float sinPhi = sincos(phi, &cosPhi);
    float ytCosPhi = yt * rsqrt(camberDiff * camberDiff + 1.0f);
    float ytSinPhi = camberDiff * ytCosPhi;

    return (float2)(t - ytSinPhi, camber + ytCosPhi);
}

uchar naca_airfoil_op(__constant float* params, float4* output, float4 coords, float4 unused) {
    float thickness = params[0];
    float maxCamber = params[1];
    float maxCamberPosition = params[2];

    float nearestTop = 0;
    float nearestTopDistanceSquared = INFINITY;
    float nearestBottom = 0;
    float nearestBottomDistanceSquared = INFINITY;
    float2 point = (float2)(coords.x, coords.y);
    for (int i = 0; i <= 100; ++i)
    {
        float t = i / 100.0f;

        float2 directionTop = naca_airfoil_point(t, false, thickness, maxCamber, maxCamberPosition);
        float distanceTopSquared = dot(directionTop, directionTop);
        if (distanceTopSquared < nearestTopDistanceSquared)
            nearestTop = t;

        float2 directionBottom = naca_airfoil_point(t, false, thickness, maxCamber, maxCamberPosition);
        float distanceBottomSquared = dot(directionBottom, directionBottom);
        if (distanceBottomSquared < nearestBottomDistanceSquared)
            nearestBottom = t;
    }

    return 3;
}

// vim: filetype=c
