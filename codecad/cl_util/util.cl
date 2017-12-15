float2 sincos2(float angle) {
    float c;
    float s = sincos(angle, &c);
    return (float2)(c, s);
}

float2 sincos2pi(float angle) {
    return sincos2(angle * M_PI);
}

float2 rotated2d(float2 point, float angle) {
    float2 sc = sincos2(angle);
    return (float2)(sc.x * point.x - sc.y * point.y,
                    sc.y * point.x + sc.x * point.y);
}
