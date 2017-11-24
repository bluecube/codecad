float2 sincos2(float angle) {
    float c;
    float s = sincos(angle, &c);
    return (float2)(c, s);
}

float2 sincos2pi(float angle) {
    return sincos2(angle * M_PI);
}
