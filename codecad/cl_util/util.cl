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

void atomic_add64(volatile __global uint *lo,
                  volatile __global uint *hi,
                  uint add)
{
    uint old = atomic_add(lo, add);
    if (old + add < old)
        atomic_inc(hi);
}
