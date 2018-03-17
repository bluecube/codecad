float sq(float x)
{
    return x * x;
}

float cb(float x)
{
    return x * x * x;
}

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

/** 32bit xorshift RNG.
 * x is generator state, returns new state.
 * x must be nonzero.
 * G1 from Numerical Recipes 3rd ed, chapter 7.1.7, page 356 */
uint xorshift32(uint x)
{
    x ^= (x >> 13);
    x ^= (x << 17);
    x ^= (x >> 5);
    return x;
}

/** Combine two uints into valid xorshift generator state (nonzero) */
uint combineState(uint a, uint b)
{
    return (a + b) % (UINT_MAX - 1) + 1;
}

/** Generate a random float between 0 and 1 using xorshift32 */
float randFloat(__private uint *state)
{
    *state = xorshift32(*state);
    return *state / (float)UINT_MAX;
}

/** Generate a random float3 between 0 and 1 by calling randFloat 3 times. */
float3 randFloat3(__private uint *state)
{
    return (float3)(randFloat(state),
                    randFloat(state),
                    randFloat(state));
}

// vim: filetype=c
