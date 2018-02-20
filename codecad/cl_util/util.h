#ifndef UTIL_H
#define UTIL_H

#define M_2PI_F (2 * M_PI_F)

float2 sincos2(float angle);
float2 sincos2pi(float angle);
float2 rotated2d(float2 point, float angle);

void atomic_add64(volatile __global uint *lo,
                  volatile __global uint *hi,
                  uint add);

// Minimalistic random number generator:

uint xorshift32(uint value);
uint combineState(uint a, uint b);
float randFloat(__private uint *state);
float3 randFloat3(__private uint *state);

#endif //UTIL_H
