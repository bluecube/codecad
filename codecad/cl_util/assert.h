#ifndef ASSERT_H
#define ASSERT_H

// Buffer for storing results of failed asserts.
// Needs to have the first 4 bytes set to 0 before use.
typedef union
{
    struct
    {
        unsigned assertCount;
        unsigned globalId[4];
        unsigned line;
    } details;
    char text[ASSERT_BUFFER_SIZE];
} AssertBuffer;

int _assert_internal(int result,
                     __global AssertBuffer* restrict assertBuffer,
                     const __constant char* restrict file,
                     int line,
                     const __constant char* restrict expr);

// Evaluates to 1 if the assertion failed, 0 otherwise, contrary to assert in
// regular C this doesn't interrupt execution if it fails (because we couldn't
// stop all the kernels anyway).
// Instead assert stores the number of registered assertions and details of the
// first one in assertBuffer.
// If macro ASSERT_ENABLED is not defined, assert does nothing (and assertBuffer
// need not be valid).
#if defined(ASSERT_ENABLED)
    #define assert(assertBuffer, x) \
        _assert_internal(x, assertBuffer, __FILE__, __LINE__, #x)
    #define WHEN_ASSERT(x) x
#else
    #define assert(assertBuffer, x) (sizeof(assertBuffer), sizeof(x), 0)
    #define WHEN_ASSERT(x)
#endif

#endif //ASSERT_H
