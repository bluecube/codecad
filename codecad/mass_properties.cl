/** Calculate volume and centroid of a given scene by evaluating the distance 
 * function on every grid point.
 * A cube is taken as inside the shape iff its center point is inside the shape.
 * Outputs count, xSum, ySum, zSum into the array `counters`.
 * [x-z]Sum are sums of indices of cubes that are inside.
 * Care must be taken when calling this not to overflow the 32bit counters. */
__kernel void mass_properties(__constant float* shape,
                              float4 boxCorner, float boxStep,
                              __global uint sum[10],
                              __global uint* intersectingCounter,
                              __global char4* list)
{
    __local uint localSumBuffer[10];
    __private uint privateSumBuffer[10];

    bool isFirstInWorkgroup = get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0;

    if (isFirstInWorkgroup)
        for (size_t i = 0; i < 10; ++i)
                localSumBuffer[i] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t i = 0; i < 10; ++i)
        privateSumBuffer[i] = 0;

    uint3 globalOffset = INNER_LOOP_SIDE * (uint3)(get_global_id(0),
                                                   get_global_id(1),
                                                   get_global_id(2));
    float distanceThreshold = boxStep * sqrt(3.0) / 2;

    for (size_t i = 0; i < INNER_LOOP_SIDE; ++i)
        for (size_t j = 0; j < INNER_LOOP_SIDE; ++j)
            for (size_t k = 0; k < INNER_LOOP_SIDE; ++k)
            {
                uint3 index = globalOffset + (uint3)(i, j, k);
                float3 point = boxCorner.xyz + convert_float3(index) * boxStep;

                float value = evaluate(shape, point).w;

                if (value <= -distanceThreshold)
                {
                    // Definitely inside; count it in
                    uint coords[4] = { index.x, index.y, index.z, 1 };
                    size_t n = 0;
                    for (size_t l = 0; l < 4; ++l)
                        for (size_t m = l; m < 4; ++m)
                            privateSumBuffer[n++] += coords[l] * coords[m];
                }
                else if (value < distanceThreshold)
                {
                    // Possibly intersecting the shape surface, needs to be split again
                    // We want to calculate what part of cube volume is covered by the
                    // unbounding sphere.
                    // This is easier to do with the cube having being 2x2x2 units big.
                    // so we scale the value
                    float radius = 2 * fabs(value) / boxStep;
                    float volume;

                    if (radius <= sqrt(2.0))
                    {
                        // Volume of the unbounding sphere itself
                        volume = radius * radius * radius * M_PI * 4 / 3.0;
                        float capHeight = radius - 1;
                        if (capHeight > 0)
                            // Subtract six spherical caps
                            volume -= capHeight * capHeight * (3 * radius - capHeight) * 2 * M_PI;
                    }
                    else
                    {
                        // Because the case where sqrt(2) < radius < sqrt(3) is too
                        // complicated, we use a lower bound on the unbounding volume
                        // in this case.
                        // The exact shape whose volume we're estimating is a cube
                        // with corners chopped of by a sphere.
                        // Instead we chop the corners off with trirectangular tetrahedrons
                        // which have more volume than the original corners
                        // (same vertices, but convex).
                        float x = 1 - sqrt(radius * radius - 2);
                        volume = 8 * (1 - x * x * x / 6);
                    }

                    // Encode volume for output
                    // Truncation when converting to char is Ok, since we only
                    // need lower bound of the volume
                    char cVolume = copysign(volume * 127 / 8.0, value);
                    //char cVolume = (volume) * 16;

                    // TODO: Figure out how to avoid the global atomic here
                    list[atomic_inc(intersectingCounter)] = (char4)(convert_char3(index),
                                                                    cVolume);
                }
            }

    for (size_t i = 0; i < 10; ++i)
        atomic_add(&localSumBuffer[i], privateSumBuffer[i]);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (isFirstInWorkgroup)
        for (size_t i = 0; i < 10; ++i)
            atom_add(&sum[i], localSumBuffer[i]);
}


// vim: filetype=c
