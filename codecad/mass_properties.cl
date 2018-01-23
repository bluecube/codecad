/** Calculate volume and centroid of a given scene by evaluating the distance 
 * function on every grid point.
 * A cube is taken as inside the shape iff its center point is inside the shape.
 * Outputs count, xSum, ySum, zSum into the array `counters`.
 * [x-z]Sum are sums of indices of cubes that are inside.
 * Care must be taken when calling this not to overflow the 32bit counters. */
__kernel void mass_properties(__constant float* shape,
                              float4 boxCorner, float boxStep,
                              float distanceThreshold,
                              __global uint sum[10],
                              __global uint* intersectingCounter,
                              __global uchar3* list)
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
                    // Possibly intersecting the shape surface, needs to be split again
                    // TODO: Figure out how to avoid the global atomic here
                    list[atomic_inc(intersectingCounter)] = convert_uchar3(index);
            }

    for (size_t i = 0; i < 10; ++i)
        atomic_add(&localSumBuffer[i], privateSumBuffer[i]);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (isFirstInWorkgroup)
        for (size_t i = 0; i < 10; ++i)
            atom_add(&sum[i], localSumBuffer[i]);
}


// vim: filetype=c
