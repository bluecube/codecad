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
                              __global uchar4* list)
{
    // Using local buffer to decrease global atomic contention
    __local uint sumBuffer[10];

    bool isFirstInWorkgroup = get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0;
    // Initialize the local buffer
    if (isFirstInWorkgroup) {
        for (size_t i = 0; i < 10; ++i)
            sumBuffer[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float3 point = as_float3(boxCorner) + boxStep * (float3)(get_global_id(0),
                                                             get_global_id(1),
                                                             get_global_id(2));

    float value = evaluate(shape, point).w;

    if (value <= -distanceThreshold)
    {
        // Definitely inside; count it in
        uint coords[4] = { get_global_id(0),
                           get_global_id(1),
                           get_global_id(2),
                           1 };
        size_t i = 0;
        for (size_t j = 0; j < 4; ++j)
            for (size_t k = j; k < 4; ++k)
                atomic_add(&sumBuffer[i++], coords[j] * coords[k]);
    }
    else if (value < distanceThreshold)
        // Possibly intersecting the shape surface, needs to be split again
        list[atomic_inc(intersectingCounter)] = (uchar4)(get_global_id(0),
                                                         get_global_id(1),
                                                         get_global_id(2),
                                                         0);

    // flush the buffers into the global result
    barrier(CLK_LOCAL_MEM_FENCE);
    if (isFirstInWorkgroup) {
        for (size_t i = 0; i < 10; ++i)
            atomic_add(&sum[i], sumBuffer[i]);
    }
}


// vim: filetype=c
