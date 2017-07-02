/* Evaluate scene on grid points starting at boxCorner and spaced by boxStep in
 * every dimension.
 * Detects if cells are provably empty (value > distanceThreshold), provably full
 * (value < -distanceThreshold) or inconclusive.
 * Coordinates of full cells are added to counters, inconclusive cell coordinates
 * are appended to output list, empty cells are skipped.
 * distanceThreshold must be specified as a parameter, setting it to 0 turns this
 * function into a leaf function for centroid and volume calculation, setting it to
 * sqrt(3) * boxStep / 2 makes this run in standard mode.
 * TODO: Figure out if distanceThreshold shouldn't be one epsilon higher.
 * Care must be taken that [xyz]Sum counters don't overflow the 32bit integers. */
__kernel void subdivision_step(__constant float* scene,
                               float4 boxCorner, float boxStep,
                               float distanceThreshold,
                               __global uint counters[5],
                               __global uchar4* list)
{
    __global uint *intersectingCounter = &counters[0];
    __global uint *innerCounter = &counters[1];
    __global uint *innerXSum = &counters[2];
    __global uint *innerYSum = &counters[3];
    __global uint *innerZSum = &counters[4];

    // Using local buffers to decrease global atomic contention
    __local uint innerCounterBuffer;
    __local uint innerXSumBuffer;
    __local uint innerYSumBuffer;
    __local uint innerZSumBuffer;

    bool isFirstInWorkgroup = get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0;
    // Initialize the local buffers
    if (isFirstInWorkgroup) {
        innerCounterBuffer = 0;
        innerXSumBuffer = 0;
        innerYSumBuffer = 0;
        innerZSumBuffer = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint3 coords = (uint3)(get_global_id(0),
                           get_global_id(1),
                           get_global_id(2));

    float3 point = as_float3(boxCorner) + boxStep * convert_float3(coords);
    float value = evaluate(scene, point).w;

    if (value <= -distanceThreshold)
    {
        // Definitely inside; count it in
        atomic_inc(&innerCounterBuffer);
        atomic_add(&innerXSumBuffer, get_global_id(0));
        atomic_add(&innerYSumBuffer, get_global_id(1));
        atomic_add(&innerZSumBuffer, get_global_id(2));
    }
    else if (value >= distanceThreshold)
    {
        // Nothing to do here, we don't report blocks that are not covered
    }
    else
        // Possibly intersecting the shape surface, needs to be split again
        list[atomic_inc(intersectingCounter)] = (uchar4)(coords.x, coords.y, coords.z, 0);

    // flush the buffers into the global result
    barrier(CLK_LOCAL_MEM_FENCE);
    if (isFirstInWorkgroup) {
        atomic_add(innerCounter, innerCounterBuffer);
        atomic_add(innerXSum, innerXSumBuffer);
        atomic_add(innerYSum, innerYSumBuffer);
        atomic_add(innerZSum, innerZSumBuffer);
    }
}

// vim: filetype=c
