/** Calculate volume and centroid of a given scene by evaluating the distance 
 * function on every grid point.
 * A cube is taken as inside the shape iff its center point is inside the shape.
 * Outputs count, xSum, ySum, zSum into the array `counters`.
 * [x-z]Sum are sums of indices of cubes that are inside.
 * Care must be taken when calling this not to overflow the 32bit counters. */
__kernel void volume(__constant float* shape,
                     float4 boxCorner, float boxStep,
                     __global uint counters[4])
{
    // Using local buffers to decrease global atomic contention
    __local uint counterBuffer;
    __local uint xSumBuffer;
    __local uint ySumBuffer;
    __local uint zSumBuffer;

    // Initialize the local buffers
    if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) {
        counterBuffer = 0;
        xSumBuffer = 0;
        ySumBuffer = 0;
        zSumBuffer = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float3 point = as_float3(boxCorner) + boxStep * (float3)(get_global_id(0),
                                                             get_global_id(1),
                                                             get_global_id(2));
    float value = evaluate(shape, point).w;

    if (value < 0)
    {
        // Inside
        atomic_inc(&counterBuffer);
        atomic_add(&xSumBuffer, get_global_id(0));
        atomic_add(&ySumBuffer, get_global_id(1));
        atomic_add(&zSumBuffer, get_global_id(2));
    }

    // flush the buffers into the global result
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) {
        atomic_add(&counters[0], counterBuffer);
        atomic_add(&counters[1], xSumBuffer);
        atomic_add(&counters[2], ySumBuffer);
        atomic_add(&counters[3], zSumBuffer);
    }
}

// vim: filetype=c
