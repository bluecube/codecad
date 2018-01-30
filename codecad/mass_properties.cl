/* Mass properties are calculated in two stages:
 *
 * stage 1: Evaluate the distance function (possibly skipping boring inner blocks),
 * store the result, calculate number of intersecting sub-blocks.
 *
 * stage 2: At this point we know the allowed error per intersecting block.
 * Sum all internal non intersecting blocks and all intersecting blocks that
 * will cause small enough error, output the list of coordinates that need to be split.  */


/** Number of sub blocks calculated inside one work item */
#define INNER_LOOP_COUNT (INNER_LOOP_SIDE * INNER_LOOP_SIDE * INNER_LOOP_SIDE)

/** Half of main diagonal of cube with side length 1
 * Threshold value for chopping of sub blocks. */
#define THRESHOLD (sqrt(3.0) / 2.0)

/** Index of the center inner sub block  */
#define INNER_LOOP_MID ((INNER_LOOP_SIDE - 1) / 2)

/** Size of counters array. The last 10 items is for coordinate sums. */
#define COUNTERS (10 + 1)
#define INTERSECTING (0)

/** Wrapper for evaluate that calculates the actual position from index and
 * stores it in temp array. Also the value is made relative to boxStep. */
static float mp_evaluate(__constant float* shape,
                         float3 boxCorner, float boxStep,
                         __global float* values,
                         uint3 index)
{
    float3 point = boxCorner + convert_float3(index) * boxStep;
    float value = evaluate(shape, point).w / boxStep;
    values[INDEX3(get_global_size(0) * INNER_LOOP_SIDE,
                  get_global_size(1) * INNER_LOOP_SIDE,
                  get_global_size(2) * INNER_LOOP_SIDE,
                  index.x, index.y, index.z)] = value;
    return value;
}

static void process_sample(uint3 index, float value, float maxError,
                           __private uint privateCounters[10],
                           __global uchar3* splitList,
                           __global uint* splitCounter)
{
    uint multiplier;

    if (value > THRESHOLD)
        multiplier = 0; // Definitely outside
    if (value <= -THRESHOLD)
        multiplier = MAX_WEIGHT_MULTIPLIER; // Definitely inside
    else if (value < THRESHOLD)
    {
        // Possibly intersecting the shape surface, might need to be split again
        // We want to calculate what part of cube volume is covered by the
        // unbounding sphere.
        // This is easier to do with the cube having being 2x2x2 units big.
        // so we scale the value, and later rescale the volume back to 0-1 range.
        float radius = 2 * fabs(value);
        float volume;

        if (radius <= sqrt(2.0))
        {
            // Volume of the unbounding sphere itself
            volume = radius * radius * radius * M_PI * 4 / 3.0;
            if (radius < 1)
            {
                // Subtract six spherical caps
                float capHeight = radius - 1;
                volume -= capHeight * capHeight * (3 * radius - capHeight) * 2 * M_PI;
            }
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

        volume /= 16; // Scale the volume back to a 1**3 cube

        // Error is also influenced by the resolution of multiplier
        float error = (1 - volume + 1.0 / MAX_WEIGHT_MULTIPLIER) / 2;

        if (error < maxError)
        {
            // No need to expand the sub block, just add it to output with proper weight
            float weight = (1 - copysign(volume, value)) / 2;
            multiplier = MAX_WEIGHT_MULTIPLIER * weight + 0.5;
        }
        else
        {
            // TODO: Figure out how to avoid the global atomic here
            splitList[atomic_inc(splitCounter)] = convert_uchar3(index);
            return;
        }
    }

    uint coords[4] = { index.x, index.y, index.z, 1 };
    size_t i = 0;
    for (size_t j = 0; j < 4; ++j)
        for (size_t k = j; k < 4; ++k)
            privateCounters[i++] += multiplier * coords[j] * coords[k];
}

/** Calculate volume and centroid of a given scene by evaluating the distance 
 * function on every grid point.
 * `counters` array contains index sums of the intersecting sub-blocks and number
 * of intersecting sub blocks. */
__kernel void mass_properties_stage1(__constant float* shape,
                                     float3 boxCorner, float boxStep,
                                     __global float* values,
                                     __global uint counters[22])
{
    __local uint localIntersecting;
    __private uint privateIntersecting = 0;

    bool isFirstInWorkgroup = get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0;

    if (isFirstInWorkgroup)
        localIntersecting = 0;

    uint3 globalOffset = INNER_LOOP_SIDE * (uint3)(get_global_id(0),
                                                   get_global_id(1),
                                                   get_global_id(2));

    uint3 midIndex = globalOffset + INNER_LOOP_MID;
    float midValue = mp_evaluate(shape, boxCorner, boxStep, values, midIndex);

    if (fabs(midValue) < THRESHOLD * INNER_LOOP_SIDE)
    {
        if (fabs(midValue) < THRESHOLD)
            privateIntersecting++;

        for (size_t i = 0; i < INNER_LOOP_SIDE; ++i)
            for (size_t j = 0; j < INNER_LOOP_SIDE; ++j)
                for (size_t k = 0; k < INNER_LOOP_SIDE; ++k)
                {
                    if (i == INNER_LOOP_MID &&
                        j == INNER_LOOP_MID &&
                        k == INNER_LOOP_MID)
                        continue;

                    uint3 index = globalOffset + (uint3)(i, j, k);
                    float value = mp_evaluate(shape, boxCorner, boxStep, values, index);

                    if (fabs(value) < THRESHOLD)
                        privateIntersecting++;
                }
    }

    atomic_add(&localIntersecting, privateIntersecting);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (isFirstInWorkgroup)
        atomic_add(&counters[0], localIntersecting);
}

__kernel void mass_properties_stage2(float maxTotalError,
                                     __global float* values,
                                     __global uchar3* splitList,
                                     __global uint counters[22])
{
    __local uint localCounters[10];
    __private uint privateCounters[10];

    bool isFirstInWorkgroup = get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0;

    if (isFirstInWorkgroup)
        for (size_t i = 0; i < 10; ++i)
                localCounters[i] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t i = 0; i < 10; ++i)
        privateCounters[i] = 0;

    float maxError = maxTotalError / counters[0]; // Maximal error per intersecting sub-block

    uint3 globalOffset = INNER_LOOP_SIDE * (uint3)(get_global_id(0),
                                                   get_global_id(1),
                                                   get_global_id(2));

    uint3 midIndex = globalOffset + INNER_LOOP_MID;
    float midValue = values[INDEX3(get_global_size(0) * INNER_LOOP_SIDE,
                                   get_global_size(1) * INNER_LOOP_SIDE,
                                   get_global_size(2) * INNER_LOOP_SIDE,
                                   midIndex.x, midIndex.y, midIndex.z)];

    if (midValue <= -THRESHOLD * INNER_LOOP_SIDE)
    {
        // All inner sub-blocks are covered by this sample, add them to the sums
        uint coords[4] = { midIndex.x, midIndex.y, midIndex.z, 1 };
        size_t n = 0;
        for (size_t l = 0; l < 4; ++l)
            for (size_t m = l; m < 4; ++m)
            {
                if (l == m && l != 3)
                    privateCounters[n] += MAX_WEIGHT_MULTIPLIER * INNER_LOOP_COUNT * (INNER_LOOP_COUNT * INNER_LOOP_COUNT - 1) / 12;
                privateCounters[n++] += MAX_WEIGHT_MULTIPLIER * INNER_LOOP_COUNT * coords[l] * coords[m];
            }
    }
    else if (midValue < THRESHOLD * INNER_LOOP_SIDE)
    {
        process_sample(midIndex, midValue, maxError,
                       privateCounters, splitList, &counters[1]);

        for (size_t i = 0; i < INNER_LOOP_SIDE; ++i)
            for (size_t j = 0; j < INNER_LOOP_SIDE; ++j)
                for (size_t k = 0; k < INNER_LOOP_SIDE; ++k)
                {
                    if (i == INNER_LOOP_MID &&
                        j == INNER_LOOP_MID &&
                        k == INNER_LOOP_MID)
                        continue;

                    uint3 index = globalOffset + (uint3)(i, j, k);
                    float value = values[INDEX3(get_global_size(0) * INNER_LOOP_SIDE,
                                                get_global_size(1) * INNER_LOOP_SIDE,
                                                get_global_size(2) * INNER_LOOP_SIDE,
                                                index.x, index.y, index.z)];

                    process_sample(index, value, maxError,
                                   privateCounters, splitList, &counters[1]);
                }
    }

    for (size_t i = 0; i < 10; ++i)
        atomic_add(&localCounters[i], privateCounters[i]);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (isFirstInWorkgroup)
        for (size_t i = 0; i < COUNTERS; ++i)
            // Using the emulated 64bit atomic_add
            atomic_add64(&counters[2 * i + 2], &counters[2 * i + 3],
                         localCounters[i]);
}


// vim: filetype=c
