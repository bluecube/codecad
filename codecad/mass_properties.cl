/* Mass properties are calculated in two stages:
 *
 * stage 1: Evaluate the distance function (possibly skipping boring inner blocks),
 * store the result, calculate number of intersecting sub-blocks.
 *
 * stage 2: At this point we know the allowed error per intersecting block.
 * Sum all internal non intersecting blocks and all intersecting blocks that
 * will cause small enough error, output the list of coordinates that need to be split.  */

// #define TREE_SIZE from python

// #define WG_SIZE 64

/*** Actual child count of the tree node */
#define TREE_CHILD_COUNT (TREE_SIZE * TREE_SIZE * TREE_SIZE)

static float get_unbounding_volume(float value, float stepSize)
{
    float radius = value / stepSize;

    if (radius <= -sqrt(3.0) / 2)
        return -1.0;
    else if (radius >= sqrt(3.0) / 2)
        return 1.0f;
    else
    {
        float absRadius = fabs(radius);
        if (absRadius <= sqrt(2.0) / 2)
        {
            // Volume of the unbounding sphere itself, already scaled to cube with volume 1
            float volume = radius * radius * radius * M_PI * 4.0 / 3.0;
            if (absRadius > 0.5)
            {
                // Subtract six spherical caps
                float capHeight = radius - copysign(0.5, radius);
                volume -= capHeight * capHeight * (3 * radius - capHeight) * M_PI * 2.0;
            }

            return volume;
        }
        else
        {
            // Because the case where sqrt(2) < absRadius < sqrt(3) is too
            // complicated, we use a lower bound on the unbounding volume
            // in this case.
            // The exact shape whose volume we're estimating is a cube
            // with corners chopped of by a sphere.
            // Instead we chop the corners off with trirectangular tetrahedrons
            // which have more volume than the original corners
            // (same vertices, but convex).
            float x = 0.5 - sqrt(radius * radius - 0.5);

            return copysign(1 - x * x * x * 4.0 / 3.0, radius);
        }
    }
}

#ifdef CODECAD_TEST
__kernel void test_mass_properties_unbounding_volume(float value, __global float* volume)
{
    *volume = get_unbounding_volume(value, 1);
}
#endif

/** Calculate volume and centroid of a given scene by evaluating the distance 
 * function on grid points.
 *
 * This is done by recursively descending down a n**3-tree, processing several
 * nodes in parallel.
 *
 * shape: Shape we're working with
 * startOffset: Index of the first work item in the input arrays.
 *              Should be a multiple of 8 (or 16?) to keep memory access aligned.
 * locationQueueSize: Size of the location queue to wrap the indices
 * allowedErrorPerVolume: Error allowed per unit of block volume.
 * locations: xyz coordinates of a corner of tree nodes to process,
 *            w is size of the child node (w * TREE_SIZE is size of this node).
 *            get_global_size(0) items large.
 * tempLocations: locations get copied in here for use in stage3
 *                get_global_size(0) items large.
 * integral1 - integral3: Partial integral values that need to get summed.
 *                        //get_num_groups(0) items large.
 *                        get_global_size(0) items large.
 *                        integral1.xyz ~ integral point.xyz
 *                        integral1.w ~ integral 1
 *                        integral2.xyz ~ integral point.xyz**2
 *                        integral2.w ~ integralAll (all volume processed, for progress reporting)
 *                        integral3.xyz ~ integral point.zxy * point.yzx
 *                        integral3.w ~ total error used
 * splitCounts - Numbers of children that need to be split. Must be prefix summed
 *               in stage 2 to generate placement indices for the new locations.
 *               get_global_size(0) items large.
 * splitMasks - Bitmasks determining which children need to be split in stage 3.
 *              in stage 2 to generate placement indices for the new locations.
 *              get_global_size(0) items large.
 */
__kernel void mass_properties_evaluate(__constant float* restrict shape,

                                       uint startOffset,
                                       uint locationQueueSize,
                                       float allowedErrorPerVolume,

                                       __global float4* restrict locations,
                                       __global float4* restrict tempLocations,

                                       __global float4* restrict integral1,
                                       __global float4* restrict integral2,
                                       __global float4* restrict integral3,

                                       __global uint* restrict splitCounts,
                                       __global uint* restrict splitMasks,

                                       __global AssertBuffer* restrict assertBuffer)
{
    uint index = startOffset + get_global_id(0) % locationQueueSize;
    float4 location = locations[index];

    tempLocations[get_global_id(0)] = location;

    float s = location.w;
    float s3 = s * s * s;

    assert(assertBuffer, s > 0);
    assert(assertBuffer, all(location.xyz + s != location.xyz));

    if (any(location.xyz + s == location.xyz))
        allowedErrorPerVolume = 1; // Avoid issues with too small values of s

    float4 volumesAndPositions[TREE_CHILD_COUNT];

    float integralOne = 0;
    float3 integralX = 0;
    float3 integralXX = 0;
    float3 integralXY = 0;
    float integralAll = 0;
    float totalError = 0;
    uint splitCount = 0;
    uint splitMask = 0;

    uint n = 0;
    #pragma unroll
    for (uint i = 0; i < TREE_SIZE; ++i)
        #pragma unroll
        for (uint j = 0; j < TREE_SIZE; ++j)
            #pragma unroll
            for (uint k = 0; k < TREE_SIZE; ++k)
            {
                float3 center = location.xyz + (float3)(i, j, k) * s + s / 2;
                float value = evaluate(shape, center).w;
                float volume = get_unbounding_volume(value, s);

                float error = (1 - fabs(volume)) / 2.0;
                //printf("%i, %i, %i (%i): center = %f, %f, %f, value = %f, volume = %f, error = %f (%s %f)\n",
                //       i, j, k, n, center.x, center.y, center.z, value, volume, error,
                //       maxErrorPerChild1, (error < maxErrorPerChild1 ? "<" : ">="));

                if (error < allowedErrorPerVolume)
                {
                    float weight = (1 - volume) / 2.0;

                    integralOne += weight;
                    integralX += weight * center;
                    integralXX += weight * (center * center + s * s / 12);
                    integralXY += weight * center.zxy * center.yzx;
                    integralAll += 1;
                    totalError += error;
                }
                else
                {
                    splitCount += 1;
                    splitMask |= 1 << n;
                }
                ++n;
            }

    integral1[get_global_id(0)] = s3 * (float4)(integralX, integralOne);
    integral2[get_global_id(0)] = s3 * (float4)(integralXX, integralAll);
    integral3[get_global_id(0)] = s3 * (float4)(integralXY, totalError);
    splitCounts[get_global_id(0)] = splitCount;
    splitMasks[get_global_id(0)] = splitMask;

    /*
    // Do a first step of parallel reduction in this function already
    // Pack the integrals for storage and prepare for the first step of summing.
    __local float4 localIntegral1[WG_SIZE];
    __local float4 localIntegral2[WG_SIZE];
    __local float4 localIntegral3[WG_SIZE];
    __local uint localSplitCouts[WG_SIZE];

    localIntegral1[get_local_id(0)] = (float4)(integralX, integralOne);
    localIntegral2[get_local_id(0)] = (float4)(integralXX, integralXY.z);
    localIntegral3[get_local_id(0)] = integralXY.xy;
    localSplitCouts[get_local_id(0)] = splitCount;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) == 0)
    {
        float4 localIntegralSum1 = 0;
        float4 localIntegralSum2 = 0;
        float2 localIntegralSum3 = 0;
        uint localSplitCoutsSum = 0;

        for (uint i = 1; i < WG_SIZE; ++i)
        {
            // TODO: Do this in parallel too
            localIntegralSum1 += localIntegral1[i];
            localIntegralSum2 += localIntegral2[i];
            localIntegralSum3 += localIntegral3[i];
            localSplitCoutsSum += localSplitCoutsSum[i];
        }

        integral1[get_group_id(0)] = localIntegralSum1;
        integral2[get_group_id(0)] = localIntegralSum2;
        integral3[get_group_id(0)] = localIntegralSum3;
        splitCounts[get_global_id(0)] = localSplitCoutsSum;
    }
    else
        splitCounts[get_global_id(0)] = splitCount;
    */
}

/** Sum up the integrals and split counts. */
__kernel void mass_properties_sum(__global float4* restrict integral1,
                                  __global float4* restrict integral2,
                                  __global float4* restrict integral3,

                                  __global uint* restrict splitCounts,

                                  __global AssertBuffer* restrict assertBuffer)
{
    // TODO: This is just proof of concept sequential implementation
    if (get_global_id(0) != 0)
        return;

    float4 sum1 = integral1[0];
    float4 sum2 = integral2[0];
    float4 sum3 = integral3[0];
    unsigned sc = splitCounts[0];
    for (uint i = 1; i < get_global_size(0); ++i)
    {
        sum1 += integral1[i];
        sum2 += integral2[i];
        sum3 += integral3[i];
        sc += splitCounts[i];
        splitCounts[i] = sc;
    }

    integral1[get_global_size(0) - 1] = sum1;
    integral2[get_global_size(0) - 1] = sum2;
    integral3[get_global_size(0) - 1] = sum3;
}

/** Collect integrals and final split count to single output buffer for python to read from. */
__kernel void mass_properties_output(uint n,
                                     __global float4* restrict integral1,
                                     __global float4* restrict integral2,
                                     __global float4* restrict integral3,
                                     __global uint* restrict splitCounts,
                                     __global float* restrict outputBuffer)
{
    uint last = n - 1;
    outputBuffer[0] = integral1[last].w; // integral 1
    vstore3(integral1[last].xyz, 0, outputBuffer + 1); // integral xyz
    vstore3(integral2[last].xyz, 0, outputBuffer + 4); // integral xyz**2
    vstore3(integral3[last].xyz, 0, outputBuffer + 7); // integral xyz * yzx
    outputBuffer[10] = integral2[last].w; // integralAll
    outputBuffer[11] = integral3[last].w; // totalError
    outputBuffer[12] = splitCounts[last]; // split count
}

/** Write new locations and allowed errors to the stack. */
__kernel void mass_properties_prepare_next(uint startOffset,
                                           uint locationQueueSize,
                                           __global float4* restrict locations,
                                           __global float4* restrict tempLocations,
                                           __global uint* restrict splitCounts,
                                           __global uint* restrict splitMasks,

                                           __global AssertBuffer* restrict assertBuffer)
{
    float4 location = tempLocations[get_global_id(0)];

    uint index = startOffset;
    if (get_global_id(0) > 0)
        index += splitCounts[get_global_id(0) - 1]; // TODO: This makes the mem access misaligned

    uint splitMask = splitMasks[get_global_id(0)];

    //printf("prepare: %f, %f, %f / %f, nextAllowedError = %f, index = %i (-%i), splitMask = %i\n",
    //       location.x, location.y, location.z, location.w,
    //       nextAllowedError, index, startOffset, splitMask);

    float s = location.w;
    float nextS = s / TREE_SIZE;

    for (uint i = 0; i < TREE_SIZE; ++i)
        for (uint j = 0; j < TREE_SIZE; ++j)
            for (uint k = 0; k < TREE_SIZE; ++k)
            {
                bool isSplit = splitMask & 1;
                splitMask >>= 1;

                if (!isSplit)
                    continue;

                float3 corner = location.xyz + (float3)(i, j, k) * s;
                locations[index % locationQueueSize] = (float4)(corner, nextS);
                ++index;
            }
}

// vim: filetype=c
