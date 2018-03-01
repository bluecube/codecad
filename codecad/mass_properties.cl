// #define TREE_SIZE from python

/*** Actual child count of the tree node */
#define TREE_CHILD_COUNT (TREE_SIZE * TREE_SIZE * TREE_SIZE)

/*** How many standard deviaitons are counted as error for plane splits */
#define PLANE_SPLIT_STDEV_MULTIPLE 2

#define PLANE_SPLIT_MIN_SAMPLES 10
#define PLANE_SPLIT_MAX_SAMPLES 50

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
 * monteCarloLeafThrreshold - Maximom value of cell size to allow using monte carlo integrals
 *                            instead of further splitting cells.
 * seed - 32bit random number for seeding the random generator.
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
                                       float bonusAllowedError,
                                       uint keepRemainingError,
                                       float monteCarloLeafThrreshold,
                                       uint seed,

                                       __global float4* restrict locations,
                                       __global float* restrict allowedErrors,
                                       __global float4* restrict tempLocations,
                                       __global float* restrict tempAllowedErrors,

                                       __global float4* restrict integral1,
                                       __global float4* restrict integral2,
                                       __global float4* restrict integral3,

                                       __global uint* restrict splitCounts,
                                       __global uint* restrict splitMasks,

                                       __global AssertBuffer* restrict assertBuffer)
{
    uint index = (startOffset + get_global_id(0)) % locationQueueSize;

    float4 location = locations[index];
    float allowedErrorFromLocation = allowedErrors[index];
    float allowedError = allowedErrorFromLocation + bonusAllowedError; // == absAllowedErrorPerChild / s3 [units**3 error / units**3 volume]

    float3 cellOrigin = location.xyz;
    float s = location.w;
    float s3 = s * s * s; // This is volume of a child sub node

    //printf("allowedError = %f, bonusAllowedError = %f, s3 = %f\n", allowedError, bonusAllowedError, s3);

    assert(assertBuffer, s > 0);
    assert(assertBuffer, all(cellOrigin + s != cellOrigin));
    assert(assertBuffer, all(cellOrigin + s / TREE_SIZE != cellOrigin)); // debug only, this can happen IRL

    if (any(cellOrigin + s / TREE_SIZE == cellOrigin))
        // If step is too small, expanding the cell will not help, because
        // coordinates of the expanded items will end up at identical positions.
        // Just accept any error and don't expand anything.
        allowedError = 1; // Avoid issues with too small values of s

    float volumes[TREE_CHILD_COUNT];
    uint n = 0;
    uint potentialSplitCount = 0;
    float remainingAllowedError = allowedError * TREE_CHILD_COUNT;

    float4 firstPlane;
    float4 avgPlaneSum = 0;
    float maxPlaneNormalError = 0;
    float maxPlaneDistanceError = 0;
    float volumeEstimateSum = 0;

    #pragma unroll
    for (uint i = 0; i < TREE_SIZE; ++i)
        #pragma unroll
        for (uint j = 0; j < TREE_SIZE; ++j)
            #pragma unroll
            for (uint k = 0; k < TREE_SIZE; ++k)
            {
                float3 ijk = (float3)(i, j, k);
                float3 center = cellOrigin + ijk * s + s / 2;
                float4 value = evaluate(shape, center);
                float unboundingVolume = get_unbounding_volume(value.w, s);

                // Preparations for splitting sub nodes
                volumes[n] = unboundingVolume;
                float error = (1 - fabs(unboundingVolume)) / 2.0;
                if (error < allowedError)
                    remainingAllowedError -= error;
                else
                    potentialSplitCount++;

                // Preparations for using Monte Carlo integration with single plane
                float4 currentPlane = value;
                currentPlane.w -= s * dot(ijk - ((float3)(TREE_SIZE, TREE_SIZE, TREE_SIZE) - 1) / 2, value.xyz);
                avgPlaneSum += currentPlane;
                volumeEstimateSum += (1 - unboundingVolume) / 2.0;
                if (n == 0)
                    firstPlane = currentPlane;
                else
                {
                    maxPlaneNormalError = max(maxPlaneNormalError, 1 - dot(firstPlane.xyz, currentPlane.xyz));
                    maxPlaneDistanceError = max(maxPlaneDistanceError, fabs(firstPlane.w - currentPlane.w) / s);
                }

                //printf("%i, %i, %i, normal: (%f, %f, %f), distance: %f, correctedDistance: %f, s=%f, unboundingVolume = %f\n",
                //       i, j, k, value.x, value.y, value.z, value.w, currentPlane.w, s, unboundingVolume);

                n++;
            }

    float allowedError2 = remainingAllowedError / potentialSplitCount;
    assert(assertBuffer, allowedError2 >= allowedError);

    float integralOne = 0;
    float3 integralX = 0;
    float3 integralXX = 0;
    float3 integralXY = 0;
    float integralAll = 0;
    float totalError = -allowedErrorFromLocation * TREE_CHILD_COUNT * s3;
    uint splitCount = 0;
    uint splitMask = 0;

    float volumeEstimate = volumeEstimateSum / TREE_CHILD_COUNT; // range: 0 - 1
    uint sampleCountEstimate = PLANE_SPLIT_STDEV_MULTIPLE * PLANE_SPLIT_STDEV_MULTIPLE *
                               volumeEstimate * (1 - volumeEstimate) / (allowedError * allowedError);

    bool usePlane = s < monteCarloLeafThrreshold &&
                    allowedError2 < 0.5 && // If no splitting is necessary, it's better to not use monte carlo
                    maxPlaneNormalError < 1e-4 && // TODO: Magic number here!
                    maxPlaneDistanceError < 1e-4 && // TODO: Magic number here!
                    sampleCountEstimate < PLANE_SPLIT_MAX_SAMPLES;


    if (usePlane)
    {
        /// Calculate the integrals using monte carlo and don't split any nodes.

        float3 normal = normalize(avgPlaneSum.xyz);
        float distance = avgPlaneSum.w / TREE_CHILD_COUNT;
        float3 cellCenter = cellOrigin + (float3)(0.5 * s * TREE_SIZE);

        //printf("normal = (%f, %f, %f), distance=%f, cellCenter=(%f, %f, %f), volumeEstimate=%f, totalAllowedError=%f\n",
        //       normal.x, normal.y, normal.z, distance, cellCenter.x, cellCenter.y, cellCenter.z, volumeEstimate, totalAllowedError);

        uint rngState = combineState(seed, get_global_id(0));
        uint sumOne = 0;
        float3 sumX = 0;
        float3 sumXX = 0;
        float3 sumXY = 0;

        uint sampleCount = 0;

        float errorThreshold = (allowedError / PLANE_SPLIT_STDEV_MULTIPLE);
        errorThreshold *= errorThreshold;
        while (1)
        {
            float3 coords = (randFloat3(&rngState) - 0.5) * s * TREE_SIZE;
            bool isInside = dot(coords, normal) < -distance;
            //printf("	%f. %f, %f -> %i\n", coords.x, coords.y, coords.z, isInside);

            if (isInside)
            {
                coords += cellCenter;

                sumOne += 1;
                sumX += coords;
                sumXX += coords * coords;
                sumXY += coords.zxy * coords.yzx;
            }
            sampleCount += 1;

            float errorTrigger = sumOne * (sampleCount - sumOne);
            errorTrigger /= sampleCount;
            errorTrigger /= sampleCount;
            errorTrigger /= sampleCount;

            if (sampleCount >= PLANE_SPLIT_MIN_SAMPLES && errorTrigger <= errorThreshold)
                break;
        }

        integralAll = TREE_CHILD_COUNT * s3;
        float scale = integralAll / sampleCount;

        integralOne = sumOne * scale;
        integralX = sumX * scale;
        integralXX = sumXX * scale;
        integralXY = sumXY * scale;
        totalError += PLANE_SPLIT_STDEV_MULTIPLE * sqrt(sumOne * (sampleCount - sumOne) / (float)sampleCount) * scale;

        if (totalError > integralAll * bonusAllowedError)
            printf("%e > %e * %e = %e\n", totalError, integralAll, bonusAllowedError, integralAll * bonusAllowedError);
    }
    else
    {
        // Potentially split nodes.

        n = 0;
        float errorSum = 0;
        #pragma unroll
        for (uint i = 0; i < TREE_SIZE; ++i)
            #pragma unroll
            for (uint j = 0; j < TREE_SIZE; ++j)
                #pragma unroll
                for (uint k = 0; k < TREE_SIZE; ++k)
                {
                    float3 center = cellOrigin + (float3)(i, j, k) * s + s / 2;
                    float volume = volumes[n];

                    float error = (1 - fabs(volume)) / 2.0;
                    //printf("%i, %i, %i (%i): center = %f, %f, %f, value = %f, volume = %f, error = %f (%s %f)\n",
                    //       i, j, k, n, center.x, center.y, center.z, value, volume, error,
                    //       maxErrorPerChild1, (error < maxErrorPerChild1 ? "<" : ">="));

                    if (error < allowedError2)
                    {
                        float weight = (1 - volume) / 2.0;

                        integralOne += weight;
                        integralX += weight * center;
                        integralXX += weight * (center * center + s * s / 12);
                        integralXY += weight * center.zxy * center.yzx;
                        errorSum += error;
                    }
                    else
                    {
                        splitCount += 1;
                        splitMask |= 1 << n;
                    }
                    ++n;
                }

        integralAll = s3 * (TREE_CHILD_COUNT - splitCount);
        integralOne *= s3;
        integralX *= s3;
        integralXX *= s3;
        integralXY *= s3;
        totalError += s3 * errorSum;

        if (splitCount > 0)
        {
            if (keepRemainingError)
            {
                float totalAllowedError = integralAll * bonusAllowedError; // This way bonusAllowedError will not decrease over time
                float allowedError3 = (totalAllowedError - totalError) / (splitCount * s3);

                //assert(assertBuffer, allowedError3 >= allowedError2);
                tempAllowedErrors[get_global_id(0)] = allowedError3;
                totalError = totalAllowedError; // The remaining allowed error gets passed to children
            }
            else
                tempAllowedErrors[get_global_id(0)] = 0;
        }
        tempLocations[get_global_id(0)] = location;
    }

    assert(assertBuffer, totalError <= integralAll * bonusAllowedError);

    integral1[get_global_id(0)] = (float4)(integralX, integralOne);
    integral2[get_global_id(0)] = (float4)(integralXX, integralAll);
    integral3[get_global_id(0)] = (float4)(integralXY, totalError);
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
                                           __global float* restrict allowedErrors,
                                           __global float4* restrict tempLocations,
                                           __global float* restrict tempAllowedErrors,
                                           __global uint* restrict splitCounts,
                                           __global uint* restrict splitMasks,
                                           __global AssertBuffer* restrict assertBuffer)
{
    float4 location = tempLocations[get_global_id(0)];
    float allowedError = tempAllowedErrors[get_global_id(0)];

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
                allowedErrors[index % locationQueueSize] = allowedError;
                ++index;
            }
}

// vim: filetype=c
