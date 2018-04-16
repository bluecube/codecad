// #define TREE_SIZE from python

/** Actual child count of the tree node */
#define TREE_CHILD_COUNT (TREE_SIZE * TREE_SIZE * TREE_SIZE)

/** Minimal dot product between directions within a cell to consider using
 * plane split on it (instead of breaking it down into sub-cells) */
#define PLANE_SPLIT_MIN_DOT cos(radians(1.0f))

/** Return fraction of volume of a cube with side `stepSize` that is covered by
 * a sphere with radius `value`. Both centered at origin. */
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

/** Return volume of a unit cube intersected with half-space dot(X, normal) < distance.
 * See ../doc/mass_properties/cube_halfspace.ipynb for details on how this works. */
static float cube_halfspace_volume(float3 normal, float distance,
                                   __global AssertBuffer* restrict assertBuffer)
{
    bool negativeDist = distance < 0;
    if (negativeDist)
    {
        distance = -distance;
        normal = -normal;
    }

    normal = fabs(normal);

    if (normal.z < normal.x)
        normal = normal.zyx;
    if (normal.z < normal.y)
        normal = normal.xzy;
    if (normal.y < normal.x)
        normal = normal.yxz;

    // Check the normalization assumptions
    assert(assertBuffer, normal.z >= normal.y);
    assert(assertBuffer, normal.y >= normal.x);
    assert(assertBuffer, normal.x >= 0);
    assert(assertBuffer, distance >= 0);

    // Rescale the distance, so that we can work with a cube [-1, 1]**3
    distance *= 2;

    float volume;

    if (-normal.x - normal.y + normal.z >= distance)
    {
        // Case 0
        assert(assertBuffer, normal.x - normal.y + normal.z >= distance);
        volume = 4 * (distance / normal.z + 1);
    }
    else if (normal.x - normal.y + normal.z >= distance)
    {
        // case 1
        assert(assertBuffer, -normal.x + normal.y + normal.z >= distance);

        float tmp = normal.x + normal.y - normal.z;

        // case 1a
        volume = -cb(distance + tmp);

        if (tmp >= distance)
        {
            // case 1b
            volume -= cb(distance - tmp);
        }

        volume = 4 * (distance / normal.z + 1) + volume / (6 * normal.x * normal.y * normal.z);
    }
    else if (-normal.x + normal.y + normal.z >= distance)
    {
        // case 2
        assert(assertBuffer, normal.x + normal.y + normal.z >= distance);
        volume = 8 - (sq(normal.x) / 3 + sq(normal.y + normal.z - distance)) / (normal.y * normal.z);
    }
    else if (normal.x + normal.y + normal.z >= distance)
    {
        // case 3
        volume = 8 - cb(normal.x + normal.y + normal.z - distance) / (6 * normal.x * normal.y * normal.z);
    }
    else
    {
        // case 4
        volume = 8;
    }

    // Scale back to a unit cube
    volume /= 8;

    if (negativeDist)
        return 1 - volume;
    else
        return volume;
}

#ifdef CODECAD_TEST
__kernel void test_mass_properties_unbounding_volume(float value, __global float* volume)
{
    *volume = get_unbounding_volume(value, 1);
}

__kernel void test_mass_properties_cube_halfspace_volume(float4 plane, __global float* volume,
                                                         __global AssertBuffer* restrict assertBuffer)
{
    *volume = cube_halfspace_volume(plane.xyz, plane.w, assertBuffer);
}
#endif

/** Returns a float between 0 and 1 indicating how close we are to running out
 * of floating point precision with step size, relative to current coords. */
static float desperation_factor(float3 coords, float s)
{
    coords = fabs(coords);
    float scale = max(max(coords.x, coords.y), coords.z);
    float logarithm = native_log2(s / scale); // Precision is not too important here
    float p = 0.9;
    return max(0.0f, -logarithm / ((1 - p) * (FLT_MANT_DIG - 1)) - p / (1 - p));
}

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
 * bonusAllowedError: Error allowed per unit of block volume.
 * planeSplitFudgeFactor: cell size times this value gets added to calculated error
 *                        of the plane split approximation. This serves either
 *                        to compensate for over-eagerness of the plane split
 *                        approximation or to completely disable it.
 * locations: xyz coordinates of a corner of tree nodes to process,
 *            w is size of the child node (w * TREE_SIZE is size of this node).
 *            get_global_size(0) items large.
 * tempLocations: locations get copied in here for use in prepare_next stage
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
                                       float planeSplitFudgeFactor,

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

    // Verify that the input location is valid and invalidate the used input
    assert(assertBuffer, !isnan(allowedErrorFromLocation));
    WHEN_ASSERT(allowedErrors[index] = NAN);

    float3 cellOrigin = location.xyz;
    float s = location.w;
    float s3 = s * s * s; // This is volume of a child sub node

    float desperationFactor = desperation_factor(cellOrigin, s);
    bonusAllowedError += desperationFactor;
    float allowedError = allowedErrorFromLocation + bonusAllowedError; // == absAllowedErrorPerChild / s3 [units**3 error / units**3 volume]

    assert(assertBuffer, s > 0);
    assert(assertBuffer, all(cellOrigin + s != cellOrigin));

    float volumes[TREE_CHILD_COUNT];
    uint n = 0;
    uint potentialSplitCount = 0;
    float remainingAllowedError = allowedError * TREE_CHILD_COUNT;

    float3 sampledGradient = 0;

    float3 firstDirection;
    float minDirectionDot = 1;
    float minPlaneSplitVolume, maxPlaneSplitVolume;

    #pragma unroll
    for (uint i = 0; i < TREE_SIZE; ++i)
        #pragma unroll
        for (uint j = 0; j < TREE_SIZE; ++j)
            #pragma unroll
            for (uint k = 0; k < TREE_SIZE; ++k)
            {
                float3 ijk = (float3)(i, j, k);
                float3 delta = s * (ijk - (TREE_SIZE - 1) / 2);
                float3 center = cellOrigin + ijk * s + s / 2;
                float4 value = evaluate(shape, center);

                sampledGradient += select(0, value.w / delta, delta != 0);

                float unboundingVolume = get_unbounding_volume(value.w, s);
                // Preparations for splitting sub nodes
                volumes[n] = unboundingVolume;
                float error = (1 - fabs(unboundingVolume)) / 2.0;
                if (error < allowedError)
                    remainingAllowedError -= error;
                else
                    potentialSplitCount++;

                assert(assertBuffer, length(value.xyz) < 1.0001);
                assert(assertBuffer, length(value.xyz) > 0.9999);
                float3 direction = value.xyz;
                // value.w is distance of the plane from the sub-cell, we need
                // the compensation added is to move the origin of the distance to the cell center
                float distance = value.w - dot(delta, direction);
                // .. and to rescale it into a unit cube
                float planeSplitVolume = cube_halfspace_volume(direction, -distance / (s * TREE_SIZE), assertBuffer);
                if (n == 0)
                {
                    firstDirection = direction;
                    minPlaneSplitVolume = planeSplitVolume;
                    maxPlaneSplitVolume = planeSplitVolume;
                }
                else
                {
                    minDirectionDot = min(minDirectionDot, dot(direction, firstDirection));
                    minPlaneSplitVolume = min(minPlaneSplitVolume, planeSplitVolume);
                    maxPlaneSplitVolume = max(maxPlaneSplitVolume, planeSplitVolume);
                }

                n++;
            }
    sampledGradient /= TREE_SIZE * TREE_SIZE * (TREE_SIZE / 2);
    //printf("sampledGradient: %e, %e, %e (%e)\n", sampledGradient.x, sampledGradient.y, sampledGradient.z, length(sampledGradient));

    float allowedError2 = remainingAllowedError / potentialSplitCount;
    assert(assertBuffer, allowedError2 * (1 + 1e-6) >= allowedError);
    allowedError2 = max(allowedError2, allowedError);

    float integralOne = 0;
    float3 integralX = 0;
    float3 integralXX = 0;
    float3 integralXY = 0;
    float integralAll = 0;
    float totalError = 0;
    uint splitCount = 0;
    uint splitMask = 0;

    float planeSplitError = (maxPlaneSplitVolume - minPlaneSplitVolume) / 2 +
                            planeSplitFudgeFactor * TREE_SIZE * s;

    if (minDirectionDot > PLANE_SPLIT_MIN_DOT && planeSplitError < allowedError)
    {
        // Error by approximating the cell as being divided by a single plane is acceptable

        float3 center = cellOrigin + (float3)(0.5 * s * TREE_SIZE);
        float volume = (maxPlaneSplitVolume + minPlaneSplitVolume) / 2;

        integralAll = TREE_CHILD_COUNT * s3;
        integralOne = integralAll * volume;
        integralX = integralOne * center;
        integralXX = integralOne * (center * center + s * s * TREE_SIZE * TREE_SIZE / 12);
        integralXY = integralOne * center.zxy * center.yzx;
        totalError += (planeSplitError - allowedErrorFromLocation) * integralAll;
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

        if (splitCount > 0 && keepRemainingError)
        {
            // We use everything we've been given, remaining allowed error gets passed to children

            float availableError = TREE_CHILD_COUNT * allowedError;
            if (allowedErrorFromLocation > 0)
                availableError -= splitCount * bonusAllowedError; // Don't get any bonus for sub-cells that still need splitting

            float allowedError3 = (availableError - errorSum) / splitCount;

            tempAllowedErrors[get_global_id(0)] = allowedError3;
            totalError = (availableError - TREE_CHILD_COUNT * allowedErrorFromLocation) * s3;
        }
        else
        {
            // We use only the errror actually used by splits, nothing else gets passed to children
            totalError = (errorSum - TREE_CHILD_COUNT * allowedErrorFromLocation) * s3;
            if (splitCount > 0)
                tempAllowedErrors[get_global_id(0)] = 0;
        }
        tempLocations[get_global_id(0)] = location;
    }

    if (keepRemainingError && allowedErrorFromLocation == 0)
        assert(assertBuffer, totalError <= allowedError * TREE_CHILD_COUNT * s3);
    else
        assert(assertBuffer, totalError <= integralAll * bonusAllowedError * (1 + 1e-3));

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

                index %= locationQueueSize;

                // Check that we're not overwriting any existing valid data
                assert(assertBuffer, isnan(allowedErrors[index]));

                locations[index] = (float4)(corner, nextS);
                allowedErrors[index] = allowedError;
                ++index;
            }
}

// vim: filetype=c
