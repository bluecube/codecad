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
                               __global uint* intersectingCounter,
                               __global uchar4* list)
{
    uint3 coords = (uint3)(get_global_id(0),
                           get_global_id(1),
                           get_global_id(2));

    float3 point = as_float3(boxCorner) + boxStep * convert_float3(coords);
    float value = evaluate(scene, point).w;

    if (value > -distanceThreshold && value < distanceThreshold)
    {
        // Possibly intersecting the shape surface, needs to be split again
        list[atomic_inc(intersectingCounter)] = (uchar4)(coords.x, coords.y, coords.z, 0);
    }
}

// vim: filetype=c
