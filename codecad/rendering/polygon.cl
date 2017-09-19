/* Encode index with information about overflowing the block.
 * If coord is outside of a block (size specified by get_global_id),
 * this function prepends information about where this overflow happened on a
 * boundary of the block. */
static uint encode_index(int2 coord, uint index)
{
    const uint indexSize = 20; // max size of index in bits

    index = index & ((1 << indexSize) - 1);
        // index must be masked because an invalid value might be larger than
        // the allowed range

    // Find which coordinate is the outside one and swap it to x
    bool y;
    int2 swappedCoord;
    if (coord.x < 0 || coord.x >= get_global_size(0))
    {
        y = false;
        swappedCoord = coord;
    }
    else if (coord.y < 0 || coord.y >= get_global_size(1))
    {
        y = true;
        swappedCoord = coord.yx;
    }
    else
        return index; // Coordinates are inside the block. no need to do anything

    return 0x80000000 | // out of block flag
           (y ? 0x40000000 : 0) | // x/y
           (swappedCoord.x < 0 ? 0x20000000: 0) | // negative
           (swappedCoord.y << indexSize) | // row/column at which the overflow happened
           index;
}

static float2 place_vertex(float2 cornerPositions[3],
                           float4 cornerValues[3])
{
    // Instead of mass point we use average of corner positions weighted by
    // (1/(1 + distance)) It's much simpler and gives only a tiny bit worse result.
    float2 average = (float2)(0, 0);
    float2 center = (float2)(0, 0);
    float weight = 0;
    for (int i = 0; i < 3; ++i)
    {
        float w = 1 / (1 + fabs(cornerValues[i].w));
        average += cornerPositions[i] * w;
        center += cornerPositions[i];
        weight += w;
    }
    average /= weight;
    center /= 3;

    // Run gradient search with fixed number of iterations
    float2 p = average; // The candidate point
    for (int i = 0; i < 8; ++i)
    {
        float2 toCenter = p - average;
        float2 gradient = 2 * toCenter;
        float residualSum = dot(toCenter, toCenter);
        for (int j = 0; j < 3; ++j)
        {
            float2 surfaceNormal = cornerValues[j].xy;
            // If the shape is properly 2D, then the gradient is inside Z=0 plane.
            // so that surfaceNormal will be already normalized and cornerValues[j].w
            // will also be correct.

            float tmp = dot(surfaceNormal, p - cornerPositions[j]) + cornerValues[j].w;
            residualSum += tmp * tmp;
            gradient += surfaceNormal * tmp;
        }

        if (residualSum < 1e-3)
            break;

        float gradientLengthSquared = dot(gradient, gradient);

        if (gradientLengthSquared < 1e-8)
            break;

        p -= gradient * (residualSum / gradientLengthSquared);
    }
    return p;
}

__kernel void process_polygon(float2 boxCorner, float boxStep,
                              __global float4* corners, /* Input, evaluated shape on grid corners */
                              __global float2* vertices, /* Output, vertex coordinates corresponding to each cell. */
                              __global uint* links, /* Output, encoded indices of cells that follows this one along the edge or -1 */
                              __global uint* starts, /* Output, encoded indices of cells that start non-looped chains
                                    Overflow specification in start index says should match overflow spec in links of
                                    a neighboring block to connect the two */
                              __global uint* startCounter /* Output, count of items in starts */)
{
    uint2 offsets[3] = {{0, 0}, {1, 1}, {get_global_id(2), 1 - get_global_id(2)}};
    uint cellType = 0;

    for (uint i = 0; i < 3; ++i)
    {
        uint2 coords = (uint2)(get_global_id(0), get_global_id(1)) + offsets[i];
        size_t cornerIndex = INDEX2(get_global_size(0) + 1, coords.x, coords.y);
        cellType = cellType << 1 | (corners[cornerIndex].w <= 0 ? 1 : 0);

    }
    uint index = INDEX3_GG;

    if (cellType == 0 || cellType == 7)
    {
        // This cell is completely inside or completely outside, nothing to generate
        links[index] = -1;
        return;
    }

    uint tmpCellType = cellType;

    int2 forwardCoords;
    int2 reverseCoords;

    bool backwards = cellType == 3 || cellType == 5 || cellType == 6;
    if (backwards)
        cellType = 7 - cellType; // Flip inside / outside flags in cellType
    bool flip = get_global_id(2) == 1;
    if (flip)
        backwards = !backwards;

    switch (cellType)
    {
        case 1:
            forwardCoords = (int2)(0, 1);
            reverseCoords = (int2)(-1, 0);
            break;
        case 2:
            forwardCoords = (int2)(0, 0);
            reverseCoords = (int2)(0, 1);
            break;
        case 4:
            forwardCoords = (int2)(-1, 0);
            reverseCoords = (int2)(0, 0);
            break;
    }

    if (backwards)
    {
        int2 tmp = forwardCoords;
        forwardCoords = reverseCoords;
        reverseCoords = tmp;
    }
    if (flip)
    {
        forwardCoords = forwardCoords.yx;
        reverseCoords = reverseCoords.yx;
    }

    forwardCoords += (int2)(get_global_id(0), get_global_id(1));
    reverseCoords += (int2)(get_global_id(0), get_global_id(1));

    links[index] = encode_index(forwardCoords,
                                INDEX3_G(forwardCoords.x,
                                         forwardCoords.y,
                                         1 - get_global_id(2)));

    uint startIndex = encode_index(reverseCoords, index);
    if (startIndex & 0x80000000) // If the overflow flag is set, then this is the start of an open chain
        starts[atomic_inc(startCounter)] = startIndex ^ 0x20000000; // Flip the +/- flag to match the end corresponding end of a chain


    float2 cornerPositions[3];
    float4 cornerValues[3];
    for (uint i = 0; i < 3; ++i)
    {
        uint2 coords = (uint2)(get_global_id(0), get_global_id(1)) + offsets[i];
        size_t cornerIndex = INDEX2(get_global_size(0) + 1, coords.x, coords.y);
        cornerPositions[i] = boxCorner + convert_float2(coords) * boxStep;
        cornerValues[i] = corners[cornerIndex];
    }
    vertices[index] = place_vertex(cornerPositions, cornerValues);
}

// vim: filetype=c
