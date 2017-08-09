__kernel void process_polygon(float2 boxCorner, float boxStep,
                              __global float4* corners, /* Input, evaluated shape on grid corners */
                              __global float2* vertices, /* Output, vertex coordinates corresponding to each cell. */
                              __global uint* links, /* Output, indices of cells that follows this one along the edge or -1 */
                              __global uint* starts, /* Output, indices of cells that start non-looped chains */
                              __global uint* startCounter /* Output, count of items in starts */)
{
    uint2 offsets[3] = {{0, 0}, {1, 1}, {get_global_id(2), 1 - get_global_id(2)}};
    float4 values[3];
    uint cellType = 0;

    for (uint i = 0; i < 3; ++i)
    {
        uint2 coords = (uint2)(get_global_id(0), get_global_id(1)) + offsets[i];
        size_t cornerIndex = INDEX2(get_global_size(0) + 1, coords.x, coords.y);

        values[i] = corners[cornerIndex];
        cellType = cellType << 1 | (values[i].w <= 0 ? 1 : 0);

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

    if (forwardCoords.x < 0 ||
        forwardCoords.x >= get_global_size(0) ||
        forwardCoords.y < 0 ||
        forwardCoords.y >= get_global_size(1))
        links[index] = -1; // This is the end of an open chain
    else
        links[index] = INDEX3_G(forwardCoords.x, forwardCoords.y, 1 - get_global_id(2));

    if (reverseCoords.x < 0 ||
        reverseCoords.x >= get_global_size(0) ||
        reverseCoords.y < 0 ||
        reverseCoords.y >= get_global_size(1))
        starts[atomic_inc(startCounter)] = index; // This is the start of an open chain

    //TODO: use QEF to calculate vertex position
    float2 centerOffset = (float2)(1 + get_global_id(2), 2 - get_global_id(2)) / 3;
    vertices[index] = boxCorner +
                      (float2)(get_global_id(0), get_global_id(1)) * boxStep +
                      centerOffset * boxStep;
}

// vim: filetype=c
