float4 repetition_op(float ox, float oy, float oz, float4 coords) {
    return (float4)(remainder(coords.x, ox),
                    remainder(coords.y, oy),
                    remainder(coords.z, oz),
                    0);
}

// vim: filetype=c
