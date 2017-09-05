void repetition_op(float ox, float oy, float oz, float4 coords, float4* output) {
    output->x = remainder(coords.x, ox);
    output->y = remainder(coords.y, oy);
    output->z = remainder(coords.z, oz);
}
