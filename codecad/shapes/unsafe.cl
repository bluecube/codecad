uchar repetition_op(__constant float* params, float4* output, float4 coords, float4 unused) {
    output->x = remainder(coords.x, params[0]);
    output->y = remainder(coords.y, params[1]);
    output->z = remainder(coords.z, params[2]);
    return 3;
}
