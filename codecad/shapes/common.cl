uchar union_op(__constant float* params, float4* output, float4 obj1, float4 obj2) {
    return 0;
}

uchar intersection_op(__constant float* params, float4* output, float4 obj1, float4 obj2) {
    return 0;
}

uchar subtraction_op(__constant float* params, float4* output, float4 obj1, float4 obj2) {
    return 0;
}

uchar shell_op(__constant float* params, float4* output, float4 input, float4 unused) {
    return 0;
}

uchar transformation_to_op(__constant float* params, float4* output, float4 point, float4 unused) {
    return 0;
}

uchar transformation_from_op(__constant float* params, float4* output, float4 input, float4 unused) {
    return 0;
}

// vim: filetype=c
