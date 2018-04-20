#
#/** Sum get_group_size(0) floating point numbers from arrayIn in parallel,
#starting at position get_group_size(0) * get_group_id(0) and store the result
#in arrayOut[get_group_id(0)].
#
#Buffer will be used for temporary values and must have at least
#get_group_size(0) / 2 items available.
#
#The algorithm works by recursively summing pairs, so it should work fairly well
#with floating point precision. */
#void sum_helper(__local float* restrict buffer,
#                __global float* restrict arrayIn,
#                __global float* restrict arrayOut);

##define SUM_HELPER(TYPE, OPERATION) \
#    TYPE v = arrayIn[get_global_id(0)]; \
#    \
#    if (get_local_id(0) > get_local_size(0) / 2) \
#        buffer[get_local_id(0) - get_local_size(0) / 2] = v; \
#    \
#    for (size_t i = get_local_size(0) / 2; i > 1; i /= 2) \
#    { \
#        barrier(CLK_LOCAL_MEM_FENCE); \
#    \
#        if (get_local_id(0) >= i) \
#            continue; \
#    \
#        TYPE a = v; \
#        TYPE b = buffer[get_local_id(0)]; \
#        v = (OPERATION); \
#    \
#        if (get_local_id(0) >= i / 2) \
#            buffer[get_local_id(0) - i / 2] = v; \
#    } \
#    \
#    barrier(CLK_LOCAL_MEM_FENCE); \
#    \
#    if (get_local_id(0) == 0) \
#    { \
#        TYPE a = v; \
#        TYPE b = buffer[0]; \
#        arrayOut[get_group_id(0)] = (OPERATION); \
#    }
#
#float sum_helper(float restrict value,
#                 __local float* restrict buffer)
#{
#    SUM_HELPER(float, a + b)
#}
#
