/** Calculate a permuted prefix sum of get_local_size(0) uints in parallel.
Input values are passed into the function as the first parameter.

Local buffer will be used for results values and must have at least
get_local_size(0) items available.

Barriers are used inside, so all threads in a workgroup must go through this
function if any one goes through it. */
float indexing_prefix_sum_helper(uint value, __local uint* buffer);
