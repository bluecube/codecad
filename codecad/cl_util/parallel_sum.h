/** Calculate a permuted inclusive prefix sum of get_local_size(0) uints in parallel.
Input values are passed into the function as the first parameter, results are returned.

Local buffer will be used for temporary values and must have at least
get_local_size(0) items available.

For work item 0 this function returns complete sum of the values.

To use this result for indexing, subtract value from the return value.

Barriers are used inside, so all threads in a workgroup must go through this
function if any one goes through it. */
uint indexing_prefix_sum_helper(uint value, __local uint* buffer);
