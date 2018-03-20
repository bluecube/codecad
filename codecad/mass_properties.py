import collections
import functools
import operator
import math
import time
import logging

import numpy
import flags
import pyopencl
from pyopencl import cltypes

from . import util
from . import cl_util
from . import subdivision
from .cl_util import opencl_manager
from . import nodes


class MassPropertiesOptions(flags.Flags):
    no_monte_carlo_leafs = ()


TREE_SIZE = 3  # Cube root of tree children count (2 => octree)
MAX_WORK_SIZE = 2**16
TREE_CHILD_COUNT = TREE_SIZE**3
LOCATION_QUEUE_SIZE = 2**26 # More is faster, but we want even low end gpus to not have any problems

assert TREE_CHILD_COUNT < 32, "Set of node children must be representable by uint bitmask"
assert TREE_CHILD_COUNT * MAX_WORK_SIZE <= 2**23, "Maximum number of split nodes per run must be exactly representable as float"

c = opencl_manager.add_compile_unit()
c.append_define("TREE_SIZE", TREE_SIZE)
c.append_file("mass_properties.cl")

logger = logging.getLogger(__name__)


class MassProperties(collections.namedtuple("MassProperties", "volume centroid inertia_tensor volume_error")):
    """ Contains volume of a body, position of its centroid and its inertia tensor.
    The inertia tensor is referenced to the centroid, not origin! """
    __slots__ = ()


def mass_properties(shape,
                    abs_allowed_error=1e-1, rel_allowed_error=1e-4,
                    options=MassPropertiesOptions.no_flags):
    """ Calculate volume, centroid and inertia tensor of the shape.
    Iteratively subdivides the shape until
    abs(actual_volume - computed_volume) < allowed_error """
    # Inertia tensor info:
    # http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node64.html

    #TODO: Default absolute error assumes that the model dimensions are in milimeters.
    # We should have a global default scale defined somewhere and start from that.

    if shape.dimension() != 3:
        raise ValueError("2D objects are not supported yet")
    if abs_allowed_error <= 0:
        raise ValueError("Absolute allowed error must be positive")
    if rel_allowed_error <= 0:
        raise ValueError("Absolute allowed error must be positive")

    opencl_manager.get_program() # Force build here so that it doesn't skew timings later

    box = shape.bounding_box()
    box_size = box.size()
    initial_step_size = box_size.max() / TREE_SIZE
    volume_to_process = box_size.max()**3

    with cl_util.BufferList() as buffer_list:
        program_buffer = nodes.make_program_buffer(shape)
        buffer_list.add(program_buffer)
        locations = cl_util.Buffer(cltypes.float4, LOCATION_QUEUE_SIZE, pyopencl.mem_flags.READ_WRITE)
        buffer_list.add(locations)
        allowed_errors = cl_util.Buffer(cltypes.float, LOCATION_QUEUE_SIZE, pyopencl.mem_flags.READ_WRITE)
        buffer_list.add(allowed_errors)
        temp_locations = cl_util.Buffer(cltypes.float4, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
        buffer_list.add(temp_locations)
        temp_allowed_errors = cl_util.Buffer(cltypes.float, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
        buffer_list.add(temp_allowed_errors)
        integral1 = cl_util.Buffer(cltypes.float4, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
        buffer_list.add(integral1)
        integral2 = cl_util.Buffer(cltypes.float4, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
        buffer_list.add(integral2)
        integral3 = cl_util.Buffer(cltypes.float4, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
        buffer_list.add(integral3)
        split_counts = cl_util.Buffer(cltypes.uint, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
        buffer_list.add(split_counts)
        split_masks = cl_util.Buffer(cltypes.uint, MAX_WORK_SIZE, pyopencl.mem_flags.READ_WRITE)
        buffer_list.add(split_masks)
        output_buffer = cl_util.Buffer(cltypes.float, 13, pyopencl.mem_flags.WRITE_ONLY | pyopencl.mem_flags.ALLOC_HOST_PTR)
        buffer_list.add(output_buffer)
        assert_buffer = cl_util.AssertBuffer()
        buffer_list.add(assert_buffer)

        # Initialize the first location
        ev = allowed_errors.enqueue_zero_fill_compatible()
        prepare_next_ev = locations.enqueue_write(box.a.as_float4(initial_step_size), wait_for=[ev])
        first_location = 0
        location_count = 1

        if MassPropertiesOptions.no_monte_carlo_leafs in options:
            monte_carlo_leaf_threshold = cltypes.float(0)
        else:
            monte_carlo_leaf_threshold = cltypes.float("inf")

        integral_one = util.KahanSummation()
        integral_x = util.KahanSummation()
        integral_y = util.KahanSummation()
        integral_z = util.KahanSummation()
        integral_xx = util.KahanSummation()
        integral_yy = util.KahanSummation()
        integral_zz = util.KahanSummation()
        integral_xy = util.KahanSummation()
        integral_xz = util.KahanSummation()
        integral_yz = util.KahanSummation()
        integral_all = util.KahanSummation()
        total_error = util.KahanSummation()

        time_computing = util.KahanSummation()
        time_summing = util.KahanSummation()
        iteration_count = 0
        locations_processed = 0

        prev_prepare_next_ev = None
        prev_allowed_error_per_volume = 0

        start_time = time.time()

        # If number of locations before an iteration is larger than this, the iteration
        # must be done in DFS mode
        # Set up so that the DFS mode can safely cross the whole dynamic range of float
        # (as there can be no more splitting after that).
        dfs_switch_threshold = LOCATION_QUEUE_SIZE - \
                               math.ceil(math.log(2**23, TREE_SIZE)) * MAX_WORK_SIZE * (TREE_CHILD_COUNT - 1)

        while location_count > 0:
            #logger.debug("*********************************")
            work_size = min(MAX_WORK_SIZE, location_count)

            #logger.debug("first_location: %i, location_count: %i, LOCATION_QUEUE_SIZE: %i, work size: %i",
            #             first_location, location_count, LOCATION_QUEUE_SIZE, work_size)

            if location_count <= dfs_switch_threshold:
                # If we have enough free space in the buffer, we run in BFS mode.
                # This has the advantage of possibly chopping off large areas that are 
                # covered as whole, which will let the allowed error to be distributed over
                # the whole calculation better. Also BFS doesn't have the tendency to get
                # stuck in hard to solve corners.

                start_offset = first_location
                first_location_move = work_size
                bfs_mode = True
            else:
                # Otherwise we run in a DFS mode, because it tends to completely solve
                # an area before continuing elsewhere and thus saves buffer space.
                # We also limit work size for the case where we couldn't handle all
                # the nodes expanding at once.

                # Limit the work size if the location buffer is almost full
                work_size_limit = (LOCATION_QUEUE_SIZE - location_count) // (TREE_CHILD_COUNT - 1)
                if work_size_limit == 0:
                    raise Exception("Maximum location buffer size exceeded. "
                                    "This is likely a bug in CodeCad. "
                                    "As a workaround try specifying lower precision.")
                elif work_size_limit < work_size:
                    work_size = work_size_limit
                    logger.warning("Limiting work size to %i to avoid exceeding location buffer size. "
                                   "This is likely a bug in CodeCad, but should only affect performance.",
                                   work_size)

                start_offset = (first_location + location_count - work_size) % LOCATION_QUEUE_SIZE
                first_location_move = 0
                bfs_mode = False

            assert work_size > 0
            assert work_size <= MAX_WORK_SIZE
            assert work_size <= location_count

            allowed_error = max(abs_allowed_error, integral_one.result * rel_allowed_error)

            if (allowed_error - total_error.result) > (volume_to_process - integral_all.result):
                # If we have more allowed error left than unprocessed volume,
                # we can ignore the locations remaining in the queue.
                # We don't count unprocessed volume with weight 0.5 (which would
                # decrease the error requirement), because while compensating the
                # total volume for this would be easy enough, I don't know how to
                # compensate centroid and inertia matrix. (TODO)
                break
            if volume_to_process > integral_all.result:
                allowed_error_per_volume = (allowed_error - total_error.result) / (volume_to_process - integral_all.result)
                assert allowed_error_per_volume >= prev_allowed_error_per_volume
                prev_allowed_error_per_volume = allowed_error_per_volume

            evaluate_ev = opencl_manager.k.mass_properties_evaluate([work_size], None,
                                                                    program_buffer,
                                                                    cltypes.uint(start_offset),
                                                                    cltypes.uint(LOCATION_QUEUE_SIZE),
                                                                    cltypes.float(allowed_error_per_volume),
                                                                    cltypes.uint(0 if bfs_mode else 1), # keepRemainingError
                                                                    monte_carlo_leaf_threshold,
                                                                    locations, allowed_errors, temp_locations, temp_allowed_errors,
                                                                    integral1, integral2, integral3,
                                                                    split_counts, split_masks,
                                                                    assert_buffer,
                                                                    wait_for=[prepare_next_ev])
            sum_ev = opencl_manager.k.mass_properties_sum([work_size], None,
                                                          integral1, integral2, integral3,
                                                          split_counts,
                                                          assert_buffer,
                                                          wait_for=[evaluate_ev])
            output_ev = opencl_manager.k.mass_properties_output([1], None,
                                                                cltypes.uint(work_size),
                                                                integral1, integral2, integral3,
                                                                split_counts,
                                                                output_buffer,
                                                                wait_for=[sum_ev])

            location_count -= work_size
            first_location = (first_location + first_location_move) % LOCATION_QUEUE_SIZE

            prepare_next_ev = opencl_manager.k.mass_properties_prepare_next([work_size], None,
                                                                            cltypes.uint((first_location + location_count) % LOCATION_QUEUE_SIZE),
                                                                            cltypes.uint(LOCATION_QUEUE_SIZE),
                                                                            locations, allowed_errors, temp_locations, temp_allowed_errors,
                                                                            split_counts, split_masks,
                                                                            assert_buffer,
                                                                            wait_for=[sum_ev])
            with output_buffer.map(pyopencl.map_flags.READ, wait_for=[output_ev]) as mapped:
                integral_one += mapped[0]
                integral_x += mapped[1]
                integral_y += mapped[2]
                integral_z += mapped[3]
                integral_xx += mapped[4]
                integral_yy += mapped[5]
                integral_zz += mapped[6]
                integral_yz += mapped[7]
                integral_xz += mapped[8]
                integral_xy += mapped[9]
                integral_all += mapped[10]
                total_error += mapped[11]
                splits = int(mapped[12])
            # assert_buffer.check()

            assert splits <= work_size * TREE_CHILD_COUNT
            location_count += splits

            # Computing time spent in OpenCL is a bit tricky here.
            # Because we try to give stage 4 as much time as possible, it might be
            # still running at this point, so we need to measure its time times one step behind.
            # The other stages are calculated immediately
            time_computing += cl_util.event_time_spent(evaluate_ev) + cl_util.event_time_spent(sum_ev) + cl_util.event_time_spent(output_ev)
            time_summing += cl_util.event_time_spent(sum_ev)
            if prev_prepare_next_ev is not None:
                time_computing += cl_util.event_time_spent(prev_prepare_next_ev)
            iteration_count += 1
            locations_processed += work_size

            prev_prepare_next_ev = prepare_next_ev

            progress = integral_all.result / volume_to_process

            time_spent = time.time() - start_time
            # logger.debug("Iteration finished: %.4f%%, total error used: %.2f, split count: %i, location count: %i, "
            #              "time elapsed: %.2f s, OpenCL time: %.2f s, OpenCL utilization: %.3f, "
            #              "iteration count: %i (%.1f iterations/s), "
            #              "locations processed: %.2e (%.0f locations/iteration, %.0f locations/s) "
            #              "%s mode, bonus allowed error: %f",
            #              100 * progress, total_error.result, splits, location_count,
            #              time_spent, time_computing.result, time_computing.result / time_spent,
            #              iteration_count, iteration_count / time_spent,
            #              locations_processed, locations_processed / iteration_count, locations_processed / time_spent,
            #              "BFS" if bfs_mode else "DFS", allowed_error_per_volume)

        total_error += max(0, volume_to_process - integral_all.result) # All remaining volume is error

        assert_buffer.check()

        prev_prepare_next_ev.wait() # Wait for the last event to finalize timing stats
        time_computing += cl_util.event_time_spent(prev_prepare_next_ev)

        time_spent = time.time() - start_time
    logger.info("Mass properties finished: "
                "time elapsed: %.2f s, OpenCL time: %.2f s, OpenCL utilization: %.3f, OpenCL relative sum time: %.3f, "
                "iteration count: %i (%.1f iterations/s), "
                "locations processed: %.2e (%.0f locations/iteration, %.0f locations/s)",
                time_spent, time_computing.result, time_computing.result / time_spent, time_summing.result / time_computing.result,
                iteration_count, iteration_count / time_spent,
                locations_processed, locations_processed / iteration_count, locations_processed / time_spent)

    # Unwrap the integral values from the KahanSummation objects
    integral_one = integral_one.result
    integral_x = integral_x.result
    integral_y = integral_y.result
    integral_z = integral_z.result
    integral_xx = integral_xx.result
    integral_yy = integral_yy.result
    integral_zz = integral_zz.result
    integral_xy = integral_xy.result
    integral_xz = integral_xz.result
    integral_yz = integral_yz.result

    volume = integral_one
    if volume == 0:
        return MassProperties(0, util.Vector.splat(0), numpy.zeros((3, 3)))
    centroid = util.Vector(integral_x, integral_y, integral_z) / integral_one

    # Prepare the inertia tensor based on the integrals, but make it referenced
    # to object centroid instead of origin
    shifted_integral_xx = integral_xx - 2 * centroid.x * integral_x + centroid.x * centroid.x * integral_one
    shifted_integral_yy = integral_yy - 2 * centroid.y * integral_y + centroid.y * centroid.y * integral_one
    shifted_integral_zz = integral_zz - 2 * centroid.z * integral_z + centroid.z * centroid.z * integral_one
    shifted_integral_xy = integral_xy - centroid.x * integral_y - centroid.y * integral_x + centroid.x * centroid.y * integral_one
    shifted_integral_xz = integral_xz - centroid.x * integral_z - centroid.z * integral_x + centroid.x * centroid.z * integral_one
    shifted_integral_yz = integral_yz - centroid.y * integral_z - centroid.z * integral_y + centroid.y * centroid.z * integral_one
    I_xx = shifted_integral_yy + shifted_integral_zz
    I_yy = shifted_integral_xx + shifted_integral_zz
    I_zz = shifted_integral_xx + shifted_integral_yy
    I_xy = -shifted_integral_xy
    I_xz = -shifted_integral_xz
    I_yz = -shifted_integral_yz
    inertia_tensor = numpy.array([[I_xx, I_xy, I_xz],
                                  [I_xy, I_yy, I_yz],
                                  [I_xz, I_yz, I_zz]])

    return MassProperties(volume, centroid, inertia_tensor, total_error.result)
