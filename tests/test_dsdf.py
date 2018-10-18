# pylint: disable=redefined-outer-name

"""
Test that the directed signed distance function behaves as expected.
"""
import pytest
import pyopencl
import pyopencl.cltypes
import numpy

import codecad

import data

codecad.cl_util.opencl_manager.add_compile_unit().append_resource("test_dsdf.cl")

with_2d_shapes = pytest.mark.parametrize("dsdf_common", data.params_2d, indirect=True)
with_all_shapes = pytest.mark.parametrize(
    "dsdf_common", data.params_2d + data.params_3d, indirect=True
)


@pytest.fixture(scope="module")
def dsdf_common(request):
    # pylint: disable=possibly-unused-variable

    shape = request.param

    size = 16

    if shape.dimension() == 3:
        size = (size, size, size)
    else:
        size = (size, size, 3)

    box_corner = -codecad.util.Vector(*size) / 2
    box_step = 1
    epsilon = 0.05

    box = shape.bounding_box()
    assert all(
        a > -s and b < s for a, b, s in zip(box.a, box.b, size)
    ), "Shape bounding box must fit into the fixed box"

    scene_buffer = codecad.nodes.make_program_buffer(shape)

    # import codecad.rendering.matplotlib_slice as ms
    # ms.render_slice(shape, 0.1)

    return locals()


@pytest.fixture(scope="module")
def eval_buffer(dsdf_common):
    size = dsdf_common["size"]
    box_corner = dsdf_common["box_corner"]
    box_step = dsdf_common["box_step"]
    scene_buffer = dsdf_common["scene_buffer"]

    b = codecad.cl_util.Buffer(
        (pyopencl.cltypes.float4, (2,)), size, pyopencl.mem_flags.READ_WRITE
    )

    ev = codecad.cl_util.opencl_manager.k.grid_eval_twice(
        size, None, scene_buffer, box_corner.as_float4(), numpy.float32(box_step), b
    )

    b.read(wait_for=[ev])
    return b


@pytest.fixture(scope="module")
def actual_distance_buffer(dsdf_common, eval_buffer):
    size = dsdf_common["size"]
    box_step = dsdf_common["box_step"]

    b = codecad.cl_util.Buffer(
        pyopencl.cltypes.float, size, pyopencl.mem_flags.WRITE_ONLY
    )

    ev = codecad.cl_util.opencl_manager.k.actual_distance_to_surface(
        size, None, numpy.float32(box_step), eval_buffer, b
    )
    b.read(wait_for=[ev])
    return b


@pytest.fixture(scope="module")
def direction_buffer(dsdf_common):
    size = dsdf_common["size"]
    box_corner = dsdf_common["box_corner"]
    box_step = dsdf_common["box_step"]
    scene_buffer = dsdf_common["scene_buffer"]
    epsilon = dsdf_common["epsilon"]

    b = codecad.cl_util.Buffer(
        (pyopencl.cltypes.float3, (2,)), size, pyopencl.mem_flags.WRITE_ONLY
    )

    ev = codecad.cl_util.opencl_manager.k.estimate_direction(
        size,
        None,
        scene_buffer,
        box_corner.as_float4(),
        numpy.float32(box_step),
        numpy.float32(epsilon),
        b,
    )
    b.read(wait_for=[ev])
    return b


@with_2d_shapes
def test_2d_direction(eval_buffer):
    """ All directions of 2D objects must have zero z coordinate """

    for v in numpy.nditer(eval_buffer.array):
        assert v["z"] == 0  # The length must be _exactly_ zero


@with_all_shapes
def test_direction_unit_length(eval_buffer):
    """ All directions must be unit length """

    for v in numpy.nditer(eval_buffer.array):
        assert v["x"] ** 2 + v["y"] ** 2 + v["z"] ** 2 == pytest.approx(1)


@with_all_shapes
@pytest.mark.skip(reason="TODO: Figure out DSDF approximation levels")
def test_dsdf_exact(eval_buffer):
    """ Following the direction ends up on the surface in a single step

    This test conditionally xfails if the DSDF of this shape is not exact.
    `test_dsdf_approximate` ensures that the approximation is correct """

    for v in eval_buffer.array:
        if v[1]["w"] != pytest.approx(0, abs=1e-5):
            pytest.xfail()


@with_all_shapes
@pytest.mark.skip(reason="TODO: Figure out DSDF approximation levels")
def test_dsdf_approximate(eval_buffer):
    """ Following the direction must get closer to the surface with each step """

    for v in eval_buffer.array:
        if v[0]["w"] == pytest.approx(0, abs=1e-5) or abs(v[1]["w"]) < abs(v[0]["w"]):
            pass
        else:
            pytest.xfail()


@with_all_shapes
def test_dsdf_valid_approximation(eval_buffer, actual_distance_buffer, dsdf_common):
    """ Distance returned is always lower or equal to the actual distance to surface.
    This condition is necessary for the raycasting algorithm to work. """

    for index in numpy.ndindex(dsdf_common["size"]):
        assert eval_buffer[index][0]["w"] <= actual_distance_buffer[index] + 1e-5


@with_all_shapes
def test_dsdf_direction(eval_buffer, direction_buffer, dsdf_common):
    """ Direction calculated by `evaluate` is close to one calculated by finite differences """

    for index in numpy.ndindex(dsdf_common["size"]):
        from_eval = codecad.util.Vector(
            eval_buffer[index][0][0], eval_buffer[index][0][1], eval_buffer[index][0][2]
        )
        from_differences1 = codecad.util.Vector(
            direction_buffer[index][0][0],
            direction_buffer[index][0][1],
            direction_buffer[index][0][2],
        )
        from_differences2 = codecad.util.Vector(
            direction_buffer[index][1][0],
            direction_buffer[index][1][1],
            direction_buffer[index][1][2],
        )

        if abs(from_differences1 - from_differences2) > 1e-4:
            # Most likely a dicontinuity and the check doesn't make sense here
            # This value might be a bit fragile
            pass
        else:
            from_differences = (from_differences1 + from_differences2) / 2
            from_differences = from_differences.normalized()

            assert abs(from_eval - from_differences) < 1e-2
            # This check doesn't need to be precise at all, we just need to make
            # sure that the value is not completely off
