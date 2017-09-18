"""
Test that the directed signed distance function behaves as expected.
"""
# TODO: This module collects some of the testing data from other test modules
# because it was the easiest thing to do, but it would be nicer if the other modules
# could inject their data into these tests

import pytest

import pyopencl
import pyopencl.cltypes
import numpy

import codecad

import test_simple2d

codecad.opencl_manager.instance.add_compile_unit().append_file("test_dsdf.cl")

nonconvex = codecad.shapes.polygon2d([(0, 0), (4, 6), (4, -2), (-4, -2), (-4, 6)])
csg_thing = codecad.shapes.cylinder(h=5, d=2, symmetrical=False).rotated((1, 2, 3), 15) & \
            codecad.shapes.sphere(d=3) + \
            codecad.shapes.box(2).translated(2, 0, 0)

shapes_2d = {"rectangle": codecad.shapes.rectangle(2, 4),
             "circle": codecad.shapes.circle(4),
             "nonconvex_offset_outside": nonconvex.offset(2),
             "nonconvex_offset_inside1": nonconvex.offset(-0.9),
             "nonconvex_offset_inside2": nonconvex.offset(-1.1),  # This one separates into two volumes
             "nonconvex_shell1": nonconvex.shell(1),
             "nonconvex_shell2": nonconvex.shell(2.5),  # Has two holes
             "gear": codecad.shapes.gears.InvoluteGear(20, 0.5),
             }
shapes_2d.update(("polygon2d_" + k, codecad.shapes.polygon2d(v)) for k, v in test_simple2d.valid_polygon_cases.items())
params_2d = [pytest.param(v, id=k) for k, v in sorted(shapes_2d.items())]

shapes_3d = {"sphere": codecad.shapes.sphere(4),
             "box": codecad.shapes.box(2, 3, 5),
             "drunk_box": codecad.shapes.box(2, 3, 5).rotated((7, 11, 13), 17),
             "translated_cylinder": codecad.shapes.cylinder(d=3, h=5).translated(0, 1, -1),
             "csg_thing": csg_thing,
             "torus": codecad.shapes.circle(d=4).translated(3, 0).revolved(),
             "empty_intersection": codecad.shapes.sphere() & codecad.shapes.sphere().translated(5, 0, 0)}
params_3d = [pytest.param(v, id=k) for k, v in sorted(shapes_3d.items())]

with_2d_shapes = pytest.mark.parametrize("dsdf_common", params_2d, indirect=True)
with_all_shapes = pytest.mark.parametrize("dsdf_common", params_2d + params_3d, indirect=True)


@pytest.fixture(scope="module")
def dsdf_common(request):
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
    assert all(a > -s and b < s for a, b, s in zip(box.a, box.b, size)), \
        "Shape bounding box must fit into the fixed box"

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

    b = codecad.util.cl_util.Buffer(codecad.opencl_manager.instance.queue,
                                    (pyopencl.cltypes.float4, (2,)),
                                    size,
                                    pyopencl.mem_flags.READ_WRITE)

    ev = codecad.opencl_manager.instance.k.grid_eval_twice(size, None,
                                                           scene_buffer,
                                                           box_corner.as_float4(),
                                                           numpy.float32(box_step),
                                                           b.buffer)

    b.read(wait_for=[ev])
    return b


@pytest.fixture(scope="module")
def actual_distance_buffer(dsdf_common, eval_buffer):
    size = dsdf_common["size"]
    box_step = dsdf_common["box_step"]

    b = codecad.util.cl_util.Buffer(codecad.opencl_manager.instance.queue,
                                    pyopencl.cltypes.float,
                                    size,
                                    pyopencl.mem_flags.WRITE_ONLY)

    ev = codecad.opencl_manager.instance.k.actual_distance_to_surface(size, None,
                                                                      numpy.float32(box_step),
                                                                      eval_buffer.buffer,
                                                                      b.buffer)
    b.read(wait_for=[ev])
    return b


@pytest.fixture(scope="module")
def direction_buffer(dsdf_common):
    size = dsdf_common["size"]
    box_corner = dsdf_common["box_corner"]
    box_step = dsdf_common["box_step"]
    scene_buffer = dsdf_common["scene_buffer"]
    epsilon = dsdf_common["epsilon"]

    b = codecad.util.cl_util.Buffer(codecad.opencl_manager.instance.queue,
                                    (pyopencl.cltypes.float3, (2,)),
                                    size,
                                    pyopencl.mem_flags.WRITE_ONLY)

    ev = codecad.opencl_manager.instance.k.estimate_direction(size, None,
                                                              scene_buffer,
                                                              box_corner.as_float4(),
                                                              numpy.float32(box_step),
                                                              numpy.float32(epsilon),
                                                              b.buffer)
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
        assert v["x"]**2 + v["y"]**2 + v["z"]**2 == pytest.approx(1)


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
def test_dsdf_valid_approximation(eval_buffer, actual_distance_buffer):
    """ Distance returned is always lower or equal to the actual distance to surface.
    This condition is necessary for the raycasting algorithm to work. """

    for v, actual_distance in zip(eval_buffer.array, actual_distance_buffer.array):
        assert v[0]["w"] <= actual_distance + 1e-5


@with_all_shapes
def test_dsdf_direction(eval_buffer, direction_buffer, dsdf_common):
    """ Direction calculated by `evaluate` is close to one calculated by finite differences """

    for v, direction in zip(eval_buffer.array, direction_buffer.array):
        from_eval = codecad.util.Vector(v[0]["x"], v[0]["y"], v[0]["z"])
        from_differences1 = codecad.util.Vector(direction[0]["x"], direction[0]["y"], direction[0]["z"])
        from_differences2 = codecad.util.Vector(direction[1]["x"], direction[1]["y"], direction[1]["z"])

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
