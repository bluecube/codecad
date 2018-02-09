import math
import numpy
import scipy.integrate

import pytest
from pytest import approx
import pyopencl
from pyopencl import cltypes

import codecad
import codecad.util

drunk_box_matrix = codecad.util.Quaternion.from_degrees((7, 11, 13), 17).as_matrix()[:3, :3]


# Names added so that test failures have clear identification of the shape
@pytest.mark.parametrize("shape, volume, centroid, inertia_tensor", [
    pytest.param(codecad.shapes.box(1),
                 1, codecad.util.Vector(0, 0, 0),
                 numpy.identity(3) * 2 / 12,
                 id="unit_box"),
    pytest.param(codecad.shapes.cylinder(h=2, r=4, symmetrical=False),
                 math.pi * 32, codecad.util.Vector(0, 0, 1),
                 numpy.diag([(3 * 4**2 + 2**2) / 6, (3 * 4**2 + 2**2) / 6, 4**2]) * math.pi * 16,
                 id="cylinder"),
    pytest.param(codecad.shapes.sphere(d=2),
                 4 * math.pi / 3, codecad.util.Vector(0, 0, 0),
                 numpy.identity(3) * (4 * math.pi / 3) * 2 / 5,
                 id="sphere"),
    pytest.param(codecad.shapes.box(2).translated(-15, 0, 0) + codecad.shapes.box(2).translated(15, 0, 0),
                 16, codecad.util.Vector(0, 0, 0),
                 None,
                 id="two_boxes"),
    pytest.param(codecad.shapes.sphere(r=2) - codecad.shapes.half_space(),
                 2 * math.pi * 2**3 / 3, codecad.util.Vector(0, -6 / 8, 0),
                 None,
                 id="hemisphere"),
    pytest.param(codecad.shapes.sphere(d=2).translated(10, 11, 7),
                 4 * math.pi / 3, codecad.util.Vector(10, 11, 7),
                 None,
                 id="translated_sphere"),
    pytest.param((codecad.shapes.sphere(r=2) - codecad.shapes.half_space()).translated(2, 0, 0).rotated((1, 0, 0), 90),
                 2 * math.pi * 2**3 / 3, codecad.util.Vector(2, 0, -6 / 8),
                 None,
                 id="translated_and_rotated_hemisphere"),
    pytest.param(codecad.shapes.box(4).translated(0, 0, 2) + codecad.shapes.box(2, 2, 9).translated(0, 0, -3.5),
                 96, codecad.util.Vector(0, 0, 0),
                 numpy.diag([1120, 1120, 192]),
                 id="not_hammer"),
    pytest.param(codecad.shapes.box(2, 3, 5).rotated((7, 11, 13), 17),
                 2 * 3 * 5, codecad.util.Vector(0, 0, 0),
                 drunk_box_matrix * (numpy.diag([3**2 + 5**2, 2**2 + 5**2, 2**2 + 3**2]) * 2 * 3 * 5 / 12) * drunk_box_matrix.T,
                 id="drunk_box"),
])
def test_mass_properties(shape, volume, centroid, inertia_tensor):
    result = codecad.mass_properties(shape, shape.bounding_box().volume() * 1e-3)

    print(result)
    assert result.volume == approx(volume, abs=result.volume_error)
    assert result.centroid == approx(centroid)

    if inertia_tensor is not None:
        assert numpy.allclose(result.inertia_tensor, inertia_tensor)


@pytest.mark.parametrize("radius", [0.2 * x - 1 for x in range(11)])
def test_mass_properties_unbounding_volume(radius):
    """ Test of a helper function inside mass properties CL code """

    integral = scipy.integrate.tplquad(lambda z, y, x: 1,
                                       0, min(0.5, abs(radius)),
                                       lambda x: 0,
                                       lambda x: min(0.5, math.sqrt(radius**2 - x**2)),
                                       lambda x, y: 0,
                                       lambda x, y: min(0.5, math.sqrt(radius**2 - x**2 - y**2)),
                                       epsabs=1e-5,
                                       epsrel=1e-5)

    expected = math.copysign(integral[0] * 8, radius)

    b = codecad.cl_util.Buffer(pyopencl.cltypes.float, 1, pyopencl.mem_flags.WRITE_ONLY)
    ev = codecad.cl_util.opencl_manager.k.test_mass_properties_unbounding_volume([1], None, numpy.float32(radius), b)
    b.read(wait_for=[ev])

    assert abs(b[0]) <= abs(expected) * (1 + 1e-3)
    assert b[0] == pytest.approx(expected, abs=0.1)


def test_mass_properties_stage2_and_stage4():
    size = 10
    integral1 = codecad.cl_util.Buffer(cltypes.float4, size, pyopencl.mem_flags.READ_WRITE)
    integral2 = codecad.cl_util.Buffer(cltypes.float4, size, pyopencl.mem_flags.READ_WRITE)
    integral3 = codecad.cl_util.Buffer(cltypes.float4, size, pyopencl.mem_flags.READ_WRITE)
    split_counts = codecad.cl_util.Buffer(cltypes.uint, size, pyopencl.mem_flags.READ_WRITE)
    output_buffer = codecad.cl_util.Buffer(cltypes.float, 13, pyopencl.mem_flags.WRITE_ONLY)
    assert_buffer = codecad.cl_util.AssertBuffer()

    evs = []
    evs.append(integral1.enqueue_write(numpy.full(size, 1, dtype=integral1.dtype)))
    evs.append(integral2.enqueue_write(numpy.full(size, 2, dtype=integral2.dtype)))
    evs.append(integral3.enqueue_write(numpy.full(size, 3, dtype=integral3.dtype)))
    evs.append(split_counts.enqueue_write(numpy.ones(shape=size, dtype=split_counts.dtype)))

    ev = codecad.cl_util.opencl_manager.k.mass_properties_sum([size], None,
                                                              integral1, integral2, integral3,
                                                              split_counts,
                                                              assert_buffer,
                                                              wait_for=evs)
    assert_buffer.check(wait_for=[ev])

    integral1.read()
    integral2.read()
    integral3.read()
    split_counts.read()

    assert integral1[-1] == numpy.full(1, 1 * size, dtype=integral1.dtype)[0]
    assert integral2[-1] == numpy.full(1, 2 * size, dtype=integral2.dtype)[0]
    assert integral3[-1] == numpy.full(1, 3 * size, dtype=integral3.dtype)[0]
    assert (split_counts[:] == numpy.arange(1, size + 1)).all()

    ev = codecad.cl_util.opencl_manager.k.mass_properties_output([1], None,
                                                                 cltypes.uint(size),
                                                                 integral1, integral2, integral3,
                                                                 split_counts,
                                                                 output_buffer)
    output_buffer.read(wait_for=[ev])
    assert (output_buffer[:] == [10, 10, 10, 10, 20, 20, 20, 30, 30, 30, 20, 30, 10]).all()
