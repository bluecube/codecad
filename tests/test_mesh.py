import functools

import pytest
import trimesh

import codecad
import codecad.rendering.mesh

shapes = [codecad.shapes.box(10),
          codecad.shapes.sphere(10)]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("grid_size", [2, 12, 16])
def test_watertight(shape, grid_size):
    blocks = codecad.rendering.mesh.triangular_mesh(shape,
                                                    subdivision_grid_size=grid_size)

    mesh = functools.reduce(trimesh.util.concatenate,
                            (trimesh.Trimesh(vertices=vertices, faces=indices)
                             for vertices, indices in blocks))
    mesh.process()  # Deduplicate vertices
    # mesh.show()

    assert mesh.is_watertight
    # TODO: Check that there are no coplanar faces ... or something
