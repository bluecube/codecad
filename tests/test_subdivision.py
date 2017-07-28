import itertools
import math

import pytest

import codecad
import codecad.subdivision

@pytest.mark.parametrize('box_size', [codecad.util.Vector(10, 20, 30), codecad.util.Vector(16, 16, 16)])
@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('grid_size', [2, 21, 256])
@pytest.mark.parametrize('overlap', [True, False])
def test_block_sizes(box_size, dimension, grid_size, overlap):
    box = codecad.util.BoundingBox(-box_size / 2, box_size / 2)
    bs = codecad.subdivision.calculate_block_sizes(box, dimension, 1, grid_size, overlap)

    assert bs[-1][0] == 1, "Final block size must have the preset resolution"
    for i, (level_resolution, level_size) in enumerate(bs):
        if i > 0:
            for j in range(dimension):
                assert level_size[j] == grid_size, "Non-top level blocks must have the preset grid resolution"
        if dimension == 2:
            assert level_size[2] == 1, "2D blocks must have just a single layer in Z"
        assert level_size[0] > 1 or level_size[1] > 1 or level_size[2] > 1, "All levels must have more than one sub-blocks"


    if overlap and len(bs) == 1:
        for i in range(dimension):
            assert bs[0][1][i] >= box_size[i] / bs[0][0] + 1, "Top level block must cover the whole box (overlap)"
            assert box_size[i] / bs[0][0] + 1 > (bs[0][1][i] - 1), "Top level block must not be too large (overlap)"
    else:
        for i in range(dimension):
            assert bs[0][1][i] >= box_size[i] / bs[0][0], "Top level block must cover the whole box"
            assert box_size[i] / bs[0][0] > (bs[0][1][i] - 1), "Top level block must not be too large"

    for level_number, ((level_resolution, level_size), (prev_level_resolution, prev_level_size)) in enumerate(zip(bs[:-1], bs[1:])):
        for i in range(dimension):
            if overlap and level_number == len(bs) - 2:
                assert level_resolution == prev_level_resolution * (prev_level_size[i] - 1), "Each level must exactly cover the next one (overlap)"
            else:
                assert level_resolution == prev_level_resolution * prev_level_size[i], "Each level must exactly cover the next one"

def test_block_corners_cube():
    _, _, blocks = codecad.subdivision.subdivision(codecad.shapes.box(10),
                                                   1,
                                                   grid_size=4,
                                                   overlap_edge_samples=True)

    corners = { block[0] for block in blocks }
    expected = { codecad.util.Vector(*coords) for coords in itertools.product([-5.5, -2.5, 0.5, 3.5], repeat=3) } - \
               { codecad.util.Vector(*coords) for coords in itertools.product([-2.5, 0.5], repeat=3) }
        # Subdivision skips blocks inside the shape

    assert corners == expected

def test_block_corners_circle():
    grid_size = 8
    step = grid_size - 1
    diameter = grid_size * step - 1
    radius = diameter / 2
    threshold = math.sqrt(2) * step / 2

    _, _, blocks = codecad.subdivision.subdivision(codecad.shapes.circle(diameter),
                                                   1,
                                                   grid_size=grid_size,
                                                   overlap_edge_samples=True)

    r = []
    for i in range(grid_size):
        r.append(-radius - 0.5 + i * step)

    corners = { block[0] for block in blocks }
    expected = set()
    for coords in itertools.product(r, repeat=2):
        corner = codecad.util.Vector(*coords)
        center = corner + codecad.util.Vector(2.5, 2.5, 0)
        if radius - threshold < abs(center) < radius + threshold:
            expected.add(corner)

    assert corners == expected
