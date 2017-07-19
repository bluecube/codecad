import itertools

import pytest

import codecad
import codecad.subdivision

@pytest.mark.parametrize('box_size', [codecad.util.Vector(10, 20, 30), codecad.util.Vector(16, 16, 16)])
@pytest.mark.parametrize('grid_size', [2, 21, 256])
@pytest.mark.parametrize('overlap', [True, False])
def test_block_sizes(box_size, grid_size, overlap):
    box = codecad.util.BoundingBox(-box_size / 2, box_size / 2)
    bs = codecad.subdivision.calculate_block_sizes(box, 1, grid_size, overlap)

    assert bs[-1][0] == 1, "Final block size must have the preset resolution"
    for i, (level_resolution, level_size) in enumerate(bs):
        if i > 0:
            assert level_size[0] == level_size[1] == level_size[2] == grid_size, "Non-top level blocks must have the preset grid resolution"
        assert level_size[0] > 1 or level_size[1] > 1 or level_size[2] > 1, "All levels must have more than one sub-blocks"

    if overlap and len(bs) == 1:
        for i in range(3):
            assert bs[0][1][i] >= box_size[i] / bs[0][0] + 1, "Top level block must cover the whole box (overlap)"
            assert box_size[i] / bs[0][0] + 1 > (bs[0][1][i] - 1), "Top level block must not be too large (overlap)"
    else:
        for i in range(3):
            assert bs[0][1][i] >= box_size[i] / bs[0][0], "Top level block must cover the whole box"
            assert box_size[i] / bs[0][0] > (bs[0][1][i] - 1), "Top level block must not be too large"

    for level_number, ((level_resolution, level_size), (prev_level_resolution, prev_level_size)) in enumerate(zip(bs[:-1], bs[1:])):
        for i in range(3):
            if overlap and level_number == len(bs) - 2:
                assert level_resolution == prev_level_resolution * (prev_level_size[i] - 1), "Each level must exactly cover the next one (overlap)"
            else:
                assert level_resolution == prev_level_resolution * prev_level_size[i], "Each level must exactly cover the next one"

def test_block_corners():
    _, _, blocks = codecad.subdivision.subdivision(codecad.shapes.box(10),
                                                   1,
                                                   grid_size=4,
                                                   overlap_edge_samples=True)

    corners = { block[0] for block in blocks }
    expected = { codecad.util.Vector(*coords) for coords in itertools.product([-5.5, -2.5, 0.5, 3.5], repeat=3) } - \
               { codecad.util.Vector(*coords) for coords in itertools.product([-2.5, 0.5], repeat=3) }
        # Subdivision skips blocks inside the shape

    assert corners == expected
