import itertools
import math

import pytest

import codecad
import codecad.subdivision

def set_has_approx_item(s1, i2, *args, **kwargs):
    for i1 in s1:
        if i1 == pytest.approx(i2, *args, **kwargs):
            return True

    return False

class set_approx_equals:
    def __init__(self, s1, s2, *args, **kwargs):
        self.s1_extra = set()
        for item1 in s1:
            if not set_has_approx_item(s2, item1, *args, **kwargs):
                self.s1_extra.add(item1)

        self.s2_extra = set()
        for item2 in s2:
            if not set_has_approx_item(s1, item2, *args, **kwargs):
                self.s2_extra.add(item2)

    def __bool__(self):
        return len(self.s1_extra) == len(self.s2_extra) == 0

    def __str__(self):
        return " ".join(["Extra values in s1:", str(self.s1_extra),
                         "Extra values in s2:", str(self.s2_extra)])

@pytest.mark.parametrize('box_size', [codecad.util.Vector(10, 20, 30), codecad.util.Vector(16, 16, 16)])
@pytest.mark.parametrize('dimension', [2, 3])
@pytest.mark.parametrize('resolution', [1, 0.1])
@pytest.mark.parametrize('grid_size', [2, 21, 256])
@pytest.mark.parametrize('overlap', [True, False])
def test_block_sizes(box_size, dimension, resolution, grid_size, overlap):
    box = codecad.util.BoundingBox(-box_size / 2, box_size / 2)
    bs = codecad.subdivision.calculate_block_sizes(box, dimension, resolution, grid_size, overlap)

    assert bs[-1][0] == 1, "Final block size must have block size 1"
    for i, (level_resolution, level_size) in enumerate(bs):
        if i > 0:
            for j in range(dimension):
                assert level_size[j] == grid_size, "Non-top level blocks must have the preset grid resolution"
        if dimension == 2:
            assert level_size[2] == 1, "2D blocks must have just a single layer in Z"
        assert level_size[0] > 1 or level_size[1] > 1 or level_size[2] > 1, "All levels must have more than one sub-blocks"


    real_block_size = bs[0][0] * resolution
    if overlap and len(bs) == 1:
        for i in range(dimension):
            assert bs[0][1][i] >= box_size[i] / real_block_size + 1, "Top level block must cover the whole box (overlap)"
            assert box_size[i] /real_block_size + 1 > (bs[0][1][i] - 1), "Top level block must not be too large (overlap)"
    else:
        for i in range(dimension):
            assert bs[0][1][i] >= box_size[i] / (bs[0][0] * resolution), "Top level block must cover the whole box"
            assert box_size[i] / real_block_size > (bs[0][1][i] - 1), "Top level block must not be too large"

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

    assert blocks[0][2] == 1
    assert blocks[0][4] == 1

    corners = { block[1] for block in blocks }
    expected = { codecad.util.Vector(*coords) for coords in itertools.product([-5.5, -2.5, 0.5, 3.5], repeat=3) } - \
               { codecad.util.Vector(*coords) for coords in itertools.product([-2.5, 0.5], repeat=3) }
        # Subdivision skips blocks inside the shape

    assert corners == expected

def test_block_corners_circle():
    resolution = 0.1 # Size of single cell
    grid_size = 8
    step = resolution * (grid_size - 1) #size of the inner block
    diameter = grid_size * step - resolution
    radius = diameter / 2
    threshold = math.sqrt(2) * step / 2

    _, _, blocks = codecad.subdivision.subdivision(codecad.shapes.circle(diameter),
                                                   resolution,
                                                   grid_size=grid_size,
                                                   overlap_edge_samples=True)

    assert blocks[0][2] == resolution
    assert blocks[0][4] == 1

    r = []
    for i in range(grid_size):
        r.append(-radius - 0.5 * resolution + i * step)

    corners = { block[1] for block in blocks }

    expected = set()
    for coords in itertools.product(r, repeat=2):
        corner = codecad.util.Vector(*coords)
        center = corner + codecad.util.Vector(step/2, step/2, 0)
        if radius - threshold < abs(center) < radius + threshold:
            expected.add(corner)

    assert set_approx_equals(corners, expected)
