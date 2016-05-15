from . import util

class _VolumeRendererState:
    def __init__(self):
        self.volume = 0
        self.centroid = util.Vector(0, 0, 0)

    def _visit(self, box, values, is_large, is_intersecting):
        if is_intersecting:
            if is_large:
                # Large box intersecting the surface must be expanded
                return True
            else:
                # If the box intersects the surface, but we won't be
                # expanding it because it is too small, estimate the volume

                # This algorithm is based only on my gut feeling and completely
                # without proof.
                # It calculates ratio of sphere volumes around the outside and inside
                # vertices and splits the box volume in this ratio
                # TODO: Figure out if it is reasonable
                outside = 0
                inside = 0
                for val in values.flat:
                    volume = val * val * val
                    if val > 0:
                        outside += volume
                    else:
                        inside += volume

                current_volume = box.volume() * inside / (inside + outside)
        else:
            if values[0, 0, 0] <= 0:
                # if the box does not intersect the surface and it is inside,
                # just calculate its volume
                current_volume = box.volume()
            else:
                # If it does not intersect surface and is outside, we can just
                # skip it
                return False


        # At this point we are stopping the box splitting with some volume inside
        # the shape, so we need to actually calculat the volume and centroid
        # TODO: use something like fsum for the summation here
        self.volume += current_volume
        self.centroid += box.centroid() * current_volume
        return False


def volume_and_centroid(shape, resolution):
    state = _VolumeRendererState()
    util.rendering.shape_apply(state._visit, shape, resolution)
    state.centroid /= state.volume
    return state

