from .. import util


def _color(c):
    return util.Vector(*((c >> shift) & 0xff for shift in [16, 8, 0]))


background = _color(0xd1e7f2)
surface = _color(0x9bf29f)

# Since we always render just a single object, we can assume its coefficient
# of ambient reflection is 1 and that ambient light is colourless
ambient = 0.1  # Ambient light intensity
light = util.Vector(1, 3, -1).normalized()  # Light direction and intensity
