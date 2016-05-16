from .. import util

def _color(c):
    return util.Vector(*((c >> shift) & 0xff for shift in [16, 8, 0]))


background = _color(0xd1e7f2)
ambient = _color(0x0c3d0f)
surface = _color(0x9bf29f) - ambient

light = util.Vector(1, 1, -1).normalized() # Light direction and intensity
