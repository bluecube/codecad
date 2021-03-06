import math

from .. import util
from . import base
from . import simple2d
from ..cl_util import opencl_manager

_c_file = opencl_manager.add_compile_unit()
_c_file.append_resource("common.h")
_c_file.append_resource("gears.cl")


class InvoluteGearBase(base.Shape2D):
    """ A 2D shape of a external involute gear that has no top land (sharp tooth tips)
    and no bottom land """

    def __init__(self, tooth_count, pressure_angle):
        self.tooth_count = tooth_count
        self.pressure_angle = math.radians(pressure_angle)

    def bounding_box(self):
        # TODO: Calculate corect bounding box
        return util.BoundingBox(util.Vector(-1.5, -1.5), util.Vector(1.5, 1.5))

    def feature_size(self):
        return 0.5 * math.pi / self.tooth_count  # 1/2 tooth thickness

    def get_node(self, point, cache):
        return cache.make_node(
            "involute_gear", [self.tooth_count, self.pressure_angle], [point]
        )


class InvoluteGear(simple2d.Union2D):
    """ A 2D shape of a external or internal involute gear.
    In case of an internal gear, this is the negative shape to be subtracted.

    The following member variables are set:
    n, module, addendum modules, dedendum_modules, pressure_angle, backlash, clearance, internal.
    pitch_diameter, root_circle_diameter, addendum_circle_diameter """

    def __init__(
        self,
        n,
        module,
        addendum_modules=1,
        dedendum_modules=1,
        pressure_angle=20,
        backlash=0,
        clearance=0,
        internal=False,
    ):

        self.n = n
        self.module = module
        self.addendum_modules = addendum_modules
        self.dedendum_modules = dedendum_modules
        self.pressure_angle = pressure_angle
        self.backlash = backlash
        self.clearance = clearance
        self.internal = internal

        self.pitch_diameter = n * module

        pitch_radius = self.pitch_diameter / 2

        if internal:
            inside_radius = pitch_radius - addendum_modules * module
            outside_radius = pitch_radius + dedendum_modules * module + clearance

            self.inside_diameter = inside_radius * 2
            self.root_diameter = outside_radius * 2

            backlash = -backlash
        else:
            inside_radius = pitch_radius - dedendum_modules * module - clearance
            outside_radius = pitch_radius + addendum_modules * module

            self.outside_diameter = outside_radius * 2
            self.root_diameter = inside_radius * 2

        gear_base = InvoluteGearBase(n, pressure_angle).scaled(pitch_radius)

        if backlash != 0:
            gear_base = gear_base.offset(-backlash)

        inner_circle = simple2d.Circle(r=inside_radius)
        outer_circle = simple2d.Circle(r=outside_radius)

        super().__init__([gear_base & outer_circle, inner_circle])
