import math

from .. import util
from . import simple2d

class InvoluteGearBase(simple2d.Shape2D):
    def __init__(self, tooth_count, pressure_angle):
        self.tooth_count = tooth_count
        self.pressure_angle = math.radians(pressure_angle)

    def bounding_box(self):
        #TODO: Calculate corect bounding box
        return util.BoundingBox(util.Vector(-1.5, -1.5),
                                util.Vector(1.5, 1.5))

    def get_node(self, point, cache):
        return cache.make_node("involute_gear",
                                [self.tooth_count, self.pressure_angle],
                                [point])

def involute_gear(tooth_count, module,
                  addendum_modules = 1,
                  dedendum_modules = 1,
                  pressure_angle = 20,
                  backlash = 0,
                  clearance = 0,
                  internal = False):
    """ Generates a 2D shape of a external or internal involute gear. """

    pitch_radius = tooth_count * module / 2
    root_radius = pitch_radius - dedendum_modules * module
    outside_radius = pitch_radius + addendum_modules * module

    if internal:
        outside_radius += clearance
        backlash = -backlash
    else:
        root_radius -= clearance

    base = InvoluteGearBase(tooth_count, pressure_angle).scaled(pitch_radius)

    if backlash != 0:
        base = base.offset(-backlash)

    root_circle = simple2d.Circle(r=root_radius)
    outside_circle = simple2d.Circle(r=outside_radius)

    return (base & outside_circle) + root_circle

def involute_gear_pair(tooth_count1, tooth_count2, module,
                       working_depth_modules = 2,
                       pressure_angle = 20,
                       backlash = 0,
                       clearance = 0):
    """ Returns tuple (gear1, gear2, center_distance) """
    # TODO Profile and backlash shifting
    gear1 = involute_gear(tooth_count1, module,
                          working_depth_modules / 2, working_depth_modules / 2,
                          pressure_angle, backlash / 2, clearance)
    gear2 = involute_gear(tooth_count2, module,
                          working_depth_modules / 2, working_depth_modules / 2,
                          pressure_angle, backlash / 2, clearance)

    return gear1, gear2, (tooth_count1 + tooth_count2) * module / 2
