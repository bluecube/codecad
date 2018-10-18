#!/usr/bin/env python3

""" Showing off gear generator and (very basic) assemblies.

This is a part of an older scrapped project, so the modelling is a bit outdated
(but the resulting piece is still nice). """

import math
import itertools
import datetime

import codecad
import codecad.shapes as s


class Planetary:
    # Setting parameters which are independent of tooth counts as class variables

    # First some desired joint parameters
    target_rps = 0.25
    target_torque = 20
    joint_od = 100

    # The arm is supposed to be 3D printable, so we specify some printer parameters
    line_width = 0.45
    layer_height = 0.2
    min_wall_thickness = 2 * line_width
    wall_thickness = 3
    min_floor_thickness = layer_height * math.ceil(0.6 / layer_height)

    # Some general gear parameters
    min_module = 1  # Minimum gear module for most gears in the system
    min_module_ring = (
        1.5
    )  # Minimum module for the ring gear, to maximize transferable torque
    tip_clearance = 0.5
    backlash = 0.1
    profile_shift = 0.15  # Profile shift in modules

    motor_gear_thickness = 7
    ring_thickness = 15
    sun_thickness = 10

    shroud_overlap = 1.6

    # General clearance between moving parts.
    clearance = 0.8

    # Clearance between shaft and printed parts
    shaft_clearance = 0.1

    # # 624 bearing
    # bearing_id = 4
    # bearing_od = 13
    # bearing_thickness = 5
    # bearing_shoulder_size = 1.5
    # bearing_clearance = 0

    # 608 bearing
    bearing_id = 8
    bearing_od = 22
    bearing_thickness = 7
    bearing_shoulder_size = 1.5
    bearing_clearance = 0

    # M3 screw
    screw_diameter = 3

    motor_shaft_diameter = 3

    @classmethod
    def optimize(cls, n=20):
        """ Go through plausible looking combinations of tooth counts and print
        n combinations with highest transmission ratios. """
        best_ratio = 1
        solutions = []
        started = datetime.datetime.now()

        def r(a, b):
            if a > b:
                return range(a, b - 1, -1)
            else:
                return range(a, b + 1)

        # There is just a single option for the motor gears, since we can find
        # the optimal values (or close to it) manually easily and don't have to
        # slow the rest down
        ranges = [r(11, 11), r(60, 60), r(10, 15), r(20, 80), r(10, 30), r(30, 70)]
        candidate_count = 1
        for r in ranges:
            candidate_count *= len(r)

        try:
            for i, counts in enumerate(itertools.product(*ranges)):
                if i % 1000 == 0:
                    progress = i / candidate_count
                    elapsed = datetime.datetime.now() - started

                    if progress > 0:
                        eta = elapsed * (1 / progress - 1)
                    else:
                        eta = "???"

                    print(
                        "  {:.3f}%, found {} solutions, best ratio 1 / {:.2f}, elapsed {}, ETA {}".format(
                            progress * 100, len(solutions), 1 / best_ratio, elapsed, eta
                        ),
                        end="\r",
                    )

                try:
                    solution = cls(*counts)
                except AssertionError:
                    continue

                ratio = solution.get_ratio()
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio

                solutions.append(solution)
        except KeyboardInterrupt:
            print("Interrupted")

        print()
        print("best:")

        solutions_sorted = sorted(solutions, key=lambda x: x.get_ratio())
        for solution in solutions_sorted[:n]:
            solution.print_details()

    def __init__(
        self,
        n_motor_gear1,
        n_motor_gear2,
        n_sun_gear,
        n_planet_gear1,
        n_planet_gear2,
        n_ring_gear,
    ):
        # This would fail later, but also would cause division by zero
        assert n_ring_gear > n_planet_gear2

        # Chosing module for the motor gear so that the sun gear always has enough space for the shaft
        self.module_motor = max(
            self.min_module,
            (
                self.motor_shaft_diameter
                + 2
                * (self.min_wall_thickness + self.shaft_clearance + self.tip_clearance)
            )
            / (n_motor_gear1 - 2 * (1 - self.profile_shift)),
        )

        # The same for sun gear
        self.module_sun = max(
            self.min_module,
            (
                self.bearing_id
                + 2
                * (self.min_wall_thickness + self.shaft_clearance + self.tip_clearance)
            )
            / (n_sun_gear - 2 * (1 - self.profile_shift)),
        )

        # Choosing module for the ring gear so that the pitch circle of the planet matches
        self.module_ring = (
            self.module_sun
            * (n_sun_gear + n_planet_gear1)
            / (n_ring_gear - n_planet_gear2)
        )

        assert self.module_ring >= self.min_module_ring

        # The gears themselves
        self.motor_gear1 = s.gears.InvoluteGear(
            n_motor_gear1,
            self.module_motor,
            1 + self.profile_shift,
            1 - self.profile_shift,
            backlash=self.backlash,
            clearance=self.tip_clearance,
        )
        self.motor_gear2 = s.gears.InvoluteGear(
            n_motor_gear2,
            self.module_motor,
            1 - self.profile_shift,
            1 + self.profile_shift,
            backlash=self.backlash,
            clearance=self.tip_clearance,
        )

        self.sun_gear = s.gears.InvoluteGear(
            n_sun_gear,
            self.module_sun,
            1 + self.profile_shift,
            1 - self.profile_shift,
            backlash=self.backlash,
            clearance=self.tip_clearance,
        )
        self.planet_gear1 = s.gears.InvoluteGear(
            n_planet_gear1,
            self.module_sun,
            1 - self.profile_shift,
            1 + self.profile_shift,
            backlash=self.backlash,
            clearance=self.tip_clearance,
        )
        self.planet_gear2 = s.gears.InvoluteGear(
            n_planet_gear2,
            self.module_ring,
            1 + self.profile_shift,
            1 - self.profile_shift,
            backlash=self.backlash,
            clearance=self.tip_clearance,
        )
        self.ring_gear = s.gears.InvoluteGear(
            n_ring_gear,
            self.module_ring,
            1 - self.profile_shift,
            1 + self.profile_shift,
            internal=True,
            backlash=self.backlash,
            clearance=self.tip_clearance,
        )

        # Fit the ring gear into the intended max housing diameter
        assert (
            self.ring_gear.root_diameter + 2 * self.min_wall_thickness < self.joint_od
        )

        # Fit the motor gear into the intended housing diameter
        assert self.motor_gear2.outside_diameter < self.joint_od - 2 * (
            self.wall_thickness + self.clearance
        )

        # Be able to slide the ring over the planet bearing housing
        assert (
            self.planet_gear2.root_diameter
            > self.bearing_od + 2 * self.min_wall_thickness
        )

        self._check_gear_on_shaft(self.motor_gear1, self.motor_shaft_diameter)
        self._check_gear_on_shaft(self.motor_gear2)
        self._check_gear_on_shaft(self.sun_gear)
        self._check_gear_on_shaft(self.planet_gear1)
        self._check_gear_on_shaft(self.planet_gear2)
        self._check_gear_pair(self.sun_gear, self.planet_gear1)
        self._check_gear_pair(self.planet_gear2, self.ring_gear)

        # Pitch diameters add up properly
        assert (
            self.sun_gear.pitch_diameter
            + self.planet_gear1.pitch_diameter
            + self.planet_gear2.pitch_diameter
            == self.ring_gear.pitch_diameter
        )
        # This condition should hold regardless of tooth counts because of how ring module is chosen

        # Some intermediate values
        self.bearing_shoulder_thickness = self.min_floor_thickness
        self.bearing_wall_thickness = (
            self.bearing_thickness + self.bearing_shoulder_thickness
        )
        self.inner_carrier_half_length = 20 + self.ring_thickness + self.clearance
        self.outer_carrier_length = (
            self.ring_thickness + 2 * self.clearance + self.bearing_wall_thickness
        )
        self.planet_radius = (
            self.sun_gear.pitch_diameter + self.planet_gear1.pitch_diameter
        ) / 2

        self.planet_count = math.floor(
            math.pi
            / math.asin(
                (self.planet_gear1.outside_diameter + self.clearance)
                / (2 * self.planet_radius)
            )
        )
        self.outer_carrier_od = (
            2 * self.planet_radius + self.bearing_od + 2 * self.min_wall_thickness
        )
        self.outer_carrier_screw_radius = (
            self.outer_carrier_od / 2 - self.wall_thickness - self.screw_diameter / 2
        )

        # Fit the large planet into the intended max housing diameter
        assert (
            2 * self.planet_radius + self.planet_gear1.outside_diameter
            <= self.joint_od - self.min_wall_thickness - self.tip_clearance
        )

        # Some often needed shapes
        self.bearing_hole = s.cylinder(
            d=self.bearing_od + 2 * self.bearing_clearance,
            h=2 * self.bearing_thickness,
            symmetrical=False,
        ) + s.cylinder(
            d=self.bearing_od - 2 * self.bearing_shoulder_size, h=float("inf")
        )
        self.screw_holes = (
            s.cylinder(h=float("inf"), d=self.screw_diameter)
            .translated(self.outer_carrier_screw_radius, 0, 0)
            .rotated((0, 0, 1), 180 / self.planet_count)
            .rotated((0, 0, 1), 360, self.planet_count)
        )
        self.shaft_hole = s.cylinder(
            h=float("inf"), d=self.bearing_id + 2 * self.shaft_clearance
        )
        self.outer_circle = s.circle(self.joint_od)

    @staticmethod
    def _check_gear_pair(g1, g2):
        assert g1.module == g2.module

        # Load sharing between teeth
        assert math.gcd(g1.n, g2.n) == 1

    @classmethod
    def _check_gear_on_shaft(cls, g, shaft_diameter=None):
        if shaft_diameter is None:
            shaft_diameter = cls.bearing_id
        assert g.root_diameter >= shaft_diameter + 2 * (
            cls.min_wall_thickness + cls.shaft_clearance
        )

    def get_ratio(self):
        return (
            (self.motor_gear1.n / self.motor_gear2.n)
            * (self.sun_gear.n / self.planet_gear1.n)
            * (self.planet_gear2.n / self.ring_gear.n)
        )

    def print_details(self):
        ratio = self.get_ratio()
        print(
            "Tooth counts: {} {} {} {} {} {}".format(
                self.motor_gear1.n,
                self.motor_gear2.n,
                self.sun_gear.n,
                self.planet_gear1.n,
                self.planet_gear2.n,
                self.ring_gear.n,
            )
        )
        print("Transmission ratio: 1 / {}".format(1 / ratio))
        print(
            "  {:.0f} motor RPM for {} RPS arm rotation".format(
                60 * self.target_rps / ratio, self.target_rps
            )
        )
        print(
            "  {:.0f} mNm motor torque for {} nm arm torque".format(
                1e3 * self.target_torque * ratio, self.target_torque
            )
        )
        print(
            "Planet count: {}, motor module: {:.2f}, sun module: {:.2f}, ring module: {:.2f}".format(
                self.planet_count, self.module_motor, self.module_sun, self.module_ring
            )
        )

    def make_planet(self):
        # Gears are profile shifted by 0.5 module (smaller gear has larger addendum)
        h = self.sun_thickness
        gear1 = self.planet_gear1.extruded(h, symmetrical=False)

        h += self.clearance
        h += self.ring_thickness
        gear2 = self.planet_gear2.extruded(h, symmetrical=False)

        h += self.clearance
        spacer = s.cylinder(
            h, self.bearing_id + 2 * self.bearing_shoulder_size, symmetrical=False
        )

        return s.union([gear1, gear2, spacer]) - self.shaft_hole

    def make_sun(self):
        h = self.motor_gear_thickness
        gear1 = self.motor_gear2.extruded(h, symmetrical=False)

        h += self.bearing_wall_thickness + self.clearance
        spacer1 = s.cylinder(h, self.sun_gear.outside_diameter, symmetrical=False)

        h += self.sun_thickness + 2 * self.clearance
        gear2 = self.sun_gear.extruded(h, symmetrical=False)

        h += self.ring_thickness + self.clearance
        spacer2 = s.cylinder(
            h, self.bearing_id + 2 * self.bearing_shoulder_size, symmetrical=False
        )

        return s.union([spacer1, spacer2, gear1, gear2]) - self.shaft_hole

    def make_carrier_inner(self):
        shroud_skip = self.shroud_overlap + self.clearance
        side = (
            (self.outer_circle - s.circle(self.joint_od - 2 * self.wall_thickness))
            .extruded(self.inner_carrier_half_length - shroud_skip, symmetrical=False)
            .translated(0, 0, shroud_skip)
        )
        cap = s.circle(
            self.joint_od - 2 * (self.min_wall_thickness + self.clearance)
        ).extruded(
            self.bearing_wall_thickness + self.clearance + self.sun_thickness,
            symmetrical=False,
        )

        sun_hole = s.cylinder(
            d=self.sun_gear.outside_diameter + 2 * self.clearance, h=float("inf")
        )
        cap -= sun_hole

        moved_bearing_hole = self.bearing_hole.rotated((1, 0, 0), 180).translated(
            0, 0, self.bearing_thickness + self.sun_thickness + self.clearance
        )
        cap -= moved_bearing_hole.translated(self.planet_radius, 0, 0).rotated(
            (0, 0, 1), 360, self.planet_count
        )

        planet_hole = s.cylinder(
            2 * self.sun_thickness + 2 * self.clearance,
            self.planet_gear1.outside_diameter + 2 * self.clearance,
        )
        cap -= planet_hole.translated(self.planet_radius, 0, 0).rotated(
            (0, 0, 1), 360, self.planet_count
        )

        cap -= self.screw_holes

        return side + cap

    def make_carrier_outer(self):
        side = s.cylinder(
            h=self.outer_carrier_length, d=self.outer_carrier_od, symmetrical=False
        )
        cap = s.cylinder(
            h=self.bearing_wall_thickness, d=self.outer_carrier_od, symmetrical=False
        )

        sun_hole = s.cylinder(
            h=float("inf"), d=self.outer_carrier_od - 2 * self.wall_thickness
        )
        side -= sun_hole

        screw_tubes = (
            s.cylinder(
                h=self.outer_carrier_length,
                d=self.screw_diameter + 2 * self.wall_thickness,
                symmetrical=False,
            )
            .translated(self.outer_carrier_screw_radius, 0, 0)
            .rotated((0, 0, 1), 180 / self.planet_count)
            .rotated((0, 0, 1), 360, self.planet_count)
        )
        side += screw_tubes

        moved_bearing_hole = self.bearing_hole.translated(
            0, 0, self.bearing_shoulder_thickness
        )
        cap -= moved_bearing_hole
        cap -= moved_bearing_hole.translated(self.planet_radius, 0, 0).rotated(
            (0, 0, 1), 360, self.planet_count
        )

        planet_hole = s.cylinder(
            float("inf"), self.planet_gear2.outside_diameter + 2 * self.clearance
        )
        side -= planet_hole.translated(
            self.planet_radius, 0, self.outer_carrier_length
        ).rotated((0, 0, 1), 360, self.planet_count)

        body = side + cap
        # TODO: Rounded unions

        body -= self.screw_holes

        return body

    def make_ring(self):
        h = self.bearing_wall_thickness
        cap = s.cylinder(h=h, d=self.joint_od, symmetrical=False)
        cap -= self.bearing_hole.translated(0, 0, self.bearing_shoulder_thickness)

        h += self.bearing_wall_thickness
        h += self.clearance
        outer = (
            self.outer_circle - s.circle(self.outer_carrier_od + 2 * self.clearance)
        ).extruded(h, symmetrical=False)

        h += self.ring_thickness
        h += self.clearance
        gear = (self.outer_circle - self.ring_gear).extruded(h, symmetrical=False)

        h += self.clearance
        h += self.shroud_overlap
        shroud = (
            self.outer_circle - s.circle(self.joint_od - 2 * self.min_wall_thickness)
        ).extruded(h, symmetrical=False)

        return cap + outer + gear + shroud

    def make_assembly(self):
        planet = (
            self.make_planet()
            .make_part("planet")
            .translated(
                self.planet_radius,
                0,
                self.inner_carrier_half_length - self.sun_thickness,
            )
        )
        return codecad.assembly(
            "gearbox",
            [
                self.make_carrier_inner()
                .make_part("inner_carrier")
                .rotated_x(180)
                .translated_z(self.inner_carrier_half_length),
                self.make_carrier_outer()
                .make_part("outer_carrier")
                .rotated_x(180)
                .translated_z(
                    self.outer_carrier_length + self.inner_carrier_half_length
                ),
                self.make_sun()
                .make_part("sun_gear")
                .translated_z(
                    self.inner_carrier_half_length
                    - self.bearing_wall_thickness
                    - 2 * self.clearance
                    - self.sun_thickness
                    - self.motor_gear_thickness
                ),
                self.make_ring()
                .make_part("ring")
                .rotated_x(180)
                .translated_z(
                    self.outer_carrier_length
                    + self.inner_carrier_half_length
                    + self.bearing_wall_thickness
                    + self.clearance
                ),
            ]
            + [
                planet.rotated_z(i * 360 / self.planet_count)
                for i in range(self.planet_count)
            ],
        )

    def make_overview(self):
        """ Take the assembly and turn it into a piece good for showing off """
        return (self.make_assembly().shape() & s.half_space()).rotated_x(30)


if __name__ == "__main__":
    # Planetary.optimize()
    p = Planetary(11, 60, 13, 41, 18, 53)
    p.print_details()
    codecad.commandline_render(p.make_assembly())
