"""Independent exact-plane section/material acceptance for DD-287.

The reference includes the cross-bore. A geometric support-surface solution
must retain both material islands and the material matrices must consume
their signed areas. No field plotting or private input files are needed.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from magnelio import geo
from magnelio.geo._occ_backend import (
    _PlanarSectionEngine,
    compute_face_material_areas,
    cross_section_polygons,
)
from magnelio.geo._polygon_clip import polygon_area
from magnelio.materials.material import Material

R, BORE, HEIGHT = 2.3e-3, 0.7e-3, 10e-3
DEFLECTION = 2.5e-8
SCALE = 1024.0


def fixture(axis):
    body = geo.Cylinder(
        origin=(0, 0, -HEIGHT / 2), axis="z", radius=R, height=HEIGHT, material="pec"
    )
    start = (-2 * R, 0, 0) if axis == "x" else (0, -2 * R, 0)
    bore = geo.Cylinder(origin=start, axis=axis, height=4 * R, radius=BORE, material="pec")

    def square(z, side):
        return geo.Profile.polygon(
            [
                (u, v, z)
                for u, v in (
                    (-side / 2, -side / 2),
                    (side / 2, -side / 2),
                    (side / 2, side / 2),
                    (-side / 2, side / 2),
                )
            ]
        )

    loft = geo.Loft(square(0.003, 1.2 * R), square(0.009, 0.6 * R), blend="ruled", material="pec")
    return (body - bore) + loft


def reference_cell(distance, rectangle):
    half = np.sqrt(distance * (2 * R - distance))
    x0, z0, x1, z1 = rectangle
    a, b = max(x0, -half), min(x1, half)
    z0, z1 = max(z0, -HEIGHT / 2), min(z1, HEIGHT / 2)
    if b <= a or z1 <= z0:
        return 0.0

    def material_length(x):
        h = np.sqrt(BORE * BORE - x * x)
        return z1 - z0 - max(min(z1, h) - max(z0, -h), 0)

    return quad(material_length, a, b, epsabs=1e-20, epsrel=1e-12)[0]


def main():
    for axis in "xy":
        body = fixture(axis)
        occ = body._occ_shape(SCALE)
        engine = _PlanarSectionEngine(occ, scale=SCALE, deflection=DEFLECTION)
        for distance in (1e-7, 1e-9):
            position = R - distance
            expected = reference_cell(distance, (-R, -HEIGHT / 2, R, HEIGHT / 2))
            answers = (
                cross_section_polygons(occ, axis, position, deflection=DEFLECTION, scale=SCALE),
                engine.section("xyz".index(axis), position),
            )
            for polygons in answers:
                assert polygons is not None and len(polygons) == 2
                measured = abs(sum(polygon_area(p - p[0]) for p in polygons))
                assert abs(measured / expected - 1) < 1e-6, (axis, distance, measured, expected)
    body = fixture("y")
    dx, dz = 25e-6, 250e-6
    materials = {0: Material.air(), 1: Material.pec()}
    for distance in (1e-7, 1e-9):
        rectangles = [
            (x, z, x + dx, z + dz)
            for x in np.arange(-50e-6, 50e-6, dx)
            for z in np.arange(-HEIGHT / 2, HEIGHT / 2, dz)
        ]
        specs = np.array([(R - distance, *r) for r in rectangles])
        exact = np.array([reference_cell(distance, r) for r in rectangles])
        booked = np.zeros(len(specs))
        epsilon = compute_face_material_areas(
            [(body, 1)],
            materials,
            specs,
            np.ones(len(specs), dtype=np.int32),
            deflection=DEFLECTION,
            scale=SCALE,
            pec_area_out=booked,
        )
        coverage_error = np.max(np.abs(booked - exact)) / (dx * dz)
        epsilon_error = np.max(np.abs(epsilon - (1 - exact / (dx * dz))))
        assert coverage_error < 1e-5 and epsilon_error < 1e-5
        print(
            f"d={distance:.0e} m: coverage/epsilon errors {coverage_error:.3e}/{epsilon_error:.3e}"
        )
    print("PASS: exact-plane trimmed sections and signed material matrices")


if __name__ == "__main__":
    main()
