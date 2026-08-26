"""Regression test for the conformal round-coax port floor (DD-053).

The z-translation-invariant round coax (D_i = 0.41 mm, D_a = 5 mm,
eps_r = 9) on a conformal 19x19x25 mesh is THE reproducer for the
pair-consistent conformal correction: before DD-053 the independently
computed E-edge (eps) and Krietenstein H-face (mu) sub-cell corrections
broke the co-located pair identity ``M_eps*M_mu = eps*mu*dz*dz~`` (pair
spread 3.1 %) and the conformal classifier left the longitudinal
z-edges at conductor-surface nodes unmasked — together capping the
port floor at max |S11| = -32.5 dB (median -35.8).

With the LC-consistent mu coupling + tangential-surface re-masking the
measured floor is max -44.1 dB / median -61.4 at unchanged conformal
z_line 48.12 Ohm (staircase reference at the same resolution: -45.3 /
-60.6 dB but z_line 44.73 Ohm; analytic 49.97 Ohm) — the construction-
site-0 acceptance: staircase-level port floor WITH conformal impedance
accuracy.  Doubled resolution reaches -56.3 / -73.6 dB at z_line
49.60 Ohm (not exercised here for runtime).

Since the short-interval grading keeps the fine-end cell at ``h_fine``
(DD-193, v0.4.6) the three cells spanning the inner conductor are
3 x 0.137 mm instead of the former 0.121 / 0.168 / 0.121 mm ramp; the
rest of the grid is unchanged.  Re-measured on that grid: z_line
48.94 Ohm (closer to the analytic 49.97), max |S11| = -135.6 dB,
median -153.9 (CPU backend, 2026-08-26).  The pinned impedance below
is that value; the staircase 44.73 Ohm still fails the 1 % bound.
"""

from __future__ import annotations

import numpy as np

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.geo import Cylinder, Difference, GeometryModel
from magnelio.mesh import BoxFace
from magnelio.ports import PortSpecMultiConductor

D_I = 0.41e-3
D_A = 5.0e-3
EPS_R = 9.0
LENGTH = 10.0e-3
F_MAX = 10e9


def _build_analysis() -> AnalysisScatteringTD:
    pec = Material.pec()
    diel = Material.from_isotropic(name="dielectric", epsilon=EPS_R)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=D_A / 2, height=LENGTH, axis="z", material=diel)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=D_I / 2, height=LENGTH, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)

    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=3,
        growth_factor=1.4,
        conformal=True,
        max_cell_size=0.4e-3,
        min_cell_size=50e-6,
        min_feature_gap=20e-6,
    )
    mesh = Mesh.from_geometry(model, control, f_max=F_MAX)

    specs = [
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=EPS_R, n_modes=1),
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, epsilon_r=EPS_R, n_modes=1),
    ]
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions({}),
        ports=specs,
        f_max=F_MAX,
        verbose=False,
    )


def test_conformal_coax_port_floor_and_impedance():
    """max |S11| < -110 dB, z_line within 1 % of 48.94 Ohm.

    DD-053 measurement (19x19x25): max -44.06 dB, median -61.42,
    z_line 48.116 Ohm.  WP-R2 (exact DTBC termination + discrete
    de-stagger; the DD-053 pair coupling makes the conformal section
    pass the DTBC pair-product gate): max |S11| = -131.0 dB at
    unchanged z_line — the former floor was absorber- and
    measurement-chain-limited, not conformal-geometric.  DD-193 grid
    (see module docstring): z_line 48.94 Ohm, max |S11| = -135.6 dB.
    The z_line bound still guards the *conformal* half of the
    acceptance — a staircase-level impedance (44.7 Ohm) must fail.
    """
    analysis = _build_analysis()

    reports = analysis.solve_ports()
    z_line = reports["port1"].z_line_num
    assert abs(z_line - 48.94) / 48.94 < 0.01, (
        f"conformal z_line regression: {z_line:.3f} Ohm (expect ~48.94)"
    )

    f_axis = np.linspace(F_MAX / 40, F_MAX, 81)
    result = analysis.run(f_axis=f_axis, excited=["port1"])
    s11_db = 20 * np.log10(np.abs(result.S("port1", "port1")) + 1e-30)
    s21_db = 20 * np.log10(np.abs(result.S("port2", "port1")) + 1e-30)

    assert np.all(np.isfinite(s11_db))
    assert s11_db.max() < -110.0, (
        f"conformal port floor regression: max |S11| = {s11_db.max():.2f} dB "
        f"at {f_axis[np.argmax(s11_db)] / 1e9:.2f} GHz (bound: -110 dB; "
        f"reflection-free acceptance line is -100 dB)"
    )
    assert np.max(np.abs(s21_db)) < 0.1, (
        f"|S21| deviates from 0 dB by {np.max(np.abs(s21_db)):.3f} dB"
    )
