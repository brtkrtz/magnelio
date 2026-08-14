"""WP-R2 acceptance: TEM port floors with the exact DTBC termination.

Measures max/median in-band |S11| (and the |S21| deviation from 0 dB)
on the three straight-line acceptance geometries of the
reflection-free plan WP-R2 (plan retired to git history, see DD-054),
all through the public high-level API:

* parallel plate, uniform transversal mesh (BC-PEC plates, PMC sides);
* parallel plate, growth-1.4 graded transversal mesh (the session-74
  grading exposure);
* rectangular coax (PTFE-filled, eps_r 2.1, staircase — the WP1/F6
  reproducer geometry);
* conformal round coax (eps_r 9, the DD-053 reproducer — the
  construction-site-0 geometry).

The two coax lines are dielectric-filled, so the numbers below cover
homogeneous-dielectric TEM feeds, not just vacuum.

Acceptance: |S11| < -100 dB across the evaluated band for the first
three; the conformal coax is measured and its residual attributed
(conformal-geometric vs. absorber-related).

Results (session 85, first DTBC production measurement):

    parallel plate uniform   max -138.7 dB   median -164.0 dB
    parallel plate graded    max -136.1 dB   median -158.1 dB
    rect coax                max -159.3 dB   median -159.4 dB
    conformal round coax     max -131.0 dB   median -131.3 dB
                             (z_line 48.12 Ohm, unchanged vs DD-053)

All four sit 30+ dB below the acceptance line.  The conformal coax
passes the DTBC pair-product gate (DD-053 makes M_eps*M_mu exact per
co-located pair), and its former -44.1 dB floor drops by ~87 dB at
unchanged impedance: the residual measured through session 83 was
entirely absorber- and measurement-chain-limited, not
conformal-geometric.

Both WP-R2 legs are required for these numbers:

* the exact DTBC termination replaces modal Mur-1 (boundary
  reflection to float noise, propagating and evanescent content), and
* the discrete de-stagger factor ``lambda^{1/2}(e^{j w dt})`` replaces
  the continuum ``e^{-gamma dz/2}`` in the a/b decomposition — the
  continuum factor's grid-dispersion gap ``~ (beta dz/2)^3 (1-r^2)/6``
  alone caps measured floors near -70 dB on lambda/20 meshes (that gap,
  not Mur, set the old parallel-plate -71.9 dB record).

Run:  python validation/dtbc_tem_port_floors.py
"""

from __future__ import annotations

import numpy as np

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecMultiConductor


def _graded_axis(lo: float, hi: float, n_cells: int, growth: float) -> np.ndarray:
    half = n_cells // 2
    d = growth ** np.arange(half, dtype=float)
    if n_cells % 2 == 0:
        d_all = np.concatenate([d, d[::-1]])
    else:
        d_all = np.concatenate([d, [d[-1] * growth], d[::-1]])
    d_all = d_all / d_all.sum() * (hi - lo)
    return lo + np.concatenate([[0.0], np.cumsum(d_all)])


def parallel_plate(graded: bool) -> AnalysisScatteringTD:
    width_a, gap_b, length, f_max = 10e-3, 5e-3, 20e-3, 10e9
    y = (
        _graded_axis(-gap_b / 2, gap_b / 2, 12, 1.4)
        if graded
        else np.linspace(-gap_b / 2, gap_b / 2, 6)
    )
    grid = GridLines(
        x=np.linspace(-width_a / 2, width_a / 2, 11),
        y=y,
        z=np.linspace(-length / 2, length / 2, 41),
    )
    return AnalysisScatteringTD(
        mesh=Mesh.from_grid(
            grid,
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            },
        ),
        ports=[
            PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, n_modes=1),
        ],
        f_max=f_max,
        verbose=False,
    )


def rect_coax() -> AnalysisScatteringTD:
    a_inner, b_outer, eps_r, length, f_max = 2e-3, 10e-3, 2.1, 50e-3, 10e9
    ptfe = Material.from_isotropic("PTFE", epsilon=eps_r)
    pec = Material.pec()
    model = GeometryModel(background=pec)
    ptfe_full = Brick(
        origin=(-b_outer / 2, -b_outer / 2, 0),
        size=(b_outer, b_outer, length),
        material=ptfe,
    )
    inner = Brick(
        origin=(-a_inner / 2, -a_inner / 2, 0),
        size=(a_inner, a_inner, length),
        material=pec,
    )
    model.add(Difference(ptfe_full, inner, material=ptfe, name="PTFE"))
    model.add(inner)
    control = MeshControl(min_nodes_per_wavelength=20, max_cell_size=0.8e-3)
    mesh = Mesh.from_geometry(model, control, f_max=f_max)
    specs = [
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=eps_r, n_modes=1),
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, epsilon_r=eps_r, n_modes=1),
    ]
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "xmin": "PEC",
                "xmax": "PEC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            }
        ),
        ports=specs,
        f_max=f_max,
        verbose=False,
    )


def conformal_round_coax() -> AnalysisScatteringTD:
    d_i, d_a, eps_r, length, f_max = 0.41e-3, 5.0e-3, 9.0, 10.0e-3, 10e9
    pec = Material.pec()
    diel = Material.from_isotropic(name="dielectric", epsilon=eps_r)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=d_a / 2, height=length, axis="z", material=diel)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=d_i / 2, height=length, axis="z", material=pec)
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
    mesh = Mesh.from_geometry(model, control, f_max=f_max)
    specs = [
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=eps_r, n_modes=1),
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, epsilon_r=eps_r, n_modes=1),
    ]
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions({}),
        ports=specs,
        f_max=f_max,
        verbose=False,
    )


def measure(name: str, ana: AnalysisScatteringTD) -> None:
    report = ana.solve_ports()["port1"]
    res = ana.run()
    s11 = res.db("port1", "port1")
    s21 = res.db("port2", "port1")
    print(
        f"  {name:24s} max|S11| {np.nanmax(s11):7.1f} dB"
        f"   median {np.nanmedian(s11):7.1f} dB"
        f"   |S21| dev {np.nanmax(np.abs(s21)):7.4f} dB"
        f"   z_line {report.z_line_num:6.2f} Ohm"
    )


def main() -> None:
    print("WP-R2 TEM port floors (exact DTBC + discrete de-stagger), band 0.25-10 GHz:")
    measure("parallel plate uniform", parallel_plate(graded=False))
    measure("parallel plate graded", parallel_plate(graded=True))
    measure("rect coax", rect_coax())
    measure("conformal round coax", conformal_round_coax())


if __name__ == "__main__":
    main()
