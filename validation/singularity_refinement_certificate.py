"""DD-194 certificate: the impedance error of a microstrip lives in the edge cell.

At a conductor edge the field and the surface current are singular
(``r^(-1/3)`` at the 90° edge of a strip of finite thickness), and the
line impedance a mode solver reads off the grid converges only slowly
in the cell that holds the edge.  ``MeshControl(singularity_refinement=k)``
starts the grading at the planes holding such an edge at ``h_fine / k``
instead of ``h_fine`` (both sides, DD-194).

Fixture: the 50 Ω microstrip of the mesh-convergence how-to (0.8 mm
FR4, 1.2 x 0.2 mm strip, half-modelled on its PMC symmetry plane),
port mode solved at 15 GHz without a time-domain run.  Ladder over
``min_nodes_per_wavelength`` with ``min_cells_per_feature = mnpw // 4``
(the how-to's rung), factors 1, 2 and 4.

Checks:

1. Z0 rises monotonically with the ladder for every factor and is
   the *same* (within 0.1 Ω) wherever the edge cell is the same —
   factor 1 at mnpw 64, factor 2 at mnpw 32 and factor 4 at mnpw 16
   all hold a 12.5 µm edge cell — although the cell counts differ by
   almost two.  The edge cell, not the cell count, sets the error.
2. Factor 2 two rungs lower lands within 0.1 Ω of factor 1's finest
   rung at two thirds of the cells, and the first-order extrapolation
   in the edge cell puts the limit about 0.35 Ω (0.7 %) above that
   finest plain rung — an offset a ladder over the bulk size alone
   never shows, because every rung of it moves the edge cell too.

What the certificate does NOT claim: a runtime gain at fixed
accuracy.  The edge cell also bounds the time step, so on the
S-parameter ladder of the how-to the three factors lie on one
cost-versus-error curve (measured 2026-08-25, DD-194) — the factor
redistributes resolution from the bulk to the edges; it pays where
the impedance or the effective permittivity is the quantity of
interest, or where the time step is bound by a floor anyway.

Measured (2026-08-25): see the table the script prints.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUPY_ACCELERATORS", "")

import numpy as np  # noqa: E402

import magnelio as mio  # noqa: E402
from magnelio import geo, ports  # noqa: E402

H_SUB, W_STRIP, T_STRIP = 0.8e-3, 1.2e-3, 0.2e-3
W_BOX, H_BOX, L = 8.0e-3, 5.0e-3, 4.0e-3
F_MAX = 15.0e9
FR4 = mio.Material.from_isotropic(name="FR4", epsilon=4.3)
LADDER = (16, 32, 64)
FACTORS = (1.0, 2.0, 4.0)


def model():
    substrate = geo.Brick(origin=(-W_BOX / 2, 0, 0), size=(W_BOX, H_SUB, L), material=FR4)
    air = geo.Brick(origin=(-W_BOX / 2, H_SUB, 0), size=(W_BOX, H_BOX - H_SUB, L), material="air")
    strip = geo.Brick(origin=(-W_STRIP / 2, H_SUB, 0), size=(W_STRIP, T_STRIP, L), material="pec")
    m = mio.GeometryModel(boundary_conditions={"xmin": "SymmetryPMC"})
    m.add(substrate)
    m.add(air - strip)
    m.add(strip)
    m.add_port(ports.PortWaveguide(name="port1", plane="zmin", n_modes=1))
    return m


def z0_on(mnpw: int, k: float):
    control = mio.MeshControl(
        min_nodes_per_wavelength=mnpw,
        min_cells_per_feature=max(2, mnpw // 4),
        singularity_refinement=k,
    )
    mesh = mio.Mesh.from_geometry(model(), control, f_max=F_MAX)
    report = mio.AnalysisScatteringTD(mesh=mesh, verbose=False).solve_ports()["port1"]
    edge_cell = float(np.diff(mesh.grid.y).min())
    return report.z_line_num, mesh.Nx * mesh.Ny, edge_cell


def main() -> None:
    rows: dict[tuple[float, int], tuple[float, int, float]] = {}
    print("factor  mnpw   cells   edge cell [um]   Z0 [ohm]")
    for k in FACTORS:
        for mnpw in LADDER:
            z0, cells, edge = z0_on(mnpw, k)
            rows[(k, mnpw)] = (z0, cells, edge)
            print(f"{k:5.0f}  {mnpw:5d}  {cells:6d}   {edge * 1e6:12.1f}   {z0:9.4f}")

    ok = True
    # 1. monotone in the ladder, equal where the edge cell is equal
    for k in FACTORS:
        z = [rows[(k, m)][0] for m in LADDER]
        if not all(b > a for a, b in zip(z, z[1:])):
            ok = False
            print(f"FAIL: factor {k:g} is not monotone: {z}")
    same_edge = [(1.0, 64), (2.0, 32), (4.0, 16)]
    z_same = [rows[key][0] for key in same_edge]
    e_same = [rows[key][2] for key in same_edge]
    cells_same = [rows[key][1] for key in same_edge]
    spread = max(z_same) - min(z_same)
    print(
        f"same edge cell {[round(e * 1e6, 1) for e in e_same]} um: "
        f"Z0 spread {spread:.3f} ohm over {cells_same} cells"
    )
    if max(e_same) - min(e_same) > 1e-9 or spread > 0.1:
        ok = False
        print("FAIL: equal edge cells should give equal Z0 within 0.1 ohm")
    if not cells_same[0] > cells_same[1] > cells_same[2]:
        ok = False
        print("FAIL: the refined ladders should reach the edge cell with fewer cells")

    # 2. the plain ladder's finest rung vs factor 2 two rungs lower
    z1 = rows[(1.0, 64)][0]
    z2 = rows[(2.0, 32)][0]
    # first-order extrapolation in the edge cell from the factor-4 ladder
    a, b = rows[(4.0, 32)], rows[(4.0, 64)]
    z_inf = b[0] + (b[0] - a[0]) * b[2] / (a[2] - b[2])
    print(f"extrapolated Z0 (first order in the edge cell): {z_inf:.3f} ohm")
    print(f"factor 1 @ 64: {z_inf - z1:+.3f} ohm   factor 2 @ 32: {z_inf - z2:+.3f} ohm")
    if abs(z1 - z2) > 0.1:
        ok = False
        print("FAIL: factor 2 two rungs lower should match factor 1's finest rung")
    if not 0.2 < z_inf - z1 < 0.6:
        ok = False
        print("FAIL: the extrapolated limit should sit 0.2-0.6 ohm above the plain ladder")

    print("CERTIFICATE PASSED" if ok else "CERTIFICATE FAILED")


if __name__ == "__main__":
    main()
