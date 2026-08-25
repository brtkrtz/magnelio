"""DD-191 certificate: a chamfer below the grid is reported, above it resolved.

The dielectric-resonator worksheet (internal record
`investigations/dr_filter/MEASUREMENTS.md`, M4/M4a) found the chamfer
of a ceramic puck to have *no* effect on its resonance — three
bit-identical eigenfrequencies for 0 / 0.2 / 0.5 mm, then a 16 %
jump at 0.8 mm.  Not a defect of the conformal material matrices:
the DD-051 entry averages the permittivity over the dual face
*transverse* to a grid edge, so a feature that thins the puck *along*
the z-edges inside the top and bottom cell layer has no lever until
it reaches the layer's midplane.  The mesher simply never saw the
chamfer — a chamfer face is a cone, and the face pass reads planes,
cylinders and spheres only.

DD-191 adds geometry-edge planes: wherever a B-rep edge lies flat in
an axis-normal plane (here the circle where the cone meets the
cylinder, at z = c and z = H - c) the grid gets a plane, so the
chamfer occupies one cell layer of its own and the layer's dual faces
see it.  Edge planes are a soft class, floored at
``h_max / max_edge_refinement`` (default 4) and reported when
dropped.

Fixture: the worksheet's coarse grid — one puck (8 mm / 4 mm bore /
6 mm, eps_r 45) in a 20 x 20 x 6 mm PEC housing, mnpw 12 at 3.5 GHz
(h_max = 1.064 mm, edge floor 0.266 mm at the default ratio, 0.133 mm
at ratio 8), lowest eigenmode around a 2.6 GHz shift.

Checks:

1. ``max_edge_refinement = 0`` reproduces the worksheet: 0 / 0.2 /
   0.5 mm give the same f0 (the chamfer is invisible), 0.8 mm jumps.
2. Default ratio: chamfers below the edge floor (0.1, 0.2 mm) warn
   and give the plain-puck f0 bit-for-bit — dropped means dropped,
   not half-applied; chamfers above it (0.3, 0.5, 0.8 mm) put grid
   nodes at z = c and H - c and move f0 monotonically.
3. Ratio 8: 0.2 mm joins the resolved chain; the whole chain
   0 < 0.2 < 0.3 < 0.5 < 0.8 mm is strictly monotone in f0, with no
   plateau and no jump.

Measured (2026-08-25): see the table the script prints.
"""

import warnings

import numpy as np

import magnelio as mio
from magnelio import geo

R_OUT, R_BORE, H, W = 4e-3, 2e-3, 6e-3, 20e-3
EPS_R = 45.0
F_MAX, MNPW, F_SHIFT = 3.5e9, 12, 2.6e9
CHAMFERS = (0.0, 0.1e-3, 0.2e-3, 0.3e-3, 0.5e-3, 0.8e-3)


def resonator(chamfer: float):
    ceramic = mio.Material("ceramic", epsilon=(EPS_R,) * 3)
    body = geo.Cylinder(origin=(0, 0, 0), radius=R_OUT, height=H, axis="z", material=ceramic)
    bore = geo.Cylinder(origin=(0, 0, 0), radius=R_BORE, height=H, axis="z")
    puck = body - bore
    if chamfer > 0:
        puck = puck.chamfered(edges="all", distance=chamfer)
    box = geo.Brick(origin=(-W / 2, -W / 2, 0), size=(W, W, H), material="air")
    model = mio.GeometryModel(background="pec")
    model.add(box - puck)
    model.add(puck)
    return model


def f0(chamfer: float, ratio: float):
    control = mio.MeshControl(min_nodes_per_wavelength=MNPW, max_edge_refinement=ratio)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = mio.Mesh.from_geometry(resonator(chamfer), control, f_max=F_MAX)
    dropped = [w for w in caught if "geometry-edge plane" in str(w.message)]
    z = np.asarray(mesh.grid.z)
    on_grid = chamfer > 0 and bool(
        np.any(np.abs(z - chamfer) < 1e-12) and np.any(np.abs(z - (H - chamfer)) < 1e-12)
    )
    result = mio.AnalysisEigenmode(
        mesh=mesh, n_modes=2, sigma=(2 * np.pi * F_SHIFT) ** 2, verbose=False
    ).run()
    return float(result.frequencies[0]), mesh.Nx * mesh.Ny * mesh.Nz, on_grid, len(dropped)


def main() -> int:
    failures: list[str] = []
    table: dict[float, dict[float, tuple]] = {}
    for ratio in (0.0, 4.0, 8.0):
        table[ratio] = {c: f0(c, ratio) for c in CHAMFERS}

    head = "f0 [GHz]   cells  on  warn"
    print("chamfer [mm] | ratio 0 (legacy)          | ratio 4 (default)          | ratio 8")
    print(f"             | {head} | {head} | {head}")
    for c in CHAMFERS:
        row = f"{c * 1e3:12.1f} |"
        for ratio in (0.0, 4.0, 8.0):
            f, cells, on, n_warn = table[ratio][c]
            row += f" {f / 1e9:9.5f} {cells:6d}  {'y' if on else '-'}  {n_warn:4d} |"
        print(row)

    def f_of(ratio, c):
        return table[ratio][c][0]

    # 1. legacy plateau + jump
    plateau = [f_of(0.0, c) for c in (0.0, 0.2e-3, 0.5e-3)]
    if max(plateau) - min(plateau) > 1e-6 * plateau[0]:
        failures.append(f"legacy: 0/0.2/0.5 mm are not a plateau: {plateau}")
    if f_of(0.0, 0.8e-3) < 1.05 * f_of(0.0, 0.0):
        failures.append("legacy: 0.8 mm did not jump")

    # 2. default ratio: dropped means dropped, resolved means on grid + monotone
    for c in (0.1e-3, 0.2e-3):
        f, _cells, on, n_warn = table[4.0][c]
        if n_warn == 0 or on:
            failures.append(f"default: {c * 1e3} mm chamfer should be dropped with a warning")
        # Same grid, same operator — the eigensolver's last digit may
        # still differ between runs (ARPACK iteration noise).
        if abs(f - f_of(4.0, 0.0)) > 1e-12 * f:
            failures.append(
                f"default: dropped {c * 1e3} mm chamfer changed f0 ({f} vs {f_of(4.0, 0.0)})"
            )
    chain = [f_of(4.0, c) for c in (0.0, 0.3e-3, 0.5e-3, 0.8e-3)]
    for c in (0.3e-3, 0.5e-3, 0.8e-3):
        if not table[4.0][c][2] or table[4.0][c][3]:
            failures.append(
                f"default: {c * 1e3} mm chamfer should be on the grid without a warning"
            )
    if not all(a < b for a, b in zip(chain, chain[1:])):
        failures.append(f"default: resolved chain is not strictly monotone: {chain}")

    # 3. ratio 8: the full chain from 0.2 mm on
    chain8 = [f_of(8.0, c) for c in (0.0, 0.2e-3, 0.3e-3, 0.5e-3, 0.8e-3)]
    if not all(a < b for a, b in zip(chain8, chain8[1:])):
        failures.append(f"ratio 8: chain is not strictly monotone: {chain8}")
    # No single resolved step may carry what the legacy grid delivered as
    # one 0 -> 0.8 mm jump; the steps themselves are uneven (the first
    # tenths of a millimetre matter most — the worksheet's fine grid
    # showed the same), so evenness is not a criterion.
    legacy_jump = f_of(0.0, 0.8e-3) - f_of(0.0, 0.0)
    if np.diff(chain8).max() > 0.5 * legacy_jump:
        failures.append(f"ratio 8: a single step carries the legacy jump: {np.diff(chain8)}")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("PASS: sub-floor chamfers reported, resolved chamfers monotone (DD-191).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
