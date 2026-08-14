"""WP-M2 acceptance: thin-sheet microstrip impedance vs resolved reference.

The session-91 reproducer geometry (shielded microstrip, 0.635 mm
eps_r = 4.3 substrate, 1.8 mm wide strip of t = 35 um) is meshed two
ways and the QTEM fundamental is solved through the public modal-port
factory on both:

* **resolved** — no ``min_cell_size`` floor; both metallization faces
  are critical planes and the 35 um layer is resolved by real cells
  (the pre-WP-M2 behaviour, here the reference).
* **thin-sheet** — ``min_cell_size = 100 um``; the WP-M2 pipeline gives
  the strip ONE grid plane at its substrate-side face (tangential-E
  mask) and the metal volume enters through the DD-051 sub-cell
  classification of the adjacent cells.

Both branches use the same transverse ``max_cell_size`` so the
comparison isolates the sheet-model error from ordinary transverse
discretization error (measured session 92: at a shared coarse bulk
the deltas are dominated by the bulk, 6.5 % on Z_0; at matched
100 um transverse resolution the sheet model itself is at
0.04 % on Z_0 / 1.06 % on eps_eff with the full WP-M3/M4 mesher —
hard floor + per-axis h_fine; 1.1 % / 1.5 % at the WP-M3 state).

The thin-sheet model must not silently change the line impedance
beyond the sub-cell approximation error.  Gate: 2 % (the architecture
stretch target) — accepted by the developer 2026-07-10 (session 92)
from the measured numbers.

Run with: ``mamba run --no-capture-output -n mio python
validation/thin_sheet_impedance_sanity.py``.
"""

from __future__ import annotations

import numpy as np

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BoxFace,
    PortSpecMultiConductor,
    RegionConductor,
    WallConductor,
    build_modal_port,
)
from magnelio.solver.stability import courant_dt

# Session-91 reproducer parameters.
H_SUB = 0.635e-3
T_MET = 35e-6
W_STRIP = 1.8e-3
EPS_R = 4.3
W_BOX = 6.0e-3
H_AIR = 3.0e-3
L_X = 2.0e-3
F_MAX = 10.0e9
FLOOR = 100e-6

ACCEPTED_GATE = 0.02  # 2 % — developer sign-off 2026-07-10 (session 92)


def _model() -> GeometryModel:
    sub = Material(name="substrate", epsilon=(EPS_R,) * 3)
    pec = Material.pec()
    air = Material.air()
    strip = Brick(
        origin=(0.0, -W_STRIP / 2, H_SUB),
        size=(L_X, W_STRIP, T_MET),
        material=pec,
    )
    air_brick = Brick(
        origin=(0.0, -W_BOX / 2, H_SUB),
        size=(L_X, W_BOX, H_AIR),
        material=air,
    )
    model = GeometryModel()
    model.add(
        Brick(
            origin=(0.0, -W_BOX / 2, 0.0),
            size=(L_X, W_BOX, H_SUB),
            material=sub,
        )
    )
    model.add(Difference(air_brick, strip))
    model.add(strip)
    return model


def _solve(mesh: Mesh, strip_z: tuple[float, float]):
    mesh = mesh.with_boundary_conditions(
        {
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PEC",
            "xmin": "PMC",
            "xmax": "PMC",
        }
    )
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(mesh.grid, accuracy="normal")
    spec = PortSpecMultiConductor(
        name="microstrip",
        plane=BoxFace.X_MIN,
        conductors=(
            WallConductor(face=BoxFace.Z_MIN),  # shield floor = ground
            RegionConductor(
                axis_a_range=(-W_STRIP / 2, W_STRIP / 2),
                axis_b_range=strip_z,
            ),
        ),
        epsilon_r=None,  # QTEM dispatch
    )
    op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=5e9)
    mode = op.discrete_modes[0].mode
    return mode.epsilon_r, mode.z_line


# Shared transverse resolution target = the floor: the thin-sheet
# branch then meshes at ~uniform 100 um (the finest the floor allows)
# and the resolved branch refines below it only around the 35 um
# layer — the remaining delta is the sheet-model error.  No interior
# x-planes -> the x-axis meshes as ONE uniform interval, giving the
# >= 3 equidistant port-normal cells the modal factory requires.
_H_MAX = FLOOR


def resolved_mesh() -> Mesh:
    control = MeshControl(
        min_nodes_per_wavelength=10,
        min_cells_per_feature=2,
        growth_factor=1.3,
        max_cell_size=_H_MAX,
    )
    return Mesh.from_geometry(_model(), control, f_max=F_MAX)


def thin_sheet_mesh() -> Mesh:
    control = MeshControl(
        min_nodes_per_wavelength=10,
        min_cells_per_feature=2,
        growth_factor=1.3,
        max_cell_size=_H_MAX,
        min_cell_size=FLOOR,
    )
    return Mesh.from_geometry(_model(), control, f_max=F_MAX)


def main() -> None:
    print("thin-sheet impedance sanity (WP-M2 acceptance)")
    print("=" * 60)

    mesh_ref = resolved_mesh()
    d_ref = min(mesh_ref.grid.dx.min(), mesh_ref.grid.dy.min(), mesh_ref.grid.dz.min())
    print(
        f"resolved:   {mesh_ref.Nx} x {mesh_ref.Ny} x {mesh_ref.Nz} "
        f"cells, d_min = {d_ref * 1e6:.1f} um, "
        f"dt = {courant_dt(mesh_ref.grid, 'normal') * 1e12:.4f} ps"
    )
    eps_ref, z_ref = _solve(mesh_ref, (H_SUB, H_SUB + T_MET))

    mesh_ts = thin_sheet_mesh()
    d_ts = min(mesh_ts.grid.dx.min(), mesh_ts.grid.dy.min(), mesh_ts.grid.dz.min())
    gz = np.asarray(mesh_ts.grid.z)
    assert H_SUB in gz, "sheet plane missing"
    assert not np.any(np.abs(gz - (H_SUB + T_MET)) < 1e-9), "far-side face leaked into the grid"
    print(
        f"thin-sheet: {mesh_ts.Nx} x {mesh_ts.Ny} x {mesh_ts.Nz} "
        f"cells, d_min = {d_ts * 1e6:.1f} um, "
        f"dt = {courant_dt(mesh_ts.grid, 'normal') * 1e12:.4f} ps"
    )
    eps_ts, z_ts = _solve(mesh_ts, (H_SUB, H_SUB))

    d_eps = abs(eps_ts - eps_ref) / eps_ref
    d_z = abs(z_ts - z_ref) / z_ref
    print()
    print(f"  eps_eff:  resolved = {eps_ref:.4f}, thin-sheet = {eps_ts:.4f}  ({d_eps * 100:.2f} %)")
    print(
        f"  Z_0:      resolved = {z_ref:6.2f} Ohm, "
        f"thin-sheet = {z_ts:6.2f} Ohm  ({d_z * 100:.2f} %)"
    )
    print()

    if d_eps > ACCEPTED_GATE or d_z > ACCEPTED_GATE:
        raise SystemExit(
            f"FAIL ({ACCEPTED_GATE:.0%} gate): eps_eff off {d_eps:.2%}, Z_0 off {d_z:.2%}."
        )
    print(f"PASS: within the accepted {ACCEPTED_GATE:.0%} gate.")


if __name__ == "__main__":
    main()
