"""Certificate: MonitorFarField against textbook antennas (DD-173).

Three gates on the thin-wire dipole family:

A. **Half-wave dipole, full model** — peak directivity at the
   textbook 2.15 dBi (±0.15 dB) and the lossless power closure
   ``P_rad == 1 − |S11|²`` per incident watt (< 3 %; measured
   2.2 % surface-quadrature defect at introduction).
B. **Quarter-wave monopole on a PEC ground** — the tutorial-08
   topology: box floor replaced by image theory, pattern masked
   below the plane, horizon directivity = 2× the dipole's (< 5 %).
C. **Symmetry parity** — SymmetryPEC half model with the DD-172
   lumped feed on the plane: full-model pattern within 8 %, P_rad
   within 5 % (floor: the KB-023 CPML mirror asymmetry).

Run from the repository root:

    mamba run --no-capture-output -n mio python \
        validation/farfield_dipole_certificate.py
"""

import numpy as np

from magnelio import AnalysisScatteringTD, Mesh
from magnelio.circuit.rasterize import EdgePath
from magnelio.mesh._thin_wire import apply_thin_wire_path
from magnelio.mesh.grid import GridLines
from magnelio.mesh.indexing import edge_index_Ez
from magnelio.monitors import MonitorFarField
from magnelio.ports import PortSpecLumped

D = 1e-3
A_RADIUS = 0.05 * D
H_CELLS = 15
PAD, PML = 4, 8
Z0_FEED = 73.0
N_STEPS = 8000
F_AXIS = np.linspace(3.2e9, 6.2e9, 121)
F0 = 4.6e9


def _wire_mesh(z_nodes, t_nodes, wire_ks):
    grid = GridLines(x=t_nodes, y=t_nodes.copy(), z=z_nodes)
    mesh = Mesh.from_grid(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    i0, j0 = Nx // 2, Ny // 2
    path = EdgePath(
        axes=["z"] * len(wire_ks),
        ijk=[(i0, j0, k) for k in wire_ks],
        signs=[1] * len(wire_ks),
        dls=[float(grid.dz[k]) for k in wire_ks],
        flat_indices=[n_Ex + n_Ey + edge_index_Ez(i0, j0, k, Nx, Ny, Nz) for k in wire_ks],
    )
    for flat in path.flat_indices:
        mesh.pec_mask_edges[2, flat - n_Ex - n_Ey] = True
    apply_thin_wire_path(mesh, path, A_RADIUS, name="dipole")
    return mesh, grid, i0, j0


def _run(mesh, grid, i0, j0, s, e, bc):
    x0, y0 = float(grid.x[i0]), float(grid.y[j0])
    ff = MonitorFarField(freqs=[F0], margin_cells=2, name="pattern")
    ana = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(bc),
        ports=[PortSpecLumped(name="feed", start=(x0, y0, s), end=(x0, y0, e), Z0=Z0_FEED)],
        f_max=8e9,
        verbose=False,
        monitors=(ff,),
    )
    res = ana.run(f_axis=F_AXIS, excited=["feed"], total_time_steps=N_STEPS, energy_stop_db=None)
    return res, ff.result(F0)


def _db(x):
    return 10.0 * np.log10(x)


def main() -> bool:
    # Full dipole (2-cell centre gap).
    span = 2 * H_CELLS
    nz, nt = span + 2 * (PAD + PML), 1 + 2 * (PAD + PML)
    z = np.linspace(0, nz * D, nz + 1)
    t = np.linspace(0, nt * D, nt + 1)
    k_c = PAD + PML + H_CELLS
    wire_ks = [k for k in range(PAD + PML, PAD + PML + span) if k not in {k_c - 1, k_c}]
    mesh, grid, i0, j0 = _wire_mesh(z, t, wire_ks)
    bc = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    res, pat_full = _run(mesh, grid, i0, j0, grid.z[k_c - 1], grid.z[k_c + 1], bc)

    d_peak = float(np.max(pat_full.directivity))
    s11 = float(np.interp(F0, F_AXIS, np.abs(res.S("feed", "feed"))))
    accepted = 1.0 - s11**2
    closure = abs(pat_full.P_rad / accepted - 1.0)
    ok_a = abs(_db(d_peak) - 2.15) < 0.15 and closure < 3e-2
    print(
        f"[{'PASS' if ok_a else 'FAIL'}] A dipole: D = {_db(d_peak):.2f} dBi, "
        f"P_rad/accepted - 1 = {closure:.2e}"
    )

    # Monopole on PEC ground.
    nz_h = H_CELLS + PAD + PML
    z_h = np.linspace(0, nz_h * D, nz_h + 1)
    mesh_m, grid_m, i0, j0 = _wire_mesh(z_h, t, list(range(1, H_CELLS)))
    bc_m = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmax")}
    bc_m["zmin"] = "PEC"
    _, pat_mono = _run(mesh_m, grid_m, i0, j0, 0.0, float(grid_m.z[1]), bc_m)
    i_eq = int(np.argmin(np.abs(pat_mono.theta - np.pi / 2)))
    d_hor = float(np.max(pat_mono.directivity[i_eq, :]))
    masked = pat_mono.U[pat_mono.theta > np.pi / 2 + 1e-9, :].max() == 0.0
    ok_b = masked and abs(d_hor / (2.0 * d_peak) - 1.0) < 5e-2
    print(
        f"[{'PASS' if ok_b else 'FAIL'}] B monopole: horizon D = {_db(d_hor):.2f} dBi "
        f"(dipole + 3 dB = {_db(2 * d_peak):.2f}), lower half masked = {masked}"
    )

    # SymmetryPEC half model with the lumped feed on the plane.
    mesh_s, grid_s, i0, j0 = _wire_mesh(z_h, t, list(range(1, H_CELLS)))
    bc_s = dict(bc_m)
    bc_s["zmin"] = "ForceSymmetryPEC"
    _, pat_half = _run(mesh_s, grid_s, i0, j0, 0.0, float(grid_s.z[1]), bc_s)
    dp = abs(pat_half.P_rad / pat_full.P_rad - 1.0)
    dg = float(
        np.max(np.abs(pat_half.realized_gain - pat_full.realized_gain))
        / np.max(pat_full.realized_gain)
    )
    ok_c = pat_half.physical_mask is None and dp < 5e-2 and dg < 8e-2
    print(
        f"[{'PASS' if ok_c else 'FAIL'}] C symmetry parity: dP_rad = {dp:.2e}, "
        f"max pattern deviation = {dg:.2e}"
    )
    return ok_a and ok_b and ok_c


if __name__ == "__main__":
    ok = main()
    print("certificate:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
