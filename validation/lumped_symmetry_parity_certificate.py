"""Certificate: lumped ports/elements on symmetry planes (DD-172).

Three gates pin the full-model power semantics of a lumped device cut
by a symmetry plane:

A. **Exact restriction (PEC crossing)** — a center-fed thin-wire
   dipole in an all-PEC cavity, run full and as a half model with the
   electric symmetry plane through the feed (double precision, half
   run pinned to the full run's dt).  The Thévenin split makes the
   half update the exact discrete restriction of the full one:
   V_full == sqrt(2) * V~_half and I_full == I~_half / sqrt(2) to
   machine noise (measured 5e-16 / 8e-16 at introduction).

B. **Open-boundary parity (PEC crossing)** — the same dipole under
   CPML.  Parity here is physics-level, floored by the CPML min/max
   mirror asymmetry (KB-023): measured max |S11_half - S11_full|
   ~ 2.1e-2 at introduction, against the O(0.3) error of an unscaled
   feed (the monopole trap).

C. **Parallel cut (PMC containment)** — a passive lumped load lying
   in a magnetic symmetry plane presents the internally doubled
   trapezoidal impedance at its terminals: the meshed half is one of
   two parallel branches (measured < 1e-8 relative at introduction).

Run from the repository root:

    mamba run --no-capture-output -n mio python \
        validation/lumped_symmetry_parity_certificate.py
"""

import numpy as np

import magnelio.analysis.scattering_td as std
from magnelio import AnalysisScatteringTD, Mesh
from magnelio.circuit import SeriesRLC
from magnelio.circuit.rasterize import EdgePath
from magnelio.mesh import BoxFace
from magnelio.mesh._thin_wire import apply_thin_wire_path
from magnelio.mesh.grid import GridLines
from magnelio.mesh.indexing import edge_index_Ez
from magnelio.ports import PortSpecLumped, PortSpecMultiConductor

D = 1e-3
A_RADIUS = 0.05 * D
H_CELLS = 15
PAD, PML = 4, 8
Z0_FEED = 73.0
F_AXIS = np.linspace(3.2e9, 6.2e9, 121)


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


def _dipole_run(mesh, grid, i0, j0, s, e, bc, *, precision=None, n_steps=8000, f_axis=F_AXIS):
    x0, y0 = float(grid.x[i0]), float(grid.y[j0])
    ana = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(bc),
        ports=[PortSpecLumped(name="feed", start=(x0, y0, s), end=(x0, y0, e), Z0=Z0_FEED)],
        f_max=8e9,
        verbose=False,
        precision=precision,
    )
    return ana.run(f_axis=f_axis, excited=["feed"], total_time_steps=n_steps, energy_stop_db=None)


def gate_a_exact_restriction():
    span = 2 * H_CELLS
    nz, nt = span + 2 * PAD, 1 + 2 * PAD
    z = np.linspace(0, nz * D, nz + 1)
    t = np.linspace(0, nt * D, nt + 1)
    k_c = PAD + H_CELLS
    wire_ks = [k for k in range(PAD, PAD + span) if k not in {k_c - 1, k_c}]
    mesh_f, grid_f, i0, j0 = _wire_mesh(z, t, wire_ks)
    bc = {f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    f_ax = np.array([4e9, 5e9])
    res_f = _dipole_run(
        mesh_f,
        grid_f,
        i0,
        j0,
        grid_f.z[k_c - 1],
        grid_f.z[k_c + 1],
        bc,
        precision="double",
        n_steps=1500,
        f_axis=f_ax,
    )
    z_h = np.linspace(0, (H_CELLS + PAD) * D, H_CELLS + PAD + 1)
    mesh_h, grid_h, _, _ = _wire_mesh(z_h, t, list(range(1, H_CELLS)))
    bc_h = dict(bc)
    bc_h["zmin"] = "ForceSymmetryPEC"
    orig = std.spectral_dt
    std.spectral_dt = lambda *a, **k: res_f.dt
    try:
        res_h = _dipole_run(
            mesh_h,
            grid_h,
            i0,
            j0,
            0.0,
            float(grid_h.z[1]),
            bc_h,
            precision="double",
            n_steps=1500,
            f_axis=f_ax,
        )
    finally:
        std.spectral_dt = orig
    V_f, I_f = res_f.signals[("feed", 0)][("feed", 0)]
    V_h, I_h = res_h.signals[("feed", 0)][("feed", 0)]
    n = min(len(V_f.values), len(V_h.values))
    dV = np.max(np.abs(V_f.values[:n] - np.sqrt(2.0) * V_h.values[:n])) / np.max(
        np.abs(V_f.values[:n])
    )
    dI = np.max(np.abs(I_f.values[:n] - I_h.values[:n] / np.sqrt(2.0))) / np.max(
        np.abs(I_f.values[:n])
    )
    ok = dV < 1e-12 and dI < 1e-12
    print(f"[{'PASS' if ok else 'FAIL'}] A exact restriction: dV = {dV:.2e}, dI = {dI:.2e}")
    return ok


def gate_b_open_boundary_parity():
    span = 2 * H_CELLS
    nz, nt = span + 2 * (PAD + PML), 1 + 2 * (PAD + PML)
    z = np.linspace(0, nz * D, nz + 1)
    t = np.linspace(0, nt * D, nt + 1)
    k_c = PAD + PML + H_CELLS
    wire_ks = [k for k in range(PAD + PML, PAD + PML + span) if k not in {k_c - 1, k_c}]
    mesh_f, grid_f, i0, j0 = _wire_mesh(z, t, wire_ks)
    bc_f = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    s_full = _dipole_run(mesh_f, grid_f, i0, j0, grid_f.z[k_c - 1], grid_f.z[k_c + 1], bc_f).S(
        "feed", "feed"
    )
    z_h = np.linspace(0, (H_CELLS + PAD + PML) * D, H_CELLS + PAD + PML + 1)
    mesh_h, grid_h, _, _ = _wire_mesh(z_h, t, list(range(1, H_CELLS)))
    bc_h = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmax")}
    bc_h["zmin"] = "ForceSymmetryPEC"
    s_half = _dipole_run(mesh_h, grid_h, i0, j0, 0.0, float(grid_h.z[1]), bc_h).S("feed", "feed")
    err = float(np.max(np.abs(s_half - s_full)))
    ok = err < 5e-2
    print(
        f"[{'PASS' if ok else 'FAIL'}] B open-boundary parity: "
        f"max |dS11| = {err:.2e} (KB-023 floor ~1e-2, unscaled feed ~3e-1)"
    )
    return ok


def gate_c_pmc_containment():
    element = SeriesRLC(R=75.0, L=5e-9, C=2e-12)
    gap, width, length = 5e-3, 16e-3, 60e-3
    grid = GridLines(
        x=np.linspace(0.0, width / 2, 5),
        y=np.linspace(-gap / 2, gap / 2, 6),
        z=np.linspace(-length / 2, length / 2, 121),
    )
    bc = {
        "xmin": "ForceSymmetryPMC",
        "xmax": "PMC",
        "ymin": "PEC",
        "ymax": "PEC",
        "zmin": "PEC",
        "zmax": "PMC",
    }
    f_axis = np.array([0.8e9, 1.7e9, 3.1e9])
    ana = AnalysisScatteringTD(
        mesh=Mesh.from_grid(grid, boundary_conditions=bc),
        ports=[
            PortSpecMultiConductor(name="m1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecLumped(
                name="load",
                start=(0.0, -gap / 2, grid.z[-3]),
                end=(0.0, gap / 2, grid.z[-3]),
                Z0=50.0,
                element=element,
            ),
        ],
        f_max=6e9,
        verbose=False,
    )
    res = ana.run(f_axis=f_axis, excited=["m1"], total_time_steps=20000, energy_stop_db=None)
    V, I = res.signals[("m1", 0)][("load", 0)]
    z_meas = -V.at_frequencies(f_axis) / I.at_frequencies(f_axis)
    jwt = 1j * (2.0 / V.dt) * np.tan(2.0 * np.pi * f_axis * V.dt / 2.0)
    z_ref = 2.0 * (element.R + jwt * element.L) + 1.0 / (jwt * (element.C / 2.0))
    rel = float(np.max(np.abs(z_meas - z_ref) / np.abs(z_ref)))
    ok = rel < 1e-6
    print(f"[{'PASS' if ok else 'FAIL'}] C PMC containment: rel Z defect = {rel:.2e}")
    return ok


if __name__ == "__main__":
    results = [gate_a_exact_restriction(), gate_b_open_boundary_parity(), gate_c_pmc_containment()]
    print("certificate:", "PASS" if all(results) else "FAIL")
    raise SystemExit(0 if all(results) else 1)
