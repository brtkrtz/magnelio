"""DD-172 parity gates: lumped ports/elements on symmetry planes.

Two gates pin the full-model power semantics:

1. **PEC crossing** — a center-fed thin-wire dipole run full and as a
   half model with an electric symmetry plane through the feed.  The
   Thévenin split (half source in series with half the internal
   impedance) makes the half-model update the exact discrete
   restriction of the full model, so S11 and Z_in must agree to solver
   noise — not to a factor of two, which is what an unscaled feed
   produces (the historic monopole-vs-dipole trap).
2. **PMC containment** — a passive lumped load lying in a magnetic
   symmetry plane presents the internally doubled trapezoidal
   impedance at its terminals (one of two parallel branches carries
   the meshed half), measured through the same per-step KVL identity
   as the DD-077 gates.
"""

from __future__ import annotations

import numpy as np

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
H_CELLS = 15  # one dipole arm incl. its gap cell [cells]
PAD, PML = 4, 8
Z0_FEED = 73.0
N_STEPS = 8000
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


def _s11(mesh, grid, i0, j0, feed_start_z, feed_end_z, bc):
    x0, y0 = float(grid.x[i0]), float(grid.y[j0])
    ana = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(bc),
        ports=[
            PortSpecLumped(
                name="feed",
                start=(x0, y0, feed_start_z),
                end=(x0, y0, feed_end_z),
                Z0=Z0_FEED,
            )
        ],
        f_max=8e9,
        verbose=False,
    )
    res = ana.run(f_axis=F_AXIS, excited=["feed"], total_time_steps=N_STEPS, energy_stop_db=None)
    return res.S("feed", "feed")


def test_pec_crossing_exact_restriction_in_cavity(monkeypatch):
    """The half-model port update is the exact restriction of the full one.

    In an all-PEC cavity (no absorber) the half model with the electric
    symmetry plane through the feed is algebraically the restriction of
    the full model: the Thévenin split gives i_half == i_full per step,
    and the recorded √2-scaled half voltage equals half the full gap
    voltage times √2 — i.e. V_full == √2 · Ṽ_half exactly.  Run in
    double precision with the half run pinned to the full run's time
    step (the Lanczos dt of the half operator differs slightly, which
    would turn the identity into a discretisation comparison).
    """
    span = 2 * H_CELLS
    nz = span + 2 * PAD
    nt = 1 + 2 * PAD
    z = np.linspace(0, nz * D, nz + 1)
    t = np.linspace(0, nt * D, nt + 1)
    k_c = PAD + H_CELLS
    gap_ks = {k_c - 1, k_c}
    wire_ks = [k for k in range(PAD, PAD + span) if k not in gap_ks]
    mesh_f, grid_f, i0, j0 = _wire_mesh(z, t, wire_ks)
    bc_pec = {f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}

    def run(mesh, grid, s, e, bc):
        x0, y0 = float(grid.x[i0]), float(grid.y[j0])
        ana = AnalysisScatteringTD(
            mesh=mesh.with_boundary_conditions(bc),
            ports=[PortSpecLumped(name="feed", start=(x0, y0, s), end=(x0, y0, e), Z0=Z0_FEED)],
            f_max=8e9,
            verbose=False,
            precision="double",
        )
        return ana.run(
            f_axis=np.array([4e9, 5e9]),
            excited=["feed"],
            total_time_steps=1500,
            energy_stop_db=None,
        )

    res_f = run(mesh_f, grid_f, grid_f.z[k_c - 1], grid_f.z[k_c + 1], bc_pec)

    import magnelio.analysis.scattering_td as std

    monkeypatch.setattr(std, "spectral_dt", lambda *a, **k: res_f.dt)
    nz_h = H_CELLS + PAD
    z_h = np.linspace(0, nz_h * D, nz_h + 1)
    mesh_h, grid_h, _, _ = _wire_mesh(z_h, t, list(range(1, H_CELLS)))
    bc_half = dict(bc_pec)
    bc_half["zmin"] = "ForceSymmetryPEC"
    res_h = run(mesh_h, grid_h, 0.0, float(grid_h.z[1]), bc_half)

    V_f, I_f = res_f.signals[("feed", 0)][("feed", 0)]
    V_h, I_h = res_h.signals[("feed", 0)][("feed", 0)]
    n = min(len(V_f.values), len(V_h.values))
    scale_v = np.max(np.abs(V_f.values[:n]))
    dV = np.max(np.abs(V_f.values[:n] - np.sqrt(2.0) * V_h.values[:n])) / scale_v
    scale_i = np.max(np.abs(I_f.values[:n]))
    dI = np.max(np.abs(I_f.values[:n] - I_h.values[:n] / np.sqrt(2.0))) / scale_i
    assert dV < 1e-12, f"relative V restriction defect = {dV:.2e}"
    assert dI < 1e-12, f"relative I restriction defect = {dI:.2e}"


def test_pec_crossing_full_half_parity_open_boundary():
    """Half dipole + SymmetryPEC feed == full dipole, not the monopole.

    With CPML boundaries the parity is physics-level, not exact: the
    min/max absorber profiles are sampled at cell centres for both the
    node-registered E and the cell-registered H updates, so the two
    faces are not mirror images and their residual reflections differ
    (KB entry).  The resonant dipole recycles that ~1e-4 field-level
    residual into ~1e-2 on S11 — the tolerance sits above it, and far
    below the O(0.3) error of an unscaled feed (the monopole trap the
    gate exists to catch).
    """
    span = 2 * H_CELLS
    nz = span + 2 * (PAD + PML)
    nt = 1 + 2 * (PAD + PML)
    z = np.linspace(0, nz * D, nz + 1)
    t = np.linspace(0, nt * D, nt + 1)
    k_c = PAD + PML + H_CELLS  # center node
    gap_ks = {k_c - 1, k_c}
    wire_ks = [k for k in range(PAD + PML, PAD + PML + span) if k not in gap_ks]
    mesh_f, grid_f, i0, j0 = _wire_mesh(z, t, wire_ks)
    bc_full = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    s_full = _s11(mesh_f, grid_f, i0, j0, grid_f.z[k_c - 1], grid_f.z[k_c + 1], bc_full)

    # Half model: as-built halved geometry, gap edge at the plane,
    # feed declared from the plane with the FULL-model Z0.
    nz_h = H_CELLS + PAD + PML
    z_h = np.linspace(0, nz_h * D, nz_h + 1)
    mesh_h, grid_h, i0, j0 = _wire_mesh(z_h, t, list(range(1, H_CELLS)))
    bc_half = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmax")}
    bc_half["zmin"] = "ForceSymmetryPEC"
    s_half = _s11(mesh_h, grid_h, i0, j0, 0.0, float(grid_h.z[1]), bc_half)

    err = np.max(np.abs(s_half - s_full))
    assert err < 5e-2, f"max |S11_half - S11_full| = {err:.2e}"

    # The headline numbers users read: resonance and feed resistance.
    zin_full = Z0_FEED * (1 + s_full) / (1 - s_full)
    zin_half = Z0_FEED * (1 + s_half) / (1 - s_half)
    rel = np.max(np.abs(zin_half - zin_full) / np.abs(zin_full))
    assert rel < 1e-1, f"max relative Z_in deviation = {rel:.2e}"


GAP, WIDTH, LENGTH = 5e-3, 16e-3, 60e-3


def test_pmc_containment_presents_doubled_trapezoidal_impedance():
    """A load in a magnetic symmetry plane carries half the current."""
    element_full = SeriesRLC(R=75.0, L=5e-9, C=2e-12)
    # Half plate line: x in [0, W/2], magnetic symmetry plane at x = 0.
    grid = GridLines(
        x=np.linspace(0.0, WIDTH / 2, 5),
        y=np.linspace(-GAP / 2, GAP / 2, 6),
        z=np.linspace(-LENGTH / 2, LENGTH / 2, 121),
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
                start=(0.0, -GAP / 2, grid.z[-3]),
                end=(0.0, GAP / 2, grid.z[-3]),
                Z0=50.0,
                element=element_full,
            ),
        ],
        f_max=6e9,
        verbose=False,
    )
    res = ana.run(f_axis=f_axis, excited=["m1"], total_time_steps=20000, energy_stop_db=None)
    V, I = res.signals[("m1", 0)][("load", 0)]
    z_meas = -V.at_frequencies(f_axis) / I.at_frequencies(f_axis)

    # Internal device = 2 x the declared full-model element.
    dt = V.dt
    jwt = 1j * (2.0 / dt) * np.tan(2.0 * np.pi * f_axis * dt / 2.0)
    z_ref = 2.0 * (element_full.R + jwt * element_full.L) + 1.0 / (jwt * (element_full.C / 2.0))
    rel = np.abs(z_meas - z_ref) / np.abs(z_ref)
    assert np.all(rel < 1e-6), f"measured Z = {z_meas} vs doubled trapezoidal {z_ref}"
