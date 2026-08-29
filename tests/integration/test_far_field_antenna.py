"""DD-173 antenna gates: MonitorFarFieldFrequency against textbook antennas.

Three solver-level gates:

1. **Half-wave dipole (full model)** — peak directivity at the
   textbook 2.15 dBi, and the lossless power closure
   ``P_rad ≈ 1 − |S11|²`` (both per watt of incident power — the
   like-for-like check; the time-domain flux monitor integrates the
   broadband pulse and is not comparable per frequency).
2. **Quarter-wave monopole on a PEC ground** — the tutorial-08
   topology: the box face on the ground is replaced by image theory,
   the pattern is masked below the plane, the horizon directivity
   doubles the dipole's.
3. **Symmetry parity** — the dipole as a SymmetryPEC half model with
   the lumped feed on the plane (DD-172): pattern and radiated power
   match the full model, closing the three-feature loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Mesh
from magnelio.circuit.rasterize import EdgePath
from magnelio.mesh._thin_wire import apply_thin_wire_path
from magnelio.mesh.grid import GridLines
from magnelio.mesh.indexing import edge_index_Ez
from magnelio.monitors import MonitorFarFieldFrequency
from magnelio.ports import PortSpecLumped

D = 1e-3
A_RADIUS = 0.05 * D
H_CELLS = 15
PAD, PML = 4, 8
Z0_FEED = 73.0
N_STEPS = 8000
F_AXIS = np.linspace(3.2e9, 6.2e9, 121)
F0 = 4.6e9  # close to the thin-dipole resonance of this geometry


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


def _run(mesh, grid, i0, j0, feed_start_z, feed_end_z, bc):
    x0, y0 = float(grid.x[i0]), float(grid.y[j0])
    ff = MonitorFarFieldFrequency(freqs=[F0], margin_cells=2, name="pattern")
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
        monitors=(ff,),
    )
    res = ana.run(f_axis=F_AXIS, excited=["feed"], total_time_steps=N_STEPS, energy_stop_db=None)
    return res, ff


def _full_dipole():
    span = 2 * H_CELLS
    nz = span + 2 * (PAD + PML)
    nt = 1 + 2 * (PAD + PML)
    z = np.linspace(0, nz * D, nz + 1)
    t = np.linspace(0, nt * D, nt + 1)
    k_c = PAD + PML + H_CELLS
    gap_ks = {k_c - 1, k_c}
    wire_ks = [k for k in range(PAD + PML, PAD + PML + span) if k not in gap_ks]
    mesh, grid, i0, j0 = _wire_mesh(z, t, wire_ks)
    bc = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    return _run(mesh, grid, i0, j0, grid.z[k_c - 1], grid.z[k_c + 1], bc)


_CACHE: dict = {}


def _cached_full():
    if "full" not in _CACHE:
        res, ff = _full_dipole()
        _CACHE["full"] = (res, ff.result(F0))
    return _CACHE["full"]


def _db(x):
    return 10.0 * np.log10(x)


def test_dipole_directivity_and_power_closure():
    res, pattern = _cached_full()
    d_peak = float(np.max(pattern.directivity))
    assert abs(_db(d_peak) - 2.15) < 0.15, f"dipole D = {_db(d_peak):.2f} dBi"
    # Pattern shape: donut around the wire axis (z): peak at the
    # equator, null along the axis.
    i_eq = np.argmin(np.abs(pattern.theta - np.pi / 2))
    assert np.max(pattern.directivity[i_eq, :]) == pytest.approx(d_peak, rel=1e-2)
    assert np.max(pattern.directivity[0, :]) < 0.05 * d_peak

    # Lossless closure per incident watt: P_rad == accepted power.
    s11 = np.interp(F0, F_AXIS, np.abs(res.S("feed", "feed")))
    accepted = 1.0 - s11**2
    assert pattern.P_rad == pytest.approx(accepted, rel=3e-2), (
        f"P_rad = {pattern.P_rad:.4f} W vs accepted {accepted:.4f} W"
    )
    # The analysis wired the same number into the result.
    assert pattern.accepted_power == pytest.approx(accepted, rel=1e-6)
    assert pattern.radiation_efficiency == pytest.approx(1.0, abs=3e-2)
    g = float(np.max(pattern.gain))
    assert g == pytest.approx(d_peak, rel=3e-2)


def test_monopole_on_ground_plane():
    nz = H_CELLS + PAD + PML
    nt = 1 + 2 * (PAD + PML)
    z = np.linspace(0, nz * D, nz + 1)
    t = np.linspace(0, nt * D, nt + 1)
    mesh, grid, i0, j0 = _wire_mesh(z, t, list(range(1, H_CELLS)))
    bc = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmax")}
    bc["zmin"] = "PEC"
    _, ff = _run(mesh, grid, i0, j0, 0.0, float(grid.z[1]), bc)
    pattern = ff.result(F0)

    # Physically a half space: masked below the ground.
    assert pattern.physical_mask is not None
    below = pattern.theta > np.pi / 2 + 1e-9
    assert pattern.U[below, :].max() == 0.0
    # Horizon directivity: the monopole doubles the dipole value.
    res_full, pattern_full = _cached_full()
    i_eq = np.argmin(np.abs(pattern.theta - np.pi / 2))
    d_horizon = float(np.max(pattern.directivity[i_eq, :]))
    d_dipole = float(np.max(pattern_full.directivity))
    assert d_horizon == pytest.approx(2.0 * d_dipole, rel=5e-2), (
        f"monopole horizon D = {_db(d_horizon):.2f} dBi "
        f"vs dipole + 3 dB = {_db(2 * d_dipole):.2f} dBi"
    )


def test_symmetry_half_model_parity():
    """Half dipole + SymmetryPEC + lumped feed on the plane (DD-172)."""
    res_full, pattern_full = _cached_full()

    nz = H_CELLS + PAD + PML
    nt = 1 + 2 * (PAD + PML)
    z = np.linspace(0, nz * D, nz + 1)
    t = np.linspace(0, nt * D, nt + 1)
    mesh, grid, i0, j0 = _wire_mesh(z, t, list(range(1, H_CELLS)))
    bc = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmax")}
    bc["zmin"] = "ForceSymmetryPEC"
    _, ff = _run(mesh, grid, i0, j0, 0.0, float(grid.z[1]), bc)
    pattern = ff.result(F0)

    # A symmetry plane is not a ground: the full sphere is physical.
    assert pattern.physical_mask is None
    # Pattern and radiated power match the full model.  The tolerance
    # sits above the CPML min/max mirror asymmetry (KB-023) and far
    # below the factor-2 error of an unbooked symmetry cut.
    assert pattern.P_rad == pytest.approx(pattern_full.P_rad, rel=5e-2)
    g_half = pattern.realized_gain
    g_full = pattern_full.realized_gain
    assert np.max(np.abs(g_half - g_full)) < 0.08 * np.max(g_full)
