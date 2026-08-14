"""DD-080 antenna gates: thin-wire dipole + PMC mirror (T4, T5).

T4 — a center-gap-fed thin-wire dipole (2h = 30 cells, a = 0.05 cells,
     Omega = 2 ln(2h/a) ~ 14) resonates (first Im Zin = 0) near the
     classical thin-dipole length ~0.48 lambda with a feed resistance
     in the textbook window.  This exercises BOTH open ends.
T5 — the PEC electric mirror: half the dipole as a monopole on a PEC
     wall resonates at the full dipole's frequency with half its feed
     resistance.  Image theory: the H field of a z-directed current is
     azimuthal (tangential to a z-wall), so a PEC wall (E_tan = 0)
     mirrors a PERPENDICULAR current co-directed — current maximum at
     the wall.  A PMC wall mirrors it ANTI-directed (current null):
     for a perpendicular wire a PMC boundary is an ideal OPEN, tested
     quantitatively on the transmission line in
     test_thin_wire_line.py::test_t5_pmc_end_is_ideal_open.
"""

from __future__ import annotations

import numpy as np

from magnelio import AnalysisScatteringTD, Mesh
from magnelio.circuit.rasterize import EdgePath
from magnelio.mesh._thin_wire import apply_thin_wire_path
from magnelio.mesh.grid import GridLines
from magnelio.mesh.indexing import edge_index_Ez
from magnelio.ports import PortSpecLumped

C0 = 299_792_458.0
D = 1e-3
A_RADIUS = 0.05 * D
H_CELLS = 15  # one dipole arm [cells]
PAD, PML = 4, 8  # air gap + CPML thickness [cells]
Z0_FEED = 73.0
N_STEPS = 8000

_CACHE: dict = {}


def _wire_mesh(z_nodes, x_nodes, y_nodes, wire_ks, gap_ks):
    grid = GridLines(x=x_nodes, y=y_nodes, z=z_nodes)
    mesh = Mesh.from_grid(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    i0, j0 = Nx // 2, Ny // 2
    ks = [k for k in wire_ks if k not in gap_ks]
    path = EdgePath(
        axes=["z"] * len(ks),
        ijk=[(i0, j0, k) for k in ks],
        signs=[1] * len(ks),
        dls=[float(grid.dz[k]) for k in ks],
        flat_indices=[n_Ex + n_Ey + edge_index_Ez(i0, j0, k, Nx, Ny, Nz) for k in ks],
    )
    for flat in path.flat_indices:
        mesh.pec_mask_edges[2, flat - n_Ex - n_Ey] = True
    apply_thin_wire_path(mesh, path, A_RADIUS, name="dipole")
    return mesh, grid, i0, j0


def _zin(mesh, grid, i0, j0, k_gap, bc, f_axis):
    x0, y0 = float(grid.x[i0]), float(grid.y[j0])
    ana = AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(bc),
        ports=[
            PortSpecLumped(
                name="feed",
                start=(x0, y0, grid.z[k_gap]),
                end=(x0, y0, grid.z[k_gap + 1]),
                Z0=Z0_FEED,
            )
        ],
        f_max=8e9,
        verbose=False,
    )
    res = ana.run(f_axis=f_axis, excited=["feed"], total_time_steps=N_STEPS, energy_stop_db=None)
    s11 = res.S("feed", "feed")
    return Z0_FEED * (1 + s11) / (1 - s11)


def _first_resonance(f_axis, zin):
    """First zero crossing of Im Zin (capacitive -> inductive)."""
    im = zin.imag
    idx = np.nonzero((im[:-1] < 0) & (im[1:] >= 0))[0]
    assert idx.size, "no Im Zin zero crossing in the scanned band"
    i = idx[0]
    # Linear interpolation between the bracketing samples.
    f0, f1, y0, y1 = f_axis[i], f_axis[i + 1], im[i], im[i + 1]
    f_res = f0 - y0 * (f1 - f0) / (y1 - y0)
    r_res = float(np.interp(f_res, f_axis, zin.real))
    return float(f_res), r_res


def _full_dipole():
    if "full" in _CACHE:
        return _CACHE["full"]
    span = 2 * H_CELLS
    nz = span + 2 * (PAD + PML)
    nt = 1 + 2 * (PAD + PML)
    z = np.linspace(0, nz * D, nz + 1)
    t = np.linspace(0, nt * D, nt + 1)
    k_lo = PAD + PML  # wire start node
    k_gap = k_lo + H_CELLS - 1  # center gap edge... see below
    # Arms: [k_lo, k_gap) and (k_gap, k_lo+span); gap edge at the center.
    wire_ks = list(range(k_lo, k_lo + span))
    mesh, grid, i0, j0 = _wire_mesh(z, t, t.copy(), wire_ks, {k_gap})
    f_axis = np.linspace(3.2e9, 6.2e9, 121)
    bc = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    zin = _zin(mesh, grid, i0, j0, k_gap, bc, f_axis)
    _CACHE["full"] = (f_axis, zin)
    return _CACHE["full"]


def test_t4_dipole_resonance():
    f_axis, zin = _full_dipole()
    f_res, r_res = _first_resonance(f_axis, zin)
    # Thin-dipole resonance: 2h ~ 0.48 lambda (a few % below half-wave;
    # Omega = 2 ln(2h/a) ~ 14.4).  Window covers the classical 0.455
    # ... 0.50 shortening range plus the O(1 cell) staircase/end bias.
    f_half = C0 / (2 * (2 * H_CELLS * D))  # exact half-wave: 5.0 GHz
    assert 0.44 * 2 * f_half / 1.0 <= f_res <= 0.50 * 2 * f_half, (
        f"dipole resonates at {f_res / 1e9:.3f} GHz "
        f"({f_res / (2 * f_half):.3f} lambda dipole length)"
    )
    assert 50.0 <= r_res <= 90.0, f"R_in at resonance = {r_res:.1f} ohm"


def test_t5_pec_mirror_monopole():
    """A monopole on a PEC wall == half the center-fed dipole."""
    f_axis, zin_full = _full_dipole()
    f_full, r_full = _first_resonance(f_axis, zin_full)

    nz = H_CELLS + PAD + PML
    nt = 1 + 2 * (PAD + PML)
    z = np.linspace(0, nz * D, nz + 1)
    t = np.linspace(0, nt * D, nt + 1)
    # Feed gap edge AT the PEC wall (k = 0): the wall node closes the
    # current path; the image completes the center-fed dipole (a
    # 2-cell gap vs. the full model's 1-cell gap — a sub-percent
    # detuning absorbed in the tolerance).
    wire_ks = list(range(1, H_CELLS))
    mesh, grid, i0, j0 = _wire_mesh(z, t, t.copy(), wire_ks, set())
    bc = {f: "CPML" for f in ("xmax", "xmin", "ymin", "ymax", "zmax")}
    bc["zmin"] = "PEC"
    zin_half = _zin(mesh, grid, i0, j0, 0, bc, f_axis)
    f_half_res, r_half = _first_resonance(f_axis, zin_half)

    assert abs(f_half_res - f_full) <= 0.02 * f_full, (
        f"PEC monopole at {f_half_res / 1e9:.3f} GHz vs full dipole {f_full / 1e9:.3f} GHz"
    )
    # Image theory: half the gap voltage at the same current
    # -> R_in(monopole) ~ R_in(dipole) / 2.
    assert abs(r_half - 0.5 * r_full) <= 0.2 * (0.5 * r_full), (
        f"R_in monopole = {r_half:.1f} vs dipole/2 = {0.5 * r_full:.1f} ohm"
    )
