"""DD-227 gates for ``SourceCurrentPath``: an impressed current filament.

Two solver-level gates, one exact and one physical:

1. **Discrete charge continuity** — the FIT update carries
   ``d/dt(S d̂) = −S ĵ`` because ``S·C̃ᵀ = 0``, so an open filament
   must accumulate exactly ``∓∫I dt`` on its two end nodes and nothing
   in between.  This is an identity of the operators, so it holds to
   double round-off — and it pins the *sign* of the injection without
   reference to any far-field phase convention.
2. **Hertzian dipole** — a filament two cells long radiates the
   textbook short-dipole pattern: ``P_rad = η₀ (k I L)² / (6π)`` in
   the library's effective-amplitude convention, peak directivity
   1.5, a ``sin θ`` pattern and no ``E_φ``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from magnelio import AnalysisTD, Excitation, Mesh
from magnelio._operators.material_matrices import build_M_eps
from magnelio.boundaries.pec import PECBoundary
from magnelio.constants import C0, ETA0
from magnelio.mesh.grid import GridLines
from magnelio.monitors import MonitorFarFieldFrequency
from magnelio.signals import WaveformGaussian
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt
from magnelio.sources import SourceCurrentPath

D = 1e-3  # cell size [m]
FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _grid(n_t, n_z):
    return GridLines(x=np.arange(n_t + 1) * D, y=np.arange(n_t + 1) * D, z=np.arange(n_z + 1) * D)


def _centre(grid):
    return float(grid.x[-1]) / 2, float(grid.y[-1]) / 2, float(grid.z[-1]) / 2


def _filament(grid, half_cells=1, name="fil"):
    """A z-directed filament of ``2·half_cells`` cells through the centre."""
    x0, y0, z0 = _centre(grid)
    return SourceCurrentPath(
        name=name,
        path=[(x0, y0, z0 - half_cells * D), (x0, y0, z0 + half_cells * D)],
    )


def test_charge_continuity_is_exact():
    """``q(end) = ±∫I dt`` to round-off — and positive at the +z end."""
    n, n_steps = 12, 200
    grid = _grid(n, n)
    mesh = Mesh.from_grid(grid)
    src = _filament(grid, half_cells=2)
    src.set_excitation(WaveformGaussian(f_max=15e9), amplitude=1.0)

    dt = courant_dt(grid, accuracy="normal")
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={f: PECBoundary(f) for f in FACES},
        dt=dt,
        total_time_steps=n_steps,
        verbose=False,
        precision="double",
        sources=[src],
    )
    solver.setup()
    solver.run()

    d = build_M_eps(mesh) * np.asarray(solver._fields.e_flat, dtype=float)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    # Charge on the dual cell of a node: the outgoing flux of the D
    # components on the six primal edges meeting there.
    q = np.zeros((Nx + 1, Ny + 1, Nz + 1))
    for lo, hi, block, shape in (
        (np.s_[:-1, :, :], np.s_[1:, :, :], d[:n_Ex], (Nx, Ny + 1, Nz + 1)),
        (np.s_[:, :-1, :], np.s_[:, 1:, :], d[n_Ex : n_Ex + n_Ey], (Nx + 1, Ny, Nz + 1)),
        (np.s_[:, :, :-1], np.s_[:, :, 1:], d[n_Ex + n_Ey :], (Nx + 1, Ny + 1, Nz)),
    ):
        comp = block.reshape(shape)
        q[lo] += comp
        q[hi] -= comp

    # The current is injected at the E half-steps t = (n + ½)·dt.
    charge = float(np.sum([src._drive((n_ + 0.5) * dt) for n_ in range(n_steps)]) * dt)
    i0 = int(np.argmin(np.abs(np.asarray(grid.x) - _centre(grid)[0])))
    j0 = int(np.argmin(np.abs(np.asarray(grid.y) - _centre(grid)[1])))
    z = np.asarray(grid.z)
    z0 = _centre(grid)[2]
    k_lo = int(np.argmin(np.abs(z - (z0 - 2 * D))))
    k_hi = int(np.argmin(np.abs(z - (z0 + 2 * D))))

    assert q[i0, j0, k_hi] == pytest.approx(charge, rel=1e-9)
    assert q[i0, j0, k_lo] == pytest.approx(-charge, rel=1e-9)
    # Nothing along the filament, and nothing in the radiating interior:
    # a source-free region carries no charge.  (The PEC walls do — that
    # is the induced image charge, and the whole box stays neutral.)
    interior = q[1:-1, 1:-1, 1:-1].copy()
    interior[i0 - 1, j0 - 1, k_hi - 1] = 0.0
    interior[i0 - 1, j0 - 1, k_lo - 1] = 0.0
    assert np.abs(interior).max() < 1e-12 * abs(charge)
    assert abs(q.sum()) < 1e-12 * abs(charge)


def test_short_filament_radiates_a_hertzian_dipole():
    """Absolute power, directivity and pattern of a two-cell filament."""
    d, pad, pml = 2e-3, 22, 8
    n = 2 * (pad + pml)
    line = np.arange(n + 1) * d
    grid = GridLines(x=line, y=line.copy(), z=line.copy())
    f0 = 5e9  # λ = 60 mm: 0.73 λ of clearance between filament and absorber
    c = float(line[-1]) / 2
    src = SourceCurrentPath(name="fil", path=[(c, c, c - d), (c, c, c + d)])
    ff = MonitorFarFieldFrequency(freqs=[f0], margin_cells=2, name="pattern")

    mesh = Mesh.from_grid(grid).with_boundary_conditions(dict.fromkeys(FACES, "CPML"))
    result = AnalysisTD(mesh=mesh, sources=[src], monitors=[ff], f_max=2 * f0, verbose=False).run(
        excitations=[Excitation("fil", waveform=WaveformGaussian(f_max=2 * f0), amplitude=1.0)],
        total_time_steps=6000,
        energy_stop_db=60,
    )
    result.renormalize("fil")
    with warnings.catch_warnings():  # the box clearance is the point, see below
        warnings.simplefilter("ignore")
        pattern = ff.result(f0)

    k = 2 * np.pi * f0 / C0
    length = 2 * d
    # Library phasors are effective amplitudes (U = |E|²/η, no ½), so the
    # reference is the RMS-current form of the Hertzian power.
    p_hertz = ETA0 * (k * length) ** 2 / (6 * np.pi)

    # The transform closes to a few percent at this clearance; the
    # convergence ladder towards 1 is the validation certificate
    # ``validation/current_path_hertzian_dipole.py`` (DD-227).
    assert pattern.power_balance > 0.94
    assert pattern.P_rad == pytest.approx(p_hertz, rel=0.06), (
        f"P_rad = {pattern.P_rad:.4e} W vs Hertzian {p_hertz:.4e} W"
    )
    assert float(pattern.directivity.max()) == pytest.approx(1.5, rel=0.03)

    # sin θ pattern, no φ dependence, no E_φ.  The residual in both is
    # a smooth few-percent bulge towards the cube corners (θ ≈ 45°/135°,
    # φ ≈ 45° at the equator) — the same first-order box-closure error
    # that puts P_rad 3 % high while leaving the self-normalised
    # directivity exact.
    equator = int(np.argmin(np.abs(pattern.theta - np.pi / 2)))
    peak = float(np.abs(pattern.E_theta).max())
    profile = np.abs(pattern.E_theta).mean(axis=1)
    np.testing.assert_allclose(profile / profile[equator], np.sin(pattern.theta), atol=0.05)
    assert np.ptp(np.abs(pattern.E_theta[equator])) < 0.06 * peak
    assert np.abs(pattern.E_phi).max() < 0.05 * peak
