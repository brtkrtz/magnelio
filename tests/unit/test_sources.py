"""Unit tests for PlaneWaveSource."""

import math

import numpy as np
import pytest

from magnelio.sources.plane_wave import _C0, _ETA0, PlaneWaveSource, _classify_axis

# ── helper ──────────────────────────────────────────────────────────────────


def _make_solver(Nx=6, Ny=6, Nz=6, f_max=5e9):
    """Return a minimal set-up FITTimeDomainSolver for attach() tests."""
    from magnelio.mesh.grid import GridLines
    from magnelio.mesh.mesher import Mesh
    from magnelio.solver.fit_td import FITTimeDomainSolver
    from magnelio.solver.stability import courant_dt

    L = 6e-3
    grid = GridLines(
        x=np.linspace(0, L, Nx + 1),
        y=np.linspace(0, L, Ny + 1),
        z=np.linspace(0, L, Nz + 1),
    )
    mesh = Mesh.from_grid(grid)
    dt = courant_dt(grid, accuracy="normal")
    solver = FITTimeDomainSolver(mesh=mesh, total_time_steps=10, dt=dt, verbose=False)
    solver.setup()
    return solver


# ── waveform ─────────────────────────────────────────────────────────────────


class TestPlaneWaveWaveform:
    def test_gaussian_peak_at_t0(self):
        pw = PlaneWaveSource(f_max=5e9, waveform="gaussian")
        t0 = 4.0 / 5e9
        assert pw.excitation(t0) == pytest.approx(1.0, rel=1e-6)

    def test_gaussian_decays_far_from_peak(self):
        pw = PlaneWaveSource(f_max=5e9, waveform="gaussian")
        assert abs(pw.excitation(0.0)) < 1e-3

    def test_sine_zero_at_t0(self):
        pw = PlaneWaveSource(f_center=1e9, waveform="sine")
        assert pw.excitation(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_sine_amplitude(self):
        pw = PlaneWaveSource(f_center=1e9, waveform="sine", amplitude=2.5)
        assert abs(pw.excitation(0.25 / 1e9)) == pytest.approx(2.5, rel=1e-6)

    def test_unknown_waveform_raises(self):
        pw = PlaneWaveSource.__new__(PlaneWaveSource)
        pw.direction = (0.0, 0.0, 1.0)
        pw.polarization = (1.0, 0.0, 0.0)
        pw.corners = None
        pw.amplitude = 1.0
        pw.waveform = "bogus"
        pw.f_center = None
        pw.f_max = None
        with pytest.raises(ValueError):
            pw.excitation(0.0)


# ── polarisation normalisation ───────────────────────────────────────────────


class TestPlaneWaveInit:
    def test_direction_normalised(self):
        pw = PlaneWaveSource(direction=(0, 0, 3))
        assert np.linalg.norm(pw.direction) == pytest.approx(1.0)

    def test_polarisation_orthogonalised(self):
        # Give a pol with a component along k; should be projected out
        pw = PlaneWaveSource(direction=(0, 0, 1), polarization=(1, 0, 1))
        p = np.array(pw.polarization)
        k = np.array(pw.direction)
        assert np.dot(p, k) == pytest.approx(0.0, abs=1e-12)
        assert np.linalg.norm(p) == pytest.approx(1.0)

    def test_parallel_pol_raises(self):
        with pytest.raises(ValueError, match="parallel"):
            PlaneWaveSource(direction=(0, 0, 1), polarization=(0, 0, 1))


# ── axis classification ───────────────────────────────────────────────────────


class TestClassifyAxis:
    @pytest.mark.parametrize(
        "d,ax,sign",
        [
            ((1, 0, 0), 0, 1),
            ((-1, 0, 0), 0, -1),
            ((0, 1, 0), 1, 1),
            ((0, -1, 0), 1, -1),
            ((0, 0, 1), 2, 1),
            ((0, 0, -1), 2, -1),
        ],
    )
    def test_aligned(self, d, ax, sign):
        a, s = _classify_axis(np.array(d, dtype=float))
        assert a == ax
        assert s == sign

    def test_oblique_raises(self):
        with pytest.raises(NotImplementedError):
            _classify_axis(np.array([1.0, 1.0, 0.0]) / math.sqrt(2))


# ── incident field ────────────────────────────────────────────────────────────


class TestIncidentField:
    def _make_attached(self, direction=(0, 0, 1), polarization=(1, 0, 0), f_max=5e9):
        pw = PlaneWaveSource(direction=direction, polarization=polarization, f_max=f_max)
        solver = _make_solver()
        pw.attach(solver)
        return pw

    def test_E_along_polarisation(self):
        pw = self._make_attached()
        t0 = 4.0 / 5e9  # peak of Gaussian
        r = np.array([3e-3, 3e-3, 0.0])
        E = pw.incident_E(r, t0)
        # Must point in x direction only
        assert abs(E[0]) > 0.5
        assert E[1] == pytest.approx(0.0, abs=1e-10)
        assert E[2] == pytest.approx(0.0, abs=1e-10)

    def test_H_perpendicular(self):
        pw = self._make_attached()
        t0 = 4.0 / 5e9
        r = np.array([3e-3, 3e-3, 0.0])
        E = pw.incident_E(r, t0)
        H = pw.incident_H(r, t0)
        k = np.array(pw.direction)
        # k × E should be parallel to H
        cross = np.cross(k, E)
        assert np.allclose(cross / np.linalg.norm(cross), H / np.linalg.norm(H), atol=1e-10)

    def test_impedance_ratio(self):
        pw = self._make_attached()
        t0 = 4.0 / 5e9
        r = np.array([3e-3, 3e-3, 0.0])
        E = pw.incident_E(r, t0)
        H = pw.incident_H(r, t0)
        ratio = np.linalg.norm(E) / np.linalg.norm(H)
        assert ratio == pytest.approx(_ETA0, rel=1e-6)

    def test_retardation(self):
        """Field at distance d is delayed by d/c₀."""
        pw = self._make_attached()
        t0 = 4.0 / 5e9
        r0 = np.array([3e-3, 3e-3, 0.0])
        d = 1e-3
        r1 = r0 + np.array([0, 0, d])
        E0 = pw.incident_E(r0, t0)
        E1 = pw.incident_E(r1, t0 + d / _C0)
        assert np.allclose(E0, E1, rtol=1e-6)


# ── box snapping ──────────────────────────────────────────────────────────────


class TestBoxSnapping:
    def test_default_box(self):
        pw = PlaneWaveSource()
        solver = _make_solver(Nx=6, Ny=6, Nz=6)
        pw.attach(solver)
        ix0, ix1, iy0, iy1, iz0, iz1 = pw._box
        assert ix0 == 2 and ix1 == solver.mesh.Nx - 1
        assert iz0 == 2 and iz1 == solver.mesh.Nz - 1

    def test_explicit_box_snapped(self):
        solver = _make_solver(Nx=6, Ny=6, Nz=6)
        x = np.asarray(solver.mesh.grid.x)
        # Request box that spans roughly the middle two nodes
        box = ((x[2], x[2], x[2]), (x[4], x[4], x[4]))
        pw = PlaneWaveSource(corners=box)
        pw.attach(solver)
        ix0, ix1, iy0, iy1, iz0, iz1 = pw._box
        assert 1 <= ix0 < ix1 <= solver.mesh.Nx
        assert 1 <= iz0 < iz1 <= solver.mesh.Nz
