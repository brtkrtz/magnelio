"""Unit tests for SourcePlaneWave / SourceFieldIncident (DD-224)."""

import math

import numpy as np
import pytest

from magnelio.signals import WaveformGaussian, WaveformSine
from magnelio.sources import SourceFieldIncident, SourcePlaneWave
from magnelio.sources.base import Source
from magnelio.sources.plane_wave import _C0, _ETA0, _classify_axis

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


def _pw(**kwargs):
    kwargs.setdefault("name", "pw")
    return SourcePlaneWave(**kwargs)


# ── contract ─────────────────────────────────────────────────────────────────


class TestContract:
    def test_hierarchy_and_unit(self):
        pw = _pw()
        assert isinstance(pw, SourceFieldIncident)
        assert isinstance(pw, Source)
        assert pw.amplitude_unit == "V/m"
        assert pw.excitable is True

    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            SourceFieldIncident(name="inc")  # type: ignore[abstract]

    def test_name_required(self):
        with pytest.raises(TypeError):
            SourcePlaneWave(direction=(0, 0, 1))  # type: ignore[call-arg]
        with pytest.raises(TypeError, match="name"):
            SourcePlaneWave(name="")

    def test_repr_is_declarative(self):
        r = repr(_pw(direction=(0, 0, 3)))
        assert r.startswith("SourcePlaneWave(name='pw'")
        assert "_waveform" not in r and "np.float64" not in r


# ── excitation binding ───────────────────────────────────────────────────────


class TestExcitationBinding:
    def test_drive_is_amplitude_times_delayed_waveform(self):
        pw = _pw()
        w = WaveformGaussian(f_max=5e9)
        pw.set_excitation(w, amplitude=2.5, delay=1e-10)
        assert pw.waveform is w
        t0 = 4.0 / 5e9
        assert pw._drive(t0 + 1e-10) == pytest.approx(2.5)
        assert pw._drive(t0) == pytest.approx(2.5 * w(t0 - 1e-10))

    def test_array_drive(self):
        pw = _pw()
        pw.set_excitation(WaveformSine(f=1e9), amplitude=3.0)
        t = np.array([0.25e-9, 0.75e-9])
        np.testing.assert_allclose(pw._drive(t), [3.0, -3.0])

    def test_clear(self):
        pw = _pw()
        pw.set_excitation(WaveformGaussian(f_max=5e9), amplitude=2.0)
        pw.clear_excitation()
        assert pw.waveform is None
        with pytest.raises(ValueError, match="no waveform"):
            pw._drive(0.0)

    def test_validation(self):
        pw = _pw()
        with pytest.raises(TypeError, match="Waveform"):
            pw.set_excitation("gaussian")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="delay"):
            pw.set_excitation(WaveformGaussian(f_max=5e9), delay=-1.0)
        with pytest.raises(ValueError, match="amplitude"):
            pw.set_excitation(WaveformGaussian(f_max=5e9), amplitude=math.nan)

    def test_attach_requires_waveform(self):
        pw = _pw()
        with pytest.raises(ValueError, match="no waveform"):
            pw.attach(_make_solver())


# ── polarisation normalisation ───────────────────────────────────────────────


class TestPlaneWaveInit:
    def test_direction_normalised(self):
        pw = _pw(direction=(0, 0, 3))
        assert np.linalg.norm(pw.direction) == pytest.approx(1.0)
        assert all(isinstance(v, float) for v in pw.direction)

    def test_polarisation_orthogonalised(self):
        # Give a pol with a component along k; should be projected out
        pw = _pw(direction=(0, 0, 1), polarization=(1, 0, 1))
        p = np.array(pw.polarization)
        k = np.array(pw.direction)
        assert np.dot(p, k) == pytest.approx(0.0, abs=1e-12)
        assert np.linalg.norm(p) == pytest.approx(1.0)

    def test_parallel_pol_raises(self):
        with pytest.raises(ValueError, match="parallel"):
            _pw(direction=(0, 0, 1), polarization=(0, 0, 1))

    def test_from_ranges(self):
        pw = SourcePlaneWave.from_ranges(name="pw", x1=1e-3, x2=9e-3, z1=1e-3, z2=19e-3)
        assert pw.name == "pw"
        lo, hi = pw.corners
        assert (lo[0], hi[0]) == (1e-3, 9e-3)
        assert (lo[2], hi[2]) == (1e-3, 19e-3)


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
    def _make_attached(self, direction=(0, 0, 1), polarization=(1, 0, 0), f_max=5e9, amp=1.0):
        pw = _pw(direction=direction, polarization=polarization)
        pw.set_excitation(WaveformGaussian(f_max=f_max), amplitude=amp)
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

    def test_amplitude_scales_the_field(self):
        pw = self._make_attached(amp=2.5)
        t0 = 4.0 / 5e9
        r = np.array([3e-3, 3e-3, 0.0])
        assert pw.incident_E(r, t0)[0] == pytest.approx(2.5)

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
        pw = _pw()
        pw.set_excitation(WaveformGaussian(f_max=5e9))
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
        pw = _pw(corners=box)
        pw.set_excitation(WaveformGaussian(f_max=5e9))
        pw.attach(solver)
        ix0, ix1, iy0, iy1, iz0, iz1 = pw._box
        assert 1 <= ix0 < ix1 <= solver.mesh.Nx
        assert 1 <= iz0 < iz1 <= solver.mesh.Nz
