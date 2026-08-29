"""Power balance of the far-field transform against the surface power.

The tangential fields on the Huygens box carry a definite real power
out of it (``Re ∮ E × H* · n̂ dS``); the transform must radiate the
same power for a lossless exterior.  ``FarFieldResult.surface_power``
carries that flux, ``power_balance`` the ratio, and the monitor warns
when the ratio leaves its tolerance — the signature of a box that
samples the radiator's near zone too closely.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from magnelio.monitors.far_field import _CLOSURE_TOLERANCE, MonitorFarFieldFrequency
from magnelio.post.far_field import FarFieldResult, SurfacePatchSet, ntff_transform, surface_power
from magnelio.signals.signal_1d import Signal1D
from tests.unit.test_monitor_far_field import _cw_state, _mesh
from tests.unit.test_ntff_transform import ETA0, F0, N_PATCH, P_ANALYTIC, L, box_patches


class TestSurfacePower:
    def test_analytic_dipole_box_balances(self):
        """Exact near fields: surface power == analytic power == P_rad."""
        patches = box_patches((-L, -L, -L), (L, L, L), N_PATCH)
        p_surf = surface_power(patches)
        assert p_surf == pytest.approx(P_ANALYTIC, rel=5e-3)
        res = ntff_transform(patches, [], F0, surface_power=p_surf)
        assert res.surface_power == p_surf
        assert res.power_balance == pytest.approx(1.0, abs=1e-2)

    def test_inconsistent_fields_do_not_balance(self):
        """H rotated by 60° against E: flux and far field part ways."""
        patches = box_patches((-L, -L, -L), (L, L, L), N_PATCH)
        skewed = [
            SurfacePatchSet(
                centers=s.centers,
                normals=s.normals,
                areas=s.areas,
                E=s.E,
                H=s.H * np.exp(1j * np.pi / 3),
            )
            for s in patches
        ]
        # The flux of a near-field box also picks up the reactive part
        # under the rotation, so only the balance is pinned.
        p_surf = surface_power(skewed)
        assert 0.0 < p_surf < P_ANALYTIC
        res = ntff_transform(skewed, [], F0, surface_power=p_surf)
        assert abs(res.power_balance - 1.0) > _CLOSURE_TOLERANCE

    def test_power_balance_needs_the_surface_power(self):
        res = ntff_transform(box_patches((-L, -L, -L), (L, L, L), N_PATCH), [], F0)
        assert res.surface_power is None
        with pytest.raises(ValueError, match="surface_power"):
            res.power_balance

    def test_result_dataclass_default(self):
        res = FarFieldResult(
            f=F0,
            theta=np.array([0.0]),
            phi=np.array([0.0]),
            E_theta=np.zeros((1, 1), dtype=complex),
            E_phi=np.zeros((1, 1), dtype=complex),
        )
        assert res.surface_power is None


class TestMonitorWarning:
    @staticmethod
    def _monitor_with_aperture(amplitude: float, **bc):
        """A CW-recorded monitor whose bins hold one plane-wave aperture.

        Only the ``zmax`` face carries a field: E_x = a, H_y = a/η.  A
        sub-wavelength aperture on the 10 mm test mesh cannot radiate
        its flux, so the balance is off by construction — the test pins
        the plumbing (surface power carried, warning raised), not
        physics.
        """
        f0 = 1e9
        omega = 2.0 * np.pi * f0
        dt = (2.0 * np.pi / omega) / 64.0
        n_steps = 640
        mesh = _mesh(pml_cells=2, **bc)
        mon = MonitorFarFieldFrequency(freqs=[f0], margin_cells=1, name="box")
        mon.attach(mesh)
        for n in range(n_steps):
            mon.record(_cw_state(mesh.grid, 0.0, 0.0, n * dt, dt, omega), n, n * dt, dt)
        waveform = np.cos(omega * np.arange(n_steps) * dt)
        mon.renormalize(Signal1D(t=np.arange(n_steps) * dt, values=waveform, dt=dt))
        spectrum = mon._source_spectrum[0]
        mon._acc["zmax"]["Ex"].result[0][...] = amplitude * spectrum
        mon._acc["zmax"]["Hy"].result[0][...] = amplitude * spectrum / ETA0
        return mon

    def test_surface_power_is_carried_and_the_shortfall_warns(self):
        mon = self._monitor_with_aperture(1.0)
        bf = next(b for b in mon._faces if b.name == "zmax")
        expected = float(np.outer(bf.w1, bf.w2).sum()) / ETA0
        with pytest.warns(UserWarning, match="power leaving the recording box"):
            res = mon.result(1e9)
        assert res.surface_power == pytest.approx(expected, rel=1e-9)
        assert abs(res.power_balance - 1.0) > _CLOSURE_TOLERANCE

    def test_empty_box_does_not_warn(self):
        """No field on the box: no power either way, nothing to compare."""
        mon = self._monitor_with_aperture(0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = mon.result(1e9)
        assert res.surface_power == 0.0
        assert res.P_rad == 0.0

    def test_symmetry_plane_books_the_mirrored_half(self):
        """Half model: the flux through the half box counts twice.

        ``P_rad`` of a symmetry model is the full-model power over the
        whole sphere; the surface power must be in the same watts.  A
        real ground plane (``PEC``) doubles nothing — both powers are
        the half-space's.
        """
        full = self._monitor_with_aperture(1.0)
        half = self._monitor_with_aperture(1.0, zmin="SymmetryPEC")
        ground = self._monitor_with_aperture(1.0, zmin="PEC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_full = full.result(1e9).surface_power
            p_half = half.result(1e9).surface_power
            p_ground = ground.result(1e9).surface_power
        assert p_full > 0.0
        # Same zmax aperture on the same grid: the wall only changes the booking.
        assert p_half == pytest.approx(2.0 * p_full, rel=1e-9)
        assert p_ground == pytest.approx(p_full, rel=1e-9)
