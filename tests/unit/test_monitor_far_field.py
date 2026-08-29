"""DD-173: MonitorFarFieldFrequency — box placement, face booking, DFT phase.

The transform physics is gated in ``test_ntff_transform.py``; here the
monitor mechanics are pinned: where the Huygens box lands relative to
the absorber, which faces are omitted and booked as image planes, that
the running DFT compensates the leapfrog half step of H, and that the
dump/restore round trip preserves the accumulators.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldState
from magnelio.boundaries.boundary_conditions import BoundaryConditions
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.monitors.far_field import MonitorFarFieldFrequency
from magnelio.signals.signal_1d import Signal1D

H = 1e-3
N = 10  # cells per axis of the test grid


def _mesh(pml_cells=2, **bc) -> Mesh:
    ax = np.arange(N + 1) * H
    defaults = {f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    defaults.update(bc)
    mesh = Mesh.from_grid(
        GridLines(ax.copy(), ax.copy(), ax.copy()),
        boundary_conditions=BoundaryConditions(**defaults),
    )
    # Simulate the mesher's absorber bookkeeping on the hand grid.
    mesh._pml_cells = {f: pml_cells for f, t in defaults.items() if t == "CPML"}
    return mesh


class TestPlacement:
    def test_box_sits_inside_absorber_plus_margin(self):
        mon = MonitorFarFieldFrequency(freqs=[1e9], margin_cells=1)
        mon.attach(_mesh(pml_cells=2))
        names = {bf.name for bf in mon._faces}
        assert names == {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
        for bf in mon._faces:
            # Node index 3 from the low side, N-3 from the high side.
            expected = 3 * H if bf.sign < 0 else (N - 3) * H
            assert bf.plane == pytest.approx(expected)
        assert mon._image_planes == []

    def test_ground_plane_is_omitted_and_booked(self):
        mon = MonitorFarFieldFrequency(freqs=[1e9], margin_cells=1)
        mon.attach(_mesh(pml_cells=2, zmin="PEC"))
        names = {bf.name for bf in mon._faces}
        assert "zmin" not in names and len(names) == 5
        (plane,) = mon._image_planes
        assert plane.axis == 2 and plane.kind == "PEC" and plane.at_low
        assert plane.position == pytest.approx(0.0)
        assert plane.physical_halfspace
        # Side faces reach down to the wall: their z-cell range starts at 0.
        xmin = next(bf for bf in mon._faces if bf.name == "xmin")
        assert xmin.slab[2].start == 0

    def test_symmetry_plane_keeps_the_full_sphere(self):
        mon = MonitorFarFieldFrequency(freqs=[1e9], margin_cells=1)
        mon.attach(_mesh(pml_cells=2, zmin=("SymmetryPEC", 0.0)))
        (plane,) = mon._image_planes
        assert not plane.physical_halfspace

    def test_pmc_wall_position_is_the_natural_magnetic_wall(self):
        mon = MonitorFarFieldFrequency(freqs=[1e9], margin_cells=1)
        mon.attach(_mesh(pml_cells=2, zmin="PMC"))
        (plane,) = mon._image_planes
        assert plane.position == pytest.approx(-0.5 * H)

    def test_periodic_face_is_rejected(self):
        mon = MonitorFarFieldFrequency(freqs=[1e9], margin_cells=1)
        with pytest.raises(ValueError, match="[Pp]eriodic"):
            mon.attach(_mesh(pml_cells=2, xmin="Periodic", xmax="Periodic"))

    def test_all_wall_cavity_is_rejected(self):
        mon = MonitorFarFieldFrequency(freqs=[1e9])
        bc = {f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
        with pytest.raises(ValueError, match="cavity"):
            mon.attach(_mesh(pml_cells=0, **bc))

    def test_too_small_interior_is_rejected(self):
        mon = MonitorFarFieldFrequency(freqs=[1e9], margin_cells=3)
        with pytest.raises(ValueError, match="physical volume"):
            mon.attach(_mesh(pml_cells=2))

    def test_margin_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="margin_cells"):
            MonitorFarFieldFrequency(freqs=[1e9], margin_cells=0)


def _cw_state(grid, E0, H0, t, dt, omega):
    """Uniform CW field as FIT grid states at the leapfrog time levels."""
    from magnelio.monitors.base import _solver_dual_widths

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    ce = np.cos(omega * t)
    ch = np.cos(omega * (t + 0.5 * dt))
    dxa = _solver_dual_widths(grid.dx)
    dya = _solver_dual_widths(grid.dy)
    dza = _solver_dual_widths(grid.dz)
    return FieldState(
        Ex=np.full((Nx, Ny + 1, Nz + 1), 0.0) + E0 * ce * grid.dx[:, None, None],
        Ey=np.full((Nx + 1, Ny, Nz + 1), 0.0) + E0 * ce * grid.dy[None, :, None],
        Ez=np.full((Nx + 1, Ny + 1, Nz), 0.0) + E0 * ce * grid.dz[None, None, :],
        Hx=np.full((Nx + 1, Ny, Nz), 0.0) + H0 * ch * dxa[:, None, None],
        Hy=np.full((Nx, Ny + 1, Nz), 0.0) + H0 * ch * dya[None, :, None],
        Hz=np.full((Nx, Ny, Nz + 1), 0.0) + H0 * ch * dza[None, None, :],
    )


class TestAccumulation:
    def test_cw_amplitude_and_half_step_phase(self):
        """A uniform CW state renormalises to its amplitude, in phase.

        H states live half a step after E; the accumulator stamps them
        with t + dt/2.  A missing compensation would leave a residual
        phase of ω·dt/2 on H — the imaginary part pins it.
        """
        f0 = 1e9
        omega = 2.0 * np.pi * f0
        dt = (2.0 * np.pi / omega) / 64.0  # 64 steps per period
        n_steps = 640  # exactly 10 periods: oscillating sums cancel
        E0, H0 = 2.5, 0.75

        mesh = _mesh(pml_cells=2)
        mon = MonitorFarFieldFrequency(freqs=[f0], margin_cells=1)
        mon.attach(mesh)
        for n in range(n_steps):
            mon.record(_cw_state(mesh.grid, E0, H0, n * dt, dt, omega), n, n * dt, dt)
        waveform = np.cos(omega * np.arange(n_steps) * dt)
        mon.renormalize(Signal1D(t=np.arange(n_steps) * dt, values=waveform, dt=dt))

        for patches in mon._patch_sets(0):
            tang_e = patches.E[np.abs(patches.E) > 1e-12]
            tang_h = patches.H[np.abs(patches.H) > 1e-12]
            np.testing.assert_allclose(tang_e, E0 + 0j, rtol=1e-9)
            np.testing.assert_allclose(tang_h, H0 + 0j, rtol=1e-9)

    def test_result_requires_renormalization(self):
        mesh = _mesh(pml_cells=2)
        mon = MonitorFarFieldFrequency(freqs=[1e9], margin_cells=1)
        mon.attach(mesh)
        with pytest.raises(ValueError, match="incident"):
            mon.result(1e9)

    def test_unknown_frequency_is_rejected(self):
        mesh = _mesh(pml_cells=2)
        mon = MonitorFarFieldFrequency(freqs=[1e9, 2e9], margin_cells=1)
        mon.attach(mesh)
        with pytest.raises(ValueError, match="not recorded"):
            mon.result(1.5e9)
        with pytest.raises(ValueError, match="pass f="):
            mon.result()


class TestPersistence:
    def test_dump_restore_round_trip(self):
        f0 = 1e9
        omega = 2.0 * np.pi * f0
        dt = (2.0 * np.pi / omega) / 64.0
        mesh = _mesh(pml_cells=2)

        mon = MonitorFarFieldFrequency(freqs=[f0], margin_cells=1)
        mon.attach(mesh)
        for n in range(100):
            mon.record(_cw_state(mesh.grid, 1.0, 0.5, n * dt, dt, omega), n, n * dt, dt)
        waveform = np.cos(omega * np.arange(100) * dt)
        mon.renormalize(Signal1D(t=np.arange(100) * dt, values=waveform, dt=dt))
        dump = mon.result_dump()

        fresh = MonitorFarFieldFrequency(freqs=[f0], margin_cells=1)
        fresh.attach(mesh)
        fresh.load_result_dump(dump)
        a = mon.result(f0, theta=np.linspace(0, np.pi, 7), phi=np.linspace(0, np.pi, 5))
        b = fresh.result(f0, theta=np.linspace(0, np.pi, 7), phi=np.linspace(0, np.pi, 5))
        np.testing.assert_array_equal(a.E_theta, b.E_theta)
        np.testing.assert_array_equal(a.E_phi, b.E_phi)

    def test_resume_continues_the_accumulation(self):
        f0 = 1e9
        omega = 2.0 * np.pi * f0
        dt = (2.0 * np.pi / omega) / 64.0
        mesh = _mesh(pml_cells=2)

        def run(mon, lo, hi):
            for n in range(lo, hi):
                mon.record(_cw_state(mesh.grid, 1.0, 0.5, n * dt, dt, omega), n, n * dt, dt)

        whole = MonitorFarFieldFrequency(freqs=[f0], margin_cells=1)
        whole.attach(mesh)
        run(whole, 0, 128)

        first = MonitorFarFieldFrequency(freqs=[f0], margin_cells=1)
        first.attach(mesh)
        run(first, 0, 60)
        dump = first.result_dump()
        second = MonitorFarFieldFrequency(freqs=[f0], margin_cells=1)
        second.attach(mesh)
        second.load_result_dump(dump)
        run(second, 60, 128)

        for bf_name, accs in whole._acc.items():
            for comp, acc in accs.items():
                np.testing.assert_allclose(
                    second._acc[bf_name][comp].result, acc.result, rtol=1e-12
                )


class TestFeedFootprintsAndMetal:
    """DD-198: patches inside feed guides and conductors are left out."""

    def test_footprint_face_sits_at_the_absorber_interface_and_is_masked(self):
        mon = MonitorFarFieldFrequency(freqs=[1e9], margin_cells=1)
        mon._port_footprints = {"xmin": [{1: (3, 6), 2: (2, 5)}]}
        mon.attach(_mesh(pml_cells=2))
        xmin = next(bf for bf in mon._faces if bf.name == "xmin")
        assert xmin.plane == pytest.approx(2 * H)  # no margin on the feed face
        other = next(bf for bf in mon._faces if bf.name == "xmax")
        assert other.plane == pytest.approx((N - 3) * H)
        assert xmin.keep is not None
        # The box's tangential cell range starts at lo_n = 3 (2 + margin);
        # the window cells 3..5 (y) x 2..4 (z) map to local 0..2 x -1..1.
        assert np.all(xmin.keep[0:3, 0:2] == 0.0)
        assert np.all(xmin.keep[3:, :] == 1.0)
        assert other.keep is None

    def test_masked_patches_carry_no_area(self):
        mon = MonitorFarFieldFrequency(freqs=[1e9], margin_cells=1)
        mon._port_footprints = {"xmin": [{1: (3, 6), 2: (2, 5)}]}
        mon.attach(_mesh(pml_cells=2))
        mon.renormalize(Signal1D(t=np.arange(4) * 1e-12, values=np.ones(4), dt=1e-12))
        sets = mon._patch_sets(0)
        xmin = next(bf for bf in mon._faces if bf.name == "xmin")
        idx = [bf.name for bf in mon._faces].index("xmin")
        areas = sets[idx].areas.reshape(xmin.keep.shape)
        assert np.all(areas[xmin.keep == 0.0] == 0.0)
        assert np.all(areas[xmin.keep == 1.0] > 0.0)

    def test_dump_round_trip_keeps_the_weights_and_the_incident_ratio(self):
        mon = MonitorFarFieldFrequency(freqs=[1e9, 2e9], margin_cells=1)
        mon._port_footprints = {"xmin": [{1: (3, 6), 2: (2, 5)}]}
        mon.attach(_mesh(pml_cells=2))
        mon._set_incident_amplitude([0.5e9, 3e9], [0.8, 1.2])
        back = MonitorFarFieldFrequency.from_result_dump(mon.result_dump())
        a = next(bf for bf in mon._faces if bf.name == "xmin")
        b = next(bf for bf in back._faces if bf.name == "xmin")
        np.testing.assert_array_equal(a.keep, b.keep)
        np.testing.assert_allclose(back._incident_amplitude, mon._incident_amplitude)
