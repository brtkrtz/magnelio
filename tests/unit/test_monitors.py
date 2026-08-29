"""Unit tests for the monitors package."""

from __future__ import annotations

import warnings
from math import inf

import numpy as np
import pytest

from magnelio._fields.field_arrays import FieldState
from magnelio.mesh.grid import GridLines
from magnelio.monitors._dft import DFTAccumulator
from magnelio.monitors.base import (
    _cell_centres,
    _expand_field_list,
    _interp_to_cell_centres,
    _snap_range,
    resolve_plane_view,
    resolve_region,
)
from magnelio.monitors.field_frequency import MonitorFieldFrequency
from magnelio.monitors.field_time import MonitorFieldTime
from magnelio.monitors.flux import MonitorFluxTime

# -- helpers ---------------------------------------------------------------


class _FakeMesh:
    def __init__(self, grid):
        self.grid = grid


def _make_grid(Nx=4, Ny=5, Nz=6):
    x = np.linspace(0, 0.01, Nx + 1)
    y = np.linspace(0, 0.02, Ny + 1)
    z = np.linspace(0, 0.03, Nz + 1)
    return GridLines(x=x, y=y, z=z)


def _make_fields(grid):
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    rng = np.random.default_rng(42)
    return FieldState(
        Ex=rng.standard_normal((Nx, Ny + 1, Nz + 1)),
        Ey=rng.standard_normal((Nx + 1, Ny, Nz + 1)),
        Ez=rng.standard_normal((Nx + 1, Ny + 1, Nz)),
        Hx=rng.standard_normal((Nx + 1, Ny, Nz)),
        Hy=rng.standard_normal((Nx, Ny + 1, Nz)),
        Hz=rng.standard_normal((Nx, Ny, Nz + 1)),
    )


def _unit_reference(mon, dt=1e-12):
    """Give *mon* a reference whose spectrum is exactly 1 at every bin.

    A single sample of height ``1/dt`` integrates to one, so ``.data``
    equals ``.data_raw`` — the monitor is answerable about its units
    without any value moving.  For tests about DFT mechanics or plotting,
    where the excitation is beside the point.
    """
    from magnelio.signals.signal_1d import Signal1D

    mon.renormalize(Signal1D(t=np.array([0.0]), values=np.array([1.0 / dt]), dt=dt))
    return mon


# -- base tests ------------------------------------------------------------


class TestBase:
    def test_cell_centres(self):
        nodes = np.array([0.0, 1.0, 3.0, 6.0])
        cc = _cell_centres(nodes)
        np.testing.assert_allclose(cc, [0.5, 2.0, 4.5])

    def test_snap_range_point(self):
        cc = np.array([0.5, 1.5, 2.5, 3.5])
        sl = _snap_range(cc, 1.6, 1.6)
        assert sl == slice(1, 2)  # nearest to 1.6 is index 1 (value 1.5)

    def test_snap_range_interval(self):
        cc = np.array([0.5, 1.5, 2.5, 3.5])
        sl = _snap_range(cc, 1.0, 3.0)
        assert sl == slice(1, 3)

    def test_expand_field_list(self):
        assert _expand_field_list(["E"]) == ["Ex", "Ey", "Ez"]
        assert _expand_field_list(["H"]) == ["Hx", "Hy", "Hz"]
        assert _expand_field_list(["E", "H"]) == ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        assert _expand_field_list(["Ez", "Hx"]) == ["Ez", "Hx"]

    def test_expand_field_list_invalid(self):
        with pytest.raises(ValueError, match="Unknown field"):
            _expand_field_list(["Q"])

    def test_resolve_region_full_domain(self):
        grid = _make_grid(4, 5, 6)
        r = resolve_region(None, grid)
        assert r.ix == slice(0, 4)
        assert r.iy == slice(0, 5)
        assert r.iz == slice(0, 6)
        assert r.ndim == 3

    def test_resolve_region_point(self):
        grid = _make_grid(4, 5, 6)
        cx = 0.5 * (grid.x[2] + grid.x[3])
        cy = 0.5 * (grid.y[1] + grid.y[2])
        cz = 0.5 * (grid.z[3] + grid.z[4])
        r = resolve_region(((cx, cy, cz), (cx, cy, cz)), grid)
        assert r.ix.stop - r.ix.start == 1
        assert r.iy.stop - r.iy.start == 1
        assert r.iz.stop - r.iz.start == 1
        assert r.ndim == 0

    def test_resolve_region_plane(self):
        grid = _make_grid(4, 5, 6)
        cz = 0.5 * (grid.z[2] + grid.z[3])
        r = resolve_region(((None, None, cz), (None, None, cz)), grid)
        assert r.ix == slice(0, 4)
        assert r.iy == slice(0, 5)
        assert r.iz.stop - r.iz.start == 1
        assert r.ndim == 2

    @staticmethod
    def _slices(r):
        return (r.ix, r.iy, r.iz)

    def test_resolve_region_box(self):
        # cell centres: x 1.25/3.75/6.25/8.75 mm, y 2/6/10/14/18 mm,
        # z 2.5/7.5/…/27.5 mm
        grid = _make_grid(4, 5, 6)
        r = resolve_region(((0.0, 0.0, 0.0), (0.005, 0.01, 0.015)), grid)
        assert self._slices(r) == (slice(0, 2), slice(0, 3), slice(0, 3))

    def test_corner_order_does_not_matter(self):
        grid = _make_grid(4, 5, 6)
        lo, hi = (0.0, 0.0, 0.0), (0.005, 0.01, 0.015)
        assert self._slices(resolve_region((lo, hi), grid)) == self._slices(
            resolve_region((hi, lo), grid)
        )

    def test_none_reaches_the_domain_boundary(self):
        """None on one side extends to the edge, on both spans the axis."""
        grid = _make_grid(4, 5, 6)
        half = resolve_region(((0.005, None, None), (None, None, None)), grid)
        assert half.ix == slice(2, 4)  # from x = 5 mm outward
        assert half.iy == slice(0, 5)
        assert half.iz == slice(0, 6)

    def test_none_and_inf_are_equivalent(self):
        grid = _make_grid(4, 5, 6)
        cz = 0.5 * (grid.z[2] + grid.z[3])
        by_none = resolve_region(((None, None, cz), (None, None, cz)), grid)
        by_inf = resolve_region(
            ((-inf, -inf, cz), (inf, inf, cz)),
            grid,
        )
        assert self._slices(by_none) == self._slices(by_inf)

    def test_bad_corner_shape_raises(self):
        grid = _make_grid(4, 5, 6)
        with pytest.raises(ValueError, match="two opposite points"):
            resolve_region((0.0, 0.0, 0.0), grid)
        with pytest.raises(ValueError, match="three coordinates"):
            resolve_region(((0.0, 0.0), (1.0, 1.0)), grid)

    def test_interp_to_cell_centres_shapes(self):
        grid = _make_grid(4, 5, 6)
        fields = _make_fields(grid)
        ix = slice(0, 4)
        iy = slice(0, 5)
        iz = slice(0, 6)
        result = _interp_to_cell_centres(
            fields, ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], ix, iy, iz, grid
        )
        for comp in result:
            assert result[comp].shape == (4, 5, 6), f"{comp} has wrong shape: {result[comp].shape}"


# -- MonitorFieldTime tests ------------------------------------------------


class TestMonitorFieldTime:
    def test_0d_recording(self):
        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)
        fields = _make_fields(grid)

        mon = MonitorFieldTime(
            corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
            times=np.array([0.0, 1e-12, 2e-12]),
            fields=["Ez"],
            name="test_0d",
        )
        mon.attach(mesh)

        dt = 1e-12
        for n in range(3):
            t = n * dt
            mon.record(fields, n, t, dt)

        assert mon.t.shape == (3,)
        data = mon.data
        assert "Ez" in data
        assert data["Ez"].shape == (3,)  # 0D: squeezed

    def test_2d_recording(self):
        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)
        fields = _make_fields(grid)

        cz = 0.5 * (grid.z[3] + grid.z[4])
        mon = MonitorFieldTime(
            corners=((None, None, cz), (None, None, cz)),
            times=np.array([0.0, 1e-12]),
            fields=["E"],
            name="test_2d",
        )
        mon.attach(mesh)

        dt = 1e-12
        for n in range(2):
            mon.record(fields, n, n * dt, dt)

        data = mon.data
        assert "Ex" in data
        assert "Ey" in data
        assert "Ez" in data
        # 2D: should be (n_times, Nx, Ny)
        assert data["Ex"].shape[0] == 2
        assert data["Ex"].ndim == 3  # (2, Nx, Ny)

    def test_skips_non_matching_times(self):
        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)
        fields = _make_fields(grid)

        mon = MonitorFieldTime(
            corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
            times=np.array([5e-12]),
            fields=["Ez"],
            name="test_skip",
        )
        mon.attach(mesh)

        # Only call at t=0 and t=1e-12, target is t=5e-12
        mon.record(fields, 0, 0.0, 1e-12)
        mon.record(fields, 1, 1e-12, 1e-12)
        assert len(mon._recorded_times) == 0

    def test_component_accessor(self):
        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)
        fields = _make_fields(grid)

        mon = MonitorFieldTime(
            corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
            times=np.array([0.0]),
            fields=["E"],
            name="test_comp",
        )
        mon.attach(mesh)
        mon.record(fields, 0, 0.0, 1e-12)

        ez = mon.component("Ez")
        assert ez.shape == (1,)

        with pytest.raises(KeyError):
            mon.component("Hx")


class TestMonitorFieldTimeInterval:
    """``interval=`` records until the run ends — no end time to guess."""

    def _run(self, mon, n_steps, dt=1e-12):
        grid = _make_grid(4, 5, 6)
        mon.attach(_FakeMesh(grid))
        fields = _make_fields(grid)
        for n in range(n_steps):
            mon.record(fields, n, n * dt, dt)
        return mon

    def test_records_every_interval_for_as_long_as_the_run_lasts(self):
        mon = self._run(
            MonitorFieldTime(
                corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
                interval=2e-12,
                fields=["Ez"],
                name="ivl",
            ),
            n_steps=11,
        )
        # t = 0, 2, 4, 6, 8, 10 ps within an 11-step (0…10 ps) run
        np.testing.assert_allclose(mon.t, np.arange(6) * 2e-12, atol=1e-15)

    def test_run_length_is_not_capped_by_the_schedule(self):
        """The whole point: a longer run simply yields more snapshots."""
        short = self._run(
            MonitorFieldTime(
                corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
                interval=2e-12,
                fields=["Ez"],
                name="s",
            ),
            n_steps=11,
        )
        long = self._run(
            MonitorFieldTime(
                corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
                interval=2e-12,
                fields=["Ez"],
                name="l",
            ),
            n_steps=41,
        )
        assert len(short.t) == 6
        assert len(long.t) == 21

    def test_start_offsets_the_first_sample(self):
        mon = self._run(
            MonitorFieldTime(
                corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
                interval=4e-12,
                start=3e-12,
                fields=["Ez"],
                name="off",
            ),
            n_steps=13,
        )
        np.testing.assert_allclose(
            mon.t,
            np.array([3e-12, 7e-12, 11e-12]),
            atol=1e-15,
        )

    def test_schedule_is_exclusive(self):
        with pytest.raises(ValueError, match="either times="):
            MonitorFieldTime(
                corners=((0, 0, 0), (0, 0, 0)),
            )
        with pytest.raises(ValueError, match="either times="):
            MonitorFieldTime(corners=((0, 0, 0), (0, 0, 0)), times=[0.0], interval=1e-12)

    def test_interval_must_be_positive(self):
        with pytest.raises(ValueError, match="interval must be positive"):
            MonitorFieldTime(corners=((0, 0, 0), (0, 0, 0)), interval=0.0)

    def test_repr_works_on_both_schedules(self):
        ivl = MonitorFieldTime(interval=2e-12, name="ivl")
        exp = MonitorFieldTime(times=[0.0, 1e-12], name="exp")
        assert "every 2e-12 s" in repr(ivl)
        assert "n_times=2" in repr(exp)

    def test_recipe_round_trip_keeps_the_interval(self):
        from magnelio.analysis._recipe import (
            _monitor_from_dict,
            _monitor_to_dict,
        )

        mon = MonitorFieldTime(
            corners=((None, None, 0.005), (None, None, 0.005)),
            interval=0.5e-9,
            start=1e-9,
            fields=["E"],
            name="ivl",
        )
        back = _monitor_from_dict(_monitor_to_dict(mon))
        assert back.interval == 0.5e-9
        assert back.start == 1e-9
        assert back.times is None


# -- MonitorFieldFrequency tests ------------------------------------------


class TestMonitorFieldFrequency:
    def test_0d_dft_single_frequency(self):
        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)

        f0 = 1e9
        mon = MonitorFieldFrequency(
            corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
            freqs=np.array([f0]),
            fields=["Ez"],
            name="test_dft_0d",
        )
        mon.attach(mesh)

        # Feed a sinusoidal Ez signal
        dt = 1e-12
        n_steps = 2000
        omega = 2 * np.pi * f0

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        for n in range(n_steps):
            t = n * dt
            # Create fields with a uniform sinusoidal Ez
            fields = FieldState.zeros(Nx, Ny, Nz)
            fields.Ez[:] = np.sin(omega * t)
            mon.record(fields, n, t, dt)

        data = mon.data_raw
        assert "Ez" in data
        # DFT of sin(wt) at f0: |F| ≈ T/2 where T = n_steps * dt
        # T = 2000 * 1e-12 = 2e-9 s, so |F| ≈ 1e-9
        mag = np.abs(data["Ez"][0])
        expected = n_steps * dt / 2
        assert mag > 0.5 * expected, f"DFT magnitude {mag} < expected {expected}"

    def test_2d_dft_shape(self):
        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)

        cz = 0.5 * (grid.z[2] + grid.z[3])
        mon = MonitorFieldFrequency(
            corners=((None, None, cz), (None, None, cz)),
            freqs=np.array([1e9, 2e9]),
            fields=["Ez"],
            name="test_dft_2d",
        )
        mon.attach(mesh)

        fields = _make_fields(grid)
        mon.record(fields, 0, 0.0, 1e-12)

        data = mon.data_raw
        assert "Ez" in data
        # Shape: (2 freqs, Nx, Ny) — squeezed from (2, 4, 5, 1)
        assert data["Ez"].shape == (2, 4, 5)

    def test_1d_line_plot(self):
        import matplotlib.pyplot as plt

        grid = _make_grid(6, 5, 4)
        mesh = _FakeMesh(grid)

        cy = 0.5 * (grid.y[2] + grid.y[3])
        cz = 0.5 * (grid.z[1] + grid.z[2])
        mon = MonitorFieldFrequency(
            corners=((None, cy, cz), (None, cy, cz)),
            freqs=np.array([1e9]),
            fields=["Ez"],
            name="test_dft_1d",
        )
        mon.attach(mesh)
        mon.record(_make_fields(grid), 0, 0.0, 1e-12)
        _unit_reference(mon)

        # single component: instantaneous line at the given phase
        fig, ax = mon.plot(component="Ez", f=1e9)
        (line,) = ax.get_lines()
        assert len(line.get_xdata()) == grid.Nx
        assert "x" in ax.get_xlabel()
        plt.close(fig)

        # field group: amplitude line
        fig2, ax2 = mon.plot(component="E")
        assert "|E|" in ax2.get_title()
        plt.close(fig2)


# -- Renormalization tests ------------------------------------------------


class TestFreqMonitorInterval:
    """DD-140: sub-sampling the DFT, and the guards that make it safe."""

    @staticmethod
    def _accumulate(interval, n_steps=2000, dt=1e-12, f0=1e9, freqs=None):
        """Drive a uniform sinusoid through a 0D monitor."""
        grid = _make_grid(4, 5, 6)
        mon = MonitorFieldFrequency(
            corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
            freqs=np.array(freqs if freqs is not None else [f0]),
            fields=["Ez"],
            interval=interval,
            name="sub",
        )
        mon.attach(_FakeMesh(grid))
        omega = 2 * np.pi * f0
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        for n in range(n_steps):
            t = n * dt
            fields = FieldState.zeros(Nx, Ny, Nz)
            fields.Ez[:] = np.sin(omega * t)
            mon.record(fields, n, t, dt)
        return mon

    def test_default_records_every_step(self):
        mon = self._accumulate(None, n_steps=100)
        assert mon._step_stride == 1

    def test_subsampled_result_matches_every_step(self):
        # 100 samples per period against every step (1000 per period):
        # the weight tracks the spacing, so the bins agree in value, not
        # merely in shape — a wrong weight would show as a factor 10.
        ref = self._accumulate(None)
        sub = self._accumulate(10e-12)
        assert sub._step_stride == 10
        got = complex(np.asarray(sub.data_raw["Ez"]).ravel()[0])
        want = complex(np.asarray(ref.data_raw["Ez"]).ravel()[0])
        assert got == pytest.approx(want, rel=1e-3)

    def test_interval_rounds_down_never_coarser_than_asked(self):
        # 25 ps of a 10 ps step is 2.5 steps: rounding up would sample
        # at 30 ps, coarser than the caller allowed.
        mon = self._accumulate(25e-12, n_steps=50, dt=10e-12, f0=1e8)
        assert mon._step_stride == 2

    def test_too_coarse_is_rejected_not_silently_wrong(self):
        with pytest.raises(ValueError, match="samples per period"):
            self._accumulate(400e-12)  # 2.5 samples per period at 1 GHz

    def test_thin_margin_warns(self):
        with pytest.warns(UserWarning, match="samples per period"):
            self._accumulate(180e-12, n_steps=100)  # ~5.6 per period

    def test_ten_samples_per_period_is_silent(self):
        # The documented rule of thumb must not trip its own warning.
        f0 = 1e9
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            mon = self._accumulate(1.0 / (10 * f0), n_steps=100, f0=f0)
        assert mon._step_stride == 100

    def test_validation_uses_the_highest_requested_frequency(self):
        # Safe for 1 GHz, far too coarse for the 8 GHz bin sharing the
        # monitor — the top frequency is what decides.
        with pytest.raises(ValueError, match="8 GHz"):
            self._accumulate(50e-12, freqs=[1e9, 8e9])

    def test_stride_is_keyed_on_the_absolute_step_index(self):
        # A resumed run must sample the same instants as an
        # uninterrupted one, so the phase of the stride cannot depend on
        # where recording started.
        grid = _make_grid(2, 2, 2)
        mon = MonitorFieldFrequency(
            corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
            freqs=np.array([1e9]),
            fields=["Ez"],
            interval=5e-12,
            name="resumed",
        )
        mon.attach(_FakeMesh(grid))
        seen = []
        for n in range(200, 260):
            fields = FieldState.zeros(2, 2, 2)
            fields.Ez[:] = 1.0
            before = complex(np.asarray(mon.data_raw["Ez"]).ravel()[0])
            mon.record(fields, n, n * 1e-12, 1e-12)
            if complex(np.asarray(mon.data_raw["Ez"]).ravel()[0]) != before:
                seen.append(n)
        assert all(n % 5 == 0 for n in seen)
        assert seen[0] == 200


class TestFreqMonitorRenormalize:
    """Verify that 1 W renormalization correctly divides out the source spectrum."""

    def test_renormalize_source_spectrum_matches_accumulator(self):
        """renormalize() must compute the source DFT in the accumulator convention."""
        from magnelio.signals.signal_1d import Signal1D

        dt = 1e-12
        n_steps = 1000
        f0 = 5e9
        omega = 2 * np.pi * f0
        freqs = np.array([f0])

        values = np.sin(omega * np.arange(n_steps) * dt)
        sig = Signal1D(t=np.arange(n_steps) * dt, values=values, dt=dt)

        # DFT accumulator result (reference)
        acc = DFTAccumulator(freqs, ())
        for n in range(n_steps):
            acc.accumulate(values[n], n * dt, dt)
        dft_result = acc.result[0]

        # renormalize() stores the source spectrum in the same convention
        grid = _make_grid(4, 5, 6)
        mon = MonitorFieldFrequency(
            corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
            freqs=freqs,
            fields=["Ez"],
            name="test_conv",
        )
        mon.attach(_FakeMesh(grid))
        mon.renormalize(sig)

        np.testing.assert_allclose(
            mon._source_spectrum[0],
            dft_result,
            rtol=1e-10,
            err_msg="Source spectrum must match DFT accumulator convention",
        )

    def test_renormalize_cancels_source_spectrum(self):
        """After renormalization, the field/source ratio = transfer function."""
        from magnelio.signals.signal_1d import Signal1D

        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)
        f0 = 5e9
        dt = 1e-12
        n_steps = 2000
        omega = 2 * np.pi * f0

        mon = MonitorFieldFrequency(
            corners=((0.005, 0.01, 0.015), (0.005, 0.01, 0.015)),
            freqs=np.array([f0]),
            fields=["Ez"],
            name="test_renorm",
        )
        mon.attach(mesh)

        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

        # Source: amplitude-2 sinusoid
        src_amp = 2.0
        src_values = np.zeros(n_steps)
        # Field: the field is 3x the source (transfer function H = 3)
        transfer = 3.0

        for n in range(n_steps):
            t = n * dt
            src_values[n] = src_amp * np.sin(omega * t)
            fields = FieldState.zeros(Nx, Ny, Nz)
            # states are grid quantities e = E·l (DD-085): a uniform
            # physical Ez of (transfer·src) is the edge voltage E·dz
            fields.Ez[:] = transfer * src_amp * np.sin(omega * t) * float(grid.dz[0])
            mon.record(fields, n, t, dt)

        # Before renormalization: raw DFT magnitude ∝ src_amp * transfer
        raw = mon.data_raw
        raw_mag = np.abs(raw["Ez"][0])

        src_signal = Signal1D(
            t=np.arange(n_steps) * dt,
            values=src_values,
            dt=dt,
        )
        mon.renormalize(src_signal)
        assert mon.is_renormalized

        # After renormalization: normalized magnitude ≈ transfer function
        norm = mon.data
        norm_mag = np.abs(norm["Ez"][0])
        np.testing.assert_allclose(
            norm_mag,
            transfer,
            rtol=0.05,
            err_msg="Renormalized magnitude should equal transfer function",
        )

        # data_raw should still return the unnormalized DFT
        raw2 = mon.data_raw
        np.testing.assert_allclose(
            np.abs(raw2["Ez"][0]),
            raw_mag,
            rtol=1e-10,
        )

    def test_renormalize_2d_shape(self):
        """Renormalization broadcasts correctly over spatial dims."""
        from magnelio.signals.signal_1d import Signal1D

        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)
        cz = 0.5 * (grid.z[2] + grid.z[3])
        freqs = np.array([1e9, 2e9])
        dt = 1e-12
        n_steps = 100

        mon = MonitorFieldFrequency(
            corners=((None, None, cz), (None, None, cz)),
            freqs=freqs,
            fields=["Ez"],
            name="test_renorm_2d",
        )
        mon.attach(mesh)

        for n in range(n_steps):
            fields = _make_fields(grid)
            mon.record(fields, n, n * dt, dt)

        src_values = np.sin(2 * np.pi * 1.5e9 * np.arange(n_steps) * dt)
        sig = Signal1D(t=np.arange(n_steps) * dt, values=src_values, dt=dt)
        mon.renormalize(sig)

        data = mon.data
        assert data["Ez"].shape == (2, 4, 5)  # (n_freqs, Nx, Ny)


class TestDataStatesItsUnit:
    """``.data`` is fields per 1 W CW or it is nothing — never raw bins
    wearing the same name."""

    @staticmethod
    def _recorded():
        grid = _make_grid(4, 5, 6)
        mon = MonitorFieldFrequency(freqs=[1e9], fields=["Ez"], name="units")
        mon.attach(_FakeMesh(grid))
        mon.record(_make_fields(grid), 0, 0.0, 1e-12)
        return mon

    def test_data_refuses_without_a_reference(self):
        mon = self._recorded()
        with pytest.raises(RuntimeError, match="source reference"):
            _ = mon.data

    def test_the_refusal_names_both_ways_out(self):
        mon = self._recorded()
        with pytest.raises(RuntimeError) as excinfo:
            _ = mon.data
        message = str(excinfo.value)
        assert ".data_raw" in message
        assert ".renormalize(" in message

    def test_component_refuses_too(self):
        mon = self._recorded()
        with pytest.raises(RuntimeError, match="source reference"):
            mon.component("Ez")

    def test_raw_bins_stay_reachable_without_a_reference(self):
        mon = self._recorded()
        assert mon.data_raw["Ez"].shape == (1, 4, 5, 6)

    def test_the_bins_survive_renormalization(self):
        """The divisor is stored, never applied to the accumulator — so a
        repeated call cannot stack, and raw stays raw."""
        mon = self._recorded()
        before = mon.data_raw["Ez"].copy()
        _unit_reference(mon)
        _unit_reference(mon)
        np.testing.assert_array_equal(mon.data_raw["Ez"], before)
        np.testing.assert_allclose(mon.data["Ez"], before)

    def test_renormalize_all_skips_other_monitor_kinds(self):
        from magnelio.monitors.field_frequency import renormalize_all
        from magnelio.signals.signal_1d import Signal1D

        grid = _make_grid(4, 5, 6)
        freq = self._recorded()
        time = MonitorFieldTime(times=np.array([0.0]), fields=["E"], name="t")
        time.attach(_FakeMesh(grid))
        time.record(_make_fields(grid), 0, 0.0, 1e-12)

        dt = 1e-12
        renormalize_all(
            [time, freq],
            Signal1D(t=np.array([0.0]), values=np.array([1.0 / dt]), dt=dt),
        )
        assert freq.is_renormalized
        assert not hasattr(time, "is_renormalized")


# -- DFTAccumulator tests -------------------------------------------------


class TestDFTAccumulator:
    def test_dc_signal(self):
        freqs = np.array([0.0])
        acc = DFTAccumulator(freqs, (3,))
        dt = 0.1
        for n in range(100):
            acc.accumulate(np.ones(3), n * dt, dt)
        # DFT at f=0 of constant signal = signal * total_time
        expected = 100 * dt  # = 10.0
        np.testing.assert_allclose(acc.result[0], expected, rtol=1e-10)

    def test_sine_peak(self):
        f0 = 5.0
        freqs = np.array([f0])
        acc = DFTAccumulator(freqs, ())
        dt = 0.01
        n_steps = 1000
        for n in range(n_steps):
            t = n * dt
            acc.accumulate(np.sin(2 * np.pi * f0 * t), t, dt)
        # Should have significant magnitude
        assert np.abs(acc.result[0]) > 1.0


# -- MonitorFluxTime tests ------------------------------------------------


class TestMonitorFluxTime:
    def test_validation(self):
        """The plane is a normal axis plus a finite position."""
        with pytest.raises(ValueError, match="normal must be"):
            MonitorFluxTime(normal="q", position=0.0)
        with pytest.raises(ValueError, match="position must be"):
            MonitorFluxTime(normal="z", position=None)
        with pytest.raises(ValueError, match="position must be"):
            MonitorFluxTime(normal="z", position=float("nan"))

    def test_records_power(self):
        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)
        fields = _make_fields(grid)

        mon = MonitorFluxTime(
            normal="z",
            position=0.015,
            name="flux_z",
        )
        mon.attach(mesh)

        dt = 1e-12
        for n in range(10):
            mon.record(fields, n, n * dt, dt)

        assert mon.t.shape == (10,)
        assert mon.power.shape == (10,)
        # Power should be non-zero for random fields
        assert np.any(mon.power != 0)


# -- plane-view resolution and 3D slice plotting ---------------------------


class TestResolvePlaneView:
    def test_2d_region(self):
        grid = _make_grid(4, 5, 6)
        cz = 0.5 * (grid.z[3] + grid.z[4])
        r = resolve_region(((None, None, cz), (None, None, cz)), grid)
        pv = resolve_plane_view(r, None, 0.0)
        assert pv.normal_idx == 2
        assert pv.slice_index is None
        assert pv.normal_pos == pytest.approx(cz)
        assert [i for i, _ in pv.free] == [0, 1]

    def test_2d_region_normal_mismatch_raises(self):
        grid = _make_grid(4, 5, 6)
        cz = 0.5 * (grid.z[3] + grid.z[4])
        r = resolve_region(((None, None, cz), (None, None, cz)), grid)
        with pytest.raises(ValueError, match="normal 'z'"):
            resolve_plane_view(r, "x", 0.0)

    def test_3d_region_snaps_position(self):
        grid = _make_grid(4, 5, 6)
        r = resolve_region(None, grid)
        pv = resolve_plane_view(r, "y", 0.0101)
        assert pv.normal_idx == 1
        cc = 0.5 * (grid.y[:-1] + grid.y[1:])
        k = int(np.argmin(np.abs(cc - 0.0101)))
        assert pv.slice_index == k
        assert pv.normal_pos == pytest.approx(cc[k])
        assert [i for i, _ in pv.free] == [0, 2]

    def test_3d_region_requires_normal(self):
        grid = _make_grid(4, 5, 6)
        r = resolve_region(None, grid)
        with pytest.raises(ValueError, match="normal="):
            resolve_plane_view(r, None, 0.0)

    def test_take2d(self):
        grid = _make_grid(4, 5, 6)
        r = resolve_region(None, grid)
        pv = resolve_plane_view(r, "y", 0.0)
        arr = np.arange(4 * 5 * 6).reshape(4, 5, 6)
        np.testing.assert_array_equal(pv.take2d(arr), arr[:, pv.slice_index, :])


class TestMonitor3DPlotting:
    def _volume_monitor(self):
        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)
        fields = _make_fields(grid)
        mon = MonitorFieldTime(times=np.array([0.0]), fields=["E"], name="vol")
        mon.attach(mesh)
        mon.record(fields, 0, 0.0, 1e-12)
        return mon

    def test_scalar_slice_matches_data(self):
        import matplotlib.pyplot as plt

        mon = self._volume_monitor()
        fig, ax = mon.plot(component="Ez", normal="y", position=0.008, plot_type="color")
        qm = [c for c in ax.collections if hasattr(c, "get_clim")][0]
        r = mon.region
        cc = r.yc
        k = int(np.argmin(np.abs(cc - 0.008)))
        expected = mon.data["Ez"][0][:, k, :]
        np.testing.assert_allclose(np.asarray(qm.get_array()).reshape(expected.shape), expected)
        assert "y=" in ax.get_title()
        plt.close(fig)

    def test_vector_slice_with_normal_component(self):
        import matplotlib.pyplot as plt

        mon = self._volume_monitor()
        fig, ax = mon.plot(component="E", normal="z", position=0.0, plot_type="vector")
        assert fig is not None
        plt.close(fig)

    def test_missing_normal_raises(self):
        mon = self._volume_monitor()
        with pytest.raises(ValueError, match="normal="):
            mon.plot(component="Ez", plot_type="color")

    def test_freq_monitor_slice(self):
        import matplotlib.pyplot as plt

        grid = _make_grid(4, 5, 6)
        mesh = _FakeMesh(grid)
        fields = _make_fields(grid)
        mon = MonitorFieldFrequency(freqs=[1e9], fields=["E"], name="vol_f")
        mon.attach(mesh)
        dt = 1e-12
        for n in range(4):
            mon.record(fields, n, n * dt, dt)
        mon.finalize()
        _unit_reference(mon, dt)
        fig, ax = mon.plot(component="E", normal="x", position=0.004, plot_type="color")
        assert "x=" in ax.get_title()
        fig2, ax2 = mon.plot(component="E", normal="x", position=0.004, plot_type="vector")
        plt.close("all")
