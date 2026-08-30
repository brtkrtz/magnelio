"""Unit tests for ``SourceCurrentPath`` (DD-227)."""

import numpy as np
import pytest

from magnelio.circuit import rasterize_curve
from magnelio.geo import Curve
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.signals import WaveformGaussian
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt
from magnelio.sources import SourceCurrentPath
from magnelio.sources.base import Source

D = 1e-3  # cell size of the test grid


def _grid(n=6):
    line = np.arange(n + 1) * D
    return GridLines(x=line, y=line, z=line)


def _solver(mesh, sources, steps=5):
    dt = courant_dt(mesh.grid, accuracy="normal")
    return FITTimeDomainSolver(
        mesh=mesh, total_time_steps=steps, dt=dt, verbose=False, sources=sources
    )


def _attached(path, *, grid=None, amplitude=1.0, **kw):
    """A source attached to a plain air solver on the test grid."""
    grid = _grid() if grid is None else grid
    src = SourceCurrentPath(name="fil", path=path, **kw)
    src.set_excitation(WaveformGaussian(f_max=100e9), amplitude=amplitude)
    solver = _solver(Mesh.from_grid(grid), [src])
    solver.setup()
    return src, solver


class TestContract:
    def test_hierarchy_and_unit(self):
        src = SourceCurrentPath(name="fil", path=[(0, 0, 0), (0, 0, 2 * D)])
        assert isinstance(src, Source)
        assert src.amplitude_unit == "A"
        assert src.excitable is True
        assert src.has_waveform is True
        assert src.writes_initial_field is False

    def test_points_become_a_polyline(self):
        src = SourceCurrentPath(name="fil", path=[(0, 0, 0), (0, 0, D), (0, D, D)])
        assert isinstance(src.curve, Curve)

    def test_a_curve_is_taken_as_given(self):
        curve = Curve.polyline([(0, 0, 0), (0, 0, 2 * D)])
        assert SourceCurrentPath(name="fil", path=curve).curve is curve

    def test_bad_arguments(self):
        with pytest.raises(TypeError, match="name"):
            SourceCurrentPath(name="", path=[(0, 0, 0), (0, 0, D)])
        with pytest.raises(TypeError, match="path"):
            SourceCurrentPath(name="fil", path=None)
        with pytest.raises(ValueError, match="at least two"):
            SourceCurrentPath(name="fil", path=[(0, 0, 0)])
        with pytest.raises(ValueError, match="samples_per_cell"):
            SourceCurrentPath(name="fil", path=[(0, 0, 0), (0, 0, D)], samples_per_cell=1)

    def test_excitation_binding(self):
        src = SourceCurrentPath(name="fil", path=[(0, 0, 0), (0, 0, 2 * D)])
        wf = WaveformGaussian(f_max=10e9)
        src.set_excitation(wf, amplitude=2.0, delay=1e-12)
        assert src.waveform is wf
        assert src._drive(1e-12) == pytest.approx(2.0 * wf(0.0))
        src.clear_excitation()
        assert src.waveform is None


class TestRasterisation:
    def test_dipole_moment_is_the_endpoint_vector(self):
        """Σ sign·dl·â over the staircase is exactly the snapped chord.

        This is why an oblique filament still carries the right dipole
        moment: only the higher multipoles see the staircase.
        """
        grid = _grid(n=8)
        start, end = (D, D, D), (6 * D, 4 * D, 3 * D)
        path = rasterize_curve(Curve.polyline([start, end]), grid)
        moment = np.zeros(3)
        for axis, sign, dl in zip(path.axes, path.signs, path.dls):
            moment["xyz".index(axis)] += sign * dl
        np.testing.assert_allclose(moment, np.subtract(end, start), atol=1e-15)

    def test_coefficient_is_minus_beta_per_edge(self):
        """Ampère's law: an impressed current opposes the field it drives."""
        src, solver = _attached([(3 * D, 3 * D, 2 * D), (3 * D, 3 * D, 4 * D)])
        idx = np.asarray(src._idx)
        beta = np.asarray(solver._beta_E)[idx]
        assert idx.size == 2
        np.testing.assert_allclose(np.asarray(src._coef), -beta, rtol=1e-7)

    def test_a_doubled_back_segment_cancels(self):
        """Out and back on the same edges is no current at all."""
        src, _ = _attached([(3 * D, 3 * D, 2 * D), (3 * D, 3 * D, 4 * D), (3 * D, 3 * D, 2 * D)])
        assert np.asarray(src._idx).size == 0

    def test_a_closed_loop_has_no_free_end(self):
        """Every edge of a loop carries the same current, once."""
        c = 3 * D
        src, _ = _attached(
            [
                (c - D, c - D, c),
                (c + D, c - D, c),
                (c + D, c + D, c),
                (c - D, c + D, c),
                (c - D, c - D, c),
            ]
        )
        coef = np.asarray(src._coef)
        assert coef.size == 8
        # Four edges run with the axis, four against it.
        assert np.count_nonzero(coef > 0) == 4
        assert np.count_nonzero(coef < 0) == 4

    def test_path_outside_the_domain_is_refused(self):
        with pytest.raises(ValueError, match="leaves the meshed domain"):
            _attached([(3 * D, 3 * D, 2 * D), (3 * D, 3 * D, 20 * D)])

    def test_edges_held_at_zero_are_reported(self):
        """A path on the PEC domain wall carries nothing, and says so."""
        with pytest.warns(UserWarning, match="held at zero"):
            _attached([(3 * D, 0.0, 2 * D), (3 * D, 0.0, 4 * D)])


class TestInjection:
    def test_one_step_writes_minus_beta_times_the_current(self):
        """From rest, the first E step is the impressed current alone."""
        src, solver = _attached([(3 * D, 3 * D, 2 * D), (3 * D, 3 * D, 4 * D)], amplitude=3.0)
        fields = solver._fields
        fields.e_flat[:] = 0.0
        dt = solver.dt
        src.inject_E(fields, dt)
        idx = np.asarray(src._idx)
        expected = np.asarray(src._coef) * src._drive(dt / 2)
        np.testing.assert_allclose(np.asarray(fields.e_flat)[idx], expected, rtol=1e-6, atol=1e-30)
        # Nothing anywhere else.
        rest = np.delete(np.asarray(fields.e_flat), idx)
        assert np.all(rest == 0.0)

    def test_h_side_is_untouched(self):
        src, solver = _attached([(3 * D, 3 * D, 2 * D), (3 * D, 3 * D, 4 * D)])
        before = np.asarray(solver._fields.h_flat).copy()
        src.inject_H(solver._fields, solver.dt)
        np.testing.assert_array_equal(np.asarray(solver._fields.h_flat), before)


class TestStoreRoundTrip:
    def test_points_round_trip(self):
        from magnelio.io.project import _source_from_dict, _source_to_dict

        src = SourceCurrentPath(name="fil", path=[(0.0, 0.0, 0.0), (0.0, 0.0, 2 * D)])
        back = _source_from_dict(_source_to_dict(src))
        assert isinstance(back, SourceCurrentPath)
        assert back.name == "fil"
        np.testing.assert_allclose(back.path, [(0.0, 0.0, 0.0), (0.0, 0.0, 2 * D)])

    def test_a_general_curve_says_it_cannot_be_rebuilt(self):
        from magnelio.io.project import _source_from_dict, _source_to_dict

        src = SourceCurrentPath(name="fil", path=Curve.polyline([(0, 0, 0), (0, 0, 2 * D)]))
        d = _source_to_dict(src)
        assert "path" not in d
        with pytest.raises(ValueError, match="cannot rebuild"):
            _source_from_dict(d)
