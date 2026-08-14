"""Unit tests for the port-signal stall watchdog and the runtime cap.

The watchdog proves the port-signal stop threshold unreachable before
the runtime cap (band-edge cut-off content decays algebraically, so a
threshold just below the plateau level is never crossed) and ends the
run early with the same outcome the cap would deliver.  These tests
drive the pure detector on synthetic envelopes and the solver wiring
through a scripted fake port — no physics, fully deterministic.
"""

import numpy as np
import pytest

from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver.fit_td import FITTimeDomainSolver, _SignalStallDetector
from magnelio.solver.stability import courant_dt

_BC = {f: "PEC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}


class TestSignalStallDetector:
    def test_plateau_is_stalled(self):
        det = _SignalStallDetector(arm_db=40.0, window=10, cap_step=10_000)
        verdicts = [det.observe(n, -50.0, -60.0) for n in range(0, 100, 10)]
        assert verdicts[:9] == [False] * 9  # window filling
        assert verdicts[9] is True

    def test_exponential_within_cap_is_left_alone(self):
        det = _SignalStallDetector(arm_db=40.0, window=10, cap_step=1_000_000)
        # -0.01 dB/step from -45 dB: hits -60 dB after 1500 more steps,
        # far inside the cap.
        for i, n in enumerate(range(0, 300, 10)):
            assert det.observe(n, -45.0 - 0.01 * n, -60.0) is False, i

    def test_slow_exponential_beyond_cap_is_stalled(self):
        det = _SignalStallDetector(arm_db=40.0, window=10, cap_step=1_000)
        # Same slope, but the cap sits at step 1000 < the ~1500-step hit.
        verdicts = [det.observe(n, -45.0 - 0.01 * n, -60.0) for n in range(0, 100, 10)]
        assert verdicts[-1] is True

    def test_recovery_above_arming_floor_resets_window(self):
        det = _SignalStallDetector(arm_db=40.0, window=5, cap_step=10_000)
        for n in range(0, 40, 10):
            det.observe(n, -50.0, -60.0)
        det.observe(40, -30.0, -60.0)  # envelope recovered → window drops
        assert det.observe(50, -50.0, -60.0) is False  # window restarts

    def test_below_threshold_is_not_stalled(self):
        # Levels already past the threshold belong to the stop criterion,
        # never to the watchdog.
        det = _SignalStallDetector(arm_db=40.0, window=3, cap_step=10_000)
        verdicts = [det.observe(n, -65.0, -60.0) for n in range(0, 50, 10)]
        assert all(v is False for v in verdicts)


class _ScriptedPort:
    """Fake modal port: no field coupling, scripted |V| envelope."""

    name = "fake"

    def __init__(self, envelope_db):
        self._envelope_db = envelope_db
        self._polls = 0

    def update_e(self, fields, t, dt):
        pass

    def poll_signal_absmax(self):
        db = self._envelope_db(self._polls)
        self._polls += 1
        return 10.0 ** (db / 20.0)


def _solver(envelope_db, **kwargs):
    N = 6
    L = N * 1e-3
    grid = GridLines(
        x=np.linspace(0, L, N + 1),
        y=np.linspace(0, L, N + 1),
        z=np.linspace(0, L, N + 1),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC)
    bcs = {f: PECBoundary(f) for f in _BC}
    return FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        ports=[_ScriptedPort(envelope_db)],
        dt=courant_dt(grid, "draft"),
        verbose=False,
        backend="numpy",
        energy_stop_db=None,
        energy_check_interval=10,
        **kwargs,
    )


class TestSolverStallWiring:
    def test_plateau_run_stops_on_stall(self):
        # Peak at the first poll, then a flat -50 dB plateau: the -60 dB
        # criterion can never fire; the watchdog must end the run.
        solver = _solver(
            lambda i: 0.0 if i == 0 else -50.0,
            total_time_steps=None,
            port_signal_stop_db=60.0,
            max_time_steps=100_000,
        )
        with pytest.warns(RuntimeWarning, match="stalled at -50.0 dB"):
            solver.run()
        assert solver._stop_reason == "port_signal_stall"
        assert solver._final_signal_db == pytest.approx(-50.0)
        assert solver._actual_steps < 2_000  # window ~10 checks, not the cap

    def test_decaying_run_stops_on_criterion_not_stall(self):
        # -2 dB per check: crosses -60 dB at check 30, well before cap.
        solver = _solver(
            lambda i: -2.0 * i,
            total_time_steps=None,
            port_signal_stop_db=60.0,
            max_time_steps=100_000,
        )
        solver.run()
        assert solver._stop_reason == "port_signal"
        assert solver._final_signal_db <= -60.0

    def test_uncatchable_run_hits_runtime_cap(self):
        # Envelope pinned at -30 dB: above the arming floor, so neither
        # the criterion nor the watchdog fires — the cap must.
        solver = _solver(
            lambda i: 0.0 if i == 0 else -30.0,
            total_time_steps=None,
            port_signal_stop_db=60.0,
            max_time_steps=500,
        )
        with pytest.warns(RuntimeWarning, match="runtime cap of 500 steps"):
            solver.run()
        assert solver._stop_reason == "runtime_cap"
        assert solver._actual_steps == 500

    def test_bounded_run_ignores_watchdog_and_cap(self):
        solver = _solver(
            lambda i: 0.0 if i == 0 else -50.0,
            total_time_steps=200,
            port_signal_stop_db=60.0,
            max_time_steps=50,  # would truncate at 50 if it applied
        )
        solver.run()
        assert solver._stop_reason == "steps"
        assert solver._actual_steps == 200

    def test_uncapped_unbounded_run_keeps_detector_off(self):
        # max_time_steps=None is the march-forever opt-out: the plateau
        # then rides until the criterion (here: reachable) fires.
        solver = _solver(
            lambda i: -1.5 * i,
            total_time_steps=None,
            port_signal_stop_db=60.0,
            max_time_steps=None,
        )
        solver.run()
        assert solver._stop_reason == "port_signal"

    def test_cap_must_advance_past_resume_step(self):
        solver = _solver(
            lambda i: -50.0,
            total_time_steps=None,
            port_signal_stop_db=60.0,
            max_time_steps=100,
        )
        solver._resume_step = 100
        with pytest.raises(ValueError, match="max_time_steps"):
            solver.run()
