"""Resume checkpoint persistence + graceful abort — the WP-S7 gate (DD-070).

WP-S6 proved the in-RAM ``state_dict`` is complete (bit-exact same-ops
rewind).  WP-S7 adds three things this file gates:

* **On-disk round-trip** — the generic ``checkpoint.h5`` serialiser
  (``_write_state_dict_h5`` / ``_read_state_dict_h5``) is lossless, so a
  resume from *disk* is still bit-exact, not merely the in-RAM snapshot.
* **Graceful abort** — a cooperative stop (``request_stop`` / a trapped
  ``SIGINT``) breaks the march at the top of an iteration, i.e. on a
  consistent leapfrog pair; ``state_dict`` there carries ``n_completed ==
  steps_recorded`` and resumes bit-exactly.  The sink checkpoint is just
  ``_write_state_dict_h5(path, solver.state_dict())``, so those two
  lossless pieces compose into a consistent on-disk abort checkpoint.
* **Unbounded runtime** — ``total_time_steps=None`` marches until the
  energy criterion (the resume-longer engine), and is rejected without a
  stop criterion.

The determinism caveat from ``test_resume_bit_exact`` applies: two
independently built modal ports differ at ~1e-13 (ARPACK start vector),
so every bit-exact assertion rewinds *the same* solver.
"""

from __future__ import annotations

import os
import signal

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.boundaries.pec import PECBoundary
from magnelio.io.project import (
    _read_state_dict_h5,
    _write_state_dict_h5,
    open_project,
)
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal import (
    BoxFace,
    ExcitationSpec,
    PortSpecRectWG,
    build_modal_port,
)
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

WR90_A = 22.86e-3
WR90_B = 10.16e-3


def _wr90_solver(n_steps):
    """One straight WR-90 line, TE10 excited at port1 (built once)."""
    grid = GridLines(
        x=np.linspace(0.0, 30e-3, 31),
        y=np.linspace(0.0, WR90_A, 24),
        z=np.linspace(0.0, WR90_B, 11),
    )
    mesh = Mesh.from_grid(grid)
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(grid, accuracy="normal")
    excitation = ExcitationSpec(f_min=8.2e9, f_max=12.4e9, mode_index=0)
    op_src = build_modal_port(
        PortSpecRectWG(
            name="port1",
            plane=BoxFace.X_MIN,
            width_a=WR90_A,
            height_b=WR90_B,
            n_modes=1,
            excitation=excitation,
        ),
        mesh,
        m_eps,
        m_mu,
        dt=dt,
        f_calc=10.0e9,
    )
    op_load = build_modal_port(
        PortSpecRectWG(
            name="port2",
            plane=BoxFace.X_MAX,
            width_a=WR90_A,
            height_b=WR90_B,
            n_modes=1,
        ),
        mesh,
        m_eps,
        m_mu,
        dt=dt,
        f_calc=10.0e9,
    )
    return FITTimeDomainSolver(
        mesh=mesh,
        dt=dt,
        total_time_steps=n_steps,
        ports=[op_src, op_load],
        boundary_conditions={
            "ymin": PECBoundary("ymin"),
            "ymax": PECBoundary("ymax"),
            "zmin": PECBoundary("zmin"),
            "zmax": PECBoundary("zmax"),
            "xmin": "PMC",
            "xmax": "PMC",
        },
        verbose=False,
    )


class _CallAtStep:
    """Test monitor that fires a zero-arg callback once, at a given step."""

    def __init__(self, step: int, callback) -> None:
        self._step = step
        self._callback = callback
        self._fired = False

    def attach(self, mesh) -> None:  # solver.setup() contract
        pass

    def record(self, fields, n, t, dt) -> None:
        if n == self._step and not self._fired:
            self._fired = True
            self._callback()


# ═════════════════════════════════════════════════════════════════════
# On-disk checkpoint round-trip: resume from checkpoint.h5 is bit-exact
# ═════════════════════════════════════════════════════════════════════


def test_checkpoint_file_bit_exact_resume(tmp_path):
    """A resume from ``checkpoint.h5`` reproduces the run exactly.

    Same-operators rewind with a *disk* round-trip in the seam: run to
    N, write the state to HDF5, continue to N+M (reference), then read the
    file back, load it into those same operators, and re-march.  Bit-exact
    ⇒ the serialiser is lossless over the full reflection-free state (E/H,
    Mur, TF/SF ring buffer, exact DTBC convolution history).
    """
    solver = _wr90_solver(n_steps=120)
    solver.run()  # 0 -> 120
    ckpt = tmp_path / "checkpoint.h5"
    _write_state_dict_h5(ckpt, solver.state_dict())
    assert ckpt.exists() and not (tmp_path / "checkpoint.h5.tmp").exists()

    solver.total_time_steps = 260
    solver.run()  # 120 -> 260 (reference)
    ref_e = solver._fields.e_flat.copy()
    ref_h = solver._fields.h_flat.copy()

    disk_state = _read_state_dict_h5(ckpt)
    assert int(disk_state["n_completed"]) == 120
    assert disk_state["ports"]["port1"]["dtbc"]["0"]["n"] == 120

    solver.load_state_dict(disk_state)  # rewind to 120 from disk
    solver.total_time_steps = 260
    solver.run()  # replay 120 -> 260
    res_e = solver._fields.e_flat.copy()
    res_h = solver._fields.h_flat.copy()

    assert np.array_equal(ref_e, res_e), (
        f"E not bit-exact after disk resume: max|Δ| = {float(np.max(np.abs(ref_e - res_e))):.3e}"
    )
    assert np.array_equal(ref_h, res_h), (
        f"H not bit-exact after disk resume: max|Δ| = {float(np.max(np.abs(ref_h - res_h))):.3e}"
    )


# ═════════════════════════════════════════════════════════════════════
# Graceful abort lands on a consistent, resumable state
# ═════════════════════════════════════════════════════════════════════


def test_graceful_abort_consistent_and_resumable(tmp_path):
    """request_stop() at step k breaks at a consistent pair (n_completed
    = k+1) that resumes bit-exactly."""
    solver = _wr90_solver(n_steps=260)
    recorder_calls = {"stop_at": 120}
    solver.monitors = [
        _CallAtStep(recorder_calls["stop_at"], solver.request_stop),
    ]
    solver.run()  # aborts at 121

    # Broke at the top of iteration 121: the completed count is 121, the
    # state is a consistent leapfrog pair, and the run is flagged aborted.
    assert solver._aborted is True
    assert solver._resume_step == 121
    snap = solver.state_dict()
    assert int(snap["n_completed"]) == 121
    # non-trivial reflection-free state actually captured
    assert snap["ports"]["port1"]["dtbc"]["0"]["n"] == 121
    assert float(np.abs(snap["e"]).max()) > 0.0

    # Persist the abort snapshot, then prove a resume from it is bit-exact.
    ckpt = tmp_path / "abort.h5"
    _write_state_dict_h5(ckpt, snap)

    solver.monitors = []  # do not re-trip
    solver.total_time_steps = 300
    solver.run()  # 121 -> 300 (reference)
    ref_e = solver._fields.e_flat.copy()
    ref_h = solver._fields.h_flat.copy()

    solver.load_state_dict(_read_state_dict_h5(ckpt))  # rewind to 121
    solver.total_time_steps = 300
    solver.run()  # replay 121 -> 300
    assert np.array_equal(ref_e, solver._fields.e_flat)
    assert np.array_equal(ref_h, solver._fields.h_flat)


# ═════════════════════════════════════════════════════════════════════
# Unbounded runtime (total_time_steps=None) — the resume-longer engine
# ═════════════════════════════════════════════════════════════════════


def test_unbounded_run_stops_on_energy():
    """An unbounded march ends on the energy criterion at the same step a
    generously bounded one does (identical check cadence)."""
    unbounded = _wr90_solver(n_steps=None)
    unbounded.energy_stop_db = 25.0
    unbounded.run()
    assert unbounded._aborted is False
    assert 0 < unbounded._actual_steps < 5000, (
        f"unbounded run did not stop cleanly: {unbounded._actual_steps}"
    )

    # A bounded run whose cap is large enough to share check_interval=100
    # must reach the criterion at the very same step.
    bounded = _wr90_solver(n_steps=5000)
    bounded.energy_stop_db = 25.0
    bounded.run()
    assert unbounded._actual_steps == bounded._actual_steps


def test_unbounded_run_requires_stop_criterion():
    """total_time_steps=None with no energy_stop_db is a misconfiguration."""
    solver = _wr90_solver(n_steps=None)  # fixture leaves energy_stop_db=None
    with pytest.raises(ValueError, match="energy_stop_db"):
        solver.run()


# ═════════════════════════════════════════════════════════════════════
# Level A: streamed run writes a checkpoint; Ctrl-C aborts gracefully
# ═════════════════════════════════════════════════════════════════════


def _tem_analysis(tmp_path, project):
    """A small parallel-plate TEM two-port (needs OCC), streamed variant."""
    from magnelio import AnalysisScatteringTD, Material, MeshControl
    from magnelio.geo import (
        Brick,
        GeometryModel,  # noqa: PLC0415
    )
    from magnelio.ports import PortWaveguide

    a, b, L = 10.0e-3, 5.0e-3, 20.0e-3
    f_max = 12.0e9
    model = GeometryModel(
        boundary_conditions={
            "xmin": "PMC",
            "xmax": "PMC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PEC",
            "zmax": "PEC",
        }
    )
    model.add(
        Brick(
            origin=(-a / 2, -b / 2, -L / 2),
            size=(a, b, L),
            material=Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=8),
        f_max=f_max,
    )
    return AnalysisScatteringTD(
        mesh=mesh,
        ports=[
            PortWaveguide(name="port1", plane="zmin", n_modes=1),
            PortWaveguide(name="port2", plane="zmax", n_modes=1),
        ],
        f_max=f_max,
        verbose=False,
        project=project,
        geometry=model,
    )


def test_streamed_run_writes_final_checkpoint(tmp_path):
    """A completed streamed run leaves a loadable resume checkpoint whose
    n_completed matches the recorded length (enables run-longer)."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    p = tmp_path / "pp"
    proj = _tem_analysis(tmp_path, project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=400,
        checkpoint_interval=50,
    )
    assert proj.status == "done"
    ckpt = p / "runs" / "port1_mode0" / "checkpoint.h5"
    assert ckpt.exists()

    state = proj.checkpoint_state(("port1", 0))
    assert state is not None
    n_recorded = proj.runs["port1_mode0"]["n_steps"]
    assert int(state["n_completed"]) == n_recorded == 400
    # the field state is really there (not just metadata); E-edge and
    # H-face counts differ on the Yee grid, so only the ranks match.
    assert state["e"].ndim == state["h"].ndim == 1
    assert state["e"].size > 0 and state["h"].size > 0
    assert float(np.abs(state["e"]).max()) > 0.0


def test_streamed_graceful_abort_via_sigint(tmp_path):
    """A Ctrl-C during a streamed run stops it gracefully: KeyboardInterrupt
    still propagates (the program stops), but a consistent checkpoint is on
    disk, the run is marked aborted, and the previous SIGINT handler is
    restored (no dangling trap)."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    p = tmp_path / "pp"
    analysis = _tem_analysis(tmp_path, project=p)
    analysis.monitors = (_CallAtStep(30, lambda: os.kill(os.getpid(), signal.SIGINT)),)

    handler_before = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        analysis.run(
            excited=[("port1", 0)],
            energy_stop_db=None,
            total_time_steps=3000,
            checkpoint_interval=10,
        )
    # the trap is transient: the caller's SIGINT handling is intact after
    assert signal.getsignal(signal.SIGINT) is handler_before

    # the aborted run left a resumable project on disk
    proj = open_project(p)
    assert proj.runs["port1_mode0"]["state"] == "aborted"
    n_aborted = proj.runs["port1_mode0"]["n_steps"]
    assert 0 < n_aborted < 3000, f"did not abort early: {n_aborted}"

    ckpt = p / "runs" / "port1_mode0" / "checkpoint.h5"
    assert ckpt.exists()
    state = proj.checkpoint_state(("port1", 0))
    assert int(state["n_completed"]) == n_aborted
