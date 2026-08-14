"""``magnelio.resume`` — the WP-S8 acceptance (DD-070).

WP-S6/S7 proved the checkpoint is complete and lossless by rewinding the
*same* operators.  WP-S8 closes the loop end to end: :func:`magnelio.resume`
rebuilds the run's operators from the store recipe, loads the latest
``checkpoint.h5`` into that fresh solver, reopens ``results.h5``, and
marches on.  The acceptance is that a resumed run is **bit-identical to an
uninterrupted run of the same total length** — the checkpoint/reload seam
injects nothing.

The line is a TEM parallel plate on purpose: its port modes are solved by
a deterministic dense Laplace (``np.linalg.eigh``), so an independently
*rebuilt* operator is bit-identical to the original — unlike an ARPACK
TE/TM build, which differs at ~1e-13 (the caveat isolated in
``test_resume_bit_exact``).  That makes the rebuild-then-resume path here
exactly reproducible, not merely FP-floor close.
"""

from __future__ import annotations

import os
import signal

import numpy as np
import pytest

import magnelio
from magnelio import AnalysisScatteringTD, Material, MeshControl, open_project, resume
from magnelio.geo import Brick
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortWaveguide

A, B, LZ = 10.0e-3, 5.0e-3, 20.0e-3
F_MAX = 12.0e9


def _tem_analysis(project=None, monitors=()):
    """A small parallel-plate TEM two-port (needs OCC); optional store."""
    from magnelio.geo import GeometryModel  # noqa: PLC0415

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
            origin=(-A / 2, -B / 2, -LZ / 2),
            size=(A, B, LZ),
            material=Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=8),
        f_max=F_MAX,
    )
    analysis = AnalysisScatteringTD(
        mesh=mesh,
        ports=[
            PortWaveguide(name="port1", plane="zmin", n_modes=1),
            PortWaveguide(name="port2", plane="zmax", n_modes=1),
        ],
        f_max=F_MAX,
        verbose=False,
        project=project,
        geometry=model,
    )
    analysis.monitors = monitors
    return analysis


class _CallAtStep:
    """Monitor that fires a zero-arg callback once, at a given step."""

    def __init__(self, step, callback):
        self._step, self._cb, self._fired = step, callback, False

    def attach(self, mesh):
        pass

    def record(self, fields, n, t, dt):
        if n == self._step and not self._fired:
            self._fired = True
            self._cb()


def _vi(signals_for_excitation):
    """Flatten a run's channel signals to ``{chan: (V_arr, I_arr)}``."""
    return {k: (v[0].values.copy(), v[1].values.copy()) for k, v in signals_for_excitation.items()}


def _assert_bit_exact(ref_vi, got_vi, tag):
    assert set(ref_vi) == set(got_vi), f"{tag}: channel set differs"
    for chan, (rv, ri) in ref_vi.items():
        gv, gi = got_vi[chan]
        assert gv.shape == rv.shape, f"{tag} {chan}: length {gv.shape} != reference {rv.shape}"
        assert np.array_equal(rv, gv), (
            f"{tag} {chan}: V not bit-exact, max|Δ|={float(np.max(np.abs(rv - gv))):.3e}"
        )
        assert np.array_equal(ri, gi), (
            f"{tag} {chan}: I not bit-exact, max|Δ|={float(np.max(np.abs(ri - gi))):.3e}"
        )


# ═════════════════════════════════════════════════════════════════════
# Primary acceptance: a bounded resume is bit-identical to one long run
# ═════════════════════════════════════════════════════════════════════


def test_resume_bounded_bit_exact(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    n1, n_total = 120, 300

    ref = _tem_analysis().run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n_total,
    )
    ref_vi = _vi(ref.signals[("port1", 0)])

    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n1,
        checkpoint_interval=40,
    )
    assert open_project(p).runs["port1_mode0"]["n_steps"] == n1

    proj = resume(p, excited=("port1", 0), total_time_steps=n_total, verbose=False)
    assert proj.runs["port1_mode0"]["state"] == "done"
    assert proj.runs["port1_mode0"]["n_steps"] == n_total
    _assert_bit_exact(ref_vi, _vi(proj.signals[("port1", 0)]), "bounded-resume")
    # S-parameters derived on read match the uninterrupted run exactly.
    assert np.array_equal(ref.S("port1", "port1"), proj.S("port1", "port1"))
    assert np.array_equal(ref.S("port2", "port1"), proj.S("port2", "port1"))


def test_resume_port_signal_gated(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    n1 = 120

    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n1,
        checkpoint_interval=40,
    )
    assert open_project(p).runs["port1_mode0"]["n_steps"] == n1

    # Continue on the port-signal criterion alone: the DTBC line drains
    # its pulse quickly, so the envelope reaches -45 dB soon after the
    # arming guard and the run flips to done past the bounded stop.
    proj = resume(p, excited=("port1", 0), port_signal_stop_db=45.0, verbose=False)
    meta = proj.runs["port1_mode0"]
    assert meta["state"] == "done"
    assert meta["n_steps"] > n1


# ═════════════════════════════════════════════════════════════════════
# Energy-gated resume reproduces an uninterrupted energy-gated run
# ═════════════════════════════════════════════════════════════════════


def test_resume_energy_gated_bit_exact(tmp_path):
    """run1 stops BOUNDED before the pulse decays, so the energy-gated
    resume hits the same check grid as the uninterrupted energy run and
    stops at the identical step with bit-identical V/I."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")

    ref = _tem_analysis().run(
        excited=[("port1", 0)],
        energy_stop_db=60.0,
        total_time_steps=None,
    )
    ref_vi = _vi(ref.signals[("port1", 0)])
    assert ref.n_actual_steps > 120  # decays well after the bounded stub

    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=100,
        checkpoint_interval=40,
    )
    proj = resume(p, excited=("port1", 0), energy_stop_db=60.0, verbose=False)

    assert proj.runs["port1_mode0"]["n_steps"] == ref.n_actual_steps
    _assert_bit_exact(ref_vi, _vi(proj.signals[("port1", 0)]), "energy-resume")


# ═════════════════════════════════════════════════════════════════════
# Ctrl-C graceful abort, then resume to completion — bit-exact
# ═════════════════════════════════════════════════════════════════════


def test_resume_after_graceful_abort(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    n_total = 300

    ref = _tem_analysis().run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n_total,
    )
    ref_vi = _vi(ref.signals[("port1", 0)])

    p = tmp_path / "pp"
    analysis = _tem_analysis(
        project=p,
        monitors=(_CallAtStep(80, lambda: os.kill(os.getpid(), signal.SIGINT)),),
    )
    with pytest.raises(KeyboardInterrupt):
        analysis.run(
            excited=[("port1", 0)],
            energy_stop_db=None,
            total_time_steps=n_total,
            checkpoint_interval=10,
        )

    aborted = open_project(p)
    assert aborted.runs["port1_mode0"]["state"] == "aborted"
    n_ab = aborted.runs["port1_mode0"]["n_steps"]
    assert 0 < n_ab < n_total

    # A bare resume() finishes the aborted run to its original target.
    proj = resume(p, excited=("port1", 0), verbose=False)
    assert proj.runs["port1_mode0"]["state"] == "done"
    assert proj.runs["port1_mode0"]["n_steps"] == n_total
    _assert_bit_exact(ref_vi, _vi(proj.signals[("port1", 0)]), "abort-resume")


# ═════════════════════════════════════════════════════════════════════
# Fill-in: resume one run without disturbing the sibling column
# ═════════════════════════════════════════════════════════════════════


def test_resume_fill_in_leaves_sibling_untouched(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    n1, n_total = 120, 300

    p = tmp_path / "pp"
    # Column S*1 stops short; column S*2 runs to length in the same project.
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n1,
        checkpoint_interval=40,
    )
    _tem_analysis(project=p).run(
        excited=[("port2", 0)],
        energy_stop_db=None,
        total_time_steps=n_total,
        checkpoint_interval=40,
    )
    before = _vi(open_project(p).signals[("port2", 0)])

    proj = resume(p, excited=("port1", 0), total_time_steps=n_total, verbose=False)
    # port1 grew to full length; port2 is byte-for-byte the same.
    assert proj.runs["port1_mode0"]["n_steps"] == n_total
    assert proj.runs["port2_mode0"]["n_steps"] == n_total
    _assert_bit_exact(before, _vi(proj.signals[("port2", 0)]), "fill-in-sibling")
    # Both columns present ⇒ the 2×2 S-matrix is fully populated.
    assert proj.S("port1", "port1").shape == proj.f_axis.shape
    assert proj.S("port1", "port2").shape == proj.f_axis.shape


# ═════════════════════════════════════════════════════════════════════
# Guards: a resume must advance, have a checkpoint, and have a recipe
# ═════════════════════════════════════════════════════════════════════


def test_resume_guards(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=150,
        checkpoint_interval=50,
    )

    # target not past the checkpoint (150) → clear error
    with pytest.raises(ValueError, match="not past the checkpoint"):
        resume(p, excited=("port1", 0), total_time_steps=120, verbose=False)

    # bare resume of a *done* bounded run has nothing to extend (inherited
    # total == n_completed) → the same guard fires
    with pytest.raises(ValueError, match="not past the checkpoint"):
        resume(p, excited=("port1", 0), verbose=False)


def test_resume_without_checkpoint_raises(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    p = tmp_path / "pp"
    # checkpoint_interval huge + short run ⇒ no periodic checkpoint, but a
    # completed run always writes a final one — so force no-checkpoint by
    # aborting via an exception before the first interval instead.
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=150,
        checkpoint_interval=50,
    )
    # A completed run *does* have a checkpoint; assert the happy path exists,
    # then delete it to exercise the missing-checkpoint guard.
    ckpt = p / "runs" / "port1_mode0" / "checkpoint.h5"
    assert ckpt.exists()
    ckpt.unlink()
    with pytest.raises(ValueError, match="no checkpoint"):
        resume(open_project(p), excited=("port1", 0), total_time_steps=300, verbose=False)


def test_resume_without_recipe_raises(tmp_path):
    """A project whose setup carries no recipe cannot be rebuilt."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    import json  # noqa: PLC0415

    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=150,
        checkpoint_interval=50,
    )
    meta_path = p / "project.json"
    meta = json.loads(meta_path.read_text())
    meta["setup"].pop("recipe")
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="no reconstruction recipe"):
        AnalysisScatteringTD.from_project(open_project(p))


def test_resume_is_exported_at_top_level():
    assert magnelio.resume is resume
    assert callable(magnelio.open_project)


# ═════════════════════════════════════════════════════════════════════
# SIGUSR1 = snapshot-and-continue (DD-070 follow-up)
# ═════════════════════════════════════════════════════════════════════


def test_sigusr1_checkpoints_without_stopping(tmp_path):
    """SIGUSR1 mid-run writes a checkpoint but does *not* stop the march.

    The run completes to its full length and closes ``done`` — the
    defining contrast to SIGINT, which breaks and closes ``aborted``
    (``test_resume_after_graceful_abort``).  ``checkpoint_interval`` is set
    huge so no periodic checkpoint would fire on its own; the signal is the
    only out-of-schedule trigger.
    """
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    if not hasattr(signal, "SIGUSR1"):
        pytest.skip("SIGUSR1 is POSIX-only")
    n_total = 200

    p = tmp_path / "pp"
    analysis = _tem_analysis(
        project=p,
        monitors=(_CallAtStep(80, lambda: os.kill(os.getpid(), signal.SIGUSR1)),),
    )
    # No exception, no abort — the run marches straight through the signal.
    analysis.run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n_total,
        checkpoint_interval=10_000,
    )

    proj = open_project(p)
    assert proj.runs["port1_mode0"]["state"] == "done"
    assert proj.runs["port1_mode0"]["n_steps"] == n_total
    # A done run always leaves a (final) checkpoint; the mid-run write is
    # proven separately below.
    assert (p / "runs" / "port1_mode0" / "checkpoint.h5").exists()


def test_sigusr1_writes_checkpoint_mid_run(tmp_path):
    """The on-demand checkpoint is really written *during* the march.

    SIGUSR1 fires at step 83; a hard crash (monitor raising) at step 130
    then tears the run down on the ``aborted`` path, which writes **no**
    final checkpoint — so the ``checkpoint.h5`` left on disk is exactly the
    one the signal forced at the next flush.  Its ``n_completed`` lies
    strictly between the signal and the crash, and with ``checkpoint_interval``
    set huge it could only have come from the signal (no periodic write is
    due that early).
    """
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    if not hasattr(signal, "SIGUSR1"):
        pytest.skip("SIGUSR1 is POSIX-only")
    from magnelio.io.project import _read_state_dict_h5  # noqa: PLC0415

    def _boom():
        raise RuntimeError("hard crash after the on-demand checkpoint")

    p = tmp_path / "pp"
    analysis = _tem_analysis(
        project=p,
        monitors=(
            _CallAtStep(83, lambda: os.kill(os.getpid(), signal.SIGUSR1)),
            _CallAtStep(130, _boom),
        ),
    )
    with pytest.raises(RuntimeError, match="hard crash"):
        analysis.run(
            excited=[("port1", 0)],
            energy_stop_db=None,
            total_time_steps=300,
            checkpoint_interval=10_000,
        )

    ckpt = p / "runs" / "port1_mode0" / "checkpoint.h5"
    assert ckpt.exists(), "SIGUSR1 should have forced a mid-run checkpoint"
    n_completed = int(_read_state_dict_h5(ckpt)["n_completed"])
    assert 83 < n_completed <= 130, (
        f"checkpoint n_completed={n_completed} is not between signal and crash"
    )
