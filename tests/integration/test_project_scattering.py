"""End-to-end tests for the project store on a scattering run (DD-070, WP-S2).

Runs a small parallel-plate TEM two-port through the analysis both in-RAM
and with ``project=``, and asserts the read-back :class:`Project`
reproduces the in-RAM :class:`ScatteringTDResult` — S-parameters
(derived on read), power waves, raw signals, energy trace — plus the
multi-excitation fill-in flow.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

pytest.importorskip("OCC.Core.BRepPrimAPI")

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.geo import (
    Brick,
    GeometryModel,  # noqa: E402
)
from magnelio.io.project import open_project  # noqa: E402
from magnelio.ports import PortWaveguide


def _model_and_mesh():
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
    return model, mesh, f_max


def _analysis(mesh, model, f_max, project=None):
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


def test_project_roundtrip_matches_in_ram(tmp_path):
    model, mesh, f_max = _model_and_mesh()

    ref = _analysis(mesh, model, f_max).run(excited=[("port1", 0)])
    proj = _analysis(
        mesh,
        model,
        f_max,
        project=tmp_path / "pp",
    ).run(excited=[("port1", 0)])

    # returned object is a read-only Project reader
    assert proj.__class__.__name__ == "Project"
    assert proj.status == "done"
    for name in ("project.json", "mesh.h5", "geometry.brep"):
        assert (tmp_path / "pp" / name).exists(), name
    assert (tmp_path / "pp" / "runs" / "port1_mode0" / "results.h5").exists()

    # S-parameters derived on read match the in-RAM computation
    for out in ("port1", "port2"):
        np.testing.assert_allclose(
            proj.S(out, "port1"),
            ref.S(out, "port1"),
            rtol=1e-6,
            atol=1e-10,
            err_msg=f"S({out},port1)",
        )

    # power waves and raw signals match
    np.testing.assert_allclose(
        proj.a("port1").values,
        ref.a("port1").values,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        proj.b("port2").values,
        ref.b("port2").values,
        rtol=1e-6,
        atol=1e-12,
    )
    v_proj = proj.signals[("port1", 0)][("port1", 0)][0].values
    v_ref = ref.signals[("port1", 0)][("port1", 0)][0].values
    np.testing.assert_allclose(v_proj, v_ref, rtol=1e-9, atol=1e-14)

    # energy trace persisted and non-trivial
    et = proj.energy_trace(("port1", 0))
    assert et.size > 0
    assert np.all(et["energy"] >= 0.0)

    # geometry recovered
    assert proj.geometry is not None
    assert len(proj.geometry) == 1


def test_project_taper_signals_matches_in_ram(tmp_path):
    model, mesh, f_max = _model_and_mesh()

    ref = _analysis(mesh, model, f_max).run(excited=[("port1", 0)], taper_signals=True)
    proj = _analysis(
        mesh,
        model,
        f_max,
        project=tmp_path / "pp",
    ).run(excited=[("port1", 0)], taper_signals=True)

    # the flag is recorded per run and surfaces in the settings contract
    assert proj.runs["port1_mode0"]["taper_signals"] is True
    assert proj.settings.taper_signals is True

    # the reader applies the same Tukey window when deriving S
    for out in ("port1", "port2"):
        np.testing.assert_allclose(
            proj.S(out, "port1"),
            ref.S(out, "port1"),
            rtol=1e-6,
            atol=1e-10,
            err_msg=f"S({out},port1)",
        )


def test_custom_f_axis_on_read(tmp_path):
    model, mesh, f_max = _model_and_mesh()
    proj = _analysis(
        mesh,
        model,
        f_max,
        project=tmp_path / "pp",
    ).run(excited=[("port1", 0)])

    # the reader can derive S on a caller-chosen axis (the raw signals,
    # not a frozen S-matrix, are what is stored)
    custom = np.linspace(2e9, 10e9, 17)
    s = proj.S("port1", "port1", f_axis=custom)
    assert s.shape == custom.shape


def test_fill_in_second_excitation(tmp_path):
    model, mesh, f_max = _model_and_mesh()
    p = tmp_path / "pp"

    _analysis(mesh, model, f_max, project=p).run(excited=[("port1", 0)])
    # a second analysis pointing at the SAME project adds the column
    proj = _analysis(mesh, model, f_max, project=p).run(excited=[("port2", 0)])

    assert len(proj.runs) == 2
    # full 2x2 S-matrix now derivable
    for out in ("port1", "port2"):
        for inp in ("port1", "port2"):
            col = proj.S(out, inp)
            assert col.shape == proj.f_axis.shape


def test_live_streaming_concurrent_read(tmp_path):
    """A concurrent reader follows a running solve live (DD-070, WP-S5).

    The solver streams into the project from a writer thread while the
    test thread, holding an independent :class:`Project` reader, watches
    the energy trace grow and the status move ``running`` → ``done`` —
    then asserts the finished stream reproduces the in-RAM S-matrix.
    Deterministic inputs (fixed ``total_time_steps``, no energy stop) make
    the final parity exact.
    """
    model, mesh, f_max = _model_and_mesh()
    p = tmp_path / "live"
    n_steps = 3000

    # in-RAM reference with identical, deterministic inputs
    ref = _analysis(mesh, model, f_max).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n_steps,
    )

    errors: list = []

    def _writer():
        try:
            _analysis(mesh, model, f_max, project=p).run(
                excited=[("port1", 0)],
                energy_stop_db=None,
                total_time_steps=n_steps,
            )
        except BaseException as exc:  # surface into the test thread
            errors.append(exc)

    th = threading.Thread(target=_writer)
    th.start()
    seen_lengths: list = []
    seen_running = False
    partial_s_ok = False
    try:
        deadline = time.time() + 60.0
        while not (p / "project.json").exists() and time.time() < deadline:
            time.sleep(0.005)
        assert (p / "project.json").exists(), "writer never created project"

        proj = open_project(p)
        while time.time() < deadline:
            proj.refresh()  # pick up concurrently appended data
            status = proj.status
            if status == "running":
                seen_running = True
            # watcher idiom: the run index now lists planned runs as
            # ``pending`` before they start — skip those
            started = {name for name, info in proj.runs.items() if info.get("state") != "pending"}
            if started:
                n = int(proj.energy_trace(("port1", 0)).size)
                if not seen_lengths or n != seen_lengths[-1]:
                    seen_lengths.append(n)
                # a partial S is queryable live (converges as energy decays)
                if n >= 3 and not partial_s_ok:
                    assert proj.S("port2", "port1").shape == proj.f_axis.shape
                    partial_s_ok = True
            if status == "done":
                break
            time.sleep(0.005)
    finally:
        th.join(timeout=90)

    assert not errors, f"writer thread raised: {errors[0]!r}"
    assert not th.is_alive(), "writer thread did not finish"

    # liveness: observed the run in progress and the energy trace growing
    assert seen_running, "never observed status == 'running'"
    assert partial_s_ok, "never computed a live partial S-parameter"
    assert len(set(seen_lengths)) >= 2, (
        f"energy trace did not grow under concurrent read: {seen_lengths}"
    )
    assert all(b >= a for a, b in zip(seen_lengths, seen_lengths[1:])), (
        f"energy trace length not monotone: {seen_lengths}"
    )

    # completion + exact parity with the in-RAM reference
    final = open_project(p)
    assert final.status == "done"
    assert final.runs["port1_mode0"]["state"] == "done"
    for out in ("port1", "port2"):
        np.testing.assert_allclose(
            final.S(out, "port1"),
            ref.S(out, "port1"),
            rtol=1e-6,
            atol=1e-10,
            err_msg=f"final S({out},port1)",
        )


def test_multi_excitation_single_call_status(tmp_path):
    """One run() with two excitations ends done with no pending leftovers."""
    model, mesh, f_max = _model_and_mesh()
    proj = _analysis(mesh, model, f_max, project=tmp_path / "pp").run(
        excited=[("port1", 0), ("port2", 0)],
    )
    assert proj.status == "done"
    assert len(proj.runs) == 2
    assert all(i["state"] == "done" for i in proj.runs.values())


def test_pending_second_run_reader_state(tmp_path):
    """Mid-analysis reader view: run 1 done, run 2 registered but pending."""
    from magnelio.io.project import ProjectStore  # noqa: PLC0415

    model, mesh, f_max = _model_and_mesh()
    p = tmp_path / "pp"
    _analysis(mesh, model, f_max, project=p).run(excited=[("port1", 0)])
    ProjectStore(p).register_planned_runs([("port2_mode0", {"excited": ["port2", 0]})])
    proj = open_project(p)
    assert proj.status == "running"  # not "done": run 2 planned
    assert proj.runs["port2_mode0"]["state"] == "pending"
    assert proj.f_axis.size > 0
    assert proj.energy_trace(("port1", 0)).size > 0
    assert ("port1", 0) in proj.signals
    assert proj.S("port1", "port1").shape == proj.f_axis.shape
    with pytest.raises(ValueError, match="pending"):
        proj.energy_trace(("port2", 0))


def test_stop_reason_booked_in_index_and_settings(tmp_path):
    """A completed run books why it ended (and the |V| level) — DD-122."""
    model, mesh, f_max = _model_and_mesh()
    proj = _analysis(mesh, model, f_max, project=tmp_path / "pp").run(
        excited=[("port1", 0)],
    )
    info = proj.runs["port1_mode0"]
    # Well-absorbed TEM line: one of the two criteria fired normally.
    assert info["stop_reason"] in ("energy", "port_signal")
    assert proj.settings.stop_reason == info["stop_reason"]
    if info.get("final_port_signal_db") is not None:
        # 0.0 is legal: the energy criterion can fire before the port
        # envelope ever samples below its running peak.
        assert info["final_port_signal_db"] <= 0.0
        assert proj.settings.final_port_signal_db == info["final_port_signal_db"]
    # The run's wall clock, and the analysis call that contained it.
    assert info["elapsed"] > 0.0
    assert info["started"] <= info["finished"]
    assert info["pid"] == os.getpid()
    assert proj.meta["analysis"]["elapsed"] >= info["elapsed"]
    td = proj.result("port1_mode0")
    assert td.elapsed == pytest.approx(info["elapsed"])
    assert td.started <= td.finished


def test_runtime_cap_truncates_books_and_resumes(tmp_path):
    """A tiny cap truncates with a warning; a bare resume() finishes the
    run to its inherited criterion despite the 'done' state (DD-122)."""
    import magnelio as mio  # noqa: PLC0415

    model, mesh, f_max = _model_and_mesh()
    with pytest.warns(RuntimeWarning, match="runtime cap of 120 steps"):
        proj = _analysis(mesh, model, f_max, project=tmp_path / "pp").run(
            excited=[("port1", 0)],
            max_time_steps=120,
        )
    info = proj.runs["port1_mode0"]
    assert info["state"] == "done"
    assert info["stop_reason"] == "runtime_cap"
    assert info["n_steps"] == 120

    # The cap-truncated run resumes on its inherited launch criterion
    # (a fresh auto cap budget) and now ends on a real criterion.
    proj2 = mio.resume(tmp_path / "pp", excited="port1", verbose=False)
    info2 = proj2.runs["port1_mode0"]
    assert info2["stop_reason"] in ("energy", "port_signal")
    assert info2["n_steps"] > 120


def test_lumped_element_streams_and_resumes(tmp_path):
    """DD-123: a passive element survives the store round-trip.

    The element is declared on the model, travels with the mesh into
    the project, and must still act after a cap-truncated resume (the
    resume path rebuilds the analysis from the *reloaded* mesh, so a
    dropped element would silently change the physics)."""
    import magnelio as mio  # noqa: PLC0415
    from magnelio import circuit  # noqa: PLC0415

    model, mesh, f_max = _model_and_mesh()
    b = 5.0e-3
    model.add_element(
        circuit.LumpedElement(
            name="shunt",
            start=(0.0, -b / 2, 0.0),
            end=(0.0, b / 2, 0.0),
            element=circuit.SeriesRLC(R=100.0),
        )
    )
    mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=8), f_max=f_max)
    assert len(mesh.elements) == 1

    ram = _analysis(mesh, model, f_max).run(excited=[("port1", 0)])
    s11_ram = np.abs(ram.S("port1", "port1"))
    # The mid-line 100 ohm shunt reflects hard (z_line ~ 188 ohm).
    assert float(s11_ram.max()) > 0.3

    with pytest.warns(RuntimeWarning, match="runtime cap"):
        _analysis(mesh, model, f_max, project=tmp_path / "pp").run(
            excited=[("port1", 0)],
            max_time_steps=120,
        )
    proj = mio.resume(tmp_path / "pp", excited="port1", verbose=False)
    info = proj.runs["port1_mode0"]
    assert info["stop_reason"] in ("energy", "port_signal")
    s11_proj = np.abs(proj.S("port1", "port1"))
    assert np.allclose(s11_proj, s11_ram, atol=5e-3)
    assert all("shunt" not in str(c) for c in proj.channels)
