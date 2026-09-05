"""``Project.watch()`` follows a project another thread is writing.

The writer is a thread rather than a process so the test stays in one
interpreter; the reader holds its own :class:`Project` and sees only
the files, exactly as a second notebook would.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

pytest.importorskip("OCC.Core.BRepPrimAPI")

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl  # noqa: E402
from magnelio.geo import Brick, GeometryModel  # noqa: E402
from magnelio.io.project import ProjectStore, _update_meta, open_project  # noqa: E402
from magnelio.mesh.grid import GridLines  # noqa: E402
from magnelio.ports import PortWaveguide  # noqa: E402


def _analysis(project):
    a, b, L = 10.0e-3, 5.0e-3, 20.0e-3
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
    mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=8), f_max=12e9)
    return AnalysisScatteringTD(
        mesh=mesh,
        ports=[
            PortWaveguide(name="port1", plane="zmin", n_modes=1),
            PortWaveguide(name="port2", plane="zmax", n_modes=1),
        ],
        f_max=12e9,
        verbose=False,
        project=project,
        geometry=model,
    )


def _run_in_thread(project, n_steps):
    errors: list = []

    def _writer():
        try:
            _analysis(project).run(
                excited=[("port1", 0)], energy_stop_db=None, total_time_steps=n_steps
            )
        except BaseException as exc:  # surface into the test thread
            errors.append(exc)

    th = threading.Thread(target=_writer)
    th.start()
    deadline = time.time() + 60.0
    while not (project / "project.json").exists() and time.time() < deadline:
        time.sleep(0.005)
    assert (project / "project.json").exists(), "writer never created the project"
    return th, errors


def test_generator_reports_changes_until_done(tmp_path):
    p = tmp_path / "live"
    th, errors = _run_in_thread(p, 3000)
    proj = open_project(p)
    snapshots = []
    for seen in proj.watch(interval=0.02, timeout=120.0):
        assert seen is proj
        snapshots.append((seen.status, sum(r.n_energy_samples for r in seen.runs.values())))
    th.join(60.0)
    assert not errors, errors
    assert len(snapshots) >= 2, snapshots
    assert snapshots[-1][0] == "done"
    assert any(status == "running" for status, _ in snapshots), snapshots
    samples = [n for _, n in snapshots]
    assert samples == sorted(samples)
    assert samples[-1] > 0


def test_callback_form_returns_the_project(tmp_path):
    p = tmp_path / "live"
    th, errors = _run_in_thread(p, 2000)
    calls = []
    proj = open_project(p)
    out = proj.watch(interval=0.02, on_change=lambda pr: calls.append(pr.status), timeout=120.0)
    th.join(60.0)
    assert not errors, errors
    assert out is proj
    assert calls[-1] == "done"
    assert len(calls) >= 2


def _store_with_running_run(tmp_path, pid, host):
    grid = GridLines(
        x=np.linspace(0, 6e-3, 7),
        y=np.linspace(0, 4e-3, 5),
        z=np.linspace(0, 4e-3, 5),
    )
    store = ProjectStore.create(tmp_path / "proj", Mesh.from_grid(grid))
    store.register_planned_runs([("port1_mode0", {"excited": ["port1", 0]})])

    def _running(meta):
        meta["runs"]["port1_mode0"].update({"state": "running", "pid": pid, "host": host})
        meta["writer"] = {"pid": pid, "host": host}

    _update_meta(store.path, _running)
    return store


def test_finished_and_stale_projects_report_once(tmp_path):
    grid = GridLines(
        x=np.linspace(0, 6e-3, 7),
        y=np.linspace(0, 4e-3, 5),
        z=np.linspace(0, 4e-3, 5),
    )
    store = ProjectStore.create(tmp_path / "done", Mesh.from_grid(grid))
    store.register_planned_runs([("port1_mode0", {"excited": ["port1", 0]})])
    store._finalize_run("port1_mode0", 10, "done")
    assert len(list(open_project(store.path).watch(interval=0.01))) == 1

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    stale = _store_with_running_run(tmp_path / "s", proc.pid, socket.gethostname())
    t0 = time.monotonic()
    seen = list(open_project(stale.path).watch(interval=0.5))
    assert len(seen) == 1
    assert seen[0].status == "stale"
    assert time.monotonic() - t0 < 0.4  # returned without a single sleep


def test_timeout_ends_a_watch_on_a_live_run(tmp_path):
    live = _store_with_running_run(tmp_path / "l", os.getpid(), socket.gethostname())
    proj = open_project(live.path)
    t0 = time.monotonic()
    seen = list(proj.watch(interval=0.05, timeout=0.3))
    assert 0.25 < time.monotonic() - t0 < 2.0
    assert len(seen) == 1
    assert proj.status == "running"
    with pytest.raises(ValueError, match="interval"):
        proj.watch(interval=0)
