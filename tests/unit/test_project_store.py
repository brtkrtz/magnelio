"""Round-trip tests for the on-disk project store (DD-070, WP-S1).

Validates that the write-once model layer — mesh (grid, material_id,
material library, pec mask, conformal sub-cell data) and geometry
(BREP compound + per-shape materials) — reconstructs faithfully through
``ProjectStore.create`` / ``open_project``.
"""

from __future__ import annotations

import os
import socket
from dataclasses import fields as dc_fields

import numpy as np
import pytest

from magnelio.io.project import ProjectStore, open_project
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh


def _assert_conformal_equal(a, b) -> None:
    """Every dataclass-of-arrays field survives the round trip verbatim."""
    if a is None:
        assert b is None
        return
    assert b is not None
    for f in dc_fields(a):
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        if va is None:
            assert vb is None, f"{f.name}: expected None"
        else:
            np.testing.assert_array_equal(np.asarray(va), np.asarray(vb), err_msg=f"field {f.name}")


class TestMeshOnlyRoundTrip:
    """Mesh round-trip without OCC (``Mesh.from_grid``)."""

    def _mesh(self) -> Mesh:
        grid = GridLines(
            x=np.linspace(0, 6e-3, 7),
            y=np.linspace(0, 4e-3, 5),
            z=np.linspace(0, 4e-3, 5),
        )
        fr4 = Material(name="FR4", epsilon=(4.4, 4.4, 4.4), sigma=(0.0, 0.0, 0.0))
        return Mesh.from_grid(
            grid,
            regions=[(fr4, (0, 0, 0, 6e-3, 2e-3, 4e-3))],
        )

    def test_grid_and_materials(self, tmp_path):
        mesh = self._mesh()
        ProjectStore.create(tmp_path / "proj", mesh, setup={"f_max": 1e10})
        p = open_project(tmp_path / "proj")

        np.testing.assert_array_equal(p.mesh.grid.x, mesh.grid.x)
        np.testing.assert_array_equal(p.mesh.grid.y, mesh.grid.y)
        np.testing.assert_array_equal(p.mesh.grid.z, mesh.grid.z)
        np.testing.assert_array_equal(p.mesh.material_id, mesh.material_id)
        np.testing.assert_array_equal(
            p.mesh.pec_mask_edges,
            mesh.pec_mask_edges,
        )
        assert p.mesh.material_library == mesh.material_library
        assert p.setup["f_max"] == 1e10
        assert p.status == "created"

    def test_no_geometry_returns_none(self, tmp_path):
        mesh = self._mesh()
        ProjectStore.create(tmp_path / "proj", mesh)
        p = open_project(tmp_path / "proj")
        assert p.geometry is None
        assert p.meta["has_geometry"] is False

    def test_not_a_project_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            open_project(tmp_path / "empty")


class TestFullModelRoundTrip:
    """Geometry + conformal mesh round-trip (requires OCC)."""

    def _model(self):
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        from magnelio.geo import GeometryModel  # noqa: PLC0415
        from magnelio.geo.primitives import Brick  # noqa: PLC0415

        air = Material.from_isotropic(name="air", epsilon=1.0)
        diel = Material.from_isotropic(name="diel", epsilon=4.0)
        model = GeometryModel(background=air)
        # two stacked bricks → an internal dielectric interface at y=2mm
        model.add(Brick(origin=(0, 0, 0), size=(6e-3, 2e-3, 4e-3), material=diel))
        model.add(Brick(origin=(0, 2e-3, 0), size=(6e-3, 2e-3, 4e-3), material=air))
        return model

    def test_roundtrip(self, tmp_path):
        from magnelio.mesh.mesher import MeshControl  # noqa: PLC0415

        model = self._model()
        mesh = Mesh.from_geometry(
            model,
            MeshControl(min_nodes_per_wavelength=8),
            f_max=10e9,
        )
        ProjectStore.create(
            tmp_path / "proj",
            mesh,
            geometry=model,
            setup={"f_max": 10e9, "dt": 1.2e-12},
        )

        # files present
        for name in ("project.json", "mesh.h5", "geometry.brep", "geometry.vtm", "geometry.json"):
            assert (tmp_path / "proj" / name).exists(), name

        p = open_project(tmp_path / "proj")

        # mesh core
        np.testing.assert_array_equal(p.mesh.material_id, mesh.material_id)
        np.testing.assert_array_equal(p.mesh.pec_mask_edges, mesh.pec_mask_edges)
        assert p.mesh.material_library == mesh.material_library

        # conformal sub-cell data (the interesting part)
        _assert_conformal_equal(mesh.edge_material, p.mesh.edge_material)
        _assert_conformal_equal(mesh.face_material, p.mesh.face_material)
        _assert_conformal_equal(mesh.pec_surface, p.mesh.pec_surface)

        # geometry: shape count, materials, and per-shape bbox
        g = p.geometry
        assert g is not None
        assert len(g) == len(model.shapes)
        assert g.background == model.background
        for loaded, orig in zip(g.shapes, model.shapes):
            assert loaded.material == orig.material
            lo_l, hi_l = loaded.bounding_box()
            lo_o, hi_o = orig.bounding_box()
            np.testing.assert_allclose(lo_l, lo_o, atol=1e-12)
            np.testing.assert_allclose(hi_l, hi_o, atol=1e-12)


class TestEigenModeRoundTrip:
    """AnalysisEigenmode persists a one-shot result into the shared container."""

    def _mesh(self):
        grid = GridLines(
            x=np.linspace(0, 20e-3, 11),
            y=np.linspace(0, 10e-3, 7),
            z=np.linspace(0, 10e-3, 7),
        )
        return Mesh.from_grid(grid)  # air cavity, all-PEC by default

    def test_store_roundtrip_exact(self, tmp_path):
        """Writing then reading the SAME result is bit-identical.

        (Eigenvectors are gauge-arbitrary — sign / degenerate subspace —
        so store fidelity must be checked against the written object,
        not a second independent solve.)
        """
        from magnelio.analysis.eigenmode import AnalysisEigenmode  # noqa: PLC0415

        mesh = self._mesh()
        ref = AnalysisEigenmode(mesh=mesh, n_modes=3, verbose=False).run()

        store = ProjectStore.create(tmp_path / "eig", mesh, setup={})
        store.write_eigenmodes(ref)
        em = open_project(tmp_path / "eig").eigenmodes

        assert em is not None
        np.testing.assert_array_equal(em.frequencies, ref.frequencies)
        for k in range(len(ref.modes)):
            for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                np.testing.assert_array_equal(
                    getattr(em.modes[k], comp),
                    getattr(ref.modes[k], comp),
                    err_msg=f"mode {k} {comp}",
                )

    def test_level_a_wiring(self, tmp_path):
        """AnalysisEigenmode(project=) returns a Project whose frequencies match."""
        from magnelio.analysis.eigenmode import AnalysisEigenmode  # noqa: PLC0415

        mesh = self._mesh()
        ref = AnalysisEigenmode(mesh=mesh, n_modes=3, verbose=False).run()
        proj = AnalysisEigenmode(
            mesh=mesh,
            n_modes=3,
            verbose=False,
            project=tmp_path / "eig2",
        ).run()

        assert proj.__class__.__name__ == "Project"
        assert (tmp_path / "eig2" / "eigenmodes.h5").exists()
        # frequencies are gauge-invariant; fields are not (see above)
        np.testing.assert_allclose(
            proj.eigenmodes.frequencies,
            ref.frequencies,
            rtol=1e-6,
        )
        assert proj.mesh.material_id.shape == mesh.material_id.shape


class TestPlannedRunProtocol:
    """Planned-run pre-registration closes the inter-run status gap.
    Before ``register_planned_runs``, finishing run *k* while run *k+1*
    was not yet in the index flipped the project status to ``done``
    mid-analysis — a live watcher polling ``status`` raced the writer.
    """

    def _store(self, tmp_path) -> ProjectStore:
        grid = GridLines(
            x=np.linspace(0, 6e-3, 7),
            y=np.linspace(0, 4e-3, 5),
            z=np.linspace(0, 4e-3, 5),
        )
        return ProjectStore.create(tmp_path / "proj", Mesh.from_grid(grid))

    def test_status_holds_running_between_runs(self, tmp_path):
        store = self._store(tmp_path)
        store.register_planned_runs(
            [
                ("port1_mode0", {"excited": ["port1", 0]}),
                ("port2_mode0", {"excited": ["port2", 0]}),
            ]
        )
        p = open_project(store.path)
        assert p.status == "running"
        assert p.runs["port1_mode0"].state == "pending"
        assert p.runs["port2_mode0"].excited == ("port2", 0)
        store._finalize_run("port1_mode0", 100, "done")
        assert open_project(store.path).status == "running"
        store._finalize_run("port2_mode0", 100, "done")
        assert open_project(store.path).status == "done"

    def test_register_keeps_existing_entries(self, tmp_path):
        store = self._store(tmp_path)
        store.register_planned_runs([("port1_mode0", {"excited": ["port1", 0]})])
        store._finalize_run("port1_mode0", 42, "done")
        store.register_planned_runs(
            [
                ("port1_mode0", {"excited": ["port1", 0]}),
                ("port2_mode0", {"excited": ["port2", 0]}),
            ]
        )
        p = open_project(store.path)
        assert p.runs["port1_mode0"].state == "done"
        assert p.runs["port1_mode0"].n_steps == 42
        assert p.runs["port2_mode0"].state == "pending"
        assert p.status == "running"

    def test_reader_guards_on_pending(self, tmp_path):
        store = self._store(tmp_path)
        store.register_planned_runs(
            [
                ("port1_mode0", {"excited": ["port1", 0]}),
                ("port2_mode0", {"excited": ["port2", 0]}),
            ]
        )
        p = open_project(store.path)
        with pytest.raises(ValueError, match="pending"):
            p.energy_trace(("port2", 0))
        with pytest.raises(ValueError, match="no started runs"):
            _ = p.dt
        with pytest.raises(ValueError, match="no started runs"):
            _ = p.s_params
        assert p.signals == {}


class TestRunTiming:
    """Every run and every analysis call carries its wall clock."""

    def _store(self, tmp_path) -> ProjectStore:
        grid = GridLines(
            x=np.linspace(0, 6e-3, 7),
            y=np.linspace(0, 4e-3, 5),
            z=np.linspace(0, 4e-3, 5),
        )
        return ProjectStore.create(tmp_path / "proj", Mesh.from_grid(grid))

    def test_finalize_stamps_the_end_and_accumulates_elapsed(self, tmp_path):
        store = self._store(tmp_path)
        store.register_planned_runs([("port1_mode0", {"excited": ["port1", 0]})])
        store._finalize_run("port1_mode0", 100, "aborted", elapsed=1.5)
        info = open_project(store.path).runs["port1_mode0"]
        assert info.elapsed == pytest.approx(1.5)
        assert info.finished is not None
        # A resumed march adds its own wall time to the run's total.
        store._finalize_run("port1_mode0", 200, "done", elapsed=2.0)
        assert open_project(store.path).runs["port1_mode0"].elapsed == pytest.approx(3.5)

    def test_planned_runs_name_their_writer(self, tmp_path):
        store = self._store(tmp_path)
        store.register_planned_runs([("port1_mode0", {"excited": ["port1", 0]})])
        meta = open_project(store.path).meta
        assert meta["writer"]["pid"] == os.getpid()
        assert meta["writer"]["host"] == socket.gethostname()

    def test_analysis_stamp_opens_and_closes(self, tmp_path):
        store = self._store(tmp_path)
        store.mark_analysis_started()
        stamp = open_project(store.path).meta["analysis"]
        assert stamp["finished"] is None
        assert stamp["elapsed"] is None
        store.mark_analysis_finished(12.5)
        stamp = open_project(store.path).meta["analysis"]
        assert stamp["elapsed"] == pytest.approx(12.5)
        assert stamp["finished"] >= stamp["started"]


class TestRunObjects:
    """``project.runs`` hands out live run objects, not index dicts."""

    def _store(self, tmp_path) -> ProjectStore:
        grid = GridLines(
            x=np.linspace(0, 6e-3, 7),
            y=np.linspace(0, 4e-3, 5),
            z=np.linspace(0, 4e-3, 5),
        )
        return ProjectStore.create(tmp_path / "proj", Mesh.from_grid(grid))

    def test_index_is_a_mapping_of_runs(self, tmp_path):
        from collections.abc import Mapping

        from magnelio.io import Run

        store = self._store(tmp_path)
        store.register_planned_runs(
            [
                ("port1_mode0", {"excited": ["port1", 0]}),
                ("port2_mode0", {"excited": ["port2", 0]}),
            ]
        )
        p = open_project(store.path)
        assert isinstance(p.runs, Mapping)
        assert not isinstance(p.runs, dict)
        assert list(p.runs) == ["port1_mode0", "port2_mode0"]
        run = p.runs["port1_mode0"]
        assert isinstance(run, Run)
        assert p.runs["port1_mode0"] is run
        assert run.excited == ("port1", 0)
        assert run.state == "pending"
        assert run.energy_trace.size == 0
        assert run.n_energy_samples == 0
        assert run.elapsed is None
        with pytest.raises(KeyError, match="port1_mode0"):
            p.runs["nope"]
        with pytest.raises(TypeError):
            run["n_steps"]

    def test_run_is_a_live_view(self, tmp_path):
        store = self._store(tmp_path)
        store.register_planned_runs([("port1_mode0", {"excited": ["port1", 0]})])
        run = open_project(store.path).runs["port1_mode0"]
        assert run.state == "pending"
        store._finalize_run("port1_mode0", 42, "done", stop_reason="energy", elapsed=0.5)
        assert run.state == "done"
        assert run.n_steps == 42
        assert run.stop_reason == "energy"
        assert run.elapsed == pytest.approx(0.5)
        assert "Run 'port1_mode0'" in repr(run)
        assert "<table" in run._repr_html_()

    def test_tables_print_every_run(self, tmp_path):
        store = self._store(tmp_path)
        store.register_planned_runs(
            [
                ("port1_mode0", {"excited": ["port1", 0]}),
                ("port2_mode0", {"excited": ["port2", 0]}),
            ]
        )
        store._finalize_run("port1_mode0", 7, "done", stop_reason="energy", elapsed=1.0)
        p = open_project(store.path)
        text = repr(p)
        assert text.startswith(f"Project {p.path}")
        assert "status" in text
        for name in ("port1_mode0", "port2_mode0", "pending", "done", "energy"):
            assert name in text
        assert "<table" in p._repr_html_()
        assert "port2_mode0" in repr(p.runs)


class TestProjectStatus:
    def _store(self, tmp_path) -> ProjectStore:
        grid = GridLines(
            x=np.linspace(0, 6e-3, 7),
            y=np.linspace(0, 4e-3, 5),
            z=np.linspace(0, 4e-3, 5),
        )
        return ProjectStore.create(tmp_path / "proj", Mesh.from_grid(grid))

    def test_an_aborted_run_aborts_the_project_even_with_pending_siblings(self, tmp_path):
        store = self._store(tmp_path)
        store.register_planned_runs(
            [
                ("port1_mode0", {"excited": ["port1", 0]}),
                ("port2_mode0", {"excited": ["port2", 0]}),
            ]
        )
        store._finalize_run("port1_mode0", 100, "aborted", stop_reason="aborted")
        assert open_project(store.path).status == "aborted"
        # Resumed to completion, the sibling still planned: on its way again.
        store._finalize_run("port1_mode0", 200, "done", stop_reason="energy")
        assert open_project(store.path).status == "running"
        store._finalize_run("port2_mode0", 200, "done", stop_reason="energy")
        assert open_project(store.path).status == "done"

    def test_a_dead_writer_reads_stale(self, tmp_path):
        import subprocess
        import sys

        from magnelio.io.project import _update_meta

        store = self._store(tmp_path)
        store.register_planned_runs([("port1_mode0", {"excited": ["port1", 0]})])
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        dead_pid = proc.pid

        def _running(meta, pid, host):
            meta["runs"]["port1_mode0"].update({"state": "running", "pid": pid, "host": host})
            meta["writer"] = {"pid": pid, "host": host}

        _update_meta(store.path, lambda m: _running(m, dead_pid, socket.gethostname()))
        p = open_project(store.path)
        assert p.runs["port1_mode0"].state == "stale"
        assert p.status == "stale"

        _update_meta(store.path, lambda m: _running(m, os.getpid(), socket.gethostname()))
        assert p.runs["port1_mode0"].state == "running"
        assert p.status == "running"

        _update_meta(store.path, lambda m: _running(m, dead_pid, "some-other-host"))
        assert p.runs["port1_mode0"].state == "running"
        assert p.status == "running"

    def test_metadata_follows_the_file_until_the_project_is_finished(self, tmp_path):
        from magnelio.io.project import _update_meta

        store = self._store(tmp_path)
        store.register_planned_runs([("port1_mode0", {"excited": ["port1", 0]})])
        p = open_project(store.path)
        assert p.status == "running"
        store._finalize_run("port1_mode0", 5, "done")
        # No refresh(): the reader noticed the writer's change on its own.
        assert p.status == "done"
        assert p.runs["port1_mode0"].n_steps == 5

        def _note(meta):
            meta["setup"]["note"] = "later"

        _update_meta(store.path, _note)
        # Finished projects keep their parsed copy until asked.
        assert "note" not in p.setup
        assert p.refresh().setup["note"] == "later"

    def test_repr_never_raises_on_a_foreign_schema(self, tmp_path):
        import json

        store = self._store(tmp_path)
        meta = json.loads((store.path / "project.json").read_text())
        meta["schema_version"] = "1.0"
        (store.path / "project.json").write_text(json.dumps(meta))
        p = open_project(store.path)
        text = repr(p)
        assert "cannot read project.json" in text
        assert "schema" in text.lower()
        assert "cannot read project.json" in p._repr_html_()


class TestCheckpointState:
    def test_mapping_access_and_summary(self, tmp_path):
        from magnelio.io import CheckpointState
        from magnelio.io.project import _write_state_dict_h5

        grid = GridLines(
            x=np.linspace(0, 6e-3, 7),
            y=np.linspace(0, 4e-3, 5),
            z=np.linspace(0, 4e-3, 5),
        )
        store = ProjectStore.create(tmp_path / "proj", Mesh.from_grid(grid))
        store.register_planned_runs([("port1_mode0", {"excited": ["port1", 0]})])
        store._finalize_run("port1_mode0", 400, "done")
        run_dir = store.path / "runs" / "port1_mode0"
        run_dir.mkdir(parents=True)
        state = {
            "n_completed": 400,
            "peak_energy": 1.5e-9,
            "peak_signal": 2.0,
            "e": np.zeros(12345, dtype=np.float32),
            "h": np.zeros(11111, dtype=np.float32),
            "boundaries": {"zmin": {"psi": np.zeros(3)}},
        }
        _write_state_dict_h5(run_dir / "checkpoint.h5", state)
        ckpt = open_project(store.path).checkpoint_state("port1_mode0")
        assert isinstance(ckpt, CheckpointState)
        assert int(ckpt["n_completed"]) == 400
        assert "boundaries" in ckpt
        assert ckpt.get("monitors", {}) == {}
        assert ckpt["e"].shape == (12345,)
        text = repr(ckpt)
        assert "n_completed" in text
        assert "float32[12345]" in text
        assert "boundaries" in text
        assert "0." not in text.split("file")[0].replace("0.0", "")  # no field content
        assert len(text) < 600
        assert "<table" in ckpt._repr_html_()
