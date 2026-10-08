"""Selective result evaluation, shared calibration and resume-safe caches."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from magnelio.io.project import ProjectStore, _RunResultWriter, _update_meta, open_project
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal.mode import Mode, ModeType
from magnelio.post.modal_sparameters import compute_s_parameters


def _field(u, v):
    zeros = np.zeros_like(u)
    return zeros, zeros, zeros, zeros


@pytest.fixture
def project(tmp_path):
    grid = GridLines(*(np.linspace(0.0, 1e-3, 3) for _ in range(3)))
    store = ProjectStore.create(tmp_path / "network", Mesh.from_grid(grid))
    channels = [(p, 0) for p in ("p1", "p2", "p3")]
    runs = [("drive1", {"excited": ["p1", 0]}), ("drive3", {"excited": ["p3", 0]})]
    store.register_planned_runs(runs)
    mode = Mode("TEM", ModeType.TEM, 0.0, 1.0, _field, z_line=50.0)
    dt = 1e-12
    values = np.exp(-(((np.arange(256) - 70.0) / 20.0) ** 2))
    for name, info in runs:
        writer = _RunResultWriter(
            store.path / "runs" / name,
            dt=dt,
            excitations=[],
            excited=tuple(info["excited"]),
            f_axis=np.linspace(1e9, 8e9, 17),
            channels=channels,
            port_modes={p: [mode] for p, _ in channels},
            port_normal_dx={},
            port_line_params={},
        )
        writer.append(
            {c: (i + 1) * values for i, c in enumerate(channels)},
            {c: 0.01 * values for c in channels},
            values,
        )
        writer.append_energy(256, 256 * dt, 1.0)
        writer.close()
        store._finalize_run(name, 256, "done")
    return open_project(store.path)


def _reads(monkeypatch):
    reads = []
    original = h5py.Dataset.__getitem__

    def read(ds, key):
        reads.append((ds.file.filename, ds.name))
        return original(ds, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    return reads


def _unexpected(*args, **kwargs):
    pytest.fail("unrequested S-matrix derivation")


def test_metadata_and_impedance_do_not_read_time_records(project, monkeypatch):
    reads = _reads(monkeypatch)
    monkeypatch.setattr(project, "_s_params", _unexpected)
    assert project.channels == (("p1", 0), ("p2", 0), ("p3", 0))
    assert project.excitations == (("p1", 0), ("p3", 0))
    assert project.dt == 1e-12
    assert project.f_axis.size == 17
    assert project.settings.dt == 1e-12
    np.testing.assert_array_equal(project.reference_impedance("p2"), 50.0)
    assert not any(path == "/reference" or path.startswith("/channels/") for _, path in reads)


def test_single_s_reads_only_involved_channels_and_run(project, monkeypatch):
    reads = _reads(monkeypatch)
    original_load = project._load_run

    def selected_load(name, **kwargs):
        assert name == "drive1", "selected S access read an unrelated run header"
        return original_load(name, **kwargs)

    monkeypatch.setattr(project, "_load_run", selected_load)
    s = project.S("p2", "p1")
    assert s.shape == (17,)
    records = [(file, path) for file, path in reads if path.startswith("/channels/")]
    assert len(records) == 4
    assert all("drive1" in file for file, _ in records)
    assert {path.split("/")[2] for _, path in records} == {"ch0", "ch1"}
    reads.clear()
    project.S("p3", "p1")
    records = [path for _, path in reads if path.startswith("/channels/")]
    assert records == ["/channels/ch2/V", "/channels/ch2/I"]


@pytest.mark.parametrize("taper", [False, True])
def test_selected_entries_equal_complete_matrix(project, taper):
    _update_meta(
        project.path, lambda meta: [r.update(taper_signals=taper) for r in meta["runs"].values()]
    )
    selected = project.S("p2", "p1")
    data = project._load_run("drive1")
    expected = compute_s_parameters(
        data["signals"],
        data["port_modes"],
        data["excited"],
        data["reference"],
        data["f_axis"],
        taper_signals=taper,
    )[("p2", 0)]
    np.testing.assert_allclose(selected, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(selected, project.s_params.S("p2", "p1"), atol=1e-12)
    assert set(project.signals) == {("p1", 0), ("p3", 0)}
    assert project.result("drive1").signals.keys() == data["signals"].keys()


def test_custom_axis_cache_reuses_incident_wave(project, monkeypatch):
    import magnelio.post.modal_sparameters as post

    original = post._modal_spectral_waves
    evaluated = []

    def evaluate(records, *args, **kwargs):
        evaluated.append(tuple(records))
        return original(records, *args, **kwargs)

    monkeypatch.setattr(post, "_modal_spectral_waves", evaluate)
    axis = np.linspace(1.5e9, 7.5e9, 23)
    expected = project.S("p2", "p1", f_axis=axis)
    project.phase("p2", "p1", f_axis=axis.copy())
    project.db("p2", "p1", f_axis=axis.copy())
    assert evaluated == [(("p1", 0), ("p2", 0))]
    project.S("p3", "p1", f_axis=axis)
    assert evaluated[-1] == (("p3", 0),)
    project.S("p2", "p1", f_axis=axis + 1e6)
    assert len(evaluated) == 3
    altered = project.S("p2", "p1", f_axis=axis)
    altered[:] = 0.0
    np.testing.assert_array_equal(project.S("p2", "p1", f_axis=axis), expected)


def test_time_plot_computes_wave_pair_once(project, monkeypatch):
    import matplotlib.pyplot as plt

    import magnelio.post.modal_sparameters as post

    original = post.destaggered_power_waves
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(post, "destaggered_power_waves", evaluate)
    monkeypatch.setattr(project, "_s_params", _unexpected)
    with pytest.raises(ValueError, match="excited"):
        project.plot_time_signals("p2")
    fig, ax = project.plot_time_signals("p2", excited="p1")
    assert len(ax.lines) == 2
    assert len(calls) == 1
    plt.close(fig)
    signal = project.a("p2", excited="p1")
    expected = signal.values.copy()
    signal.values[:] = 0.0
    np.testing.assert_array_equal(project.a("p2", excited="p1").values, expected)


def test_unselected_short_stream_still_limits_common_prefix(project):
    with h5py.File(project.path / "runs" / "drive1" / "results.h5", "r+") as file:
        file["channels/ch2/I"].resize((200,))
    project.S("p2", "p1")
    assert project._load_run("drive1", channels=(("p2", 0),))["n_steps"] == 200
    assert project.a("p2", excited="p1").values.size == 200


@pytest.mark.parametrize("state", ["done", "running"])
def test_changed_run_invalidates_derived_data(project, state):
    before = project.S("p2", "p1")
    wave = project.b("p2", excited="p1").values
    matrix = project.s_params.S("p2", "p1")
    with h5py.File(project.path / "runs" / "drive1" / "results.h5", "r+") as file:
        file["channels/ch1/V"][:] *= 2.0
        file["channels/ch1/I"][:] *= 2.0
    _update_meta(
        project.path, lambda meta: meta["runs"]["drive1"].update(state=state, finished="changed")
    )
    np.testing.assert_allclose(project.S("p2", "p1"), 2.0 * before, atol=1e-12)
    np.testing.assert_allclose(project.b("p2", excited="p1").values, 2.0 * wave, atol=1e-12)
    np.testing.assert_allclose(project.s_params.S("p2", "p1"), 2.0 * matrix, atol=1e-12)
    if state == "running":
        with h5py.File(project.path / "runs" / "drive1" / "results.h5", "r+") as file:
            file["channels/ch1/V"][:] *= 2.0
            file["channels/ch1/I"][:] *= 2.0
        np.testing.assert_allclose(project.S("p2", "p1"), 4.0 * before, atol=1e-12)


def test_reference_reads_only_longest_runs_reference(project, monkeypatch):
    reads = _reads(monkeypatch)
    assert project.reference_signal.values.size == 256
    assert sum(path == "/reference" for _, path in reads) == 1
    assert not any(path.startswith("/channels/") for _, path in reads)


def test_single_run_time_plot_resolves_metadata_without_s(project, monkeypatch):
    import matplotlib.pyplot as plt

    _update_meta(project.path, lambda meta: meta["runs"].pop("drive3"))
    reads = _reads(monkeypatch)
    monkeypatch.setattr(project, "_s_params", _unexpected)
    fig, ax = project.plot_time_signals("p2")
    assert len(ax.lines) == 2
    assert len([path for _, path in reads if path.startswith("/channels/")]) == 2
    plt.close(fig)


def test_band_selection_retains_coupled_projections(project, monkeypatch):
    import json
    from types import SimpleNamespace

    import magnelio.io.project as io
    import magnelio.ports._modal.dispersion as dispersion
    import magnelio.post.modal_sparameters as post

    with h5py.File(project.path / "runs" / "drive1" / "results.h5", "r+") as file:
        extra = file["channels"].create_group("ch3")
        extra.attrs["name"] = "p2"
        extra.attrs["mode"] = 1
        for label in ("V", "I"):
            extra.create_dataset(label, data=file[f"channels/ch1/{label}"][:])
        modes = json.loads(file["ports/p2"].attrs["modes"])
        file["ports/p2"].attrs["modes"] = json.dumps(modes + modes)
        for port in file["ports"]:
            file[f"ports/{port}"].create_group("band")
    _update_meta(project.path, lambda meta: meta["runs"]["drive1"].update(port_model="band"))

    def band_record(group, label):
        return SimpleNamespace(
            name=label,
            n_modes=2 if label == "p2" else 1,
            me_u=1,
            me_v=1,
            mh_u=1,
            mh_v=1,
            curl_slice=1,
        )

    evaluations = []

    def evaluate(records, bands, axis, **kwargs):
        evaluations.append(tuple(records))
        assert all((band.name, m) in records for band in bands for m in range(band.n_modes))
        return {
            key: (np.ones_like(axis), 2.0 * np.ones_like(axis), 50.0 * np.ones_like(axis))
            for key in records
        }

    monkeypatch.setattr(io, "_read_band_decomposition", band_record)
    monkeypatch.setattr(post, "_band_spectral_waves", evaluate)
    monkeypatch.setattr(
        dispersion,
        "solve_port_dispersion",
        lambda band, axis, **kwargs: SimpleNamespace(
            z_line=np.full((band.n_modes, len(axis)), 50.0)
        ),
    )
    monkeypatch.setattr(project, "_band_mesh_operators", _unexpected)
    reads = _reads(monkeypatch)
    np.testing.assert_array_equal(project.reference_impedance("p2", 1), 50.0)
    assert not any(path == "/reference" or path.startswith("/channels/") for _, path in reads)
    np.testing.assert_array_equal(project.S("p2", "p1", mode_out=1), 2.0)
    assert set(evaluations[0]) == {("p1", 0), ("p2", 0), ("p2", 1)}
    project.S("p2", "p1", mode_out=0)
    assert len(evaluations) == 1
    project.S("p3", "p1")
    assert evaluations[-1] == (("p3", 0),)


def test_incident_normalization_reads_only_excited_channel(project, monkeypatch):
    import json

    from magnelio.analysis.scattering_td import incident_amplitude_ratio

    path = project.path / "runs" / "drive1" / "results.h5"
    with h5py.File(path, "r+") as file:
        modes = json.loads(file["ports/p1"].attrs["modes"])
        modes[0].update(mode_type="TE", omega_c=2.0 * np.pi * 0.3e9, z_line=None)
        file["ports/p1"].attrs["modes"] = json.dumps(modes)
    reads = _reads(monkeypatch)
    monkeypatch.setattr(project, "_s_params", _unexpected)
    axis, ratio = project._incident_ratio("drive1")
    assert [p for _, p in reads if p.startswith("/channels/")] == [
        "/channels/ch0/V",
        "/channels/ch0/I",
    ]
    data = project._load_run("drive1")
    _, incident = compute_s_parameters(
        data["signals"],
        data["port_modes"],
        data["excited"],
        data["reference"],
        axis,
        return_incident=True,
    )
    np.testing.assert_allclose(
        ratio,
        incident_amplitude_ratio(
            data["reference"], axis, incident, data["port_modes"], data["excited"]
        ),
        atol=1e-12,
    )
    with h5py.File(path, "r+") as file:
        file["channels/ch0/V"][:] *= 2.0
        file["channels/ch0/I"][:] *= 2.0
    _update_meta(project.path, lambda meta: meta["runs"]["drive1"].update(finished="resumed"))
    np.testing.assert_allclose(project._incident_ratio("drive1")[1], 2.0 * ratio, atol=1e-12)
    project.refresh()
    assert not project._incident_cache


def test_tem_incident_ratio_needs_no_spectral_work(project, monkeypatch):
    reads = _reads(monkeypatch)
    monkeypatch.setattr(project, "_spectral_waves", _unexpected)
    assert project._incident_ratio("drive1") is None
    assert not any(path == "/reference" or path.startswith("/channels/") for _, path in reads)
