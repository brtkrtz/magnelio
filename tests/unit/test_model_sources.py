"""Sources on the model and the mesh, and the store's waveform / monitor codecs (DD-224)."""

from __future__ import annotations

import json
import warnings

import h5py
import numpy as np
import pytest

import magnelio as mio
from magnelio import Excitation, GeometryModel, Mesh, monitors, signals, sources
from magnelio.analysis._recipe import (
    _monitor_from_dict,
    _monitor_to_dict,
    _waveform_from_dict,
    _waveform_to_dict,
)
from magnelio.circuit import LumpedElement, SeriesRLC
from magnelio.io.project import _load_mesh, _save_mesh
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortWaveguide


def _grid(n=4, L=4e-3):
    ax = np.linspace(0.0, L, n + 1)
    return GridLines(x=ax, y=ax, z=ax)


def _element(name):
    return LumpedElement(name=name, start=(0, 0, 0), end=(1e-3, 0, 0), element=SeriesRLC(R=50.0))


# ── GeometryModel.add_source ────────────────────────────────────────────────


class TestAddSource:
    def test_add_and_chain(self):
        model = GeometryModel()
        pw = sources.SourcePlaneWave(name="pw", direction=(0, 0, 1))
        assert model.add_source(pw) is model
        assert model.sources == [pw]

    def test_type_check(self):
        with pytest.raises(TypeError, match="add_source"):
            GeometryModel().add_source(PortWaveguide(name="p", plane="zmin"))

    def test_one_name_namespace(self):
        model = GeometryModel()
        model.add_port(PortWaveguide(name="a", plane="zmin"))
        model.add_element(_element("b"))
        model.add_source(sources.SourcePlaneWave(name="c"))
        with pytest.raises(ValueError, match="duplicate source name 'a'"):
            model.add_source(sources.SourcePlaneWave(name="a"))
        with pytest.raises(ValueError, match="duplicate source name 'b'"):
            model.add_source(sources.SourcePlaneWave(name="b"))
        with pytest.raises(ValueError, match="duplicate port name 'c'"):
            model.add_port(PortWaveguide(name="c", plane="zmax"))
        with pytest.raises(ValueError, match="duplicate element name 'c'"):
            model.add_element(_element("c"))


# ── Mesh.sources ────────────────────────────────────────────────────────────


class TestMeshSources:
    def test_from_grid_has_none(self):
        assert Mesh.from_grid(_grid()).sources == ()

    def test_with_sources_and_namespace(self):
        mesh = Mesh.from_grid(_grid()).with_ports([PortWaveguide(name="p", plane="zmin")])
        pw = sources.SourcePlaneWave(name="pw")
        m2 = mesh.with_sources([pw])
        assert m2.sources == (pw,)
        assert mesh.sources == ()  # the original is untouched
        with pytest.raises(ValueError, match="source names must be unique"):
            mesh.with_sources([sources.SourcePlaneWave(name="p")])
        with pytest.raises(ValueError, match="port names must be unique"):
            m2.with_ports([PortWaveguide(name="pw", plane="zmin")])

    def test_sources_survive_mesh_copies(self):
        pw = sources.SourcePlaneWave(name="pw")
        mesh = Mesh.from_grid(_grid()).with_sources([pw])
        assert mesh.with_boundary_conditions({"zmin": "PMC"}).sources == (pw,)
        assert mesh.with_pec_boundaries(faces=("xmin",)).sources == (pw,)

    def test_from_geometry_carries_sources(self):
        from magnelio.geo import Brick

        model = GeometryModel()
        model.add(Brick(origin=(0, 0, 0), size=(2e-3, 2e-3, 2e-3), material="pec"))
        model.add_source(sources.SourcePlaneWave(name="pw", direction=(0, 0, -1)))
        mesh = Mesh.from_geometry(model, mio.MeshControl(), f_max=10e9)
        assert len(mesh.sources) == 1
        assert mesh.sources[0].name == "pw"
        assert mesh.sources[0].direction == (0.0, 0.0, -1.0)

    def test_store_round_trip(self, tmp_path):
        pw = sources.SourcePlaneWave(
            name="pw",
            direction=(1, 0, 0),
            polarization=(0, 1, 1),
            corners=((None, 1e-3, 1e-3), (None, 3e-3, 3e-3)),
        )
        mesh = Mesh.from_grid(_grid()).with_sources([pw])
        path = tmp_path / "mesh.h5"
        with h5py.File(path, "w") as f:
            _save_mesh(f, mesh)
        with h5py.File(path, "r") as f:
            tags = [d["type"] for d in json.loads(f["mesh"].attrs["sources"])]
            back = _load_mesh(f)
        assert tags == ["SourcePlaneWave"]
        (src,) = back.sources
        assert isinstance(src, sources.SourcePlaneWave)
        assert src.name == "pw"
        assert src.direction == pw.direction
        assert src.polarization == pw.polarization
        assert src.corners == pw.corners
        assert src.waveform is None  # run state is not model data

    def test_store_without_sources_reads_empty(self, tmp_path):
        path = tmp_path / "mesh.h5"
        with h5py.File(path, "w") as f:
            _save_mesh(f, Mesh.from_grid(_grid()))
        with h5py.File(path, "r") as f:
            assert "sources" not in f["mesh"].attrs
            assert _load_mesh(f).sources == ()


# ── recipe codecs ───────────────────────────────────────────────────────────


class TestWaveformRecipe:
    @pytest.mark.parametrize(
        "wf",
        [
            signals.WaveformGaussian(f_max=10e9),
            signals.WaveformGaussianModulated(f_min=8.2e9, f_max=12.4e9),
            signals.WaveformSine(f=2e9, phase=30.0, rise_time=1e-9),
            signals.WaveformSine(f=2e9),
            signals.WaveformStep(rise_time=1e-10, hold=1e-9, fall_time=2e-10),
            signals.WaveformStep(rise_time=1e-10),
            signals.WaveformTable(
                t=[0.0, 1e-9, 2e-9], values=[0.0, 1.0, 0.0], f_max=3e9, f_center=1e9
            ),
        ],
    )
    def test_round_trip(self, wf):
        d = _waveform_to_dict(wf)
        assert d["type"] == type(wf).__name__
        json.dumps(d)  # JSON-able
        back = _waveform_from_dict(d)
        assert type(back) is type(wf)
        t = np.linspace(0.0, 3e-9, 31)
        np.testing.assert_array_equal(back(t), wf(t))
        assert (back.f_max, back.f_min, back.f_center, back.t_end) == (
            wf.f_max,
            wf.f_min,
            wf.f_center,
            wf.t_end,
        )

    def test_none(self):
        assert _waveform_to_dict(None) is None
        assert _waveform_from_dict(None) is None

    def test_legacy_excitation_dict_maps_onto_a_waveform(self):
        from magnelio.analysis._recipe import _waveform_from_recipe

        assert _waveform_from_recipe({"waveform": None}) is None
        assert _waveform_from_recipe({}) is None
        gauss = _waveform_from_recipe(
            {"excitation": {"f_min": 0.0, "f_max": 9e9, "mode_index": 0, "waveform": "gaussian"}}
        )
        assert gauss == signals.WaveformGaussian(f_max=9e9)
        mod = _waveform_from_recipe(
            {
                "excitation": {
                    "f_min": 8.2e9,
                    "f_max": 12.4e9,
                    "mode_index": 1,
                    "waveform": "modulated_gaussian",
                }
            }
        )
        assert mod == signals.WaveformGaussianModulated(f_min=8.2e9, f_max=12.4e9)

    def test_function_waveform_is_written_but_not_rebuilt(self):
        wf = signals.WaveformFunction(lambda t: 0.0, f_max=5e9, t_end=1e-9)
        d = _waveform_to_dict(wf)
        assert d == {
            "type": "WaveformFunction",
            "f_max": 5e9,
            "f_min": 0.0,
            "f_center": None,
            "t_end": 1e-9,
        }
        with pytest.raises(NotImplementedError, match="cannot be resumed"):
            _waveform_from_dict(d)


class TestMonitorRecipe:
    def test_flux_and_wall_loss_write_normal_position(self):
        flux = monitors.MonitorFluxTime(normal="y", position=2e-3, name="f")
        d = _monitor_to_dict(flux)
        assert (d["normal"], d["position"]) == ("y", 2e-3)
        back = _monitor_from_dict(d)
        assert (back.normal, back.position, back.name) == ("y", 2e-3, "f")
        wl = monitors.MonitorWallLoss(freqs=[1e9], normal="x", position=1e-3, sigma=5.8e7)
        d = _monitor_to_dict(wl)
        assert (d["normal"], d["position"]) == ("x", 1e-3)
        back = _monitor_from_dict(d)
        assert (back.normal, back.position) == ("x", 1e-3)

    def test_legacy_plane_pairs_still_read(self):
        flux = _monitor_from_dict({"type": "MonitorFluxTime", "plane": ["z", 5e-3], "name": "f"})
        assert (flux.normal, flux.position) == ("z", 5e-3)
        wl = _monitor_from_dict(
            {
                "type": "MonitorWallLoss",
                "freqs": [1e9],
                "reference_plane": ["z", 1e-3],
                "sigma": 5.8e7,
                "mu": 1.0,
                "roughness": None,
                "bc_faces": ["zmin"],
                "name": "wl",
            }
        )
        assert (wl.normal, wl.position) == ("z", 1e-3)

    def test_far_field_tag_and_legacy_alias(self):
        ff = monitors.MonitorFarFieldFrequency(freqs=[2e9], margin_cells=2, name="ff")
        d = _monitor_to_dict(ff)
        assert d["type"] == "MonitorFarFieldFrequency"
        assert isinstance(_monitor_from_dict(d), monitors.MonitorFarFieldFrequency)
        legacy = dict(d, type="MonitorFarField")
        back = _monitor_from_dict(legacy)
        assert isinstance(back, monitors.MonitorFarFieldFrequency)
        assert back.margin_cells == 2


# ── the scattering analysis takes waveform=, not excitations= ───────────────


class TestScatteringWaveformArgument:
    def _analysis(self, **kwargs):
        mesh = Mesh.from_grid(_grid()).with_ports([PortWaveguide(name="p", plane="zmin")])
        return mio.AnalysisScatteringTD(mesh=mesh, f_max=10e9, verbose=False, **kwargs)

    def test_waveform_type_checked(self):
        with pytest.raises(TypeError, match="Waveform"):
            self._analysis(waveform="gaussian")

    def test_waveform_above_band_warns(self):
        with pytest.warns(UserWarning, match="exceeds the analysis band"):
            self._analysis(waveform=signals.WaveformGaussian(f_max=20e9))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._analysis(waveform=signals.WaveformGaussian(f_max=10e9))

    def test_excitations_rejected(self):
        with pytest.raises(TypeError, match="excited="):
            self._analysis().run(excitations=[Excitation("p")])
