"""Physical and persistence boundaries of the public naming revision."""

import math

import numpy as np
import pytest

import magnelio as mio
from magnelio import analysis, fields, geo, ports, post
from magnelio.analysis._recipe import (
    _spec_from_dict,
    _spec_to_dict,
    _waveform_from_dict,
    _waveform_to_dict,
    excitation_from_dict,
    excitation_to_dict,
)
from magnelio.mesh import GridLines
from magnelio.ports.declarative import resolve_declarative_port
from magnelio.signals import Signal1D


@pytest.mark.parametrize("method", ["gamma", "z_wave", "z_modal"])
def test_mode_hz_boundary_and_old_unit_calls_fail(method):
    mode = ports.Mode(
        name="TEM",
        mode_type=ports.ModeType.TEM,
        omega_c=0.0,
        epsilon_r=1.0,
        field_evaluator=lambda u, v: (u, v, u, v),
    )
    evaluate = getattr(mode, method)
    value = evaluate(f=1e9)
    if method == "gamma":
        assert value.imag == pytest.approx(2 * math.pi * 1e9 / 299792458.0)
    with pytest.raises(TypeError):
        evaluate(2 * math.pi * 1e9)
    with pytest.raises(TypeError):
        evaluate(omega=2 * math.pi * 1e9)


@pytest.mark.parametrize("plane", ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"])
def test_rectangular_origin_keeps_global_corner_on_each_face(plane):
    port = ports.PortAnalytical(
        name="p",
        plane=plane,
        family="rect_wg",
        width=0.003,
        height=0.002,
        origin=(0.011, 0.023, 0.037),
    )
    spec = resolve_declarative_port(port, None)
    axis = "xyz".index(plane[0])
    expected = tuple(x for i, x in enumerate(port.origin) if i != axis)
    assert spec.origin == expected
    assert (spec.width, spec.height) == (0.003, 0.002)


@pytest.mark.parametrize("family,wrong", [("rect_wg", "center"), ("coax", "origin")])
def test_wrong_family_anchor_rejected_even_at_default_point(family, wrong):
    with pytest.raises(TypeError, match=wrong):
        ports.PortAnalytical(
            name="p",
            plane="zmin",
            family=family,
            width=1,
            height=1,
            inner_radius=0.1,
            outer_radius=0.2,
            **{wrong: (0, 0, 0)},
        )


def test_legacy_rectangular_recipe_retains_corner_and_disk_keys():
    legacy = {
        "type": "PortSpecRectWG",
        "name": "legacy",
        "plane": "x_max",
        "width_a": 0.003,
        "height_b": 0.002,
        "center": [0.011, 0.023],
        "epsilon_r": 2.0,
        "n_modes": 1,
    }
    spec = _spec_from_dict(legacy)
    assert spec.origin == (0.011, 0.023)
    assert (spec.width, spec.height) == (0.003, 0.002)
    assert _spec_to_dict(spec) == legacy
    with pytest.raises(TypeError):
        ports.PortSpecRectWG(name="p", plane=spec.plane, width=1, height=1, center=(0, 0))


def test_legacy_phase_recipes_keep_degree_values_and_delay():
    legacy_wave = {"type": "WaveformSine", "f": 1e9, "phase": 31.0}
    wave = _waveform_from_dict(legacy_wave)
    assert wave.phase_deg == 31.0
    assert _waveform_to_dict(wave)["phase"] == 31.0
    legacy = {
        "source": "p",
        "mode": 0,
        "waveform": legacy_wave,
        "amplitude": 2.0,
        "delay": 7e-9,
        "phase": 90.0,
    }
    exc = excitation_from_dict(legacy)
    assert exc.phase_deg == 90.0
    assert exc.effective_delay() == pytest.approx(7.25e-9)
    assert excitation_to_dict(exc)["phase"] == 90.0


def test_curated_homes_and_removed_calls():
    from magnelio.post._surface_current import SurfaceCurrent
    from magnelio.post.wall_loss import WallLossQ
    from magnelio.solver.eigenmode_result import EigenmodeResult

    assert fields.SurfaceCurrent is SurfaceCurrent
    assert post.WallLossQ is WallLossQ
    assert analysis.EigenmodeResult is EigenmodeResult
    assert {"Bend", "Wrap", "ImportedSheet"} <= set(geo.__all__)
    assert not hasattr(mio.GeometryModel, "plot")
    assert not hasattr(SurfaceCurrent, "current_through")
    assert not hasattr(geo.Solid, "tag_face")
    assert not hasattr(geo.Solid, "imprint")
    assert not hasattr(mio.AnalysisTD, "estimate")


def test_material_corners_validate_and_preserve_overwrite_order():
    grid = GridLines(x=[0, 1, 2], y=[0, 1, 2], z=[0, 1, 2])
    first = mio.Material.from_isotropic("first", epsilon=2)
    second = mio.Material.from_isotropic("second", epsilon=3)
    mesh = mio.Mesh.from_grid(
        grid,
        regions=[(first, ((2, 2, 2), (0, 0, 0))), (second, ((1, 1, 1), (2, 2, 2)))],
    )
    assert mesh.material_library[mesh.material_id[0, 0, 0]].name == "first"
    assert mesh.material_library[mesh.material_id[1, 1, 1]].name == "second"
    with pytest.raises(ValueError, match="two finite 3D corners"):
        mio.Mesh.from_grid(grid, regions=[(first, (0, 0, 0, 2, 2, 2))])


def test_spectral_axes_and_conflicting_snapshot_selector():
    grid = GridLines(x=[0, 1], y=[0, 1], z=[0, 1])
    zero = fields.FieldState.zeros(grid)
    spec = fields.FieldSpectrum(grid, f_axis=[1e9], Ex=zero.Ex[None].astype(complex))
    np.testing.assert_array_equal(spec.f_axis, [1e9])
    assert not hasattr(spec, "f") and not hasattr(spec, "frequencies")
    with pytest.raises(ValueError):
        spec.snapshot(f=1e9, frame=0)
    sig = Signal1D(t=np.arange(4), values=np.ones(4), dt=1)
    np.testing.assert_array_equal(sig.f_axis, np.fft.rfftfreq(4))
    assert not hasattr(sig, "f")


def test_loaded_geometry_show_uses_the_model_viewer(monkeypatch):
    from magnelio.io import LoadedGeometry
    from magnelio.post import plot_3d

    calls = []
    monkeypatch.setattr(plot_3d, "show_geometry", lambda model, **kw: calls.append((model, kw)))
    loaded = LoadedGeometry([], "air")
    loaded.show(render_mode="none", camera="xy")
    assert calls == [(loaded, {"render_mode": "none", "camera": "xy"})]
    assert not hasattr(loaded, "plot")


def test_export_audit_detects_an_imported_omission(monkeypatch):
    from validation.tools import check_api_surface

    monkeypatch.setattr(geo, "__all__", [name for name in geo.__all__ if name != "Bend"])
    assert check_api_surface.main() == 1
