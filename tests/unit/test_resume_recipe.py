"""Reconstruction-recipe codec — the WP-S8 serialisation surface (DD-070).

``resume()`` rebuilds a streamed run's operators from a JSON recipe
stored in ``project.json`` (the store persists the model + results, but
the operators are re-derived — WP-S6).  These tests pin the codec: every
supported port spec and boundary form round-trips through JSON
unchanged, and an unserialisable configuration raises at write time
(never a silently-different resume).  No OCC / HDF5 needed.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from magnelio.analysis._recipe import (
    _bc_to_dict,
    _monitor_from_dict,
    _monitor_to_dict,
    _spec_from_dict,
    _spec_to_dict,
    _waveform_from_dict,
    _waveform_to_dict,
)
from magnelio.boundaries.boundary_conditions import BoundaryConditions
from magnelio.boundaries.pec import PECBoundary
from magnelio.monitors import MonitorFieldTime
from magnelio.ports._lumped import PortSpecLumped
from magnelio.ports._modal.factory import (
    BoxFace,
    ModeType,
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
)
from magnelio.signals import WaveformGaussian, WaveformGaussianModulated


@pytest.mark.parametrize(
    "w",
    [
        WaveformGaussian(1e9, edge_attenuation_db=17),
        WaveformGaussianModulated(1e6, 1e9, edge_attenuation_db=6),
    ],
)
def test_gaussian_resolved_parameters_round_trip(w):
    recipe = json.loads(json.dumps(_waveform_to_dict(w)))
    restored = _waveform_from_dict(recipe)
    assert restored == w
    assert restored.tau == w.tau
    assert restored.peak_time == w.peak_time
    np.testing.assert_array_equal(
        restored(np.linspace(0, w.t_end, 501)), w(np.linspace(0, w.t_end, 501))
    )


@pytest.mark.parametrize("modulated", [False, True])
def test_legacy_gaussian_recipe_preserves_original_samples(modulated):
    recipe = {
        "type": "WaveformGaussianModulated" if modulated else "WaveformGaussian",
        "f_max": 1e9,
    }
    if modulated:
        recipe["f_min"] = 1e6
    w = _waveform_from_dict(recipe)
    width = recipe["f_max"] - recipe.get("f_min", 0.0)
    tau, peak = 2 / (math.pi * width), 4 / width
    t = np.linspace(0, 8 / width, 501)
    original = np.exp(-(((t - peak) / tau) ** 2))
    if modulated:
        original *= np.cos(2 * math.pi * (recipe["f_max"] + recipe["f_min"]) / 2 * (t - peak))
    assert w.peak_time == peak and w.tau == tau
    np.testing.assert_array_equal(w(t), original)
    # Re-saving a legacy run must retain its legacy timing as explicit metadata.
    restored = _waveform_from_dict(json.loads(json.dumps(_waveform_to_dict(w))))
    np.testing.assert_array_equal(restored(t), original)
    assert restored == w


def test_gaussian_stored_resolved_values_not_recomputed():
    recipe = _waveform_to_dict(WaveformGaussianModulated(1e6, 1e9))
    recipe["tau"] = np.nextafter(recipe["tau"], math.inf)
    recipe["peak_time"] = np.nextafter(recipe["peak_time"], math.inf)
    restored = _waveform_from_dict(recipe)
    assert restored.tau == recipe["tau"]
    assert restored.peak_time == recipe["peak_time"]


@pytest.mark.parametrize(
    "bad", [{"tau": 1e-9}, {"tau": -1, "peak_time": 1e-9, "edge_attenuation_db": 25}]
)
def test_gaussian_invalid_stored_parameters_refused(bad):
    with pytest.raises(ValueError, match="tau"):
        _waveform_from_dict({"type": "WaveformGaussian", "f_max": 1e9, **bad})


@pytest.mark.parametrize(
    "spec",
    [
        PortSpecRectWG(
            name="p_rect",
            plane=BoxFace.X_MIN,
            width=22.86e-3,
            height=10.16e-3,
            n_modes=2,
        ),
        PortSpecCoax(
            name="p_coax",
            plane=BoxFace.Z_MAX,
            inner_radius=0.5e-3,
            outer_radius=1.5e-3,
            epsilon_r=2.1,
            center=(1e-3, -2e-3),
            n_modes=1,
        ),
        PortSpecNumerical(
            name="p_num",
            plane=BoxFace.Y_MIN,
            n_modes=3,
            epsilon_r=1.0,
            mode_type=ModeType.TE,
            window=((0.0, 0.0), (1e-3, 2e-3)),
        ),
        PortSpecMultiConductor(
            name="p_mc",
            plane=BoxFace.Z_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        ),
        PortSpecLumped(
            name="p_disc",
            start=(0.0, 0.0, 0.0),
            end=(0.0, 0.0, 1e-3),
            Z0=50.0,
        ),
    ],
)
def test_spec_round_trip_through_json(spec):
    """Every supported spec survives dict → JSON string → dict unchanged."""
    d = _spec_to_dict(spec)
    restored = _spec_from_dict(json.loads(json.dumps(d)))
    assert restored == spec


def test_numerical_mode_type_none_round_trips():
    spec = PortSpecNumerical(
        name="p",
        plane=BoxFace.X_MAX,
        n_modes=1,
        epsilon_r=1.0,
        mode_type=None,
        window=None,
    )
    assert _spec_from_dict(_spec_to_dict(spec)) == spec


def test_bc_dict_and_object_and_high_level_forms():
    """String entries, BC objects, and BoundaryConditions all reduce to
    the canonical ``{face: type_str}`` map."""
    mixed = {"xmin": "PMC", "ymin": PECBoundary("ymin"), "zmin": "PEC"}
    assert _bc_to_dict(mixed) == {"xmin": "PMC", "ymin": "PEC", "zmin": "PEC"}

    hi = BoundaryConditions(xmin="PEC", xmax="CPML", ymin="PMC")
    got = _bc_to_dict(hi)
    assert got["xmin"] == "PEC" and got["xmax"] == "CPML" and got["ymin"] == "PMC"


def test_field_time_monitor_round_trips_including_unbounded_corners():
    """A monitor spec survives the recipe codec through standard JSON.

    The unbounded corner components (written as ``None``) persist as
    the ``±inf`` sentinel strings — no bare ``Infinity`` token, which
    a stricter JSON reader than Python's would reject.
    """
    mon = MonitorFieldTime(
        corners=((None, None, 0.005), (None, None, 0.005)),
        times=[1e-9, 2e-9, 3e-9],
        fields=["E"],
        name="Eplane",
    )
    d = _monitor_to_dict(mon)
    assert d["corners"][0][0] == "-inf"
    assert d["corners"][1][0] == "inf"
    assert d["corners"][0][2] == 0.005
    js = json.dumps(d)
    assert "Infinity" not in js
    r = _monitor_from_dict(json.loads(js))
    assert r.name == "Eplane"
    assert r.corners == (
        (float("-inf"), float("-inf"), 0.005),
        (float("inf"), float("inf"), 0.005),
    )
    assert np.allclose(r.times, mon.times) and r.fields == ["E"]


def test_explicit_conductor_list_refused():
    """A hand-built ConductorSpec list has no lossless recipe form yet — the
    codec refuses it at write time rather than resuming a different problem."""
    spec = PortSpecMultiConductor(
        name="p",
        plane=BoxFace.Z_MIN,
        conductors=("ground", "signal"),  # non-None ⇒ unsupported for resume
        epsilon_r=1.0,
        n_modes=1,
    )
    with pytest.raises(NotImplementedError, match="ConductorSpec"):
        _spec_to_dict(spec)
