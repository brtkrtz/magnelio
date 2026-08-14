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

import numpy as np
import pytest

from magnelio.analysis._recipe import (
    _bc_to_dict,
    _monitor_from_dict,
    _monitor_to_dict,
    _spec_from_dict,
    _spec_to_dict,
)
from magnelio.boundaries.boundary_conditions import BoundaryConditions
from magnelio.boundaries.pec import PECBoundary
from magnelio.monitors import MonitorFieldTime
from magnelio.ports._lumped import PortSpecLumped
from magnelio.ports._modal.factory import (
    BoxFace,
    ExcitationSpec,
    ModeType,
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
)


@pytest.mark.parametrize(
    "spec",
    [
        PortSpecRectWG(
            name="p_rect",
            plane=BoxFace.X_MIN,
            width_a=22.86e-3,
            height_b=10.16e-3,
            n_modes=2,
            excitation=ExcitationSpec(f_min=8.2e9, f_max=12.4e9, mode_index=1),
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
