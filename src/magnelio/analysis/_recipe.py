"""Reconstruction recipe for a project-backed time-domain analysis (DD-070, WP-S8).

A streamed run persists its *model* (``mesh.h5``) and *results*
(``results.h5`` + ``checkpoint.h5``), but the operators that terminate
and drive the run are re-built on resume — WP-S6 established that
constant operators (material matrices, DTBC kernels) are re-derived on
the resuming solver, never stored.  To rebuild them, :func:`resume`
needs the *run recipe*: the resolved port specs, the monitors, the
waveform and band settings — everything the constructor of
:class:`~magnelio.AnalysisTD` or
:class:`~magnelio.AnalysisScatteringTD` consumed; the excitations of
a run travel with the run itself (``results.h5``).

This module is the pure (JSON-serialisable) codec for that recipe.  It
knows only the declarative spec / boundary vocabulary
(``magnelio.ports`` + ``magnelio.boundaries``), not the analysis class, so
it carries no import cycle.  The recipe is stored under
``project.json`` → ``setup`` → ``recipe`` at store-creation time and read
back by :meth:`AnalysisScatteringTD.from_project`.

Scope.  The recipe covers the modal pipeline (the only project-backed
one today) with the five concrete spec types and string-typed
boundary conditions (PEC / PMC / CPML / Periodic).  A configuration it
cannot represent losslessly — an explicit ``ConductorSpec`` list, a
custom-graded ``CPMLBoundary`` object — raises at *write* time with an
actionable message, so a resume never silently rebuilds a different
problem.
"""

from __future__ import annotations

import math
from typing import Any

from magnelio.boundaries.boundary_conditions import BoundaryConditions
from magnelio.boundaries.cpml import CPMLBoundary
from magnelio.boundaries.pec import PECBoundary
from magnelio.boundaries.periodic import PeriodicBoundary
from magnelio.boundaries.pmc import PMCBoundary

# 2.0: the boundary closure left the recipe for the mesh (DD-103).
from magnelio.io._schema import SCHEMA_VERSION as RECIPE_SCHEMA_VERSION
from magnelio.ports._lumped import PortSpecLumped
from magnelio.ports._modal.factory import (
    BoxFace,
    ModeType,
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
)

# ═════════════════════════════════════════════════════════════════════
# JSON helpers (tuples <-> nested lists; enums via .value)
# ═════════════════════════════════════════════════════════════════════


def _to_json(v: Any) -> Any:
    """Tuples/lists → nested lists; numpy scalars → Python scalars."""
    if isinstance(v, (tuple, list)):
        return [_to_json(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        return v.item()
    return v


def _to_tuple(v: Any) -> Any:
    """Inverse of :func:`_to_json` for the fields stored as tuples."""
    if isinstance(v, list):
        return tuple(_to_tuple(x) for x in v)
    return v


def _num_to_json(x) -> Any:
    """Encode a float, keeping the JSON standard-clean for ``±inf``.

    Field-monitor ``size`` uses ``inf`` for a full-domain extent; encoding
    it as a sentinel string avoids the non-standard bare ``Infinity`` token
    (Python's ``json`` reads it, but a stricter reader would not)."""
    x = float(x)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _num_from_json(v) -> float:
    return float(v)  # float("inf")/float("-inf") parse the sentinels


def _corners_to_json(corners) -> list:
    """Monitor corners as ``[[x0, y0, z0], [x1, y1, z1]]``.

    Normalised through :func:`normalize_corners`, so an omitted box and
    a ``None`` component both come out as the ``±inf`` sentinels — one
    stored shape for every way of writing the same region.
    """
    from magnelio.monitors.base import normalize_corners  # noqa: PLC0415

    lo, hi = normalize_corners(corners)
    return [[_num_to_json(x) for x in lo], [_num_to_json(x) for x in hi]]


def _corners_from_json(v) -> tuple:
    return tuple(tuple(_num_from_json(x) for x in point) for point in v)


# ═════════════════════════════════════════════════════════════════════
# Waveforms (DD-224: the class name is the type tag)
# ═════════════════════════════════════════════════════════════════════


def _waveform_to_dict(wf) -> dict | None:
    """Serialise a waveform by class-name tag and init fields.

    A ``WaveformFunction`` carries a Python callable the store cannot
    hold; its bandwidth is written so the recipe stays complete, and
    :func:`_waveform_from_dict` refuses to rebuild it.
    """
    from magnelio.signals.waveforms import (  # noqa: PLC0415
        WaveformFunction,
        WaveformGaussian,
        WaveformGaussianModulated,
        WaveformSine,
        WaveformStep,
        WaveformTable,
    )

    if wf is None:
        return None
    if isinstance(wf, WaveformGaussian):
        return {"type": "WaveformGaussian", "f_max": float(wf.f_max)}
    if isinstance(wf, WaveformGaussianModulated):
        return {
            "type": "WaveformGaussianModulated",
            "f_min": float(wf.f_min),
            "f_max": float(wf.f_max),
        }
    if isinstance(wf, WaveformSine):
        return {
            "type": "WaveformSine",
            "f": float(wf.f),
            "phase": float(wf.phase),
            "rise_time": None if wf.rise_time is None else float(wf.rise_time),
        }
    if isinstance(wf, WaveformStep):
        return {
            "type": "WaveformStep",
            "rise_time": float(wf.rise_time),
            "hold": None if wf.hold is None else float(wf.hold),
            "fall_time": None if wf.fall_time is None else float(wf.fall_time),
        }
    if isinstance(wf, WaveformTable):
        return {
            "type": "WaveformTable",
            "t": [float(v) for v in wf.t],
            "values": [float(v) for v in wf.values],
            "f_max": float(wf.f_max),
            "f_min": float(wf.f_min),
            "f_center": None if wf.f_center is None else float(wf.f_center),
        }
    if isinstance(wf, WaveformFunction):
        return {
            "type": "WaveformFunction",
            "f_max": float(wf.f_max),
            "f_min": float(wf.f_min),
            "f_center": None if wf.f_center is None else float(wf.f_center),
            "t_end": None if math.isinf(wf.t_end) else float(wf.t_end),
        }
    raise NotImplementedError(
        f"resume cannot serialise waveform type {type(wf).__name__}",
    )


def _waveform_from_dict(d: dict | None):
    """Inverse of :func:`_waveform_to_dict`."""
    from magnelio.signals.waveforms import (  # noqa: PLC0415
        WaveformGaussian,
        WaveformGaussianModulated,
        WaveformSine,
        WaveformStep,
        WaveformTable,
    )

    if d is None:
        return None
    t = d["type"]
    if t == "WaveformGaussian":
        return WaveformGaussian(f_max=float(d["f_max"]))
    if t == "WaveformGaussianModulated":
        return WaveformGaussianModulated(f_min=float(d["f_min"]), f_max=float(d["f_max"]))
    if t == "WaveformSine":
        return WaveformSine(
            f=float(d["f"]),
            phase=float(d.get("phase", 0.0)),
            rise_time=None if d.get("rise_time") is None else float(d["rise_time"]),
        )
    if t == "WaveformStep":
        return WaveformStep(
            rise_time=float(d["rise_time"]),
            hold=None if d.get("hold") is None else float(d["hold"]),
            fall_time=None if d.get("fall_time") is None else float(d["fall_time"]),
        )
    if t == "WaveformTable":
        return WaveformTable(
            t=[float(v) for v in d["t"]],
            values=[float(v) for v in d["values"]],
            f_max=float(d["f_max"]),
            f_min=float(d.get("f_min", 0.0)),
            f_center=None if d.get("f_center") is None else float(d["f_center"]),
        )
    if t == "WaveformFunction":
        raise NotImplementedError(
            "this run was driven by a WaveformFunction, whose Python callable the "
            "project store cannot hold — it cannot be resumed.  Re-run with a "
            "storable waveform (WaveformGaussian, WaveformGaussianModulated, "
            "WaveformSine, WaveformStep or WaveformTable) to make it resumable.",
        )
    raise TypeError(f"unknown waveform type {t!r} in recipe")


# ═════════════════════════════════════════════════════════════════════
# Excitations (DD-224) — stored per run, next to its signals
# ═════════════════════════════════════════════════════════════════════


def excitation_to_dict(exc) -> dict:
    """Serialise an :class:`~magnelio.Excitation` (waveform by class-name tag)."""
    return {
        "source": exc.source,
        "mode": int(exc.mode),
        "waveform": _waveform_to_dict(exc.waveform),
        "amplitude": float(exc.amplitude),
        "delay": float(exc.delay),
        "phase": float(exc.phase),
    }


def excitation_from_dict(d: dict):
    """Inverse of :func:`excitation_to_dict`."""
    from magnelio.analysis.excitation import Excitation  # noqa: PLC0415

    return Excitation(
        d["source"],
        mode=int(d.get("mode", 0)),
        waveform=_waveform_from_dict(d.get("waveform")),
        amplitude=float(d.get("amplitude", 1.0)),
        delay=float(d.get("delay", 0.0)),
        phase=float(d.get("phase", 0.0)),
    )


# ═════════════════════════════════════════════════════════════════════
# Port specs
# ═════════════════════════════════════════════════════════════════════


def _spec_to_dict(spec) -> dict:
    """Serialise one resolved port spec to a JSON-able dict."""
    if isinstance(spec, PortSpecLumped):
        d = {
            "type": "PortSpecLumped",
            "name": spec.name,
            "start": _to_json(spec.start),
            "end": _to_json(spec.end),
            "Z0": float(spec.Z0),
        }
        if spec.element is not None:
            d["element"] = {
                "kind": type(spec.element).__name__,
                "R": spec.element.R,
                "L": spec.element.L,
                "C": spec.element.C,
            }
        return d
    d = {
        "type": type(spec).__name__,
        "name": spec.name,
        "plane": spec.plane.value,
        "n_modes": int(spec.n_modes),
    }
    if isinstance(spec, PortSpecCoax):
        d.update(
            inner_radius=float(spec.inner_radius),
            outer_radius=float(spec.outer_radius),
            epsilon_r=float(spec.epsilon_r),
            center=_to_json(spec.center),
        )
    elif isinstance(spec, PortSpecRectWG):
        d.update(
            width_a=float(spec.width_a),
            height_b=float(spec.height_b),
            epsilon_r=float(spec.epsilon_r),
            center=_to_json(spec.center),
        )
    elif isinstance(spec, PortSpecNumerical):
        d.update(
            epsilon_r=float(spec.epsilon_r),
            mode_type=None if spec.mode_type is None else spec.mode_type.value,
            window=_to_json(spec.window),
        )
    elif isinstance(spec, PortSpecMultiConductor):
        if spec.conductors is not None:
            raise NotImplementedError(
                f"resume cannot yet serialise the explicit ConductorSpec "
                f"list on port {spec.name!r}; it is representable only "
                f"through the auto-detected (conductors=None) form.  Run "
                f"this structure without project= (in-RAM), or drive the "
                f"resume by hand from components.",
            )
        d.update(
            conductors=None,
            epsilon_r=None if spec.epsilon_r is None else float(spec.epsilon_r),
            window=_to_json(spec.window),
        )
    else:
        raise TypeError(
            f"cannot serialise port spec of type {type(spec).__name__} "
            f"for resume; supported: PortSpecCoax / PortSpecRectWG / "
            f"PortSpecNumerical / PortSpecMultiConductor / PortSpecLumped",
        )
    return d


def _spec_from_dict(d: dict):
    """Inverse of :func:`_spec_to_dict`."""
    t = d["type"]
    if t == "PortSpecLumped":
        element = None
        el = d.get("element")
        if el is not None:
            from magnelio.circuit.companion import (  # noqa: PLC0415
                ParallelRLC,
                SeriesRLC,
            )

            cls = {"SeriesRLC": SeriesRLC, "ParallelRLC": ParallelRLC}[el["kind"]]
            element = cls(R=el["R"], L=el["L"], C=el["C"])
        return PortSpecLumped(
            name=d["name"],
            start=_to_tuple(d["start"]),
            end=_to_tuple(d["end"]),
            Z0=float(d["Z0"]),
            element=element,
        )
    plane = BoxFace(d["plane"])
    if t == "PortSpecCoax":
        return PortSpecCoax(
            name=d["name"],
            plane=plane,
            inner_radius=float(d["inner_radius"]),
            outer_radius=float(d["outer_radius"]),
            epsilon_r=float(d["epsilon_r"]),
            center=_to_tuple(d["center"]),
            n_modes=int(d["n_modes"]),
        )
    if t == "PortSpecRectWG":
        return PortSpecRectWG(
            name=d["name"],
            plane=plane,
            width_a=float(d["width_a"]),
            height_b=float(d["height_b"]),
            epsilon_r=float(d["epsilon_r"]),
            center=_to_tuple(d["center"]),
            n_modes=int(d["n_modes"]),
        )
    if t == "PortSpecNumerical":
        return PortSpecNumerical(
            name=d["name"],
            plane=plane,
            n_modes=int(d["n_modes"]),
            epsilon_r=float(d["epsilon_r"]),
            mode_type=None if d["mode_type"] is None else ModeType(d["mode_type"]),
            window=_to_tuple(d["window"]),
        )
    if t == "PortSpecMultiConductor":
        return PortSpecMultiConductor(
            name=d["name"],
            plane=plane,
            conductors=None,
            epsilon_r=None if d["epsilon_r"] is None else float(d["epsilon_r"]),
            n_modes=int(d["n_modes"]),
            window=_to_tuple(d["window"]),
        )
    raise TypeError(f"unknown port spec type {t!r} in recipe")


# ═════════════════════════════════════════════════════════════════════
# Boundary conditions  (canonical string form)
# ═════════════════════════════════════════════════════════════════════

_BC_OBJECT_TYPE = {
    PECBoundary: "PEC",
    PMCBoundary: "PMC",
    CPMLBoundary: "CPML",
    PeriodicBoundary: "Periodic",
}


def _bc_value_to_str(face: str, value) -> str:
    """Map one boundary entry (string or BC object) to its type string."""
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        _parse_symmetry_value,
    )

    parsed = _parse_symmetry_value(value)
    if parsed is not None:
        # The recipe rebuilds runtime walls, and a symmetry face is
        # physically its wall type; the symmetry semantics round-trip
        # with the mesh (DD-154), not with the recipe.
        return parsed[0]
    if isinstance(value, str):
        return value
    for cls, name in _BC_OBJECT_TYPE.items():
        if isinstance(value, cls):
            if cls is CPMLBoundary and int(getattr(value, "thickness_cells", 8)) != 8:
                # The string form re-materialises CPML with the analysis-level
                # cpml_thickness_cells + default grading; a hand-built layer
                # with a non-default profile cannot round-trip through it.
                raise NotImplementedError(
                    f"resume cannot serialise a custom CPMLBoundary on face "
                    f"{face!r} (thickness_cells="
                    f"{getattr(value, 'thickness_cells', '?')}); pass CPML as "
                    f"a string in a BoundaryConditions / dict so the layer is "
                    f"rebuilt from cpml_thickness_cells, or resume from components.",
                )
            return name
    raise TypeError(
        f"cannot serialise boundary condition {value!r} on face {face!r} "
        f"for resume; use a string type or a PEC/PMC/CPML/Periodic object",
    )


def _bc_to_dict(boundary_conditions) -> dict:
    """Serialise the boundary closure to ``{face: type_str}``."""
    if isinstance(boundary_conditions, BoundaryConditions):
        return dict(boundary_conditions.to_dict())
    if isinstance(boundary_conditions, dict):
        return {face: _bc_value_to_str(face, value) for face, value in boundary_conditions.items()}
    raise TypeError(
        f"boundary_conditions must be BoundaryConditions or dict; "
        f"got {type(boundary_conditions).__name__}",
    )


# ═════════════════════════════════════════════════════════════════════
# Field monitors  (DD-070, WP-S9 — persisted so resume rebuilds them)
# ═════════════════════════════════════════════════════════════════════


def _monitor_to_dict(mon) -> dict:
    """Serialise a supported monitor spec, or raise (never a silent drop)."""
    from magnelio.monitors.field_frequency import (  # noqa: PLC0415
        MonitorFieldFrequency,
    )
    from magnelio.monitors.field_time import MonitorFieldTime  # noqa: PLC0415
    from magnelio.monitors.flux import MonitorFluxTime  # noqa: PLC0415
    from magnelio.monitors.wall_loss import MonitorWallLoss  # noqa: PLC0415

    if isinstance(mon, MonitorFieldTime):
        return {
            "type": "MonitorFieldTime",
            "corners": _corners_to_json(mon.corners),
            # Exactly one of the two schedules is set; the open-ended
            # interval form must resume as an interval, not as the
            # instants it happened to reach before the checkpoint.
            "times": (None if mon.times is None else [float(t) for t in mon.times]),
            "interval": (None if mon.interval is None else float(mon.interval)),
            "start": float(mon.start),
            "fields": list(mon.fields),
            "name": mon.name,
        }
    if isinstance(mon, MonitorFluxTime):
        return {
            "type": "MonitorFluxTime",
            "normal": mon.normal,
            "position": float(mon.position),
            "name": mon.name,
        }
    if isinstance(mon, MonitorFieldFrequency):
        return {
            "type": "MonitorFieldFrequency",
            "corners": _corners_to_json(mon.corners),
            "freqs": [float(fr) for fr in mon.freqs],
            "fields": list(mon.fields),
            "name": mon.name,
        }
    if isinstance(mon, MonitorWallLoss):
        # Roughness reuses the store's material serialiser — one format
        # for the one concept (DD-088), imported lazily to keep the
        # recipe module free of an io dependency at import time.
        from magnelio.io.project import _roughness_to_dict  # noqa: PLC0415

        return {
            "type": "MonitorWallLoss",
            "freqs": [float(fr) for fr in mon.freqs],
            "normal": mon.normal,
            "position": _num_to_json(mon.position),
            "sigma": None if mon.sigma is None else float(mon.sigma),
            "mu": float(mon.mu),
            "roughness": (None if mon.roughness is None else _roughness_to_dict(mon.roughness)),
            "bc_faces": list(mon.bc_faces),
            "name": mon.name,
        }
    from magnelio.monitors.far_field import MonitorFarFieldFrequency  # noqa: PLC0415

    if isinstance(mon, MonitorFarFieldFrequency):
        return {
            "type": "MonitorFarFieldFrequency",
            "freqs": [float(fr) for fr in mon.freqs],
            "margin_cells": int(mon.margin_cells),
            "name": mon.name,
        }
    # Project-store monitor streaming: DD-070.
    raise NotImplementedError(
        f"resume cannot serialise monitor type {type(mon).__name__}; "
        f"only MonitorFieldTime, MonitorFluxTime, MonitorFieldFrequency, "
        f"MonitorWallLoss and MonitorFarFieldFrequency are streamed to the project "
        f"store.  Run without project= to keep it in RAM, or drop it for "
        f"the streamed run.",
    )


def _monitor_from_dict(d: dict):
    from magnelio.monitors.field_frequency import (  # noqa: PLC0415
        MonitorFieldFrequency,
    )
    from magnelio.monitors.field_time import MonitorFieldTime  # noqa: PLC0415
    from magnelio.monitors.flux import MonitorFluxTime  # noqa: PLC0415

    if d["type"] == "MonitorFieldTime":
        # Recipes predating the interval form carry "times" only.
        times = d.get("times")
        interval = d.get("interval")
        return MonitorFieldTime(
            corners=_corners_from_json(d["corners"]),
            times=None if times is None else [float(t) for t in times],
            interval=None if interval is None else float(interval),
            start=float(d.get("start", 0.0)),
            fields=list(d["fields"]),
            name=d["name"],
        )
    if d["type"] == "MonitorFluxTime":
        return MonitorFluxTime(
            normal=str(d["normal"]),
            position=float(_num_from_json(d["position"])),
            name=d["name"],
        )
    if d["type"] == "MonitorFieldFrequency":
        return MonitorFieldFrequency(
            corners=_corners_from_json(d["corners"]),
            freqs=[float(fr) for fr in d["freqs"]],
            fields=list(d["fields"]),
            name=d["name"],
        )
    if d["type"] == "MonitorWallLoss":
        from magnelio.io.project import _roughness_from_dict  # noqa: PLC0415
        from magnelio.monitors.wall_loss import MonitorWallLoss  # noqa: PLC0415

        return MonitorWallLoss(
            freqs=[float(fr) for fr in d["freqs"]],  # __post_init__ asarrays
            normal=str(d["normal"]),
            position=float(_num_from_json(d["position"])),
            sigma=None if d["sigma"] is None else float(d["sigma"]),
            mu=float(d["mu"]),
            roughness=_roughness_from_dict(d.get("roughness")),
            bc_faces=tuple(d["bc_faces"]),
            name=d["name"],
        )
    if d["type"] == "MonitorFarFieldFrequency":
        from magnelio.monitors.far_field import MonitorFarFieldFrequency  # noqa: PLC0415

        return MonitorFarFieldFrequency(
            freqs=[float(fr) for fr in d["freqs"]],
            margin_cells=int(d.get("margin_cells", 3)),
            name=d["name"],
        )
    raise TypeError(f"unknown monitor type {d['type']!r} in recipe")


# ═════════════════════════════════════════════════════════════════════
# Full recipe
# ═════════════════════════════════════════════════════════════════════


def build_recipe(analysis) -> dict:
    """Capture everything needed to reconstruct ``analysis`` for a resume.

    The mesh is *not* part of the recipe (it round-trips through
    ``mesh.h5``, ports, elements and sources with it); the returned
    dict carries the resolved port specs, the field monitors, the wall
    model and the band settings — plus, for the scattering analysis,
    its frequency axis and explicit waveform.  Stored under
    ``setup['recipe']``.

    The boundary closure is *not* here either — since DD-103 it belongs
    to the mesh, and round-trips with it through ``mesh.h5``.
    """
    # Roughness reuses the store's material serialiser (one format for
    # the one concept, DD-088) — lazily, like the monitor codec above.
    from magnelio._backend.array_api import resolve_precision  # noqa: PLC0415
    from magnelio.io.project import _roughness_to_dict  # noqa: PLC0415

    # Persist the RESOLVED precision (concrete "single"/"double"), not the
    # raw None sentinel: a resume must reproduce the dtype the run actually
    # used, independent of MAGNELIO_PRECISION at resume time (plan WP3).
    real_dtype, _ = resolve_precision(analysis.precision)
    recipe = {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "analysis": type(analysis).__name__,
        "precision": "single" if real_dtype.itemsize == 4 else "double",
        "ports": [_spec_to_dict(s) for s in analysis.ports],
        "f_max": float(analysis.f_max),
        "port_model": analysis.port_model,
        # SIBC wall model (WP-D5): the switch + overrides suffice — the
        # spec itself (surfaces, fits) is re-derived from the stored
        # mesh and this band on resume, like every constant operator.
        "wall_model": analysis.wall_model,
        "wall_sigma": (None if analysis.wall_sigma is None else float(analysis.wall_sigma)),
        "wall_mu": float(analysis.wall_mu),
        "wall_roughness": (
            None if analysis.wall_roughness is None else _roughness_to_dict(analysis.wall_roughness)
        ),
        # Every persisted monitor kind is in the recipe so a resume
        # rebuilds it (time/flux stream into results.h5; frequency and
        # wall-loss carry running accumulators in their own result
        # files).  Non-data control monitors are deliberately NOT — see
        # _serialisable_monitors.
        "monitors": [_monitor_to_dict(m) for m in _serialisable_monitors(analysis.monitors)],
    }
    # The scattering analysis adds its frequency axis and the explicit
    # waveform; the general time-domain analysis has neither.
    if hasattr(analysis, "n_freq"):
        recipe["f_min"] = float(analysis.f_min)
        recipe["n_freq"] = int(analysis.n_freq)
        recipe["waveform"] = _waveform_to_dict(analysis.waveform)
        # Band-pipeline settings.  These size the ghost excitation and
        # the subspace, so a run rebuilt without them is a *different*
        # run: n_syn in particular falls back to auto-sizing and the
        # synthesised pulse changes, which a resume would splice onto
        # the recorded one mid-record.
        if getattr(analysis, "band_options", None):
            recipe["band_options"] = {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in dict(analysis.band_options).items()
            }
    return recipe


# The name the scattering analysis used before DD-224 Phase B.
build_scattering_recipe = build_recipe


def _serialisable_monitors(monitors) -> list:
    """The recipe-persistable subset of a monitor list.

    A WHITELIST on purpose, not a try/except around
    :func:`_monitor_to_dict`: a resume must not reconstruct a non-data
    control monitor (a ``_CallAtStep`` SIGINT handler would re-fire its
    abort).  The cost is that a new persisted monitor kind has to be
    added here as well, or it is silently dropped from the rebuilt run.
    """
    from magnelio.monitors.far_field import MonitorFarFieldFrequency  # noqa: PLC0415
    from magnelio.monitors.field_frequency import (  # noqa: PLC0415
        MonitorFieldFrequency,
    )
    from magnelio.monitors.field_time import MonitorFieldTime  # noqa: PLC0415
    from magnelio.monitors.flux import MonitorFluxTime  # noqa: PLC0415
    from magnelio.monitors.wall_loss import MonitorWallLoss  # noqa: PLC0415

    return [
        m
        for m in monitors
        if isinstance(
            m,
            (
                MonitorFieldTime,
                MonitorFluxTime,
                MonitorFieldFrequency,
                MonitorWallLoss,
                MonitorFarFieldFrequency,
            ),
        )
    ]


def recipe_kwargs(recipe: dict) -> dict:
    """Reconstruct the analysis constructor kwargs (sans mesh).

    Returns the ports / monitors / wall model / band settings the
    recipe holds — the scattering keys (``f_min``, ``n_freq``,
    ``waveform``) only when present; the caller supplies ``mesh=``
    (from the store) and ``verbose=``.
    """
    from magnelio.io.project import _roughness_from_dict  # noqa: PLC0415

    kwargs = {
        "ports": [_spec_from_dict(d) for d in recipe["ports"]],
        "f_max": float(recipe["f_max"]),
        "port_model": recipe["port_model"],
        "precision": recipe["precision"],
        "monitors": tuple(_monitor_from_dict(d) for d in recipe.get("monitors", [])),
        "wall_model": recipe.get("wall_model", "perturbative"),
        "wall_sigma": (None if recipe.get("wall_sigma") is None else float(recipe["wall_sigma"])),
        "wall_mu": float(recipe.get("wall_mu", 1.0)),
        "wall_roughness": _roughness_from_dict(recipe.get("wall_roughness")),
    }
    if "n_freq" in recipe:
        kwargs["f_min"] = float(recipe["f_min"])
        kwargs["n_freq"] = int(recipe["n_freq"])
        kwargs["waveform"] = _waveform_from_dict(recipe.get("waveform"))
        band_opts = recipe.get("band_options")
        if band_opts:
            # f_band round-trips through JSON as a list; the sizing code
            # indexes it either way, but a tuple keeps the rebuilt
            # analysis equal to the one that was stored.
            kwargs["band_options"] = {
                k: (tuple(v) if isinstance(v, list) else v) for k, v in dict(band_opts).items()
            }
    return kwargs
