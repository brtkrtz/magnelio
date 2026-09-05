"""On-disk project store — the write-once model layer.

This is the foundation of the simulation/post-processing separation: a
**project directory** that persists the static model (geometry, mesh,
setup metadata) plus the streamed time-domain results and per-run
resume checkpoints.

Directory layout (the write-once model, plus the ``runs/`` and
``eigenmodes.h5`` artefacts of the streaming layers)::

    <project>/
        project.json        schema/version, setup metadata, run index, status
        geometry.brep       exact OCC geometry (ordered compound)  [optional]
        geometry.vtm        per-solid multiblock mesh for ParaView  [optional]
        geometry.json       per-shape names/materials + background (compound
                            order)
        mesh.h5             grid, material_id, material_library,
                            pec_mask, edge/face/PEC-surface material

* Geometry is stored as an ordered ``TopoDS_Compound`` via
  ``BRepTools`` (exact, lossless round-trip); the shape → material
  mapping rides in ``geometry.json`` in the same order the compound
  iterates.  BREP carries the boundary representation only, so the
  reconstructed geometry is kernel-shape-backed
  (:class:`~magnelio.geo.ImportedSolid`) —
  the original ``Brick`` / ``Difference`` CSG tree is not recovered
  (by design; the store serves visualisation, documentation and
  re-meshing, and resume never needs geometry).
* The mesh HDF5 mirrors the (retired) ``save_project`` mesh schema plus
  the material library and the conformal sub-cell material data
  (``EdgeMaterialData`` / ``FaceMaterialData`` / ``PECSurfaceData``),
  so a loaded mesh is bit-identical to the one that produced it.

See ``PROJECT_STORE_PLAN.md`` for the full plan.
"""

# Design: DD-070 (project store, unbounded runtime, bit-exact resume);
# work packages: WP-S1 (write-once model), WP-S2+ (runs/ and eigenmodes.h5
# streaming artefacts).

from __future__ import annotations

import html
import json
import os
import socket
import sys
import time
from collections.abc import Mapping
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

from magnelio._progress import (
    _is_notebook_stream,
    current_reporter,
    format_clock,
    format_seconds,
    parse_utc,
    utc_now,
)
from magnelio._repr import fmt_array, fmt_db, fmt_value, html_kv, html_table, kv_block, text_table
from magnelio.analysis.result_interface import RunSettings, ScatteringResultMixin
from magnelio.io._schema import (
    SCHEMA_VERSION,
    validate_schema,
)
from magnelio.post._energy import db_below_peak

# ═════════════════════════════════════════════════════════════════════
# Material  <->  JSON
# ═════════════════════════════════════════════════════════════════════


def _material_to_dict(mat) -> dict:
    """Serialise a :class:`~magnelio.materials.material.Material` to a dict."""
    d = {
        "name": mat.name,
        "epsilon": list(mat.epsilon),
        "mu": list(mat.mu),
        "sigma": list(mat.sigma),
        "sigma_m": list(mat.sigma_m),
        "is_pec": bool(mat.is_pec),
        "color": list(mat.color) if mat.color is not None else None,
        "alpha": float(mat.alpha),
        "visible": bool(mat.visible),
    }
    if mat.dispersion is not None:
        # Schema-additive (DD-083): absent key = non-dispersive (all
        # older stores load unchanged).
        d["dispersion"] = _dispersion_to_dict(mat.dispersion)
    if mat.dispersion_mu is not None:
        # Schema-additive (DD-089): the same pole-residue form on the
        # H side (the model's eps_inf field carries mu_inf).
        d["dispersion_mu"] = _dispersion_to_dict(mat.dispersion_mu)
    if mat.roughness is not None:
        # Schema-additive (DD-088): a tagged dict per model; absent key =
        # smooth (all older stores load unchanged).
        d["roughness"] = _roughness_to_dict(mat.roughness)
    return d


def _dispersion_to_dict(m) -> dict:
    """Serialise a :class:`DispersionModel` (complex poles flattened to
    re/im quadruples for JSON)."""
    return {
        "eps_inf": float(m.eps_inf),
        "poles": [[a.real, a.imag, r.real, r.imag] for a, r in m.poles],
        "f_band": [float(m.f_band[0]), float(m.f_band[1])],
    }


def _dispersion_from_dict(d: dict | None):
    """Reconstruct a :class:`DispersionModel` from :func:`_dispersion_to_dict`."""
    from magnelio.materials.dispersion import DispersionModel  # noqa: PLC0415

    if d is None:
        return None
    return DispersionModel(
        eps_inf=float(d["eps_inf"]),
        poles=tuple((complex(ar, ai), complex(rr, ri)) for ar, ai, rr, ri in d["poles"]),
        f_band=(float(d["f_band"][0]), float(d["f_band"][1])),
    )


def _roughness_to_dict(r) -> dict:
    """Serialise a :class:`SurfaceRoughness` to a model-tagged dict."""
    from magnelio.materials.roughness import Hammerstad, Huray  # noqa: PLC0415

    if isinstance(r, Hammerstad):
        return {"model": "hammerstad", "rms_height": float(r.rms_height)}
    if isinstance(r, Huray):
        return {
            "model": "huray",
            "radius": float(r.radius),
            "coverage": float(r.coverage),
            "base_ratio": float(r.base_ratio),
        }
    raise TypeError(f"cannot serialise roughness model: {r!r}")


def _roughness_from_dict(d: dict | None):
    """Reconstruct a roughness model from :func:`_roughness_to_dict` output."""
    from magnelio.materials.roughness import Hammerstad, Huray  # noqa: PLC0415

    if d is None:
        return None
    kind = d["model"]
    if kind == "hammerstad":
        return Hammerstad(rms_height=float(d["rms_height"]))
    if kind == "huray":
        return Huray(
            radius=float(d["radius"]),
            coverage=float(d["coverage"]),
            base_ratio=float(d.get("base_ratio", 1.0)),
        )
    raise ValueError(f"unknown roughness model in store: {kind!r}")


def _material_from_dict(d: dict):
    """Reconstruct a :class:`Material` from :func:`_material_to_dict` output."""
    from magnelio.materials.material import Material  # noqa: PLC0415

    return Material(
        name=d["name"],
        epsilon=tuple(d["epsilon"]),
        mu=tuple(d["mu"]),
        sigma=tuple(d["sigma"]),
        sigma_m=tuple(d["sigma_m"]),
        is_pec=bool(d["is_pec"]),
        dispersion=_dispersion_from_dict(d.get("dispersion")),
        dispersion_mu=_dispersion_from_dict(d.get("dispersion_mu")),
        roughness=_roughness_from_dict(d.get("roughness")),
        color=tuple(d["color"]) if d.get("color") is not None else None,
        alpha=float(d.get("alpha", 1.0)),
        visible=bool(d.get("visible", True)),
    )


# ═════════════════════════════════════════════════════════════════════
# Geometry  <->  BREP  (ordered compound)
# ═════════════════════════════════════════════════════════════════════


def write_brep(shapes: list, path: str | Path) -> None:
    """Write an ordered list of geometry shapes to a BREP compound file.

    Parameters
    ----------
    shapes : list
        Geometry shape objects (each exposing ``_occ_shape()``) or raw
        ``TopoDS_Shape`` instances.  The write order is preserved and
        recovered verbatim by :func:`read_brep`.
    path : str or Path
        Output ``.brep`` file.
    """
    from OCC.Core.BRep import BRep_Builder  # noqa: PLC0415
    from OCC.Core.BRepTools import breptools  # noqa: PLC0415
    from OCC.Core.TopoDS import TopoDS_Compound  # noqa: PLC0415

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    # DD-120: build at the model scale (raw TopoDS entries force 1.0 —
    # they are already meter-space BReps), then transform the compound
    # back so the .brep file stays in meters at any model scale.
    geo_scale = 1.0
    if shapes and all(hasattr(s, "_analytic_bbox") for s in shapes):
        from magnelio.geo._scaling import model_scale  # noqa: PLC0415

        geo_scale = model_scale(shapes)
    for s in shapes:
        topo = s._occ_shape(geo_scale) if hasattr(s, "_occ_shape") else s
        builder.Add(compound, topo)
    if geo_scale != 1.0:
        from magnelio.geo._occ_backend import occ_scale  # noqa: PLC0415

        compound = occ_scale(compound, 1.0 / geo_scale, (0.0, 0.0, 0.0))
    if not breptools.Write(compound, str(path)):
        raise IOError(f"BRepTools.Write failed for {path}")


def read_brep(path: str | Path) -> list:
    """Read a BREP compound file back into an ordered list of shapes.

    Parameters
    ----------
    path : str or Path
        A ``.brep`` file written by :func:`write_brep`.

    Returns
    -------
    list of TopoDS_Shape
        The compound's direct children, in write order.
    """
    from OCC.Core.BRep import BRep_Builder  # noqa: PLC0415
    from OCC.Core.BRepTools import breptools  # noqa: PLC0415
    from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Iterator  # noqa: PLC0415

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    if not breptools.Read(compound, str(path), builder):
        raise IOError(f"BRepTools.Read failed for {path}")
    shapes = []
    it = TopoDS_Iterator(compound)
    while it.More():
        shapes.append(it.Value())
        it.Next()
    return shapes


class LoadedGeometry:
    """Read-only geometry recovered from a project's BREP + material list.

    Mirrors the iteration surface of
    :class:`~magnelio.geo.GeometryModel` (``shapes``, ``background``,
    iteration, ``len``) so it drops into the same consumers.
    """

    def __init__(self, shapes: list, background) -> None:
        self.shapes = shapes
        self.background = background

    def __iter__(self):
        return iter(self.shapes)

    def __len__(self) -> int:
        return len(self.shapes)

    def plot(self, **kwargs):
        """Interactive 3D view — same wrapper as ``GeometryModel.plot``."""
        from magnelio.post.plot_geometry import (  # noqa: PLC0415
            show_geometry,
        )

        return show_geometry(self, **kwargs)

    def plot_cross_section(self, normal: str, position: float, **kwargs):
        """2D cross-section — same wrapper as
        ``GeometryModel.plot_cross_section``."""
        from magnelio.post.plot_geometry import (  # noqa: PLC0415
            plot_cross_section,
        )

        return plot_cross_section(
            self,
            normal,
            position,
            **kwargs,
        )

    def __repr__(self) -> str:
        bg = getattr(self.background, "name", None)
        return f"LoadedGeometry({len(self.shapes)} shapes, background={bg})"


# ═════════════════════════════════════════════════════════════════════
# Conformal sub-cell dataclasses  <->  HDF5  (generic array bundle)
# ═════════════════════════════════════════════════════════════════════


def _save_dataclass_arrays(grp, obj) -> None:
    """Write a dataclass-of-arrays into ``grp``; ``None`` fields recorded.

    Each non-None field becomes a dataset; the names of the ``None``
    fields are stored in the group attribute ``none_fields`` so the
    loader reconstructs them exactly.
    """
    none_fields = []
    for f in dc_fields(obj):
        val = getattr(obj, f.name)
        if val is None:
            none_fields.append(f.name)
        else:
            grp.create_dataset(f.name, data=np.asarray(val))
    grp.attrs["none_fields"] = json.dumps(none_fields)


def _load_dataclass_arrays(grp, cls):
    """Reconstruct a dataclass-of-arrays written by :func:`_save_dataclass_arrays`.

    Schema-additive: fields added to the dataclass after a store was
    written (e.g. ``A_face_pec``, DD-087) are absent from old groups
    and load as ``None`` — consumers carry their own fallbacks.
    """
    none_fields = set(json.loads(grp.attrs.get("none_fields", "[]")))
    kwargs = {}
    for f in dc_fields(cls):
        if f.name in none_fields or f.name not in grp:
            kwargs[f.name] = None
        else:
            kwargs[f.name] = grp[f.name][()]
    return cls(**kwargs)


# ═════════════════════════════════════════════════════════════════════
# Mesh  <->  HDF5
# ═════════════════════════════════════════════════════════════════════


def _companion_to_dict(element) -> dict:
    """Serialise a SeriesRLC/ParallelRLC as its constructor fields.

    Deliberately NOT ``dataclasses.asdict``: that would also emit the
    ``init=False`` transient-state fields (``_i``/``_vL``/…), which the
    constructor rejects on reload.  Companion state never round-trips
    through the declaration — it lives in the solver checkpoint.
    """
    from magnelio.circuit.companion import SeriesRLC  # noqa: PLC0415

    return {
        "kind": "series" if isinstance(element, SeriesRLC) else "parallel",
        "R": element.R,
        "L": element.L,
        "C": element.C,
    }


def _declarative_port_to_dict(port) -> dict:
    """Serialise a declarative port (DD-109) for the mesh round-trip."""
    import dataclasses  # noqa: PLC0415

    from magnelio.circuit.companion import ParallelRLC, SeriesRLC  # noqa: PLC0415
    from magnelio.ports.declarative import (  # noqa: PLC0415
        PortAnalytical,
        PortLumped,
        PortWaveguide,
    )

    if isinstance(port, PortLumped):
        # Not asdict on the whole port: its ``element`` needs the
        # init-fields-only treatment (see _companion_to_dict).
        d = {
            "name": port.name,
            "start": list(port.start),
            "end": list(port.end),
            "Z0": port.Z0,
            "kind": "lumped",
        }
        if port.element is not None:
            assert isinstance(port.element, (SeriesRLC, ParallelRLC))
            d["element"] = _companion_to_dict(port.element)
        return d
    d = dataclasses.asdict(port)
    d["kind"] = "waveguide" if isinstance(port, PortWaveguide) else "analytical"
    assert isinstance(port, (PortWaveguide, PortAnalytical))
    plane = port.plane
    d["plane"] = plane.value if hasattr(plane, "value") else str(plane)
    return d


def _declarative_port_from_dict(d: dict):
    """Rebuild a declarative port written by :func:`_declarative_port_to_dict`."""
    from magnelio.circuit.companion import ParallelRLC, SeriesRLC  # noqa: PLC0415
    from magnelio.ports.declarative import (  # noqa: PLC0415
        PortAnalytical,
        PortLumped,
        PortWaveguide,
    )

    d = dict(d)
    kind = d.pop("kind")
    if kind == "lumped":
        elem = d.pop("element", None)
        if elem is not None:
            elem = dict(elem)
            elem_cls = SeriesRLC if elem.pop("kind") == "series" else ParallelRLC
            elem = elem_cls(**elem)
        return PortLumped(
            name=d["name"],
            start=tuple(d["start"]),
            end=tuple(d["end"]),
            Z0=float(d["Z0"]),
            element=elem,
        )
    if d.get("corners") is not None:
        d["corners"] = tuple(tuple(c) for c in d["corners"])
    if d.get("center") is not None:
        d["center"] = tuple(d["center"])
    cls = PortWaveguide if kind == "waveguide" else PortAnalytical
    return cls(**d)


def _source_to_dict(source) -> dict:
    """Serialise a declarative source (DD-224) for the mesh round-trip.

    The class name is the type tag; the dataclass init fields are the
    payload (the excitation binding is run state, not model data).
    """
    import dataclasses  # noqa: PLC0415

    from magnelio.fields import FieldState  # noqa: PLC0415

    d = {"type": type(source).__name__}
    for f in dataclasses.fields(source):
        if not f.init:
            continue
        v = getattr(source, f.name)
        if isinstance(v, FieldState) or callable(v):
            # Array payloads travel as datasets (``_store_payload``);
            # a Python callable cannot be stored at all — the tag and
            # the remaining fields keep the recipe readable.
            continue
        if f.name == "path":
            # A point path round-trips verbatim; a general Curve carries a
            # builder callable the store cannot rebuild (like a callable
            # incident field), so the recipe keeps the tag and the rest.
            pts = getattr(source, "_points", None)
            if pts is None:
                continue
            v = [[float(c) for c in point] for point in pts]
        elif f.name == "corners" and v is not None:
            v = [[None if c is None else float(c) for c in point] for point in v]
        elif isinstance(v, tuple):
            v = [float(c) for c in v]
        d[f.name] = v
    return d


def _source_from_dict(d: dict, payload: dict | None = None):
    """Rebuild a declarative source written by :func:`_source_to_dict`.

    *payload* holds the datasets a source stored next to its recipe
    (an initial field's arrays); a source needing one is rebuilt by
    its own ``_from_store_payload``.
    """
    from magnelio.sources import (  # noqa: PLC0415
        SourceCurrentPath,
        SourceFieldIncident,
        SourceFieldInitial,
        SourcePlaneWave,
    )

    registry = {
        "SourcePlaneWave": SourcePlaneWave,
        "SourceFieldInitial": SourceFieldInitial,
        "SourceFieldIncident": SourceFieldIncident,
        "SourceCurrentPath": SourceCurrentPath,
    }
    d = dict(d)
    tag = d.pop("type")
    if tag not in registry:
        raise ValueError(f"unknown source type {tag!r} in mesh.h5")
    if tag == "SourceFieldInitial":
        if payload is None:
            raise ValueError(f"source {d.get('name')!r}: mesh.h5 holds no field arrays for it")
        return SourceFieldInitial._from_store_payload(d, payload)
    if tag == "SourceCurrentPath" and "path" not in d:
        raise ValueError(
            f"source {d.get('name')!r} is a SourceCurrentPath on a general "
            f"magnelio.geo.Curve, which the store cannot rebuild; declare it "
            f"again on the reloaded mesh, or give the path as points",
        )
    if tag == "SourceFieldIncident":
        raise ValueError(
            f"source {d.get('name')!r} is a SourceFieldIncident with a Python callable, "
            f"which the store cannot rebuild; declare it again on the reloaded mesh",
        )
    if d.get("corners") is not None:
        d["corners"] = tuple(tuple(c) for c in d["corners"])
    for key in ("direction", "polarization"):
        if key in d and d[key] is not None:
            d[key] = tuple(d[key])
    return registry[tag](**d)


def _lumped_element_to_dict(element) -> dict:
    """Serialise a declarative LumpedElement (DD-123) for the mesh round-trip."""
    return {
        "name": element.name,
        "start": list(element.start),
        "end": list(element.end),
        "element": _companion_to_dict(element.element),
    }


def _lumped_element_from_dict(d: dict):
    """Rebuild a declarative LumpedElement written by :func:`_lumped_element_to_dict`."""
    from magnelio.circuit import LumpedElement  # noqa: PLC0415
    from magnelio.circuit.companion import ParallelRLC, SeriesRLC  # noqa: PLC0415

    elem = dict(d["element"])
    elem_cls = SeriesRLC if elem.pop("kind") == "series" else ParallelRLC
    return LumpedElement(
        name=d["name"],
        start=tuple(d["start"]),
        end=tuple(d["end"]),
        element=elem_cls(**elem),
    )


def _save_mesh(f, mesh) -> None:
    """Write a :class:`Mesh` into an open h5py file/group ``f``."""
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        bc_type_entries,
        cpml_thickness_of,
        symmetry_entries,
    )

    grid = f.create_group("grid")
    grid.create_dataset("x", data=np.asarray(mesh.grid.x))
    grid.create_dataset("y", data=np.asarray(mesh.grid.y))
    grid.create_dataset("z", data=np.asarray(mesh.grid.z))

    mg = f.create_group("mesh")
    # Element type (DD-224 §6): today's Mesh is the hexahedral Yee grid;
    # the loader dispatches on this tag once tetrahedral and surface
    # meshes exist, without another schema bump.
    mg.attrs["element"] = "hexahedral"
    mg.create_dataset("material_id", data=mesh.material_id)
    mg.create_dataset("pec_mask_edges", data=mesh.pec_mask_edges)
    mg.attrs["material_library"] = json.dumps(
        {str(mid): _material_to_dict(mat) for mid, mat in mesh.material_library.items()}
    )
    # Boundary closure (DD-103): a mesh property since the declaration
    # moved to the model, so it round-trips here rather than in the
    # analysis recipe.  Stored as the canonical type map — a
    # PECBoundary's own wall material (DD-099) does not survive, the
    # same limitation the recipe route had.
    bc = mesh.boundary_conditions
    bc_json = {
        "types": bc_type_entries(bc),
        "cpml_thickness_cells": cpml_thickness_of(bc),
    }
    sym = symmetry_entries(bc)
    if sym:
        bc_json["symmetry"] = sym
    mg.attrs["boundary_conditions"] = json.dumps(bc_json)
    # Declarative ports (DD-109) travel with the mesh, so they round-
    # trip here — the analysis reads them off the reloaded mesh exactly
    # as off a fresh one.
    if mesh.ports:
        mg.attrs["ports"] = json.dumps(
            [_declarative_port_to_dict(p) for p in mesh.ports],
        )
    # Passive lumped elements (DD-123) travel with the mesh like ports.
    if getattr(mesh, "elements", ()):
        mg.attrs["elements"] = json.dumps(
            [_lumped_element_to_dict(e) for e in mesh.elements],
        )
    # Field sources (DD-224) travel with the mesh like ports.
    if getattr(mesh, "sources", ()):
        mg.attrs["sources"] = json.dumps(
            [_source_to_dict(src) for src in mesh.sources],
        )
        for src in mesh.sources:
            payload = getattr(src, "_store_payload", None)
            if payload is None:
                continue
            grp = mg.create_group(f"sources/{src.name}")
            for key, arr in payload().items():
                grp.create_dataset(key, data=np.asarray(arr))
    # Design frequency (DD-186) — travels with the mesh like the
    # closure and the ports.
    if getattr(mesh, "f_max", None) is not None:
        mg.attrs["f_max"] = float(mesh.f_max)
    # Grid-plane provenance (DD-200).  A string *dataset*, not an attr:
    # HDF5 attributes are capped at 64 KB and a CAD import with
    # hundreds of shapes lists every one of them per plane.
    if getattr(mesh, "planes", None) is not None:
        import h5py  # noqa: PLC0415

        mg.create_dataset(
            "planes_json",
            data=json.dumps(mesh.planes.as_dict(rounded=False)),
            dtype=h5py.string_dtype(),
        )

    if mesh.edge_material is not None:
        _save_dataclass_arrays(mg.create_group("edge_material"), mesh.edge_material)
    if mesh.face_material is not None:
        _save_dataclass_arrays(mg.create_group("face_material"), mesh.face_material)
    if mesh.pec_surface is not None:
        _save_dataclass_arrays(mg.create_group("pec_surface"), mesh.pec_surface)


def _load_mesh(f):
    """Reconstruct a :class:`Mesh` from an open h5py file/group ``f``."""
    from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
        BoundaryConditions,
    )
    from magnelio.geo._subcell import EdgeMaterialData, FaceMaterialData  # noqa: PLC0415
    from magnelio.mesh._conformal import PECSurfaceData  # noqa: PLC0415
    from magnelio.mesh._planes import GridPlanes  # noqa: PLC0415
    from magnelio.mesh.grid import GridLines  # noqa: PLC0415
    from magnelio.mesh.mesher import Mesh  # noqa: PLC0415

    grid = GridLines(
        x=f["grid/x"][()],
        y=f["grid/y"][()],
        z=f["grid/z"][()],
    )
    mg = f["mesh"]
    element = str(mg.attrs.get("element", "hexahedral"))
    if element != "hexahedral":
        raise ValueError(
            f"mesh.h5 holds a {element!r} mesh; this release reads hexahedral meshes only",
        )
    library = {
        int(mid): _material_from_dict(d)
        for mid, d in json.loads(mg.attrs["material_library"]).items()
    }
    edge_material = (
        _load_dataclass_arrays(mg["edge_material"], EdgeMaterialData)
        if "edge_material" in mg
        else None
    )
    face_material = (
        _load_dataclass_arrays(mg["face_material"], FaceMaterialData)
        if "face_material" in mg
        else None
    )
    pec_surface = (
        _load_dataclass_arrays(mg["pec_surface"], PECSurfaceData) if "pec_surface" in mg else None
    )
    bc_attr = mg.attrs.get("boundary_conditions")
    boundary_conditions = None
    if bc_attr is not None:
        d = json.loads(bc_attr)
        boundary_conditions = BoundaryConditions(
            **d["types"],
            cpml_thickness_cells=int(d.get("cpml_thickness_cells", 8)),
            symmetry=d.get("symmetry", {}),
        )
    ports_attr = mg.attrs.get("ports")
    ports = (
        tuple(_declarative_port_from_dict(d) for d in json.loads(ports_attr))
        if ports_attr is not None
        else ()
    )
    elements_attr = mg.attrs.get("elements")
    elements = (
        tuple(_lumped_element_from_dict(d) for d in json.loads(elements_attr))
        if elements_attr is not None
        else ()
    )
    sources_attr = mg.attrs.get("sources")
    sources = ()
    if sources_attr is not None:
        rebuilt = []
        for d in json.loads(sources_attr):
            grp = mg.get(f"sources/{d.get('name')}")
            payload = {k: grp[k][()] for k in grp} if grp is not None else None
            rebuilt.append(_source_from_dict(d, payload))
        sources = tuple(rebuilt)
    planes = None
    if "planes_json" in mg:
        raw = mg["planes_json"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        planes = GridPlanes.from_dict(json.loads(raw))
    return Mesh(
        ports=ports,
        elements=elements,
        sources=sources,
        planes=planes,
        grid=grid,
        material_id=mg["material_id"][()],
        material_library=library,
        pec_mask_edges=mg["pec_mask_edges"][()],
        edge_material=edge_material,
        face_material=face_material,
        pec_surface=pec_surface,
        boundary_conditions=boundary_conditions,
        f_max=float(mg.attrs["f_max"]) if "f_max" in mg.attrs else None,
    )


# ═════════════════════════════════════════════════════════════════════
# Atomic JSON write
# ═════════════════════════════════════════════════════════════════════


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON via a temp file + ``os.replace`` (atomic on POSIX)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _utc_now_iso() -> str:
    """The current UTC time as the ISO-8601 string every stamp in the store uses."""
    return utc_now().isoformat(timespec="seconds")


def _writer_identity() -> dict:
    """Who is writing: the process id and host of the running solver.

    A reader on the same host can ask the operating system whether that
    process still exists — the difference between a run that is
    marching and one whose kernel died with ``running`` on disk.
    """
    return {"pid": int(os.getpid()), "host": socket.gethostname()}


def _update_meta(path: Path, fn) -> None:
    """Read ``project.json``, apply ``fn`` in place, write it back atomically."""
    with open(path / "project.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    fn(meta)
    meta["modified"] = _utc_now_iso()
    _write_json_atomic(path / "project.json", meta)


# ═════════════════════════════════════════════════════════════════════
# Modes  <->  scalars  (only z_modal / gamma / mode_type are ever used
# by the S-parameter and power-wave decomposition, so a run persists a
# handful of scalars, not the transverse profiles)
# ═════════════════════════════════════════════════════════════════════


# Non-None placeholder for the reconstructed Mode's field_evaluator: the
# decomposition never samples transverse profiles, and Mode.__post_init__
# only checks that the evaluator is not None (profiles then stay unset).
_MODE_EVAL_SENTINEL = object()


class _PersistedLumpedMode:
    """Reconstructed lumped-port reference impedance (``z_modal`` = Z0)."""

    def __init__(self, z0: float) -> None:
        self.z0 = float(z0)

    def z_modal(self, omega: float) -> complex:
        del omega
        return complex(self.z0)


def _mode_to_dict(mode) -> dict:
    """Serialise the scalars a run needs to reproduce ``z_modal``/``gamma``."""
    if hasattr(mode, "mode_type"):
        return {
            "kind": "modal",
            "name": getattr(mode, "name", ""),
            "mode_type": mode.mode_type.value,
            "omega_c": float(mode.omega_c),
            "epsilon_r": float(mode.epsilon_r),
            "z_line": None if mode.z_line is None else float(mode.z_line),
        }
    return {"kind": "lumped", "z0": float(mode.z_modal(0.0).real)}


def _mode_from_dict(d: dict):
    """Reconstruct a mode (real :class:`Mode` or lumped stub) from scalars."""
    if d["kind"] == "lumped":
        return _PersistedLumpedMode(d["z0"])
    from magnelio.ports._modal.mode import Mode, ModeType  # noqa: PLC0415

    return Mode(
        name=d.get("name", ""),
        mode_type=ModeType(d["mode_type"]),
        omega_c=float(d["omega_c"]),
        epsilon_r=float(d["epsilon_r"]),
        field_evaluator=_MODE_EVAL_SENTINEL,
        z_line=None if d["z_line"] is None else float(d["z_line"]),
    )


def _write_sparse(group, name: str, matrix) -> None:
    """Store one CSC block as ``data``/``indices``/``indptr`` + shape."""
    import scipy.sparse as sp  # noqa: PLC0415

    m = sp.csc_matrix(matrix)
    g = group.create_group(name)
    g.attrs["shape"] = np.asarray(m.shape, dtype="i8")
    g.create_dataset("data", data=m.data)
    g.create_dataset("indices", data=m.indices)
    g.create_dataset("indptr", data=m.indptr)


def _read_sparse(group):
    """Rebuild the CSC block written by :func:`_write_sparse`."""
    import scipy.sparse as sp  # noqa: PLC0415

    return sp.csc_matrix(
        (group["data"][:], group["indices"][:], group["indptr"][:]),
        shape=tuple(int(v) for v in group.attrs["shape"]),
    )


def _write_band_decomposition(port_group, band) -> None:
    """Store one port's band postprocessing input (DD-230).

    Written as *datasets*, never attributes: the payload is linear in
    the port cross-section (measured 0.26 KiB per free tangential DOF,
    so 146 KiB at 536 DOFs) and passes HDF5's 64 KiB attribute ceiling
    on any production feed.  The mesh-side operators ``M_eps``,
    ``M_mu`` and the 3D curl are deliberately *not* here — they are
    functions of the grid the project already stores, and a reader
    rebuilds them rather than carrying one copy per port.
    """
    ch, pl = band.chain_inward, band.plane
    g = port_group.create_group("band")
    g.attrs["n_modes"] = int(band.n_modes)

    cg = g.create_group("chain")
    cg.attrs["n_t"] = int(ch.n_t)
    cg.attrs["dt"] = float(ch.dt)
    cg.attrs["pairing"] = str(ch.pairing)
    cg.attrs["ez_step"] = int(ch.ez_step)
    # et_step is a scalar stride on z-normal faces and a per-edge array
    # on x-/y-normal ones (the two tangential families are different E
    # components there); the flag keeps the distinction on read.
    et_step = np.asarray(ch.et_step)
    cg.attrs["et_step_scalar"] = bool(et_step.ndim == 0)
    cg.create_dataset("et_step", data=np.atleast_1d(et_step))
    for name in ("D_m1", "D_0", "D_p1"):
        _write_sparse(cg, name, getattr(ch, name))
    for name in ("w_period", "free_u", "free_v", "et_indices", "ez_indices"):
        cg.create_dataset(name, data=np.asarray(getattr(ch, name)))

    pg = g.create_group("plane")
    pg.attrs["face"] = pl.face.value
    pg.attrs["coordinate"] = float(pl.coordinate)
    pg.attrs["normal_dx"] = float(pl.normal_dx)
    for name in ("u_node_window", "v_node_window"):
        pg.attrs[name] = np.asarray(getattr(pl, name), dtype="i8")
    for name in ("u_bounds", "v_bounds"):
        pg.attrs[name] = np.asarray(getattr(pl, name), dtype=float)
    for name in (
        "e_u_indices",
        "h_v_indices",
        "u_edge_uv",
        "u_edge_lengths",
        "e_v_indices",
        "h_u_indices",
        "v_edge_uv",
        "v_edge_lengths",
        "e_u_indices_interior",
        "e_v_indices_interior",
    ):
        pg.create_dataset(name, data=np.asarray(getattr(pl, name)))

    fg = g.create_group("family")
    fg.create_dataset("freqs", data=np.asarray(band.family_freqs, dtype=float))
    fg.create_dataset("zetas", data=np.asarray(band.family_zetas, dtype=complex))

    mg = g.create_group("modes")
    for c in range(int(band.n_modes)):
        cgm = mg.create_group(f"ch{c}")
        cgm.create_dataset("e_u", data=np.asarray(band.e_u_profiles[c]))
        cgm.create_dataset("e_v", data=np.asarray(band.e_v_profiles[c]))
        cgm.create_dataset("h_u", data=np.asarray(band.h_u_profiles[c]))
        cgm.create_dataset("h_v", data=np.asarray(band.h_v_profiles[c]))
        du, dv = band.dual_e_profiles[c]
        cgm.create_dataset("dual_u", data=np.asarray(du))
        cgm.create_dataset("dual_v", data=np.asarray(dv))

    # DD-244: the plane masses in the recorder's convention and the
    # port's curl restriction make the record self-contained — the
    # reader needs no mesh-side operator for the decomposition.
    if band.me_u is not None:
        mass = g.create_group("masses")
        for name in ("me_u", "me_v", "mh_u", "mh_v"):
            mass.create_dataset(name, data=np.asarray(getattr(band, name), dtype=float))
    cs = band.curl_slice
    if cs is not None:
        sg = g.create_group("curl_slice")
        _write_sparse(sg, "c_sub", cs.c_sub)
        sg.create_dataset("mh_rows", data=np.asarray(cs.mh_rows, dtype=float))
        sg.attrs["n_h_u"] = int(cs.n_h_u)
        sg.attrs["n_period"] = int(cs.n_period)
        sg.attrs["n_t"] = int(cs.n_t)
        sg.attrs["n_edges"] = int(cs.n_edges)
        sg.attrs["port_key"] = np.asarray(cs.port_key, dtype="i8")
    if band.g_2d is not None:
        lg = g.create_group("line")
        _write_sparse(lg, "g_2d", band.g_2d)
        lg.attrs["n_signal"] = 0 if band.signal_nodes is None else len(band.signal_nodes)
        for k, nodes in enumerate(band.signal_nodes or []):
            lg.create_dataset(f"signal{k}", data=np.asarray(nodes, dtype="i8"))


def _read_band_decomposition(port_group, label: str):
    """Rebuild the :class:`BandDecomposition` written for one port."""
    from magnelio.mesh import BoxFace  # noqa: PLC0415
    from magnelio.ports._modal.band_dtbc import BandDecomposition  # noqa: PLC0415
    from magnelio.ports._modal.port_plane import PortPlane  # noqa: PLC0415
    from magnelio.ports._modal.zeta_pencil import PeriodChain  # noqa: PLC0415

    g = port_group["band"]
    cg, pg, fg, mg = g["chain"], g["plane"], g["family"], g["modes"]

    et_step = cg["et_step"][:]
    chain = PeriodChain(
        D_m1=_read_sparse(cg["D_m1"]),
        D_0=_read_sparse(cg["D_0"]),
        D_p1=_read_sparse(cg["D_p1"]),
        w_period=cg["w_period"][:],
        n_t=int(cg.attrs["n_t"]),
        free_u=cg["free_u"][:],
        free_v=cg["free_v"][:],
        et_indices=cg["et_indices"][:],
        ez_indices=cg["ez_indices"][:],
        et_step=int(et_step[0]) if bool(cg.attrs["et_step_scalar"]) else et_step,
        ez_step=int(cg.attrs["ez_step"]),
        dt=float(cg.attrs["dt"]),
        pairing=str(cg.attrs["pairing"]),
    )
    plane = PortPlane(
        face=BoxFace(pg.attrs["face"]),
        coordinate=float(pg.attrs["coordinate"]),
        e_u_indices=pg["e_u_indices"][:],
        h_v_indices=pg["h_v_indices"][:],
        u_edge_uv=pg["u_edge_uv"][:],
        u_edge_lengths=pg["u_edge_lengths"][:],
        e_v_indices=pg["e_v_indices"][:],
        h_u_indices=pg["h_u_indices"][:],
        v_edge_uv=pg["v_edge_uv"][:],
        v_edge_lengths=pg["v_edge_lengths"][:],
        e_u_indices_interior=pg["e_u_indices_interior"][:],
        e_v_indices_interior=pg["e_v_indices_interior"][:],
        normal_dx=float(pg.attrs["normal_dx"]),
        u_node_window=tuple(int(v) for v in pg.attrs["u_node_window"]),
        v_node_window=tuple(int(v) for v in pg.attrs["v_node_window"]),
        u_bounds=tuple(float(v) for v in pg.attrs["u_bounds"]),
        v_bounds=tuple(float(v) for v in pg.attrs["v_bounds"]),
    )
    n_modes = int(g.attrs["n_modes"])
    chans = [mg[f"ch{c}"] for c in range(n_modes)]
    masses = {}
    if "masses" in g:
        masses = {name: g["masses"][name][:] for name in ("me_u", "me_v", "mh_u", "mh_v")}
    curl_slice = None
    if "curl_slice" in g:
        from magnelio.ports._modal.zeta_pencil import PortCurlSlice  # noqa: PLC0415

        sg = g["curl_slice"]
        curl_slice = PortCurlSlice(
            c_sub=_read_sparse(sg["c_sub"]).tocsr(),
            mh_rows=sg["mh_rows"][:],
            n_h_u=int(sg.attrs["n_h_u"]),
            n_period=int(sg.attrs["n_period"]),
            n_t=int(sg.attrs["n_t"]),
            n_edges=int(sg.attrs["n_edges"]),
            port_key=tuple(int(v) for v in sg.attrs["port_key"]),
        )
    g_2d = None
    signal_nodes = None
    if "line" in g:
        lg = g["line"]
        g_2d = _read_sparse(lg["g_2d"]).tocsr()
        n_signal = int(lg.attrs["n_signal"])
        signal_nodes = [lg[f"signal{k}"][:] for k in range(n_signal)] if n_signal else None
    return BandDecomposition(
        name=label,
        n_modes=n_modes,
        chain_inward=chain,
        plane=plane,
        family_freqs=fg["freqs"][:],
        family_zetas=fg["zetas"][:],
        e_u_profiles=[c["e_u"][:] for c in chans],
        e_v_profiles=[c["e_v"][:] for c in chans],
        h_u_profiles=[c["h_u"][:] for c in chans],
        h_v_profiles=[c["h_v"][:] for c in chans],
        dual_e_profiles=[(c["dual_u"][:], c["dual_v"][:]) for c in chans],
        curl_slice=curl_slice,
        g_2d=g_2d,
        signal_nodes=signal_nodes,
        **masses,
    )


def _band_s_dict(run: dict, f_axis, mesh_ops: tuple) -> tuple:
    """``(S, z_ref)`` of one stored band run, derived on read (DD-230).

    The band counterpart of :func:`compute_s_parameters` in
    :meth:`Project._s_params`: the stored per-port
    ``BandDecomposition`` records replace the built operators, and the
    mesh-side operators come from the project's own mesh.
    """
    from magnelio.post.modal_sparameters import (  # noqa: PLC0415
        compute_band_s_parameters,
    )

    bands = run.get("port_band") or {}
    if not bands:
        raise ValueError(
            "this run was recorded through the band pipeline but carries "
            "no per-port band data; it predates the band project-store "
            "schema and its S-matrix cannot be re-derived — re-run it",
        )
    m_eps, m_mu, c_3d = mesh_ops
    return compute_band_s_parameters(
        run["signals"],
        list(bands.values()),
        run["excited"],
        f_axis,
        m_eps=m_eps,
        m_mu=m_mu,
        c_3d=c_3d,
        return_reference=True,
        port_reference_scale=run.get("port_reference_scale"),
    )


# ═════════════════════════════════════════════════════════════════════
# Time-domain run results  <->  runs/<name>/results.h5
# ═════════════════════════════════════════════════════════════════════


def _safe_run_name(name: str) -> str:
    """Filesystem-safe run directory name (path separators → ``_``)."""
    out = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    return out or "run"


def _line_params_json(port_line_params: dict, label: str) -> str:
    """Serialise a port's certified line parameters (``None`` → NaN)."""
    lp = {
        str(m): [float(x) if x is not None else float("nan") for x in params]
        for (lbl, m), params in port_line_params.items()
        if lbl == label
    }
    return json.dumps(lp)


class _RunResultWriter:
    """Append-only SWMR writer for one run's ``results.h5`` (DD-070, WP-S4).

    Every time series (per-channel V/I, the reference waveform, and the
    energy trace) is declared as a *resizable* dataset up front, then the
    file is switched into HDF5-SWMR mode so a separate reader process can
    follow the run live.  Two hard SWMR rules shape the schema:

    * No object may be created after the mode switch — so the full group
      tree (channels, ports, energy) and every static attribute (``dt``,
      the excited pair, the per-port mode scalars + de-stagger
      parameters) are written in :meth:`__init__`, before ``swmr_mode``.
    * No attribute may be *edited* after the switch — so the recorded
      step count is never an attribute; it is the current length of the
      ``reference`` stream (the authoritative sample count, appended last
      in every :meth:`append`), and the mutable run ``state`` lives in
      ``project.json``, not here.

    The live streaming sink (:class:`_RunSink`) drives one
    :meth:`append` per solver flush; a finished-in-RAM result would drive
    a single :meth:`append` — one schema, one reader, whichever the
    cadence.

    The per-run resume ``checkpoint.h5`` is a *separate* file
    (overwritten, not appended — see :func:`_write_state_dict_h5`); this
    writer owns only the append streams.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        dt: float,
        excitations: list,
        excited: tuple[str, int] | None,
        f_axis,
        channels: list,
        port_modes: dict,
        port_normal_dx: dict,
        port_line_params: dict,
        port_band: dict | None = None,
        port_reference_scale: dict | None = None,
        monitors=None,
        grid=None,
    ) -> None:
        import h5py  # noqa: PLC0415

        run_dir.mkdir(parents=True, exist_ok=True)
        self._f = f = h5py.File(run_dir / "results.h5", "w", libver="latest")
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["dt"] = float(dt)
        # The run's excitations (DD-224): a JSON list of Excitation
        # dicts (waveform by class-name tag).  A scattering channel run
        # additionally names its excited pair — the S-matrix column.
        f.attrs["excitations"] = json.dumps(list(excitations))
        if excited is not None:
            f.attrs["excited_name"] = excited[0]
            f.attrs["excited_mode"] = int(excited[1])
        if f_axis is not None:
            f.create_dataset("f_axis", data=np.asarray(f_axis, dtype=float))

        self._ref = f.create_dataset(
            "reference",
            shape=(0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(4096,),
        )
        # One sampled drive per excitation (amplitude and delay
        # included), in excitation order; ``reference`` duplicates the
        # first one and stays the committed-count stream.
        xg = f.create_group("excitations")
        self._exc: list = []
        for i, exc in enumerate(excitations):
            g = xg.create_group(f"ex{i}")
            g.attrs["name"] = exc["source"]
            g.attrs["mode"] = int(exc.get("mode", 0))
            ds = g.create_dataset(
                "signal",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=(4096,),
            )
            self._exc.append(ds)
        chg = f.create_group("channels")
        self._chan: list = []  # ordered [((label, mode), V_ds, I_ds), ...]
        for i, (label, mode) in enumerate(channels):
            cg = chg.create_group(f"ch{i}")
            cg.attrs["name"] = label
            cg.attrs["mode"] = int(mode)
            v_ds = cg.create_dataset(
                "V",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=(4096,),
            )
            i_ds = cg.create_dataset(
                "I",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=(4096,),
            )
            self._chan.append(((label, mode), v_ds, i_ds))

        eg = f.create_group("energy")
        self._e_step = eg.create_dataset(
            "step",
            shape=(0,),
            maxshape=(None,),
            dtype="i8",
            chunks=(1024,),
        )
        self._e_time = eg.create_dataset(
            "time",
            shape=(0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(1024,),
        )
        self._e_energy = eg.create_dataset(
            "energy",
            shape=(0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(1024,),
        )

        pg = f.create_group("ports")
        for label, modes in port_modes.items():
            p = pg.create_group(label)
            if label in port_normal_dx:
                p.attrs["normal_dx"] = float(port_normal_dx[label])
            p.attrs["modes"] = json.dumps([_mode_to_dict(m) for m in modes])
            p.attrs["line_params"] = _line_params_json(port_line_params, label)
            if port_reference_scale and label in port_reference_scale:
                p.attrs["reference_scale"] = float(port_reference_scale[label])
            # Band runs decompose per frequency from the port's own
            # chain and profiles, not from the modal line parameters
            # (DD-230); written before the SWMR switch like everything
            # else static.
            if port_band and label in port_band:
                _write_band_decomposition(p, port_band[label])

        # Field-monitor write-through (WP-S9): declare each MonitorFieldTime's
        # resizable streams up front (before SWMR), so a large monitor spills
        # to disk during the run instead of filling RAM.  Stored row-major
        # (n_steps, nz, ny, nx) for a direct ParaView/XDMF reference.
        self._mon: dict = {}
        # Flux monitors stream a tiny per-step scalar time series (like the
        # V/I channels) into an append-only flux/<name>/ group; declared up
        # front, before SWMR (DD-070 follow-up).
        self._flux: dict = {}
        if monitors:
            if grid is not None:
                self._declare_monitors(f, monitors, grid)
            self._declare_flux(f, monitors)

        f.swmr_mode = True
        self._n = 0

    def _declare_monitors(self, f, monitors, grid) -> None:
        """Create the ``monitors/<name>/`` group tree (WP-S9)."""
        from magnelio.monitors.base import (  # noqa: PLC0415
            _corners_array,
            mirrors_to_jsonable,
            resolve_region,
        )
        from magnelio.monitors.field_time import MonitorFieldTime  # noqa: PLC0415

        root = None
        for mon in monitors:
            if not isinstance(mon, MonitorFieldTime):
                continue
            region = resolve_region(mon.corners, grid)
            nx = region.ix.stop - region.ix.start
            ny = region.iy.stop - region.iy.start
            nz = region.iz.stop - region.iz.start
            components = list(mon._components)
            if root is None:
                root = f.create_group("monitors")
            mg = root.create_group(mon.name)
            mg.attrs["type"] = "MonitorFieldTime"
            mg.attrs["corners"] = _corners_array(mon.corners)
            mg.attrs["fields"] = json.dumps(list(mon.fields))
            mg.attrs["components"] = json.dumps(components)
            # Symmetry planes the region touches (DD-154) — the reload
            # reader mirrors plots without mesh access.
            if getattr(mon, "_mirrors", ()):
                mg.attrs["symmetry"] = json.dumps(mirrors_to_jsonable(mon._mirrors))
            mg.attrs["nx"] = nx
            mg.attrs["ny"] = ny
            mg.attrs["nz"] = nz
            mg.create_dataset("grid_x", data=grid.x[region.ix.start : region.ix.stop + 1])
            mg.create_dataset("grid_y", data=grid.y[region.iy.start : region.iy.stop + 1])
            mg.create_dataset("grid_z", data=grid.z[region.iz.start : region.iz.stop + 1])
            times_ds = mg.create_dataset(
                "times",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=(256,),
            )
            comp_ds = {}
            for comp in components:
                comp_ds[comp] = mg.create_dataset(
                    comp,
                    shape=(0, nz, ny, nx),
                    maxshape=(None, nz, ny, nx),
                    dtype="f8",
                    chunks=(1, nz, ny, nx),
                )
            self._mon[mon.name] = {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "comps": comp_ds,
                "times": times_ds,
                "n": 0,
            }

    def _declare_flux(self, f, monitors) -> None:
        """Create the ``flux/<name>/`` group tree (DD-070 follow-up).

        Each MonitorFluxTime streams a scalar Poynting-flux time series
        (time + power), append-only exactly like the V/I channels — no
        grid needed, the plane geometry is recoverable from the corners.
        """
        from magnelio.monitors.flux import MonitorFluxTime  # noqa: PLC0415

        root = None
        for mon in monitors:
            if not isinstance(mon, MonitorFluxTime):
                continue
            if root is None:
                root = f.create_group("flux")
            fg = root.create_group(mon.name)
            fg.attrs["type"] = "MonitorFluxTime"
            fg.attrs["plane_normal"] = mon.normal
            fg.attrs["plane_position"] = float(mon.position)
            times_ds = fg.create_dataset(
                "times",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=(4096,),
            )
            power_ds = fg.create_dataset(
                "power",
                shape=(0,),
                maxshape=(None,),
                dtype="f8",
                chunks=(4096,),
            )
            self._flux[mon.name] = {
                "times": times_ds,
                "power": power_ds,
                "n": 0,
            }

    def append_flux(self, name: str, times, power) -> None:
        """Append a batch of flux (time, power) samples (DD-070 follow-up)."""
        m = self._flux.get(name)
        if m is None or not times:
            return
        k = len(times)
        n0, n1 = m["n"], m["n"] + k
        m["times"].resize((n1,))
        m["times"][n0:n1] = np.asarray(times, dtype=float)
        m["power"].resize((n1,))
        m["power"][n0:n1] = np.asarray(power, dtype=float)
        m["n"] = n1
        self._f.flush()

    def append_monitor(self, name: str, times, comp_data: dict) -> None:
        """Append a batch of field-monitor snapshots (WP-S9).

        ``comp_data[comp]`` is ``(k, *squeezed_region)`` in the monitor's
        native (x, y, z) index order; it is reshaped to the full region
        ``(k, nx, ny, nz)`` and transposed to the row-major
        ``(k, nz, ny, nx)`` XDMF layout before the append.
        """
        m = self._mon.get(name)
        if m is None or not times:
            return
        k = len(times)
        nx, ny, nz = m["nx"], m["ny"], m["nz"]
        n0, n1 = m["n"], m["n"] + k
        for comp, ds in m["comps"].items():
            if comp not in comp_data:
                continue
            full = np.asarray(comp_data[comp], dtype=float).reshape(k, nx, ny, nz)
            ds.resize((n1, nz, ny, nx))
            ds[n0:n1] = full.transpose(0, 3, 2, 1)
        t_ds = m["times"]
        t_ds.resize((n1,))
        t_ds[n0:n1] = np.asarray(times, dtype=float)
        m["n"] = n1
        self._f.flush()

    @property
    def monitor_meta(self) -> list:
        """Per-monitor ``{name, n, nx, ny, nz, components}`` for the XDMF."""
        return [
            {
                "name": name,
                "n": m["n"],
                "nx": m["nx"],
                "ny": m["ny"],
                "nz": m["nz"],
                "components": list(m["comps"].keys()),
            }
            for name, m in self._mon.items()
        ]

    @property
    def n_written(self) -> int:
        """Number of signal samples committed so far."""
        return self._n

    def append(self, v_by_channel: dict, i_by_channel: dict, reference, excitation_blocks=()):
        """Append a block of V/I/excitation samples (equal length) and flush.

        ``reference`` (the first excitation's drive) is grown last, so
        its length is the committed sample count a live reader observes.
        """
        ref = np.asarray(reference, dtype=float)
        length = ref.shape[0]
        if length == 0:
            return
        n0, n1 = self._n, self._n + length
        for key, v_ds, i_ds in self._chan:
            v_ds.resize((n1,))
            v_ds[n0:n1] = np.asarray(v_by_channel[key], dtype=float)
            i_ds.resize((n1,))
            i_ds[n0:n1] = np.asarray(i_by_channel[key], dtype=float)
        for ds, blk in zip(self._exc, excitation_blocks):
            ds.resize((n1,))
            ds[n0:n1] = np.asarray(blk, dtype=float)
        self._ref.resize((n1,))
        self._ref[n0:n1] = ref
        self._n = n1
        self._f.flush()

    def append_energy(self, step: int, time: float, energy: float) -> None:
        """Append one energy-trace sample and flush."""
        m = self._e_step.shape[0]
        self._e_step.resize((m + 1,))
        self._e_step[m] = int(step)
        self._e_time.resize((m + 1,))
        self._e_time[m] = float(time)
        self._e_energy.resize((m + 1,))
        self._e_energy[m] = float(energy)
        self._f.flush()

    def close(self) -> None:
        """Flush and close the file (the run state lives in project.json)."""
        self._f.close()

    @classmethod
    def reopen(
        cls,
        run_dir: Path,
        *,
        n_keep: int,
        monitor_keep: dict | None = None,
        flux_keep: dict | None = None,
    ) -> "_RunResultWriter":
        """Reopen an existing ``results.h5`` to append a resumed tail (WP-S8).

        Truncates every append stream back to the checkpoint's committed
        step count ``n_keep`` — a hard kill between two checkpoints can
        leave the streams flushed a little past the last checkpoint, and a
        bit-exact resume must continue from the checkpoint, not from that
        orphaned tail.  The reopen mirrors the fresh writer's lifecycle:
        the resize-down happens in plain read-write mode, *then* the file
        re-enters SWMR (which forbids shrinking), positioned at ``n_keep``
        so the sink appends the resumed samples in place.
        """
        import h5py  # noqa: PLC0415

        self = cls.__new__(cls)
        self._f = f = h5py.File(run_dir / "results.h5", "a", libver="latest")
        self._ref = f["reference"]
        self._ref.resize((n_keep,))
        self._exc = []
        if "excitations" in f:
            xg = f["excitations"]
            for i in range(len(xg)):
                ds = xg[f"ex{i}"]["signal"]
                ds.resize((n_keep,))
                self._exc.append(ds)
        self._chan = []
        chg = f["channels"]
        for name in chg:
            cg = chg[name]
            key = (str(cg.attrs["name"]), int(cg.attrs["mode"]))
            v_ds, i_ds = cg["V"], cg["I"]
            v_ds.resize((n_keep,))
            i_ds.resize((n_keep,))
            self._chan.append((key, v_ds, i_ds))
        # Energy trace: keep only the samples strictly before the resume
        # step — the resumed march re-samples from its own next check
        # boundary (>= n_keep), so no sample is duplicated or lost.
        eg = f["energy"]
        self._e_step = eg["step"]
        self._e_time = eg["time"]
        self._e_energy = eg["energy"]
        m = min(self._e_step.shape[0], self._e_time.shape[0], self._e_energy.shape[0])
        keep = int(np.count_nonzero(np.asarray(self._e_step[:m]) < n_keep))
        self._e_step.resize((keep,))
        self._e_time.resize((keep,))
        self._e_energy.resize((keep,))
        # Field monitors (WP-S9): truncate each stream to its checkpointed
        # snapshot count (the monitor's _next_idx at the checkpoint), so the
        # resumed run appends onward from a consistent point.
        self._mon = {}
        monitor_keep = monitor_keep or {}
        if "monitors" in f:
            for name in f["monitors"]:
                mg = f["monitors"][name]
                if mg.attrs.get("type") != "MonitorFieldTime":
                    continue
                nx, ny, nz = (int(mg.attrs["nx"]), int(mg.attrs["ny"]), int(mg.attrs["nz"]))
                mk = int(monitor_keep.get(name, 0))
                comp_ds = {}
                for comp in json.loads(mg.attrs["components"]):
                    ds = mg[comp]
                    ds.resize((mk, nz, ny, nx))
                    comp_ds[comp] = ds
                mg["times"].resize((mk,))
                self._mon[name] = {
                    "nx": nx,
                    "ny": ny,
                    "nz": nz,
                    "comps": comp_ds,
                    "times": mg["times"],
                    "n": mk,
                }
        # Flux monitors (DD-070 follow-up): truncate each scalar stream to
        # its checkpointed sample count, same rationale as the field monitors.
        self._flux = {}
        flux_keep = flux_keep or {}
        if "flux" in f:
            for name in f["flux"]:
                fg = f["flux"][name]
                if fg.attrs.get("type") != "MonitorFluxTime":
                    continue
                fk = int(flux_keep.get(name, 0))
                fg["times"].resize((fk,))
                fg["power"].resize((fk,))
                self._flux[name] = {
                    "times": fg["times"],
                    "power": fg["power"],
                    "n": fk,
                }
        f.swmr_mode = True
        self._n = int(n_keep)
        return self


class _RunSink:
    """Live streaming sink attached to the FIT-TD solver (DD-070, WP-S5).

    Owns one run's :class:`_RunResultWriter` and, at every solver flush,
    pulls the newly recorded V/I tail from the
    :class:`~magnelio.ports.recorder.PortSignalRecorder`, sampling the
    excitation drives in lockstep so a separate reader process can
    derive converging S-parameters mid-run.  The solver calls
    :meth:`flush` at each energy-check interval and once at exit; the
    analysis calls :meth:`close` to flip the run to ``done`` in
    ``project.json``.
    """

    def __init__(
        self,
        writer,
        recorder,
        excitation_fns,
        dt,
        *,
        store,
        run_name,
        checkpoint_path=None,
        step_offset=0,
        field_time_monitors=(),
        flux_monitors=(),
        freq_monitors=(),
        wall_loss_monitors=(),
        far_field_monitors=(),
        grid=None,
    ) -> None:
        self._writer = writer
        # ``[(key, fn), ...]`` — one drive per excitation, the first
        # doubling as the run's ``reference`` stream.
        self._recorder = recorder
        self._excitation_fns = [fn for _, fn in excitation_fns]
        self._dt = float(dt)
        self._store = store
        self._run_name = run_name
        self._n_flushed = 0
        # Field monitors drained to disk at each flush (WP-S9): the run
        # sink pulls each MonitorFieldTime's pending snapshots and appends
        # them, so the monitor never accumulates the whole run in RAM.
        self._field_time_monitors = list(field_time_monitors)
        self._flux_monitors = list(flux_monitors)
        # Frequency monitors dump their DFT accumulator to fields_freq.h5
        # (whole-file overwrite) alongside each checkpoint (DD-070 follow-up).
        self._freq_monitors = list(freq_monitors)
        # Wall-loss monitors dump the same way (DD-082 addendum): running
        # DFT accumulators, whole-file overwrite, own result file.
        self._wall_loss_monitors = list(wall_loss_monitors)
        # Far-field monitors: same fixed-size running-DFT shape (DD-173).
        self._far_field_monitors = list(far_field_monitors)
        self._grid = grid
        # Global-time origin of the fresh recorder (DD-070, WP-S8): on a
        # resume the recorder restarts at local index 0 but corresponds to
        # global step ``step_offset``, so the reference waveform must be
        # sampled at ``(step_offset + local_k)·dt`` to stay phase-aligned
        # with the pre-resume stream.  Zero on a first run.
        self._step_offset = int(step_offset)
        # Resume-checkpoint plumbing (DD-070, WP-S7); inert until the
        # solver-side state provider is attached via enable_checkpoints.
        self._checkpoint_path = checkpoint_path
        self._freq_path = (
            checkpoint_path.parent / "fields_freq.h5" if checkpoint_path is not None else None
        )
        self._wall_loss_path = (
            checkpoint_path.parent / "wall_loss.h5" if checkpoint_path is not None else None
        )
        self._far_field_path = (
            checkpoint_path.parent / "far_field.h5" if checkpoint_path is not None else None
        )
        self._checkpoint_fn = None
        self._checkpoint_interval = 1
        self._last_ckpt_step = -1
        # On-demand checkpoint (DD-070 follow-up): a SIGUSR1 handler sets
        # this flag; the next flush writes a checkpoint out of the periodic
        # schedule *without* stopping the march, then clears it.
        self._checkpoint_requested = False

    def enable_checkpoints(self, checkpoint_fn, interval: int) -> None:
        """Attach a state provider so the run writes resume checkpoints.

        Parameters
        ----------
        checkpoint_fn : callable
            Zero-arg callable returning the solver ``state_dict`` (a
            nested dict of arrays/scalars); typically
            ``solver.state_dict``.
        interval : int
            Minimum number of leapfrog steps between periodic
            checkpoints.  A checkpoint is written at the first flush that
            reaches this stride, and a final one on a ``done`` close /
            graceful abort regardless of the stride.
        """
        self._checkpoint_fn = checkpoint_fn
        self._checkpoint_interval = max(1, int(interval))

    def request_checkpoint(self) -> None:
        """Request an out-of-schedule checkpoint at the next flush.

        Signal-safe (sets a single flag): a ``SIGUSR1`` handler calls this
        so a long run can be snapshotted *on demand* — e.g. before a
        maintenance window — without stopping the march.  The checkpoint is
        written at the next energy-check flush (a consistent leapfrog pair),
        so it is exactly as resumable as a periodic one.  A no-op if no
        state provider is attached (streaming without resume checkpoints).
        """
        self._checkpoint_requested = True

    def flush(self, energy: tuple | None = None) -> None:
        """Append the newly recorded V/I tail (+ optional energy sample).

        Parameters
        ----------
        energy : (int, float, float), optional
            ``(step, time, energy)`` sample to append to the energy
            trace.  ``None`` on the final exit flush, which only drains
            the V/I tail past the last energy check.  When present, its
            step also drives the periodic resume checkpoint.
        """
        n_rec = self._recorder.n_steps_recorded
        if n_rec > self._n_flushed:
            tail = self._recorder.tail(self._n_flushed)
            off = self._step_offset
            blocks = [
                np.array(
                    [fn((off + k) * self._dt) for k in range(self._n_flushed, n_rec)],
                    dtype=float,
                )
                for fn in self._excitation_fns
            ]
            self._writer.append(
                {key: vi[0] for key, vi in tail.items()},
                {key: vi[1] for key, vi in tail.items()},
                blocks[0],
                blocks,
            )
            self._n_flushed = n_rec
        # Drain field monitors *before* the checkpoint, so the checkpoint's
        # per-monitor cursor equals the on-disk snapshot count (WP-S9).
        for mon in self._field_time_monitors:
            times, comp_data = mon.pop_pending()
            if times:
                self._writer.append_monitor(mon.name, times, comp_data)
        for mon in self._flux_monitors:
            times, power = mon.pop_pending()
            if times:
                self._writer.append_flux(mon.name, times, power)
        if energy is not None:
            self._writer.append_energy(*energy)
            self._maybe_checkpoint(int(energy[0]))

    def _maybe_checkpoint(self, step: int) -> None:
        """Write a checkpoint if the periodic stride has been reached.

        Called from :meth:`flush` *after* the V/I tail has been appended,
        so ``results.h5`` and the checkpoint agree on the committed step
        count — the checkpoint's ``n_completed`` equals the stream length
        (both reflect the same consistent leapfrog pair).
        """
        if self._checkpoint_fn is None:
            return
        forced = self._checkpoint_requested
        if forced or step - self._last_ckpt_step >= self._checkpoint_interval:
            self.write_checkpoint(step)
            if forced:
                self._checkpoint_requested = False
                # The solver's reporter is current during a flush; its
                # note replaces the live status line cleanly and stays
                # silent when the run is.  ``step + 1`` == the
                # checkpoint's n_completed (the flush ran after
                # _resume_step advanced).
                rep = current_reporter()
                if rep is not None:
                    rep.note(
                        f"checkpoint written on request at step {step + 1} "
                        f"({self._checkpoint_path})",
                    )

    def write_checkpoint(self, step: int | None = None) -> None:
        """Force-write the current solver state to ``checkpoint.h5``.

        A no-op when no state provider is attached (streaming without
        resume).  Used for the periodic writes, the final ``done``
        checkpoint (enables run-longer), and the graceful-abort
        checkpoint.
        """
        if self._checkpoint_fn is None or self._checkpoint_path is None:
            return
        state = self._checkpoint_fn()
        _write_state_dict_h5(self._checkpoint_path, state)
        # Frequency result: written after checkpoint.h5 and tagged with the
        # same n_completed, so a crash between the two leaves fields_freq.h5
        # older — the resume step-check then rejects the stale accumulator
        # instead of integrating on from a wrong partial DFT.
        if self._freq_monitors and self._freq_path is not None:
            dumps = {m.name: m.result_dump() for m in self._freq_monitors}
            _write_freq_result_h5(
                self._freq_path,
                dumps,
                int(state["n_completed"]),
            )
        # Wall-loss results: same ordering argument as the frequency dump
        # above — written after checkpoint.h5 and tagged with the same
        # n_completed, so a crash between the two leaves wall_loss.h5
        # older and the resume step-check rejects it.
        if self._wall_loss_monitors and self._wall_loss_path is not None:
            dumps = {m.name: m.result_dump() for m in self._wall_loss_monitors}
            _write_wall_loss_h5(
                self._wall_loss_path,
                dumps,
                int(state["n_completed"]),
            )
        # Far-field results: same ordering as the dumps above.
        if self._far_field_monitors and self._far_field_path is not None:
            dumps = {m.name: m.result_dump() for m in self._far_field_monitors}
            _write_far_field_h5(
                self._far_field_path,
                dumps,
                int(state["n_completed"]),
            )
        if step is not None:
            self._last_ckpt_step = step

    def close(
        self,
        state: str = "done",
        stop_reason: str | None = None,
        final_port_signal_db: float | None = None,
        elapsed: float | None = None,
    ) -> None:
        """Drain the final tail, close the file, finalise ``project.json``.

        A ``done`` close writes a final checkpoint so the completed run
        can be resumed to run longer (DD-070).  An ``aborted`` close does
        not — the graceful-abort checkpoint was already written at the
        consistent break point by the solver.  ``stop_reason`` /
        ``final_port_signal_db`` book why the run ended (and the |V|
        envelope level below peak it reached) into the run index;
        ``elapsed`` books the march's wall time.
        """
        self.flush()
        if state == "done":
            self.write_checkpoint()
        self._write_xdmf()
        n_written = self._writer.n_written
        self._writer.close()
        self._store._finalize_run(
            self._run_name,
            n_written,
            state,
            stop_reason=stop_reason,
            final_port_signal_db=final_port_signal_db,
            elapsed=elapsed,
        )
        self._export_paraview()

    def _export_paraview(self) -> None:
        """Generate the run's ParaView session (DD-115), best-effort.

        Runs after the writer is closed and the run finalised, so the
        exporter reads a consistent ``results.h5`` / ``fields_freq.h5``.
        Visualization-only: any failure is downgraded to a warning —
        it must never invalidate a completed run.
        """
        try:
            from magnelio.io.paraview import export_run_visualization  # noqa: PLC0415

            export_run_visualization(self._store.path, self._run_name)
        except Exception as exc:  # viz-only; the run itself is complete
            import warnings  # noqa: PLC0415

            warnings.warn(
                f"ParaView session export failed (visualization only): {exc}",
                UserWarning,
                stacklevel=2,
            )

    def _write_xdmf(self) -> None:
        """Write ``fields.xdmf`` for the run's field monitors (WP-S9).

        Written once at close, when the recorded snapshot count is final
        (an early energy stop or an abort records fewer than the requested
        times).  A no-op when the run carries no field monitors.
        """
        meta = self._writer.monitor_meta
        specs = []
        for m in meta:
            if m["n"] == 0:
                continue
            times = np.asarray(self._writer._mon[m["name"]]["times"][: m["n"]], dtype=float)
            specs.append({**m, "times": times})
        if not specs:
            return
        from magnelio.io.xdmf import write_run_xdmf  # noqa: PLC0415

        run_dir = self._store.path / "runs" / self._run_name
        write_run_xdmf(run_dir / "fields.xdmf", "results.h5", specs)


_ENERGY_DTYPE = [("step", int), ("time", float), ("energy", float)]


def _read_energy_group(eg) -> np.ndarray:
    """The ``energy/{step,time,energy}`` streams as one structured array.

    Same common-prefix rule as the signals: the three sub-streams
    advance in lockstep but a live reader may see them flushed to
    different lengths, so they are sliced to their minimum.
    """
    m = min(int(eg["step"].shape[0]), int(eg["time"].shape[0]), int(eg["energy"].shape[0]))
    trace = np.empty(m, dtype=_ENERGY_DTYPE)
    if m > 0:
        trace["step"] = eg["step"][:m]
        trace["time"] = eg["time"][:m]
        trace["energy"] = eg["energy"][:m]
    return trace


def _read_energy_trace(run_dir: Path) -> np.ndarray:
    """A run's energy trace alone — the light read a live table wants.

    Opens ``results.h5`` in SWMR mode and touches only the three energy
    streams.  A run without a results file yet, or one the writer is
    still creating, reads as an empty trace rather than an error.
    """
    import h5py  # noqa: PLC0415

    path = run_dir / "results.h5"
    if not path.exists():
        return np.empty(0, dtype=_ENERGY_DTYPE)
    try:
        with h5py.File(path, "r", swmr=True) as f:
            return _read_energy_group(f["energy"])
    except (OSError, KeyError):
        return np.empty(0, dtype=_ENERGY_DTYPE)


def _count_energy_samples(run_dir: Path) -> int:
    """How many energy samples a run has flushed — one dataset shape, no data."""
    import h5py  # noqa: PLC0415

    path = run_dir / "results.h5"
    if not path.exists():
        return 0
    try:
        with h5py.File(path, "r", swmr=True) as f:
            return int(f["energy"]["step"].shape[0])
    except (OSError, KeyError):
        return 0


def _last_energy_step(run_dir: Path) -> int | None:
    """The step of the latest energy sample a run has flushed, or ``None``."""
    import h5py  # noqa: PLC0415

    path = run_dir / "results.h5"
    if not path.exists():
        return None
    try:
        with h5py.File(path, "r", swmr=True) as f:
            steps = f["energy"]["step"]
            return int(steps[-1]) if steps.shape[0] else None
    except (OSError, KeyError, IndexError):
        return None


def _read_run_results(run_dir: Path) -> dict:
    """Read a run written by :class:`_RunResultWriter` into a plain dict.

    Opens the file in SWMR mode so an in-progress (live) run can be read
    concurrently (WP-S4).  A live reader may catch the streams flushed to
    slightly different lengths — HDF5 guarantees per-dataset consistency,
    not that every stream shows the same append count — so the recorded
    step count is the **common prefix** length across the reference and
    every V/I channel, and all series are sliced to it.  That keeps a
    partial mid-flush read self-consistent (no ragged V vs reference).
    """
    import h5py  # noqa: PLC0415

    from magnelio.signals.signal_1d import Signal1D  # noqa: PLC0415

    with h5py.File(run_dir / "results.h5", "r", swmr=True) as f:
        validate_schema(
            f.attrs.get("schema_version"),
            str(run_dir / "results.h5"),
        )
        dt = float(f.attrs["dt"])
        excited = (
            (str(f.attrs["excited_name"]), int(f.attrs["excited_mode"]))
            if "excited_name" in f.attrs
            else None
        )
        excitations = json.loads(f.attrs["excitations"]) if "excitations" in f.attrs else []
        f_axis = f["f_axis"][()] if "f_axis" in f else None

        chan_items = []  # (label, mode, V_ds, I_ds)
        lengths = [int(f["reference"].shape[0])]
        exc_items = []  # (key, ds)
        if "excitations" in f:
            xg = f["excitations"]
            for i in range(len(xg)):
                g = xg[f"ex{i}"]
                ds = g["signal"]
                lengths.append(int(ds.shape[0]))
                exc_items.append(((str(g.attrs["name"]), int(g.attrs["mode"])), ds))
        for name in f["channels"]:
            cg = f["channels"][name]
            v_ds, i_ds = cg["V"], cg["I"]
            lengths.append(int(v_ds.shape[0]))
            lengths.append(int(i_ds.shape[0]))
            chan_items.append(
                (str(cg.attrs["name"]), int(cg.attrs["mode"]), v_ds, i_ds),
            )
        n_steps = min(lengths)  # common prefix across all streams
        t = np.arange(n_steps) * dt
        reference = Signal1D(
            t=t,
            values=f["reference"][:n_steps],
            dt=dt,
            label="excitation",
        )

        signals = {}
        for label, mode, v_ds, i_ds in chan_items:
            v = Signal1D(t=t, values=v_ds[:n_steps], dt=dt, label=f"{label}_mode{mode}_V")
            i = Signal1D(t=t, values=i_ds[:n_steps], dt=dt, label=f"{label}_mode{mode}_I")
            signals[(label, mode)] = (v, i)
        single = len(exc_items) == 1
        excitation_signals = {
            key: Signal1D(
                t=t,
                values=ds[:n_steps],
                dt=dt,
                label="excitation" if single else f"excitation{key}",
            )
            for key, ds in exc_items
        }

        energy_trace = _read_energy_group(f["energy"])

        port_modes = {}
        port_normal_dx = {}
        port_line_params = {}
        port_band = {}
        port_reference_scale = {}
        for label in f["ports"]:
            p = f["ports"][label]
            port_modes[label] = [_mode_from_dict(d) for d in json.loads(p.attrs["modes"])]
            if "band" in p:
                port_band[label] = _read_band_decomposition(p, label)
            if "reference_scale" in p.attrs:
                port_reference_scale[label] = float(p.attrs["reference_scale"])
            if "normal_dx" in p.attrs:
                port_normal_dx[label] = float(p.attrs["normal_dx"])
            for m_str, params in json.loads(p.attrs["line_params"]).items():
                r, q, z0 = params
                port_line_params[(label, int(m_str))] = (
                    r,
                    q,
                    None if (z0 != z0) else z0,  # nan -> None
                )

    return dict(
        excited=excited,
        excitations=excitations,
        excitation_signals=excitation_signals,
        signals=signals,
        reference=reference,
        dt=dt,
        n_steps=n_steps,
        energy_trace=energy_trace,
        port_modes=port_modes,
        port_normal_dx=port_normal_dx,
        port_line_params=port_line_params,
        port_band=port_band,
        port_reference_scale=port_reference_scale,
        f_axis=f_axis,
    )


# ═════════════════════════════════════════════════════════════════════
# Field monitors  <->  runs/<name>/results.h5  (monitors/<name>/…)
# ═════════════════════════════════════════════════════════════════════


def _field_time_monitors(monitors) -> list:
    """The ``MonitorFieldTime`` subset of a monitor list (WP-S9)."""
    if not monitors:
        return []
    from magnelio.monitors.field_time import MonitorFieldTime  # noqa: PLC0415

    return [m for m in monitors if isinstance(m, MonitorFieldTime)]


def _flux_monitors(monitors) -> list:
    """The ``MonitorFluxTime`` subset of a monitor list (DD-070 follow-up)."""
    if not monitors:
        return []
    from magnelio.monitors.flux import MonitorFluxTime  # noqa: PLC0415

    return [m for m in monitors if isinstance(m, MonitorFluxTime)]


def _freq_monitors(monitors) -> list:
    """The ``MonitorFieldFrequency`` subset of a list (DD-070 follow-up)."""
    if not monitors:
        return []
    from magnelio.monitors.field_frequency import (  # noqa: PLC0415
        MonitorFieldFrequency,
    )

    return [m for m in monitors if isinstance(m, MonitorFieldFrequency)]


def _far_field_monitors(monitors) -> list:
    """The ``MonitorFarFieldFrequency`` subset of a list (DD-173)."""
    if not monitors:
        return []
    from magnelio.monitors.far_field import MonitorFarFieldFrequency  # noqa: PLC0415

    return [m for m in monitors if isinstance(m, MonitorFarFieldFrequency)]


def _wall_loss_monitors(monitors) -> list:
    """The ``MonitorWallLoss`` subset of a list (DD-082 addendum)."""
    if not monitors:
        return []
    from magnelio.monitors.wall_loss import MonitorWallLoss  # noqa: PLC0415

    return [m for m in monitors if isinstance(m, MonitorWallLoss)]


def _list_run_monitors(run_dir: Path) -> list[str]:
    """Names of the field-time monitors streamed into a run, or ``[]``."""
    import h5py  # noqa: PLC0415

    results = run_dir / "results.h5"
    if not results.exists():
        return []
    with h5py.File(results, "r", swmr=True) as f:
        if "monitors" not in f:
            return []
        return [
            name
            for name in f["monitors"]
            if f["monitors"][name].attrs.get("type") == "MonitorFieldTime"
        ]


def _list_run_flux(run_dir: Path) -> list[str]:
    """Names of the flux monitors streamed into a run, or ``[]``."""
    import h5py  # noqa: PLC0415

    results = run_dir / "results.h5"
    if not results.exists():
        return []
    with h5py.File(results, "r", swmr=True) as f:
        if "flux" not in f:
            return []
        return [
            name for name in f["flux"] if f["flux"][name].attrs.get("type") == "MonitorFluxTime"
        ]


def _list_run_freq(run_dir: Path) -> list[str]:
    """Names of the frequency monitors persisted for a run, or ``[]``."""
    import h5py  # noqa: PLC0415

    ff = run_dir / "fields_freq.h5"
    if not ff.exists():
        return []
    with h5py.File(ff, "r") as f:
        return [name for name in f if isinstance(f[name], h5py.Group)]


class _LoadedFieldMonitor:
    """Lazy reader over one streamed ``MonitorFieldTime`` (DD-070, WP-S9).

    The monitor's snapshots live in the run's ``results.h5`` (row-major
    ``(n_steps, nz, ny, nx)``); this reader loads the small time axis and
    metadata eagerly and each component array **on demand**, transposing
    back to the monitor's native ``(n_steps, nx, ny, nz)`` order and
    squeezing singleton spatial axes — so ``.data`` / ``.component`` match
    an in-RAM monitor's exactly, without holding the whole record.
    ``.plot(...)`` hydrates a real :class:`MonitorFieldTime` to reuse the
    full plotting machinery.
    """

    def __init__(self, run_dir: Path, name: str, grid=None) -> None:
        import h5py  # noqa: PLC0415

        from magnelio.monitors.base import _corners_from_array  # noqa: PLC0415

        self._run_dir = Path(run_dir)
        self.name = name
        self._grid = grid
        with h5py.File(self._run_dir / "results.h5", "r", swmr=True) as f:
            mg = f["monitors"][name]
            self.corners = _corners_from_array(mg.attrs["corners"])
            self.fields = json.loads(mg.attrs["fields"])
            self._components = json.loads(mg.attrs["components"])
            self._nx = int(mg.attrs["nx"])
            self._ny = int(mg.attrs["ny"])
            self._nz = int(mg.attrs["nz"])
            self._t = mg["times"][()]
            sym_attr = mg.attrs.get("symmetry")
        from magnelio.monitors.base import mirrors_from_jsonable  # noqa: PLC0415

        self._mirrors = mirrors_from_jsonable(
            json.loads(sym_attr) if sym_attr is not None else None,
        )
        self._n = int(self._t.shape[0])

    @property
    def t(self) -> np.ndarray:
        """Recorded time points [s]."""
        return np.asarray(self._t, dtype=float)

    @property
    def components(self) -> list[str]:
        return list(self._components)

    def component(self, comp: str) -> np.ndarray:
        """Recorded data for one component, shape ``(n_times, <spatial>)``."""
        import h5py  # noqa: PLC0415

        if comp not in self._components:
            raise KeyError(
                f"component {comp!r} not recorded; available: {self._components}",
            )
        with h5py.File(self._run_dir / "results.h5", "r", swmr=True) as f:
            arr = f["monitors"][self.name][comp][: self._n]  # (n,nz,ny,nx)
        return np.squeeze(np.transpose(arr, (0, 3, 2, 1)))  # ->(n,nx,ny,nz)

    @property
    def data(self) -> dict[str, np.ndarray]:
        """All recorded components stacked along a leading time axis."""
        return {c: self.component(c) for c in self._components}

    def _hydrate(self):
        """Build an in-RAM :class:`MonitorFieldTime` for plotting reuse."""
        from magnelio.monitors.base import resolve_region  # noqa: PLC0415
        from magnelio.monitors.field_time import MonitorFieldTime  # noqa: PLC0415

        times = self._t if self._n > 0 else np.array([0.0])
        mon = MonitorFieldTime(
            corners=self.corners,
            times=times,
            fields=list(self.fields),
            name=self.name,
        )
        data = {c: self.component(c) for c in self._components}
        mon._recorded_times = [float(x) for x in self._t]
        mon._next_idx = self._n
        mon._snapshots = [
            {c: np.asarray(data[c][ti]) for c in self._components} for ti in range(self._n)
        ]
        if self._grid is not None:
            mon._region = resolve_region(self.corners, self._grid)
            mon._grid = self._grid
        mon._mirrors = self._mirrors
        return mon

    def plot(self, *args, **kwargs):
        """Plot the recorded field (delegates to :class:`MonitorFieldTime`)."""
        return self._hydrate().plot(*args, **kwargs)

    def interact(self, *args, **kwargs):
        """Interactive time-step slider (delegates to :class:`MonitorFieldTime`)."""
        return self._hydrate().interact(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"_LoadedFieldMonitor(name={self.name!r}, n_times={self._n}, "
            f"region=({self._nx}, {self._ny}, {self._nz}))"
        )


class _LoadedFluxMonitor:
    """Lazy reader over one streamed ``MonitorFluxTime`` (DD-070 follow-up).

    The scalar flux time series (time + power) lives in the run's
    ``results.h5`` (``flux/<name>/``).  It is small, so both axes load
    eagerly; ``.t`` / ``.power`` / ``.total_energy`` match an in-RAM
    :class:`MonitorFluxTime` exactly, and ``.plot()`` hydrates one to reuse
    its plotting machinery.
    """

    def __init__(self, run_dir: Path, name: str) -> None:
        import h5py  # noqa: PLC0415

        self._run_dir = Path(run_dir)
        self.name = name
        with h5py.File(self._run_dir / "results.h5", "r", swmr=True) as f:
            fg = f["flux"][name]
            self.normal = str(fg.attrs["plane_normal"])
            self.position = float(fg.attrs["plane_position"])
            self._t = fg["times"][()]
            self._power = fg["power"][()]

    @property
    def t(self) -> np.ndarray:
        """Time axis [s]."""
        return np.asarray(self._t, dtype=float)

    @property
    def power(self) -> np.ndarray:
        """Instantaneous Poynting flux [W] vs. time."""
        return np.asarray(self._power, dtype=float)

    @property
    def total_energy(self) -> float:
        """Time-integrated Poynting energy [J]."""
        t, p = self.t, self.power
        if len(t) < 2:
            return 0.0
        return float(np.trapezoid(p, t))

    def _hydrate(self):
        """Build an in-RAM :class:`MonitorFluxTime` for plotting reuse."""
        from magnelio.monitors.flux import MonitorFluxTime  # noqa: PLC0415

        mon = MonitorFluxTime(normal=self.normal, position=self.position, name=self.name)
        mon._times = [float(x) for x in self._t]
        mon._power = [float(x) for x in self._power]
        return mon

    def plot(self, *args, **kwargs):
        """Plot the recorded flux (delegates to :class:`MonitorFluxTime`)."""
        return self._hydrate().plot(*args, **kwargs)

    def __repr__(self) -> str:
        return f"_LoadedFluxMonitor(name={self.name!r}, n_steps={len(self._t)})"


class _LoadedFreqMonitor:
    """Lazy reader over one persisted ``MonitorFieldFrequency`` (DD-070).

    The DFT result lives in the run's ``fields_freq.h5`` (its own whole-file
    result, overwritten at each checkpoint — during a run it is a converging
    *partial* DFT, exactly like the streaming S-parameters).  Metadata loads
    eagerly, each component's complex bins on demand.

    ``.data`` / ``.component`` divide by the spectrum of the run's stored
    excitation and are therefore fields per 1 W CW, matching the in-RAM
    monitor; ``.data_raw`` returns the undivided bins.  ``.plot()``
    hydrates a real :class:`MonitorFieldFrequency`, reference included, to
    reuse its plotting machinery.
    """

    def __init__(self, run_dir: Path, name: str, reference=None, grid=None, incident=None) -> None:
        import h5py  # noqa: PLC0415

        from magnelio.monitors.base import _corners_from_array  # noqa: PLC0415

        self._run_dir = Path(run_dir)
        self.name = name
        # Only for plotting: the cell layer a field plane stands for, so
        # wires and discrete ports show up in a loaded plot as they do
        # in a live one.
        self._grid = grid
        # Signal1D, or a callable returning one — the reader stays lazy so
        # listing a project's monitors costs no run-results read.
        self._reference = reference
        self._incident = incident
        self._spectrum = None
        self._incident_ratio = None
        self._incident_loaded = False
        with h5py.File(self._run_dir / "fields_freq.h5", "r") as f:
            g = f[name]
            self._components = json.loads(g.attrs["components"])
            self.fields = json.loads(g.attrs["fields"])
            # Schema-additive (DD-140); absent means every step.
            self.interval = float(g.attrs["interval"]) if "interval" in g.attrs else None
            self.freqs = g["freqs"][()]
            self.corners = _corners_from_array(g["corners"][()])
            self._grid_x = g["grid_x"][()]
            self._grid_y = g["grid_y"][()]
            self._grid_z = g["grid_z"][()]
            sym_attr = g.attrs.get("symmetry")
        from magnelio.monitors.base import mirrors_from_jsonable  # noqa: PLC0415

        self._mirrors = mirrors_from_jsonable(
            json.loads(sym_attr) if sym_attr is not None else None,
        )

    @property
    def f(self) -> np.ndarray:
        """Frequency array [Hz]."""
        return np.asarray(self.freqs, dtype=float)

    @property
    def components(self) -> list[str]:
        return list(self._components)

    @staticmethod
    def _squeeze_spatial(arr: np.ndarray) -> np.ndarray:
        """Squeeze length-1 spatial axes, keep the frequency axis 0."""
        squeeze = tuple(ax for ax in range(1, arr.ndim) if arr.shape[ax] == 1)
        return np.squeeze(arr, axis=squeeze) if squeeze else arr

    def _source_spectrum(self) -> np.ndarray | None:
        """Spectrum of the run's excitation, in the accumulator convention.

        Sampled lazily from the run's stored reference waveform (the read
        is cached for a finished run) and memoised here, so a caller who
        only wants ``data_raw`` never pays for it.
        """
        from magnelio.monitors._dft import source_spectrum  # noqa: PLC0415

        if self._spectrum is None and self._reference is not None:
            sig = self._reference() if callable(self._reference) else self._reference
            if sig is not None:
                self._spectrum = source_spectrum(sig.values, sig.dt, self.freqs)
        return self._spectrum

    def _read_bins(self, comp: str) -> np.ndarray:
        import h5py  # noqa: PLC0415

        if comp not in self._components:
            raise KeyError(f"component {comp!r} not recorded; available: {self._components}")
        with h5py.File(self._run_dir / "fields_freq.h5", "r") as f:
            return f[self.name]["bins"][comp][()]

    def component(self, comp: str) -> np.ndarray:
        """One component per 1 W CW, shape ``(n_freqs, <spatial>)``."""
        from magnelio.monitors._dft import divide_by_spectrum  # noqa: PLC0415

        src = self._source_spectrum()
        if src is None:
            raise RuntimeError(
                f"monitor {self.name!r}: this run stores no excitation "
                f"reference, so its DFT bins cannot be expressed as fields "
                f"per 1 W CW.  Read .data_raw for the raw bins."
            )
        out = divide_by_spectrum(self._read_bins(comp), src)
        ratio = self._incident_amplitude()
        if ratio is not None:
            out = out / np.asarray(ratio, dtype=float).reshape(-1, *([1] * (out.ndim - 1)))
        return self._squeeze_spatial(out)

    def _incident_amplitude(self) -> np.ndarray | None:
        """|a(f)| / |W(f)| on the monitor frequencies (DD-198), or None.

        Read from the file when the run wrote it, else derived from the
        stored port signals through the run's incident-ratio callable.
        """
        import h5py  # noqa: PLC0415

        if self._incident_loaded:
            return self._incident_ratio
        self._incident_loaded = True
        with h5py.File(self._run_dir / "fields_freq.h5", "r") as f:
            if "incident_amplitude" in f[self.name]:
                self._incident_ratio = np.asarray(
                    f[self.name]["incident_amplitude"][()], dtype=float
                )
                return self._incident_ratio
        if self._incident is not None:
            ratio = self._incident()
            if ratio is not None:
                f_axis, values = ratio
                self._incident_ratio = np.interp(
                    np.asarray(self.freqs, dtype=float),
                    np.asarray(f_axis, dtype=float),
                    np.asarray(values, dtype=float),
                )
        return self._incident_ratio

    def renormalize(self, source_signal) -> None:
        """Set (or replace) the excitation the bins are divided by."""
        self._reference = source_signal
        self._spectrum = None

    @property
    def data(self) -> dict:
        """Recorded fields per 1 W incident CW power (E in V/m, H in A/m)."""
        return {c: self.component(c) for c in self._components}

    @property
    def data_raw(self) -> dict:
        """Raw DFT bins, in field units x seconds (undivided)."""
        return {c: self._squeeze_spatial(self._read_bins(c)) for c in self._components}

    def _hydrate(self):
        """Build an in-RAM :class:`MonitorFieldFrequency` for plotting reuse."""
        import h5py  # noqa: PLC0415

        from magnelio.monitors._dft import DFTAccumulator  # noqa: PLC0415
        from magnelio.monitors.base import MonitorRegion  # noqa: PLC0415
        from magnelio.monitors.field_frequency import (  # noqa: PLC0415
            MonitorFieldFrequency,
        )

        mon = MonitorFieldFrequency(
            corners=self.corners,
            freqs=self.freqs,
            fields=list(self.fields),
            interval=self.interval,
            name=self.name,
        )
        nx, ny, nz = (len(self._grid_x), len(self._grid_y), len(self._grid_z))
        ndim = sum(1 for n in (nx, ny, nz) if n > 1)
        mon._region = MonitorRegion(
            ix=slice(0, nx),
            iy=slice(0, ny),
            iz=slice(0, nz),
            xc=np.asarray(self._grid_x),
            yc=np.asarray(self._grid_y),
            zc=np.asarray(self._grid_z),
            ndim=ndim,
        )
        mon._accumulators = {}
        incident = None
        with h5py.File(self._run_dir / "fields_freq.h5", "r") as f:
            bg = f[self.name]["bins"]
            for comp in self._components:
                acc = DFTAccumulator(self.freqs, (nx, ny, nz))
                acc._bins[...] = bg[comp][()]
                mon._accumulators[comp] = acc
            if "incident_amplitude" in f[self.name]:
                incident = np.asarray(f[self.name]["incident_amplitude"][()], dtype=float)
        mon._mirrors = self._mirrors
        mon._grid = self._grid
        # Carry the run's reference across, or the hydrated monitor would
        # refuse to hand out data it cannot put a unit on.
        mon._source_spectrum = self._source_spectrum()
        ratio = incident if incident is not None else self._incident_amplitude()
        if ratio is not None:
            mon._incident_amplitude = np.asarray(ratio, dtype=float)
        return mon

    def plot(self, *args, **kwargs):
        """Plot the DFT field (delegates to :class:`MonitorFieldFrequency`)."""
        return self._hydrate().plot(*args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"_LoadedFreqMonitor(name={self.name!r}, "
            f"n_freqs={len(self.freqs)}, components={self._components})"
        )


# ═════════════════════════════════════════════════════════════════════
# Resume checkpoint  <->  runs/<name>/checkpoint.h5
# ═════════════════════════════════════════════════════════════════════
#
# A checkpoint is the solver's ``state_dict()`` (DD-070, WP-S6): a purely
# recursive ``dict`` of ``{str: dict | ndarray | scalar}`` — E/H, the
# completed-step count and energy peak, and every stateful boundary
# (CPML ψ) and port (Mur previous-values, TF/SF ring buffer, exact DTBC
# convolution history).  That shape maps one-to-one onto the HDF5 object
# tree (a nested dict is a group, everything else a dataset), so a single
# recursive walker serialises every holder — no per-type code.  Unlike
# ``results.h5`` (append-only SWMR), the checkpoint is *overwritten* every
# interval, so it is written to a temp file and ``os.replace``-d into
# place: a concurrent reader/resumer never observes a half-written state.


def _host_array(v):
    """Return a host ``ndarray`` view of ``v`` (GPU cupy arrays → host).

    ``state_dict`` copies live field/aux arrays verbatim, so on a GPU run
    they are ``cupy`` arrays; HDF5 needs host memory.  ``cupy.ndarray``
    exposes ``.get()`` and is not a ``numpy.ndarray``; everything else
    (numpy arrays, Python/numpy scalars) goes through ``np.asarray``.
    """
    if hasattr(v, "get") and not isinstance(v, np.ndarray):
        return v.get()
    return np.asarray(v)


def _write_state_group(grp, state: dict) -> None:
    """Recursively write a state dict into an open h5py group."""
    for key, val in state.items():
        if isinstance(val, dict):
            _write_state_group(grp.create_group(key), val)
        else:
            grp.create_dataset(key, data=_host_array(val))


def _read_state_group(grp) -> dict:
    """Inverse of :func:`_write_state_group` (subgroup → nested dict)."""
    import h5py  # noqa: PLC0415

    out = {}
    for key, item in grp.items():
        out[key] = _read_state_group(item) if isinstance(item, h5py.Group) else item[()]
    return out


def _write_state_dict_h5(path: str | Path, state: dict) -> None:
    """Atomically write a solver ``state_dict`` to ``path`` (temp+rename).

    The full state is written to ``<path>.tmp`` and then ``os.replace``-d
    onto ``path`` (atomic on POSIX within one filesystem), so a crash
    mid-write leaves the previous checkpoint intact and a live
    reader/resumer never sees a partial file.
    """
    import h5py  # noqa: PLC0415

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        _write_state_group(f, state)
    os.replace(tmp, path)


def _read_state_dict_h5(path: str | Path) -> dict:
    """Read a checkpoint written by :func:`_write_state_dict_h5`.

    Returns the same nested-dict shape the solver's ``state_dict``
    produced; scalars come back as numpy scalars (``load_state_dict``
    coerces them via ``int()`` / ``float()``).
    """
    import h5py  # noqa: PLC0415

    with h5py.File(Path(path), "r") as f:
        return _read_state_group(f)


# ═════════════════════════════════════════════════════════════════════
# Frequency monitors  <->  runs/<name>/fields_freq.h5  (DD-070 follow-up)
# ═════════════════════════════════════════════════════════════════════
#
# A MonitorFieldFrequency's DFT accumulator is a fixed-size running sum, not
# an append-only stream — it cannot live in the SWMR results.h5 (an in-place
# overwrite is not SWMR-consistent).  It gets its own result file, written
# whole and atomically (temp + os.replace) at each checkpoint: the file is
# the post-processing result the user plots *and*, transparently, the resume
# source for the accumulator.  n_completed ties it to checkpoint.h5.


def _write_freq_result_h5(path, dumps: dict, n_completed: int) -> None:
    """Atomically write MonitorFieldFrequency DFT results to *path*.

    ``dumps`` maps monitor name -> :meth:`MonitorFieldFrequency.result_dump`.
    Written to ``<path>.tmp`` then ``os.replace``-d, so a live reader or a
    resumer never sees a partial file.  ``n_completed`` is the step the
    accumulators reflect; a resume checks it against ``checkpoint.h5`` so a
    crash between the two writes cannot load a mismatched accumulator.
    """
    import h5py  # noqa: PLC0415

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["n_completed"] = int(n_completed)
        for name, dump in dumps.items():
            g = f.create_group(name)
            g.attrs["components"] = json.dumps(list(dump["components"]))
            g.attrs["fields"] = json.dumps(list(dump["fields"]))
            # Schema-additive (DD-140): the sampling the bins were
            # integrated with.  Absent in files written before it
            # existed, which is exactly "every step".
            if dump.get("interval") is not None:
                g.attrs["interval"] = float(dump["interval"])
            # Schema-additive (DD-154): symmetry planes the region
            # touches; absent means none.
            if dump.get("symmetry"):
                g.attrs["symmetry"] = json.dumps(dump["symmetry"])
            for key in ("freqs", "corners", "grid_x", "grid_y", "grid_z"):
                g.create_dataset(key, data=np.asarray(dump[key]))
            # Schema-additive (DD-198): the launched incident wave per
            # unit excitation waveform; absent means 1 (lumped/TEM feed).
            if dump.get("incident_amplitude") is not None:
                g.create_dataset("incident_amplitude", data=np.asarray(dump["incident_amplitude"]))
            bg = g.create_group("bins")
            for comp, arr in dump["bins"].items():
                bg.create_dataset(comp, data=np.asarray(arr))
    os.replace(tmp, path)


# ═════════════════════════════════════════════════════════════════════
# Wall-loss monitors  <->  runs/<name>/wall_loss.h5  (DD-082 addendum)
# ═════════════════════════════════════════════════════════════════════
#
# Same kind as a frequency monitor — a fixed-size running DFT, not an
# append stream — so it follows the same shape: its own file, written whole
# and atomically at each checkpoint, tagged with the n_completed that ties
# it to checkpoint.h5.  It differs in one way: the RESULT is a reduction
# (P_loss/P_flow per tag), not the accumulators, so the file carries the
# reduction for readers AND the raw accumulators for the resume.


def _write_wall_loss_h5(path, dumps: dict, n_completed: int) -> None:
    """Atomically write MonitorWallLoss results to *path*.

    ``dumps`` maps monitor name -> :meth:`MonitorWallLoss.result_dump`.
    Written to ``<path>.tmp`` then ``os.replace``-d, so a live reader or a
    resumer never sees a partial file.  ``n_completed`` is the step the
    accumulators reflect; a resume checks it against ``checkpoint.h5``.
    """
    import h5py  # noqa: PLC0415

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["n_completed"] = int(n_completed)
        for name, dump in dumps.items():
            g = f.create_group(name)
            # Tags are heterogeneous (material ids are ints, BC walls are
            # face names) — JSON keeps the types; the arrays below are in
            # this order.
            g.attrs["tags"] = json.dumps(list(dump["tags"]))
            g.create_dataset("freqs", data=np.asarray(dump["freqs"]))
            g.create_dataset("total", data=np.asarray(dump["total"]))
            fg = g.create_group("fraction")
            for i, arr in enumerate(dump["fraction"]):
                fg.create_dataset(str(i), data=np.asarray(arr))
            rg = g.create_group("raw")
            hg = rg.create_group("h_bins")
            for i, arr in enumerate(dump["h_bins"]):
                hg.create_dataset(str(i), data=np.asarray(arr))
            bg = rg.create_group("ref_bins")
            for key, arr in dump["ref_bins"].items():
                bg.create_dataset(key, data=np.asarray(arr))
    os.replace(tmp, path)


def _list_run_wall_loss(run_dir: Path) -> list[str]:
    """Names of the wall-loss monitors persisted for a run, or ``[]``."""
    import h5py  # noqa: PLC0415

    wl = run_dir / "wall_loss.h5"
    if not wl.exists():
        return []
    with h5py.File(wl, "r") as f:
        return [name for name in f if isinstance(f[name], h5py.Group)]


class _LoadedMonitorWallLoss:
    """Lazy reader over one persisted ``MonitorWallLoss`` (DD-082 addendum).

    Serves the same result API as the in-RAM monitor — :attr:`f`,
    :attr:`dissipated_fraction`, :meth:`power_loss` — reading the reduced
    per-tag fractions the run wrote at its last checkpoint.  Like the
    streamed S-parameters, a live run's file is a converging *partial*
    result, readable from the first dump.
    """

    def __init__(self, run_dir: Path, name: str):
        self._run_dir = Path(run_dir)
        self.name = name

    def _group(self, f):
        return f[self.name]

    @property
    def f(self) -> np.ndarray:
        """Frequency axis [Hz]."""
        import h5py  # noqa: PLC0415

        with h5py.File(self._run_dir / "wall_loss.h5", "r") as fh:
            return self._group(fh)["freqs"][()]

    @property
    def freqs(self) -> np.ndarray:
        """Alias of :attr:`f` (the in-RAM monitor's attribute name)."""
        return self.f

    @property
    def dissipated_fraction(self) -> dict:
        """Per-tag ``P_loss(f)/P_flow(f)`` (scale-free), plus ``"total"``."""
        import h5py  # noqa: PLC0415

        with h5py.File(self._run_dir / "wall_loss.h5", "r") as fh:
            g = self._group(fh)
            tags = json.loads(g.attrs["tags"])
            out = {tag: g["fraction"][str(i)][()] for i, tag in enumerate(tags)}
            out["total"] = g["total"][()]
            return out

    def power_loss(self, P_in: float = 1.0) -> dict:
        """Per-tag wall loss [W] for *P_in* Watts through the reference
        plane, plus ``"total"``."""
        return {tag: frac * P_in for tag, frac in self.dissipated_fraction.items()}

    def __repr__(self) -> str:
        return f"_LoadedMonitorWallLoss(name={self.name!r}, n_freqs={len(self.f)})"


# ═════════════════════════════════════════════════════════════════════
# Far-field monitors  <->  runs/<name>/far_field.h5  (DD-173)
# ═════════════════════════════════════════════════════════════════════
#
# Same fixed-size running-DFT shape as the frequency and wall-loss
# dumps.  The file carries the surface bins PLUS the box geometry and
# image planes, so the reader rebuilds the transform inputs without the
# mesh — reader == monitor by construction.


def _write_far_field_h5(path, dumps: dict, n_completed: int) -> None:
    """Atomically write MonitorFarFieldFrequency results to *path*.

    ``dumps`` maps monitor name -> :meth:`MonitorFarFieldFrequency.result_dump`.
    Written to ``<path>.tmp`` then ``os.replace``-d; ``n_completed``
    ties the accumulators to ``checkpoint.h5`` for the resume check.
    """
    import h5py  # noqa: PLC0415

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["n_completed"] = int(n_completed)
        for name, dump in dumps.items():
            g = f.create_group(name)
            g.attrs["margin_cells"] = int(dump["margin_cells"])
            g.attrs["image_planes"] = json.dumps(dump["image_planes"])
            g.create_dataset("freqs", data=np.asarray(dump["freqs"]))
            for key in ("source_spectrum", "accepted_power", "incident_amplitude"):
                if key in dump:
                    g.create_dataset(key, data=np.asarray(dump[key]))
            fg = g.create_group("faces")
            for face in dump["faces"]:
                sg = fg.create_group(str(face["name"]))
                sg.attrs["axis"] = int(face["axis"])
                sg.attrs["sign"] = float(face["sign"])
                sg.attrs["plane"] = float(face["plane"])
                for key in ("c1", "c2", "w1", "w2"):
                    sg.create_dataset(key, data=np.asarray(face[key]))
                bg = sg.create_group("bins")
                for comp, arr in face["bins"].items():
                    bg.create_dataset(comp, data=np.asarray(arr))
    os.replace(tmp, path)


def _read_far_field_dump(run_dir: Path, name: str) -> dict:
    """Read one monitor's dump back into the ``result_dump`` shape."""
    import h5py  # noqa: PLC0415

    with h5py.File(Path(run_dir) / "far_field.h5", "r") as f:
        g = f[name]
        dump = {
            "name": name,
            "freqs": g["freqs"][()],
            "margin_cells": int(g.attrs["margin_cells"]),
            "image_planes": json.loads(g.attrs["image_planes"]),
            "faces": [],
        }
        for key in ("source_spectrum", "accepted_power", "incident_amplitude"):
            if key in g:
                dump[key] = g[key][()]
        for face_name, sg in g["faces"].items():
            dump["faces"].append(
                {
                    "name": face_name,
                    "axis": int(sg.attrs["axis"]),
                    "sign": float(sg.attrs["sign"]),
                    "plane": float(sg.attrs["plane"]),
                    "c1": sg["c1"][()],
                    "c2": sg["c2"][()],
                    "w1": sg["w1"][()],
                    "w2": sg["w2"][()],
                    "bins": {comp: sg["bins"][comp][()] for comp in sg["bins"]},
                }
            )
    return dump


def _list_run_far_field(run_dir: Path) -> list[str]:
    """Names of the far-field monitors persisted for a run, or ``[]``."""
    import h5py  # noqa: PLC0415

    ff = run_dir / "far_field.h5"
    if not ff.exists():
        return []
    with h5py.File(ff, "r") as f:
        return [name for name in f if isinstance(f[name], h5py.Group)]


class _LoadedFarFieldMonitor:
    """Lazy reader over one persisted ``MonitorFarFieldFrequency`` (DD-173).

    Hydrates the dump into a result-serving monitor on first access and
    delegates, so the reader serves exactly the in-RAM API —
    :attr:`f`, :meth:`result` and the pattern plots.
    """

    def __init__(self, run_dir: Path, name: str, reference=None, incident=None):
        self._run_dir = Path(run_dir)
        self.name = name
        self._reference = reference
        self._incident = incident
        self._monitor = None

    def _hydrate(self):
        if self._monitor is None:
            from magnelio.monitors.far_field import MonitorFarFieldFrequency  # noqa: PLC0415

            dump = _read_far_field_dump(self._run_dir, self.name)
            self._monitor = MonitorFarFieldFrequency.from_result_dump(dump)
            if not self._monitor.is_renormalized and self._reference is not None:
                # The streamed run renormalises after its final flush, so
                # the file carries raw bins; the run's reference waveform
                # supplies the divisor on read (the _LoadedFreqMonitor
                # pattern).
                self._monitor.renormalize(self._reference())
            if self._monitor._incident_amplitude is None and self._incident is not None:
                # DD-198: the launched incident wave per unit waveform,
                # derived from the stored port signals like the S-matrix.
                ratio = self._incident()
                if ratio is not None:
                    self._monitor._set_incident_amplitude(*ratio)
        return self._monitor

    @property
    def f(self) -> np.ndarray:
        """Frequency axis [Hz]."""
        return self._hydrate().f

    @property
    def freqs(self) -> np.ndarray:
        """Alias of :attr:`f` (the in-RAM monitor's attribute name)."""
        return self._hydrate().freqs

    def renormalize(self, source_signal) -> None:
        """Set (or replace) the excitation the surface DFT is divided by."""
        self._reference = lambda: source_signal
        if self._monitor is not None:
            self._monitor.renormalize(source_signal)

    def result(self, *args, **kwargs):
        return self._hydrate().result(*args, **kwargs)

    def plot_cut(self, *args, **kwargs):
        return self._hydrate().plot_cut(*args, **kwargs)

    def plot_3d(self, *args, **kwargs):
        return self._hydrate().plot_3d(*args, **kwargs)

    def __repr__(self) -> str:
        return f"_LoadedFarFieldMonitor(name={self.name!r})"


# ═════════════════════════════════════════════════════════════════════
# Eigenmodes  <->  eigenmodes.h5
# ═════════════════════════════════════════════════════════════════════


def _save_eigenmodes(path: Path, result) -> None:
    """Write an :class:`EigenmodeResult` to ``eigenmodes.h5``."""
    import h5py  # noqa: PLC0415

    with h5py.File(path / "eigenmodes.h5", "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.create_dataset(
            "frequencies",
            data=np.asarray(result.frequencies, dtype=float),
        )
        if result.solver_info:
            f.attrs["solver_info"] = json.dumps(result.solver_info)
        for i, mode in enumerate(result.modes):
            mg = f.create_group(f"mode_{i:03d}")
            for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                mg.create_dataset(comp, data=getattr(mode, comp))


def _load_eigenmodes(path: Path, mesh=None):
    """Reconstruct an :class:`EigenmodeResult` from ``eigenmodes.h5``."""
    import h5py  # noqa: PLC0415

    from magnelio._fields.field_arrays import FieldState  # noqa: PLC0415
    from magnelio.solver.eigenmode_result import EigenmodeResult  # noqa: PLC0415

    with h5py.File(path / "eigenmodes.h5", "r") as f:
        frequencies = f["frequencies"][()]
        solver_info = json.loads(f.attrs["solver_info"]) if "solver_info" in f.attrs else {}
        modes = []
        for i in range(len(frequencies)):
            mg = f[f"mode_{i:03d}"]
            modes.append(
                FieldState(
                    Ex=mg["Ex"][()],
                    Ey=mg["Ey"][()],
                    Ez=mg["Ez"][()],
                    Hx=mg["Hx"][()],
                    Hy=mg["Hy"][()],
                    Hz=mg["Hz"][()],
                )
            )
    return EigenmodeResult(
        frequencies=frequencies,
        modes=modes,
        mesh=mesh,
        solver_info=solver_info,
    )


# ═════════════════════════════════════════════════════════════════════
# ProjectStore (writer)  /  Project (reader)
# ═════════════════════════════════════════════════════════════════════


class ProjectStore:
    """Write-once model store for a project directory.

    Persists the static model — geometry (BREP + VTM), mesh, and setup
    metadata.  Streamed time-domain results and per-run resume
    checkpoints are added by later work packages; :class:`ProjectStore`
    holds the directory handle they will attach to.

    Create a store with :meth:`create`; read one back with
    :func:`open_project`.
    """

    # Design: DD-070 WP-S1 (write-once model store).

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def create(
        cls,
        path: str | Path,
        mesh,
        *,
        geometry=None,
        setup: dict | None = None,
        paraview: bool = True,
        exist_ok: bool = True,
    ) -> "ProjectStore":
        """Create a project directory and write the static model.

        Parameters
        ----------
        path : str or Path
            Project directory (created if absent).
        mesh : Mesh
            The simulation mesh (grid, materials, conformal sub-cell
            data) — written to ``mesh.h5``.
        geometry : GeometryModel or list, optional
            Source geometry.  When given, an exact ``geometry.brep`` and
            (if *paraview*) a per-solid ``geometry.vtm`` are written,
            plus a ``geometry.json`` carrying per-shape names and
            materials in compound order.
        setup : dict, optional
            JSON-serialisable analysis metadata (dt, frequency plan,
            port/BC/excitation descriptions).  Stored under ``setup`` in
            ``project.json``; consumed by later work packages.
        paraview : bool, default True
            Also write the tessellated ``geometry.vtm`` for ParaView,
            and generate the per-run ParaView session at run close.
        exist_ok : bool, default True
            Reuse an existing directory (raise if False and it exists).

        Returns
        -------
        ProjectStore
        """
        try:
            import h5py  # noqa: PLC0415, F401
        except ImportError as exc:
            raise ImportError(
                "h5py is required for the project store. Install with: pip install h5py",
            ) from exc

        path = Path(path)
        if path.exists() and not exist_ok:
            raise FileExistsError(f"project directory already exists: {path}")
        path.mkdir(parents=True, exist_ok=True)

        store = cls(path)
        store._write_mesh(mesh)
        has_geometry = geometry is not None
        if has_geometry:
            store._write_geometry(geometry, paraview=paraview)

        from magnelio._version import __version__  # noqa: PLC0415

        now = _utc_now_iso()
        _write_json_atomic(
            path / "project.json",
            {
                "schema_version": SCHEMA_VERSION,
                "magnelio_version": __version__,
                "created": now,
                "modified": now,
                "has_geometry": has_geometry,
                "setup": setup or {},
                "runs": {},
                "status": "created",
            },
        )
        return store

    # -- internal writers ------------------------------------------------

    def _write_mesh(self, mesh) -> None:
        import h5py  # noqa: PLC0415

        with h5py.File(self.path / "mesh.h5", "w") as f:
            _save_mesh(f, mesh)

    def _write_geometry(self, geometry, *, paraview: bool) -> None:
        from magnelio.geo.wire import ThinWire  # noqa: PLC0415

        shapes = list(geometry)
        background = getattr(geometry, "background", None)
        write_brep(shapes, self.path / "geometry.brep")
        # Schema-additive metadata for non-solid shapes (DD-080): a
        # ThinWire's curve is a wire in the BREP compound; ``kinds`` /
        # ``radii`` record what it was.  The v1 reader does not
        # reconstruct ThinWire objects from these (the wire's physics
        # lives in the stored consolidated mesh; resume never re-meshes).
        _write_json_atomic(
            self.path / "geometry.json",
            {
                "materials": [_material_to_dict(s.material) for s in shapes],
                "names": [getattr(s, "name", None) for s in shapes],
                "kinds": ["wire" if isinstance(s, ThinWire) else "solid" for s in shapes],
                "radii": [s.radius if isinstance(s, ThinWire) else None for s in shapes],
                # Display colour imported from a CAD file (DD-178), if any.
                "colors": [
                    list(c) if (c := getattr(s, "color", None)) is not None else None
                    for s in shapes
                ],
                "background": (_material_to_dict(background) if background is not None else None),
            },
        )
        if paraview:
            try:
                from magnelio.io.paraview import export_vtm  # noqa: PLC0415

                solids = [s for s in shapes if not isinstance(s, ThinWire)]
                if solids:
                    export_vtm(self.path / "geometry.vtm", solids)
            except ImportError:
                pass  # VTM is viz-only; BREP is the source of truth

    def register_planned_runs(self, planned) -> None:
        """Pre-register planned runs as ``pending`` in the run index.

        ``planned`` is an iterable of ``(run_name, entry)`` pairs, one
        per run the caller is about to stream, ``entry`` the index
        payload (``{"excited": [port, mode]}`` for a scattering
        channel).  Registering them
        up front closes the status gap between sequential runs: without
        it, finishing run *k* while run *k+1* is not yet in the index
        made :meth:`_finalize_run` report ``status = "done"`` for a
        project that was still mid-analysis.  A
        ``pending`` entry counts as not-done, so the project status
        stays ``"running"`` until the last planned run finishes.

        Existing entries are left untouched (fill-in: a second analysis
        adding excitations must not clobber ``done``/``aborted`` runs).
        ``pending`` entries carry no run directory on disk; they are
        replaced wholesale when :meth:`open_run` starts the
        run.
        """

        def _upd(meta: dict) -> None:
            runs = meta.setdefault("runs", {})
            for run_name, entry in planned:
                runs.setdefault(
                    _safe_run_name(run_name),
                    {**dict(entry), "state": "pending"},
                )
            meta["status"] = "running"
            meta["writer"] = _writer_identity()

        _update_meta(self.path, _upd)

    def mark_analysis_started(self) -> None:
        """Stamp the start of an analysis call (``run()`` or ``resume()``).

        The per-run stamps cover the marches; this one covers the whole
        call, setup included, which is what the ``finished in`` line
        reports and what a reader wants to see at the top of a project.
        """

        def _upd(meta: dict) -> None:
            meta["analysis"] = {"started": _utc_now_iso(), "finished": None, "elapsed": None}
            meta["writer"] = _writer_identity()

        _update_meta(self.path, _upd)

    def mark_analysis_finished(self, elapsed: float) -> None:
        """Close the analysis stamp opened by :meth:`mark_analysis_started`."""

        def _upd(meta: dict) -> None:
            entry = meta.setdefault("analysis", {})
            entry["finished"] = _utc_now_iso()
            entry["elapsed"] = float(elapsed)

        _update_meta(self.path, _upd)

    def open_run(
        self,
        run_name: str,
        *,
        excitations: list,
        excited: tuple[str, int] | None,
        dt: float,
        f_axis,
        channels,
        port_modes: dict,
        port_normal_dx: dict,
        port_line_params: dict,
        excitation_fns,
        recorder,
        port_band: dict | None = None,
        port_model: str = "modal",
        port_reference_scale: dict | None = None,
        energy_stop_db: float | None = None,
        port_signal_stop_db: float | None = None,
        total_time_steps: int | None = None,
        taper_signals: bool = False,
        monitors=None,
        grid=None,
    ) -> "_RunSink":
        """Open a live streaming sink for one time-domain run.

        Declares the run's resizable ``results.h5`` streams (HDF5-SWMR),
        registers the run in ``project.json`` as ``running`` (replacing a
        ``pending`` pre-registration, see :meth:`register_planned_runs`),
        and returns
        a solver-attachable :class:`_RunSink`.  The solver
        appends the V/I and energy tails during ``run()``; call
        :meth:`_RunSink.close` when the run finishes to flip its
        state to ``done``.  ``run_name`` is sanitised for the filesystem.
        ``excitations`` are the run's Excitation dicts (DD-224),
        ``excited`` the S-matrix column of a scattering channel run
        (``None`` on a general time-domain run), ``excitation_fns`` the
        ``[(key, drive), ...]`` sampled alongside the signals.

        ``energy_stop_db`` / ``total_time_steps`` are recorded in the run
        index so ``resume()`` can default to the run's original
        stop criterion — e.g. finish an aborted run to the target it was
        launched with, without the caller repeating it.  ``monitors`` +
        ``grid`` declare the field-monitor write-through streams;
        the sink drains each ``MonitorFieldTime`` to disk as the run
        proceeds.
        """
        # Design: WP-S5 (live streaming sink), WP-S8 (resume defaults from the
        # run index), WP-S9 (field-monitor write-through).
        safe = _safe_run_name(run_name)
        run_dir = self.path / "runs" / safe
        writer = _RunResultWriter(
            run_dir,
            dt=dt,
            excitations=excitations,
            excited=excited,
            f_axis=f_axis,
            channels=list(channels),
            port_modes=port_modes,
            port_normal_dx=port_normal_dx,
            port_line_params=port_line_params,
            port_band=port_band,
            port_reference_scale=port_reference_scale,
            monitors=monitors,
            grid=grid,
        )

        def _upd(meta: dict) -> None:
            meta.setdefault("runs", {})[safe] = {
                "excited": None if excited is None else [excited[0], int(excited[1])],
                "excitations": [[e["source"], int(e.get("mode", 0))] for e in excitations],
                "n_steps": 0,
                "dt": float(dt),
                "port_model": port_model,
                "energy_stop_db": energy_stop_db,
                "port_signal_stop_db": port_signal_stop_db,
                "total_time_steps": total_time_steps,
                "taper_signals": bool(taper_signals),
                "state": "running",
                # Wall clock of the run (DD-253): ``elapsed`` sums the
                # marches of a resumed run; ``pid``/``host`` say who
                # is writing, so a reader can tell a live run from a
                # dead one.
                "started": _utc_now_iso(),
                "finished": None,
                "elapsed": 0.0,
                **_writer_identity(),
            }
            meta["status"] = "running"
            meta["writer"] = _writer_identity()

        _update_meta(self.path, _upd)
        return _RunSink(
            writer,
            recorder,
            excitation_fns,
            dt,
            store=self,
            run_name=safe,
            checkpoint_path=run_dir / "checkpoint.h5",
            field_time_monitors=_field_time_monitors(monitors),
            flux_monitors=_flux_monitors(monitors),
            freq_monitors=_freq_monitors(monitors),
            wall_loss_monitors=_wall_loss_monitors(monitors),
            far_field_monitors=_far_field_monitors(monitors),
            grid=grid,
        )

    def reopen_run(
        self,
        run_name: str,
        *,
        recorder,
        excitation_fns,
        dt: float,
        n_keep: int,
        step_offset: int,
        monitors=None,
        grid=None,
        monitor_keep: dict | None = None,
        flux_keep: dict | None = None,
    ) -> "_RunSink":
        """Reopen a run's streams to append a resumed tail.

        Truncates ``results.h5`` back to ``n_keep`` (the checkpoint's
        committed step count), marks the run ``running`` again, and returns
        a sink whose reference sampling is offset to global step
        ``step_offset`` so the appended tail is phase-aligned with the
        pre-resume stream.  The caller attaches it to the resuming solver
        exactly like a fresh sink.  ``monitor_keep`` truncates each field
        monitor's stream to its checkpointed snapshot count so the resumed
        run appends onward consistently.
        """
        # Design: WP-S8 (resume tail append), WP-S9 (monitor stream truncation).
        safe = _safe_run_name(run_name)
        run_dir = self.path / "runs" / safe
        writer = _RunResultWriter.reopen(
            run_dir, n_keep=int(n_keep), monitor_keep=monitor_keep, flux_keep=flux_keep
        )

        def _upd(meta: dict) -> None:
            run = meta.setdefault("runs", {}).setdefault(safe, {})
            run["state"] = "running"
            run["n_steps"] = int(n_keep)
            # A resumed march keeps ``started`` and the accumulated
            # ``elapsed``; the new segment is stamped separately and the
            # writer identity is refreshed — it is a new process.
            run.setdefault("started", _utc_now_iso())
            run["resumed"] = _utc_now_iso()
            run["finished"] = None
            run.setdefault("elapsed", 0.0)
            run.update(_writer_identity())
            meta["status"] = "running"
            meta["writer"] = _writer_identity()

        _update_meta(self.path, _upd)
        return _RunSink(
            writer,
            recorder,
            excitation_fns,
            dt,
            store=self,
            run_name=safe,
            checkpoint_path=run_dir / "checkpoint.h5",
            step_offset=int(step_offset),
            field_time_monitors=_field_time_monitors(monitors),
            flux_monitors=_flux_monitors(monitors),
            freq_monitors=_freq_monitors(monitors),
            wall_loss_monitors=_wall_loss_monitors(monitors),
            far_field_monitors=_far_field_monitors(monitors),
            grid=grid,
        )

    def _finalize_run(
        self,
        safe: str,
        n_steps: int,
        state: str,
        stop_reason: str | None = None,
        final_port_signal_db: float | None = None,
        elapsed: float | None = None,
    ) -> None:
        """Flip a streamed run to its terminal ``state`` in project.json.

        ``stop_reason`` records why the marching ended ("energy",
        "port_signal", "port_signal_stall", "runtime_cap", "steps",
        "aborted"), ``final_port_signal_db`` the |V|-envelope level below
        peak at the stop — schema-additive, absent on older projects.
        ``elapsed`` is the wall time of the march that just ended; it is
        added to what earlier marches of the same run accumulated.
        """

        def _upd(meta: dict) -> None:
            run = meta.setdefault("runs", {}).setdefault(safe, {})
            run["state"] = state
            run["n_steps"] = int(n_steps)
            if stop_reason is not None:
                run["stop_reason"] = str(stop_reason)
            if final_port_signal_db is not None:
                run["final_port_signal_db"] = float(final_port_signal_db)
            run["finished"] = _utc_now_iso()
            if elapsed is not None:
                run["elapsed"] = float(run.get("elapsed") or 0.0) + float(elapsed)
            # Every planned run done → done.  An aborted run ends the
            # analysis call (the interrupt propagates), so its pending
            # siblings never start in that call: the project is aborted,
            # not running.  Anything else is still on its way.
            states = [r.get("state") for r in meta["runs"].values()]
            if all(s == "done" for s in states):
                meta["status"] = "done"
            elif any(s == "aborted" for s in states):
                meta["status"] = "aborted"
            else:
                meta["status"] = "running"

        _update_meta(self.path, _upd)

    def write_eigenmodes(self, result) -> None:
        """Persist an :class:`EigenmodeResult` to ``eigenmodes.h5``.

        Eigenmode analysis has no time-marching state, so it produces a
        one-shot result rather than a streamable/resumable run.
        """
        _save_eigenmodes(self.path, result)

        def _upd(meta: dict) -> None:
            meta["has_eigenmodes"] = True
            meta["status"] = "done"

        _update_meta(self.path, _upd)
        self._export_eigenmode_paraview()

    def _export_eigenmode_paraview(self) -> None:
        """Generate the eigenmode ParaView session, best-effort.

        Mirrors the run-close export: visualization-only, so any
        failure is downgraded to a warning rather than invalidating a
        stored result.
        """
        try:
            from magnelio.io.paraview import (  # noqa: PLC0415
                export_eigenmode_visualization,
            )

            export_eigenmode_visualization(self.path)
        except Exception as exc:  # viz-only; the result itself is written
            import warnings  # noqa: PLC0415

            warnings.warn(
                f"ParaView eigenmode export failed (visualization only): {exc}",
                UserWarning,
                stacklevel=2,
            )


# ═════════════════════════════════════════════════════════════════════
# Runs as objects, and whether their writer is still alive (DD-254)
# ═════════════════════════════════════════════════════════════════════

# Stored statuses after which the metadata no longer changes on its own.
_TERMINAL_STATUS = ("done", "aborted")


def _file_stamp(path: Path):
    """What identifies a version of a file the writer replaces atomically."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_ino, st.st_size)


def _pid_alive(pid) -> bool | None:
    """Whether a process exists on this host; ``None`` when it cannot be known.

    POSIX answers signal 0.  Elsewhere the question is left open on
    purpose — a wrong ``False`` would call a marching run stale.
    """
    if pid is None or os.name != "posix":
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OverflowError, ValueError, OSError):
        return None
    return True


def _writer_gone(identity: dict) -> bool:
    """True when the recorded writer ran on this host and no longer exists."""
    if not identity or identity.get("host") != socket.gethostname():
        return False
    return _pid_alive(identity.get("pid")) is False


def _writer_state(entry: dict) -> str:
    """A run entry's live state: ``stale`` for a ``running`` one nobody writes."""
    state = entry.get("state", "done")
    if state == "running" and _writer_gone(entry):
        return "stale"
    return state


class Run:
    """One run of a project: its index entry, its files, its result.

    Handed out by ``project.runs[name]``.  A live view — every attribute
    reads the project's current metadata, so a run that is marching in
    another process shows its growing step count and energy without
    any call on your side.  ``pending`` runs have no directory yet and
    read as empty.
    """

    def __init__(self, project: "Project", name: str) -> None:
        self._project = project
        self.name = name

    # ── the index entry ─────────────────────────────────────────────

    @property
    def _info(self) -> dict:
        return self._project._run_info(self.name)

    @property
    def path(self) -> Path:
        """The run's directory (``runs/<name>``), whether it exists yet or not."""
        return self._project.path / "runs" / self.name

    @property
    def state(self) -> str:
        """``pending``, ``running``, ``done``, ``aborted`` — or ``stale``.

        ``stale`` is ``running`` on disk with the recorded solver process
        gone from this host: the kernel died, or the machine rebooted.
        Resume the run, or run the analysis again.
        """
        return _writer_state(self._info)

    @property
    def excited(self) -> tuple | None:
        """The excited ``(port, mode)`` of a scattering run; ``None`` on a general run."""
        excited = self._info.get("excited")
        return None if excited is None else (excited[0], int(excited[1]))

    @property
    def excitations(self) -> tuple:
        """Every ``(source, mode)`` that drove the run."""
        return tuple((e[0], int(e[1])) for e in self._info.get("excitations", ()))

    @property
    def n_steps(self) -> int:
        """Leapfrog steps committed to disk so far.

        The index books the count when a march ends; while a run
        marches, the step of its latest flushed energy sample stands
        in, so the count moves with the solver.
        """
        info = self._info
        if info.get("state") == "running":
            last = _last_energy_step(self.path)
            if last is not None:
                return last + 1
        return int(info.get("n_steps", 0) or 0)

    @property
    def dt(self) -> float | None:
        """Solver time step [s]."""
        value = self._info.get("dt")
        return None if value is None else float(value)

    @property
    def total_time_steps(self) -> int | None:
        """The fixed step count the run was given, or ``None`` for an open-ended run."""
        return self._info.get("total_time_steps")

    @property
    def port_model(self) -> str | None:
        """``"modal"`` or ``"band"`` — the port pipeline that wrote the run."""
        return self._info.get("port_model")

    @property
    def energy_stop_db(self) -> float | None:
        """The energy criterion [dB below peak] the run stops at, if any."""
        return self._info.get("energy_stop_db")

    @property
    def port_signal_stop_db(self) -> float | None:
        """The port-signal criterion [dB below peak] the run stops at, if any."""
        return self._info.get("port_signal_stop_db")

    @property
    def taper_signals(self) -> bool | None:
        return self._info.get("taper_signals")

    @property
    def stop_reason(self) -> str | None:
        """Why the marching ended: ``energy``, ``port_signal``, ``steps``, …"""
        return self._info.get("stop_reason")

    @property
    def final_port_signal_db(self) -> float | None:
        return self._info.get("final_port_signal_db")

    @property
    def pid(self) -> int | None:
        """Process id of the solver that wrote (or writes) the run."""
        return self._info.get("pid")

    @property
    def host(self) -> str | None:
        """Host name of the solver that wrote (or writes) the run."""
        return self._info.get("host")

    # ── the wall clock ──────────────────────────────────────────────

    @property
    def started(self):
        """UTC datetime the first march started, or ``None``."""
        return parse_utc(self._info.get("started"))

    @property
    def resumed(self):
        """UTC datetime the latest resumed march started, or ``None``."""
        return parse_utc(self._info.get("resumed"))

    @property
    def finished(self):
        """UTC datetime the latest march ended; ``None`` while marching."""
        return parse_utc(self._info.get("finished"))

    @property
    def elapsed(self) -> float | None:
        """Wall time of the marching [s], resumes summed.

        On a run that is still marching, the time since its current
        march started is added to what earlier marches booked.
        """
        info = self._info
        stored = info.get("elapsed")
        if info.get("state") != "running":
            return None if stored is None else float(stored)
        since = parse_utc(info.get("resumed") or info.get("started"))
        live = (utc_now() - since).total_seconds() if since is not None else 0.0
        return float(stored or 0.0) + max(0.0, live)

    # ── the data ────────────────────────────────────────────────────

    @property
    def energy_trace(self) -> np.ndarray:
        """Stored energy at the solver's check cadence — a structured array.

        Fields ``step``, ``time`` [s] and ``energy`` [J]; read afresh on
        every access while the run marches, so it grows as the solver
        flushes.  Empty on a pending run.
        """
        if self._info.get("state") == "pending":
            return np.empty(0, dtype=_ENERGY_DTYPE)
        return _read_energy_trace(self.path)

    @property
    def energy_db(self) -> float | None:
        """The latest stored energy in dB below the run's peak — the progress figure."""
        return db_below_peak(self.energy_trace)

    @property
    def n_energy_samples(self) -> int:
        """How many energy samples the run has flushed — the cheapest sign of life."""
        if self._info.get("state") == "pending":
            return 0
        return _count_energy_samples(self.path)

    @property
    def has_checkpoint(self) -> bool:
        """Whether a resume checkpoint exists on disk."""
        return (self.path / "checkpoint.h5").exists()

    def result(self):
        """The run as a :class:`~magnelio.analysis.TDResult` (see :meth:`Project.result`)."""
        return self._project.result(self.name)

    @property
    def monitors(self) -> dict:
        """The run's monitor readers by name (see :meth:`Project.monitors_for`)."""
        return self._project.monitors_for(self.name)

    def checkpoint_state(self):
        """The run's resume checkpoint (see :meth:`Project.checkpoint_state`)."""
        return self._project.checkpoint_state(self.name)

    def plot_energy(self, *, x: str = "time", ax=None):
        """Plot the run's stored energy in dB below its peak.

        The figure the progress line reports, over the whole run, with
        the run's energy criterion as a dashed line.  ``x`` is
        ``"time"`` (nanoseconds) or ``"step"``; ``ax`` draws into
        existing axes.  Returns ``(fig, ax)``.
        """
        from magnelio.post._plot_energy import plot_energy_traces  # noqa: PLC0415

        return plot_energy_traces(
            {self.name: self.energy_trace},
            energy_stop_db=self.energy_stop_db,
            x=x,
            ax=ax,
        )

    # ── how a run introduces itself ─────────────────────────────────

    def _criteria(self) -> str:
        parts = []
        if self.energy_stop_db is not None:
            parts.append(f"energy {-float(self.energy_stop_db):.0f} dB")
        if self.port_signal_stop_db is not None:
            parts.append(f"port signal {-float(self.port_signal_stop_db):.0f} dB")
        if self.total_time_steps is not None:
            parts.append(f"step {int(self.total_time_steps)}")
        return " or ".join(parts) if parts else "—"

    def _summary_rows(self) -> list[tuple[str, object]]:
        state = self.state
        steps = f"{self.n_steps}"
        if self.total_time_steps:
            steps += f" of {int(self.total_time_steps)}"
        rows: list[tuple[str, object]] = [("state", state)]
        if self.excited is not None:
            rows.append(("excited", self.excited))
        elif self.excitations:
            rows.append(("excitations", list(self.excitations)))
        rows += [
            ("steps", steps),
            ("stops at", self._criteria()),
            ("energy", f"{fmt_db(self.energy_db)} below peak"),
            ("stop reason", self.stop_reason),
            ("started", self.started),
            ("finished", self.finished),
            ("elapsed", format_seconds(self.elapsed)),
            ("dt", self.dt),
            ("checkpoint", self.has_checkpoint),
        ]
        return rows

    def _table_row(self) -> list[object]:
        return [
            self.name,
            self.excited if self.excited is not None else list(self.excitations),
            self.state,
            self.n_steps,
            fmt_db(self.energy_db),
            format_seconds(self.elapsed),
            self.stop_reason,
        ]

    def __repr__(self) -> str:
        return kv_block(f"Run {self.name!r}", self._summary_rows())

    def _repr_html_(self) -> str:
        return html_kv(f"Run {self.name!r}", self._summary_rows())


_RUN_COLUMNS = ("run", "excited", "state", "steps", "energy", "elapsed", "stop reason")
_RUN_ALIGN = "lllrrrl"


# ═════════════════════════════════════════════════════════════════════
# The live notebook panel (DD-255)
# ═════════════════════════════════════════════════════════════════════
#
# Only widget *state* is set from the refresh thread — the HTML string
# and the PNG bytes.  A notebook renders state changes of a displayed
# widget whenever they arrive; output written with ``display()`` after
# the cell returned is dropped, which is why the panel is not an
# ``Output`` widget and never prints.  The picture is rendered without
# pyplot, on an Agg canvas the thread owns, so no GUI backend and no
# global figure registry are touched off the main thread.


def _energy_png(project: "Project", x: str) -> bytes:
    """The project's energy plot as PNG bytes, rendered off the main thread."""
    from io import BytesIO  # noqa: PLC0415

    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415
    from matplotlib.figure import Figure  # noqa: PLC0415

    from magnelio.post._plot_energy import plot_energy_traces  # noqa: PLC0415

    fig = Figure(figsize=(6.4, 3.2), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    traces = project._energy_traces()
    if traces:
        plot_energy_traces(traces, energy_stop_db=project._common_energy_stop(), x=x, ax=ax)
    else:
        ax.text(0.5, 0.5, "no energy samples yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


class _InPlacePainter:
    """Show a project's summary where the last one was — replaced, not appended.

    Three surfaces, told apart the way the progress reporter tells its
    streams apart: a notebook cell, where IPython's ``clear_output``
    and ``display`` let the HTML table replace itself; a terminal,
    where the text table is redrawn over its previous lines with
    cursor movement; and anything else — a log, a pipe — where whole
    tables are appended, because there is nothing to overwrite.
    """

    def __init__(self, stream=None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        try:
            self._tty = bool(self._stream.isatty())
        except (AttributeError, ValueError):
            self._tty = False
        self._notebook = not self._tty and _is_notebook_stream(self._stream)
        self._lines = 0

    def paint(self, project: "Project") -> None:
        if self._notebook:
            try:
                from IPython.display import clear_output, display  # noqa: PLC0415
            except ImportError:
                self._notebook = False
            else:
                clear_output(wait=True)
                display(project)
                return
        text = repr(project)
        if self._tty and self._lines:
            # Up over the previous table, clear from there to the end
            # of the screen, then draw the new one in its place.
            self._stream.write(f"\x1b[{self._lines}A\x1b[J")
        self._stream.write(text + "\n")
        if not self._tty:
            self._stream.write("\n")
        self._stream.flush()
        self._lines = text.count("\n") + 1


def _build_monitor_panel(path: Path, interval: float, x: str, ipywidgets):
    """Assemble the panel and start its refresh thread."""
    import threading  # noqa: PLC0415

    project = Project(path)
    table = ipywidgets.HTML(value=project._repr_html_())
    picture = ipywidgets.Image(value=_energy_png(project, x), format="png")
    panel = ipywidgets.VBox([table, picture])
    stop = threading.Event()

    def _refresh() -> None:
        # Its own reader: no h5 handles or caches shared with the main
        # thread.  The generator yields the terminal snapshot last, so
        # the panel always ends on the final state.
        own = Project(path)
        for _ in own._watch_iter(interval, None):
            if stop.is_set():
                return
            table.value = own._repr_html_()
            picture.value = _energy_png(own, x)

    worker = threading.Thread(target=_refresh, name="magnelio-monitor", daemon=True)
    panel.stop = stop.set
    panel._magnelio_thread = worker
    worker.start()
    return panel


class _RunIndex(Mapping):
    """``project.runs``: the runs by name, printing as a table of their states."""

    def __init__(self, project: "Project") -> None:
        self._project = project

    def __getitem__(self, name: str) -> Run:
        if name not in self._project._run_index():
            raise KeyError(
                f"no run {name!r} in project {self._project.path}; "
                f"runs: {list(self._project._run_index())}",
            )
        return self._project._run_object(name)

    def __iter__(self):
        return iter(list(self._project._run_index()))

    def __len__(self) -> int:
        return len(self._project._run_index())

    def _rows(self) -> list[list[object]]:
        return [self[name]._table_row() for name in self]

    def _table_text(self) -> str:
        if len(self) == 0:
            return ""
        return text_table(_RUN_COLUMNS, self._rows(), align=_RUN_ALIGN)

    def _table_html(self) -> str:
        if len(self) == 0:
            return ""
        return html_table(_RUN_COLUMNS, self._rows(), align=_RUN_ALIGN)

    def __repr__(self) -> str:
        return self._table_text() or f"no runs in project {self._project.path}"

    def _repr_html_(self) -> str:
        return self._table_html() or html.escape(f"no runs in project {self._project.path}")


class CheckpointState(Mapping):
    """A run's resume checkpoint, as the solver's ``state_dict`` reads back.

    A read-only mapping with the nested shape ``FITTimeDomainSolver``
    wrote — ``n_completed``, the peak energy and port signal, the flat
    ``e`` and ``h`` field vectors, and a group per boundary, port and
    monitor — so ``state["e"]`` and ``"dispersion" in state`` work as on
    the plain dict, while printing the object shows its size and step
    rather than its field vectors.
    """

    def __init__(self, data: dict, *, run: str | None = None, path=None) -> None:
        self._data = dict(data)
        self.run = run
        self.path = None if path is None else Path(path)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def _summary_rows(self) -> list[tuple[str, object]]:
        rows: list[tuple[str, object]] = []
        for key in ("n_completed", "peak_energy", "peak_signal"):
            if key in self._data:
                rows.append((key, self._data[key]))
        for key in ("e", "h"):
            if key in self._data:
                rows.append((key, fmt_array(self._data[key])))
        groups = [k for k, v in self._data.items() if isinstance(v, dict)]
        if groups:
            rows.append(("groups", ", ".join(groups)))
        if self.path is not None:
            rows.append(("file", str(self.path)))
        return rows

    def __repr__(self) -> str:
        title = f"CheckpointState of run {self.run!r}" if self.run else "CheckpointState"
        return kv_block(title, self._summary_rows())

    def _repr_html_(self) -> str:
        title = f"CheckpointState of run {self.run!r}" if self.run else "CheckpointState"
        return html_kv(title, self._summary_rows())


class Project(ScatteringResultMixin):
    """Read-only view over a project directory.

    Lazily loads and caches the mesh, geometry and setup metadata on
    first access.  Implements the same scattering-result contract as
    the in-RAM :class:`~magnelio.analysis.scattering_td.ScatteringTDResult`
    (see :mod:`magnelio.analysis.result_interface`), so a separate
    post-processing script and the return value of ``run()`` are one
    and the same reader.
    """

    # Design: WP-S1 (static-model reader).

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not (self.path / "project.json").exists():
            raise FileNotFoundError(
                f"not a project directory (no project.json): {self.path}",
            )
        self._meta = None
        self._meta_stamp = None
        self._mesh = None
        self._geometry = None
        self._run_cache: dict = {}
        self._s_cache: dict = {}
        self._run_objects: dict = {}

    @property
    def meta(self) -> dict:
        """The parsed ``project.json`` contents.

        Re-read from disk whenever the file changed, for as long as the
        project is not finished — so a project opened while a solver
        writes it shows the current state without :meth:`refresh`.
        The file is small and replaced atomically by the writer, so the
        check is one ``stat`` per access and a parse only on a change.
        Once the stored status is terminal the parsed copy is kept.
        """
        if self._meta is not None and self._meta.get("status") in _TERMINAL_STATUS:
            return self._meta
        stamp = _file_stamp(self.path / "project.json")
        if self._meta is None or stamp != self._meta_stamp:
            with open(self.path / "project.json", encoding="utf-8") as fh:
                meta = json.load(fh)
            validate_schema(
                meta.get("schema_version"),
                f"{self.path}/project.json",
            )
            if self._meta is not None:
                # The index changed under a live reader: derived data
                # (S-matrix, cached runs) was computed from the old one.
                self._s_cache.clear()
            self._meta = meta
            self._meta_stamp = stamp
        return self._meta

    @property
    def setup(self) -> dict:
        """Analysis setup metadata stored at creation time."""
        return self.meta.get("setup", {})

    @property
    def status(self) -> str:
        """Project status: ``created`` → ``running`` → ``done`` / ``aborted`` / ``stale``.

        ``done`` means *every planned run* is done: the analysis
        pre-registers its runs as ``pending``, so the status holds at
        ``running`` in the gaps between sequential runs.  ``aborted``
        means a run ended on a graceful stop or an error and the
        analysis went with it — resume it, or run the analysis again.
        ``stale`` is not stored: it is ``running`` on disk with nobody
        writing — the solver process recorded on the project is gone
        from this host.  A project written on another host cannot be
        told stale and reads ``running`` until its writer finishes.
        """
        stored = self.meta.get("status", "unknown")
        if stored != "running":
            return stored
        index = self._run_index()
        if any(_writer_state(info) == "stale" for info in index.values()):
            return "stale"
        if not any(info.get("state") == "running" for info in index.values()):
            # Between runs, or planned runs only: the analysis process
            # itself is the writer to ask after.
            writer = self.meta.get("writer") or {}
            if _writer_gone(writer):
                return "stale"
        return stored

    def refresh(self) -> "Project":
        """Re-read the metadata and drop cached run / S-parameter data.

        A project that is not finished re-reads its metadata on its own
        whenever the file changes; call this to force a re-read on a
        finished project (a run resumed elsewhere), or to drop the
        cached run and S-parameter data.  The immutable model (mesh,
        geometry) is kept.  Returns ``self`` for chaining
        (``project.refresh().s_params``).
        """
        self._meta = None
        self._meta_stamp = None
        self._run_cache.clear()
        self._s_cache.clear()
        return self

    # ── wall clock (DD-253) ─────────────────────────────────────────
    #
    # The project's own clock is the marching of its runs: the first
    # start, the last finish, the summed wall time — the same three
    # numbers an in-RAM result carries, so a script reads them off
    # either.  The analysis call that produced the runs, setup
    # included, is the ``analysis`` entry of ``meta``.

    @property
    def started(self):
        """Wall-clock start (UTC datetime) of the first march, or ``None``."""
        stamps = [parse_utc(r.get("started")) for r in self._started_runs().values()]
        stamps = [s for s in stamps if s is not None]
        return min(stamps) if stamps else None

    @property
    def finished(self):
        """Wall-clock end (UTC datetime) of the last march; ``None`` while one marches."""
        runs = list(self._started_runs().values())
        if not runs or any(not r.get("finished") for r in runs):
            return None
        return max(parse_utc(r["finished"]) for r in runs)

    @property
    def elapsed(self) -> float | None:
        """Wall time of the marching [s], summed over the runs and their resumes."""
        values = [r.get("elapsed") for r in self._started_runs().values()]
        values = [float(v) for v in values if v is not None]
        return sum(values) if values else None

    @property
    def mesh(self):
        """The reconstructed :class:`~magnelio.mesh.mesher.Mesh` (lazy)."""
        if self._mesh is None:
            import h5py  # noqa: PLC0415

            with h5py.File(self.path / "mesh.h5", "r") as f:
                self._mesh = _load_mesh(f)
        return self._mesh

    @property
    def grid(self):
        """The mesh grid (shortcut for ``project.mesh.grid``)."""
        return self.mesh.grid

    @property
    def geometry(self):
        """The reconstructed geometry (:class:`LoadedGeometry`) or ``None``.

        Requires OCC.  Returns ``None`` if the project carries no
        geometry (``geometry.brep`` absent).
        """
        if self._geometry is None:
            from magnelio.geo.imported import ImportedSolid  # noqa: PLC0415

            brep = self.path / "geometry.brep"
            if not brep.exists():
                return None
            with open(self.path / "geometry.json", encoding="utf-8") as fh:
                gj = json.load(fh)
            occ_shapes = read_brep(brep)
            mats = [_material_from_dict(d) for d in gj["materials"]]
            background = (
                _material_from_dict(gj["background"]) if gj.get("background") is not None else None
            )
            # ThinWire shapes (DD-080) are stored as bare curves plus
            # ``kinds``/``radii`` metadata; v1 does not reconstruct them
            # as ThinWire objects.  The stored consolidated mesh already
            # carries their physics — only RE-meshing this loaded
            # geometry would lose the wires, hence the warning.
            kinds = gj.get("kinds", ["solid"] * len(mats))
            if "wire" in kinds:
                import warnings  # noqa: PLC0415

                warnings.warn(
                    "This project's geometry contains thin wires; the "
                    "loaded geometry exposes them as bare curve shapes "
                    "only.  Re-meshing it would drop the thin-wire "
                    "physics — use the stored mesh (project.mesh).",
                    UserWarning,
                    stacklevel=2,
                )
            names = gj.get("names", [None] * len(mats))
            colors = gj.get("colors") or [None] * len(mats)
            shapes = [
                ImportedSolid(s, m, name=n, color=tuple(c) if c is not None else None)
                for s, m, n, c, kind in zip(occ_shapes, mats, names, colors, kinds)
                if kind != "wire"
            ]
            self._geometry = LoadedGeometry(shapes, background)
        return self._geometry

    @property
    def eigenmodes(self):
        """The stored :class:`EigenmodeResult`, or ``None`` if absent (lazy)."""
        if not (self.path / "eigenmodes.h5").exists():
            return None
        return _load_eigenmodes(self.path, mesh=self.mesh)

    # -- time-domain runs ------------------------------------------------

    @property
    def runs(self) -> "_RunIndex":
        """The project's runs by name, each a :class:`Run`.

        A read-only mapping: ``project.runs["port1_mode0"]`` is the run
        object, ``list(project.runs)`` the names, and the mapping
        itself prints as a table of every run's state.  Run states:
        ``pending`` (planned by the analysis, not started — no run
        directory on disk yet) → ``running`` → ``done`` / ``aborted``;
        a ``running`` run whose writer process no longer exists reads
        as ``stale``.
        """
        return _RunIndex(self)

    def _run_index(self) -> dict:
        """The raw run index of ``project.json`` (name → entry dict)."""
        return self.meta.get("runs", {})

    def _run_info(self, name: str) -> dict:
        """One raw run entry; ``KeyError`` names the runs that exist."""
        index = self._run_index()
        try:
            return index[name]
        except KeyError:
            raise KeyError(f"no run {name!r} in project {self.path}; runs: {list(index)}") from None

    def _run_object(self, name: str) -> "Run":
        """The :class:`Run` handle of *name*, one per project and name."""
        run = self._run_objects.get(name)
        if run is None:
            run = self._run_objects[name] = Run(self, name)
        return run

    def _started_runs(self) -> dict:
        """Run-index entries whose run has started on disk (not pending)."""
        return {
            name: info
            for name, info in self._run_index().items()
            if info.get("state", "done") != "pending"
        }

    def _require_started(self, name: str) -> str:
        """Return ``name``, raising if the run is still pending."""
        if self._run_index().get(name, {}).get("state", "done") == "pending":
            raise ValueError(
                f"run {name!r} is pending (planned but not started yet); "
                f"started runs: {sorted(self._started_runs())}",
            )
        return name

    def _load_run(self, name: str) -> dict:
        # Only a finished run is safe to cache: a still-running run grows
        # on disk, so re-read it (SWMR) on every access until it is done.
        # The cache is keyed by what the index says about the run, so a
        # run resumed by another process (done → running → done, more
        # steps) is re-read rather than served from before the resume.
        info = self._run_index().get(name, {})
        state = info.get("state", "done")
        key = (info.get("n_steps"), info.get("finished"))
        cached = self._run_cache.get(name)
        if state == "done" and cached is not None and cached[0] == key:
            return cached[1]
        data = _read_run_results(self.path / "runs" / name)
        if state == "done":
            self._run_cache[name] = (key, data)
        return data

    def _all_runs(self) -> dict:
        # Pending runs have no directory on disk yet — only started
        # runs are loadable.
        return {name: self._load_run(name) for name in self._started_runs()}

    def _run_name_for_excited(
        self,
        excited: str | tuple[str, int] | None,
    ) -> str:
        """Resolve a run: by name, or by its excited ``(port, mode)`` pair."""
        index = self._run_index()
        names = list(index)
        if not names:
            raise ValueError(f"project {self.path} has no runs")
        if excited is None:
            if len(names) == 1:
                return self._require_started(names[0])
            raise ValueError(
                f"project holds {len(names)} runs {names}; pass the run name "
                f"(or a scattering run's excited pair) to select one",
            )
        if isinstance(excited, str) and excited in index:
            return self._require_started(excited)
        key = [excited, 0] if isinstance(excited, str) else [excited[0], excited[1]]
        for name, info in index.items():
            if info.get("excited") is not None and list(info["excited"]) == key:
                return self._require_started(name)
        raise KeyError(
            f"no run named {excited!r} and none excited at {tuple(key)!r}; runs: {names}",
        )

    def _run_excitations(self, name: str) -> list:
        """The Excitation dicts stored with a run (DD-224)."""
        return list(self._load_run(name).get("excitations", []))

    def result(self, name: str | tuple[str, int] | None = None):
        """One run as a :class:`~magnelio.analysis.TDResult`.

        Rebuilds the in-RAM result object of the run from the store:
        the recorded port signals, the sampled excitations, the energy
        trace and lazy readers for the run's monitors.  ``name`` is
        the run name (``run_1``, or the ``name=`` given to
        :meth:`~magnelio.AnalysisTD.run`); a scattering channel run may
        be named by its excited pair.  May be omitted when the project
        holds one run.
        """
        from magnelio.analysis._recipe import excitation_from_dict  # noqa: PLC0415
        from magnelio.analysis.time_domain import TDResult  # noqa: PLC0415

        run_name = self._run_name_for_excited(name)
        d = self._load_run(run_name)
        info = self._run_info(run_name)
        settings = RunSettings(
            **{
                **self.settings.__dict__,
                "dt": float(d["dt"]),
                "n_actual_steps": int(d["n_steps"]),
                "energy_stop_db": info.get("energy_stop_db"),
                "port_signal_stop_db": info.get("port_signal_stop_db"),
                "stop_reason": info.get("stop_reason"),
                "final_port_signal_db": info.get("final_port_signal_db"),
                "excitations": tuple(tuple(e) for e in info.get("excitations", ())),
            }
        )
        return TDResult(
            excitations=tuple(excitation_from_dict(e) for e in d.get("excitations", [])),
            dt=float(d["dt"]),
            n_steps=int(d["n_steps"]),
            signals=d["signals"],
            excitation_signals=d.get("excitation_signals", {}),
            energy_trace=d["energy_trace"],
            monitors=self.monitors_for(run_name),
            port_modes=d["port_modes"],
            port_normal_dx=d["port_normal_dx"],
            port_line_params=d["port_line_params"],
            settings=settings,
            name=run_name,
            started=parse_utc(info.get("started")),
            finished=parse_utc(info.get("finished")),
            elapsed=info.get("elapsed"),
        )

    def _first_started_run(self) -> str:
        started = self._started_runs()
        if not started:
            raise ValueError(
                f"project {self.path} has no started runs yet "
                f"(pending: {sorted(self._run_index())})",
            )
        return next(iter(started))

    @property
    def dt(self) -> float:
        return self._load_run(self._first_started_run())["dt"]

    @property
    def f_axis(self) -> np.ndarray:
        """Frequency axis of the stored S-matrix [Hz], ascending."""
        return self._load_run(self._first_started_run())["f_axis"]

    @property
    def signals(self) -> dict:
        """``{excited_key: {(port, mode): (V, I)}}`` across all runs."""
        out = {}
        for name, d in self._all_runs().items():
            out[d["excited"] if d["excited"] is not None else name] = d["signals"]
        return out

    @property
    def reference_signal(self):
        """Excitation waveform of the longest run (see ``ScatteringTDResult``)."""
        runs = self._all_runs()
        if not runs:
            raise ValueError(f"project {self.path} has no started runs")
        longest = max(runs.values(), key=lambda d: d["n_steps"])
        return longest["reference"]

    def energy_trace(self, excited: str | tuple[str, int] | None = None):
        """Stored ``(step, time, energy)`` trace of a run [structured array].

        The same as ``project.runs[name].energy_trace``; this form takes
        a scattering run's excited pair and may omit the selector on a
        one-run project.
        """
        return self._run_object(self._run_name_for_excited(excited)).energy_trace

    def _energy_traces(self) -> dict:
        """The non-empty energy traces of every started run, by run name."""
        out = {}
        for name in self._started_runs():
            trace = self._run_object(name).energy_trace
            if trace.size:
                out[name] = trace
        return out

    def _common_energy_stop(self) -> float | None:
        """The energy criterion shared by every started run, or ``None``."""
        levels = {self._run_object(n).energy_stop_db for n in self._started_runs()}
        levels.discard(None)
        return float(next(iter(levels))) if len(levels) == 1 else None

    def plot_energy(self, *, x: str = "time", ax=None):
        """Plot every run's stored energy in dB below its peak, one curve per run.

        The figure the progress line reports, for the whole project:
        the legend names the runs, and the energy criterion is a dashed
        line when every run shares one.  ``x`` is ``"time"``
        (nanoseconds) or ``"step"``; ``ax`` draws into existing axes.
        Returns ``(fig, ax)``.
        """
        from magnelio.post._plot_energy import plot_energy_traces  # noqa: PLC0415

        return plot_energy_traces(
            self._energy_traces(),
            energy_stop_db=self._common_energy_stop(),
            x=x,
            ax=ax,
        )

    # ── following a project someone else is writing (DD-255) ────────

    def _watch_signature(self) -> tuple:
        """What changes when the writer does: the index, and each run's sample count."""
        meta = self.meta
        runs = tuple((name, run.state, run.n_energy_samples) for name, run in self.runs.items())
        return (meta.get("modified"), meta.get("status"), runs)

    def _is_terminal(self) -> bool:
        return self.status in (*_TERMINAL_STATUS, "stale")

    def _watch_iter(self, interval: float, timeout: float | None):
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        last = None
        while True:
            try:
                signature = self._watch_signature()
            except OSError:
                # The writer is replacing a file this instant; look again.
                signature = last
            if signature != last:
                last = signature
                yield self
                if self._is_terminal():
                    return
            if deadline is not None and time.monotonic() >= deadline:
                return
            time.sleep(interval)

    def watch(self, interval: float = 2.0, *, on_change=None, timeout: float | None = None):
        """Follow this project while another process writes it.

        Polls the store every ``interval`` seconds and reports each
        change — a run starting or ending, a new energy sample, a
        status change — until the project is finished (``done``,
        ``aborted`` or ``stale``) or ``timeout`` seconds have passed.
        The first report comes at once, so the current state is always
        delivered, and so is the final one.

        Without ``on_change`` this is a generator yielding the project
        itself at every change::

            for proj in mio.open_project("magic_tee").watch():
                print(proj)                     # the run table, as it moves

        Print (or plot) what you want to see: inside a loop a bare
        expression such as ``proj.runs["port1_mode0"].energy_db``
        displays nothing, in a notebook as anywhere else — only the
        last expression of a cell is shown.  For the ready-made
        display that replaces itself at every change, see
        :meth:`follow`; for a panel that keeps moving while other
        cells run, :meth:`monitor`.

        With ``on_change`` the loop runs here: the callable is called
        with the project at every change, and the project is returned
        when the loop ends::

            mio.open_project("magic_tee").watch(on_change=lambda p: p.plot_energy())

        Parameters
        ----------
        interval : float, default 2.0
            Seconds between two looks at the store.
        on_change : callable, optional
            ``on_change(project)`` at every change, instead of yielding.
        timeout : float, optional
            Give up after this many seconds; the generator (or the
            call) then ends with the project in whatever state it is —
            check ``status`` to tell a finished project from one still
            marching.

        Returns
        -------
        generator, or Project
            The generator form without ``on_change``; the project
            itself with it.
        """
        if not interval > 0.0:
            raise ValueError(f"interval must be positive; got {interval!r}")
        if on_change is None:
            return self._watch_iter(float(interval), timeout)
        for _ in self._watch_iter(float(interval), timeout):
            on_change(self)
        return self

    def follow(self, interval: float = 2.0, *, timeout: float | None = None, stream=None):
        """Watch this project and show its state in place until it is finished.

        The zero-code form of :meth:`watch`: at every change the
        project's summary and run table are shown again, *replacing*
        the previous one rather than scrolling below it.  In a notebook
        the cell's output is cleared and the table redrawn; on a
        terminal the table is redrawn over its own lines; in a log
        file each table is appended.  Blocks until the project is
        finished (``done``, ``aborted`` or ``stale``) or ``timeout``
        seconds have passed, and returns the project.  Interrupting the
        kernel (Ctrl-C) ends it early.

        For a panel that keeps moving while you work in other cells,
        see :meth:`monitor`.

        Parameters
        ----------
        interval : float, default 2.0
            Seconds between two looks at the store.
        timeout : float, optional
            Give up after this many seconds.
        stream : file-like, optional
            Where the text form goes; defaults to ``sys.stdout``.  A
            notebook is recognised on its own.
        """
        if not interval > 0.0:
            raise ValueError(f"interval must be positive; got {interval!r}")
        painter = _InPlacePainter(stream)
        for _ in self._watch_iter(float(interval), timeout):
            painter.paint(self)
        return self

    def monitor(self, interval: float = 2.0, *, x: str = "time"):
        """A live panel for a notebook: the run table above the energy plot.

        Returns an ``ipywidgets`` box that a background thread refreshes
        every ``interval`` seconds — the project summary with its run
        table as HTML, and every run's stored energy in dB below the
        peak as a picture — until the project is finished.  Leave it
        as the last expression of a cell; the cell returns at once and
        the panel keeps moving.  ``panel.stop()`` ends the refresh
        early.  Needs the ``jupyter`` extra (``pip install
        'magnelio[jupyter]'``).

        Parameters
        ----------
        interval : float, default 2.0
            Seconds between refreshes.
        x : {"time", "step"}, default "time"
            Horizontal axis of the energy plot.
        """
        try:
            import ipywidgets  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Project.monitor() needs ipywidgets; install the notebook extra: "
                "pip install 'magnelio[jupyter]'",
            ) from exc
        if not interval > 0.0:
            raise ValueError(f"interval must be positive; got {interval!r}")
        return _build_monitor_panel(self.path, float(interval), x, ipywidgets)

    def monitors_for(
        self,
        excited: str | tuple[str, int] | None = None,
    ) -> dict:
        """Lazy monitor readers for one run, keyed by monitor name.

        Resolves each monitor kind by name into its matching lazy reader —
        :class:`_LoadedFieldMonitor` (MonitorFieldTime snapshots) and
        :class:`_LoadedFluxMonitor` (MonitorFluxTime scalar series) from the
        run's ``results.h5``, plus :class:`_LoadedFreqMonitor`
        (MonitorFieldFrequency DFT) from ``fields_freq.h5`` and
        :class:`_LoadedMonitorWallLoss` (per-tag dissipated fractions) from
        ``wall_loss.h5``.  The user only knows the name, not the kind or
        the file.  ``excited`` selects the run; it may be omitted
        when the project holds one run.
        """
        run_name = self._run_name_for_excited(excited)
        run_dir = self.path / "runs" / run_name
        # A scattering channel run renormalises its frequency monitors
        # with the channel's reference waveform; a general time-domain
        # run keeps them raw (several drives, no single reference —
        # ``TDResult.renormalize`` is the user's call).
        scattering = self._run_info(run_name).get("excited") is not None
        out: dict = {
            name: _LoadedFieldMonitor(run_dir, name, grid=self.grid)
            for name in _list_run_monitors(run_dir)
        }
        for name in _list_run_flux(run_dir):
            out[name] = _LoadedFluxMonitor(run_dir, name)
        for name in _list_run_freq(run_dir):
            out[name] = _LoadedFreqMonitor(
                run_dir,
                name,
                reference=(lambda rn=run_name: self._load_run(rn)["reference"])
                if scattering
                else None,
                grid=self.grid,
                incident=(lambda rn=run_name: self._incident_ratio(rn)) if scattering else None,
            )
        for name in _list_run_wall_loss(run_dir):
            out[name] = _LoadedMonitorWallLoss(run_dir, name)
        for name in _list_run_far_field(run_dir):
            out[name] = _LoadedFarFieldMonitor(
                run_dir,
                name,
                reference=(lambda rn=run_name: self._load_run(rn)["reference"])
                if scattering
                else None,
                incident=(lambda rn=run_name: self._incident_ratio(rn)) if scattering else None,
            )
        return out

    def _incident_ratio(self, run_name: str):
        """``(f_axis, |a(f)| / |W(f)|)`` of a run's excited channel (DD-198).

        The incident power wave the run launched per unit excitation
        waveform, derived from the stored port signals exactly as the
        S-matrix is; ``None`` when the run has no port signals.
        """
        from magnelio.post.modal_sparameters import (  # noqa: PLC0415
            compute_s_parameters,
        )

        cache = self.__dict__.setdefault("_incident_cache", {})
        if run_name in cache:
            return cache[run_name]
        d = self._load_run(run_name)
        signals = d.get("signals")
        if not signals or d.get("reference") is None or d.get("excited") is None:
            cache[run_name] = None
            return None
        f_axis = np.asarray(d["f_axis"], dtype=float)
        _, a_inc = compute_s_parameters(
            recorder_signals=signals,
            port_modes=d["port_modes"],
            excited=d["excited"],
            reference_signal=d["reference"],
            f_axis=f_axis,
            taper_signals=bool(self._run_info(run_name).get("taper_signals", False)),
            port_normal_dx=d["port_normal_dx"],
            port_line_params=d["port_line_params"],
            return_incident=True,
        )
        from magnelio.analysis.scattering_td import (  # noqa: PLC0415
            incident_amplitude_ratio,
        )

        ratio = incident_amplitude_ratio(
            d["reference"], f_axis, a_inc, d["port_modes"], d["excited"]
        )
        cache[run_name] = None if ratio is None else (f_axis, ratio)
        return cache[run_name]

    @property
    def monitors(self) -> dict:
        """Monitor readers of the project's sole run.

        Convenience for the common single-excitation case
        (``project.monitors["Ez_plane"].plot(t=…)``) — resolves time, flux,
        frequency and wall-loss monitors by name (see :meth:`monitors_for`).
        With more
        than one run the monitor name alone is ambiguous — use
        :meth:`monitors_for` with the excited pair.
        """
        names = list(self._run_index())
        if len(names) > 1:
            raise ValueError(
                f"project holds {len(names)} runs; monitor names are "
                f"per-run — use project.monitors_for(excited=…) to pick one",
            )
        return self.monitors_for(None)

    def export_paraview(
        self,
        excited: str | tuple[str, int] | None = None,
        *,
        glyph_percentile: float = 98.0,
        bake_state: bool = True,
    ) -> dict:
        """(Re-)generate the ready-to-open ParaView session for one run.

        The run close already generates this automatically;
        call this to regenerate with different options, or after the
        automatic export was skipped (``paraview=False``, missing
        ``pvpython``).  Writes ``paraview_open.py`` (open with
        ``paraview --script=…``), per-monitor data descriptors under
        ``paraview/``, and — when ``pvpython`` is available and
        *bake_state* — the double-clickable ``paraview.pvsm``, all in
        the run directory.

        Parameters
        ----------
        excited : str or tuple, optional
            Selects the run: a port name, or a ``(name, mode)`` pair.
            May be omitted when the project holds one run.
        glyph_percentile : float, default 98.0
            Percentile of the field-vector magnitude used as the glyph
            clip cap (edge singularities would otherwise dictate the
            arrow scaling).
        bake_state : bool, default True
            Bake ``paraview.pvsm`` via ``pvpython`` when available.

        Returns
        -------
        dict
            Written artefact paths (``script``, ``state`` or ``None``,
            ``monitors``); empty when there is nothing to visualise.
        """
        from magnelio.io.paraview import export_run_visualization  # noqa: PLC0415

        return export_run_visualization(
            self.path,
            self._run_name_for_excited(excited),
            glyph_percentile=glyph_percentile,
            bake_state=bake_state,
        )

    def export_paraview_eigenmodes(
        self,
        *,
        glyph_percentile: float = 98.0,
        bake_state: bool = True,
    ) -> dict:
        """(Re-)generate the ParaView session for the stored eigenmodes.

        The eigenmode counterpart of :meth:`export_paraview`; writing
        the eigenmode result already generates this automatically.
        Eigenmodes have no excitation and no time axis, so they belong
        to the project rather than to a run: the artefacts land in the
        project directory itself (``paraview_open.py``,
        ``paraview.pvsm``, ``paraview/eigenmodes.pvd`` and one ``.vtr``
        per mode).

        Stepping the ParaView time axis steps through the **modes** —
        degenerate pairs share an eigenfrequency exactly, so the
        frequency cannot serve as that axis; it travels as field data
        instead.  Fields are peak-normalised per mode, since an
        eigenvector carries no absolute amplitude, and the divisors are
        written alongside so the scaling stays reversible.

        Parameters
        ----------
        glyph_percentile : float, default 98.0
            Percentile of ``|E|`` used as the glyph magnitude clip cap
            (a field peak on a conductor edge would otherwise dictate
            the arrow scaling).
        bake_state : bool, default True
            Bake ``paraview.pvsm`` via ``pvpython`` when available.

        Returns
        -------
        dict
            Written artefact paths (``script``, ``state`` or ``None``,
            ``monitors``); empty when the project stores no eigenmodes.

        Examples
        --------
        >>> project = mio.AnalysisEigenmode(mesh=mesh, n_modes=4,
        ...                                 project="cavity").run()
        >>> project.export_paraview_eigenmodes()["script"]
        """
        from magnelio.io.paraview import export_eigenmode_visualization  # noqa: PLC0415

        return export_eigenmode_visualization(
            self.path,
            glyph_percentile=glyph_percentile,
            bake_state=bake_state,
        )

    def checkpoint_state(
        self,
        excited: str | tuple[str, int] | None = None,
    ) -> "CheckpointState | None":
        """Load a run's resume checkpoint (``state_dict``), or ``None``.

        Returns the solver state persisted at the last periodic / final
        / graceful-abort checkpoint as a :class:`CheckpointState` — a
        read-only mapping with the nested-dict shape of the solver's
        ``state_dict`` — or ``None`` if the run wrote no checkpoint
        (streaming without resume, or aborted before the first
        interval).  ``resume()`` feeds this straight into
        ``FITTimeDomainSolver.load_state_dict``; the ``n_completed``
        entry is the step the checkpoint corresponds to.
        """
        # Design: WP-S8 (resume from checkpoint).
        name = self._run_name_for_excited(excited)
        ckpt = self.path / "runs" / name / "checkpoint.h5"
        if not ckpt.exists():
            return None
        return CheckpointState(_read_state_dict_h5(ckpt), run=name, path=ckpt)

    def _band_mesh_operators(self) -> tuple:
        """``(M_eps, M_mu, C)`` of the stored mesh, for a band read (DD-230).

        These are the three quantities the phasor synthesis applies
        that belong to the *grid* rather than to a port.  Rebuilding
        them here is what keeps the per-port payload small — and it is
        cheap: three operator assemblies, none of the band pipeline's
        expensive construction (the contour-QZ ghost kernels drive the
        time stepping and play no part in the decomposition).
        """
        from magnelio._operators.curl import build_curl_matrix  # noqa: PLC0415
        from magnelio._operators.material_matrices import (  # noqa: PLC0415
            build_M_eps,
            build_M_mu,
        )

        mesh = self.mesh
        return (build_M_eps(mesh), build_M_mu(mesh), build_curl_matrix(mesh.grid))

    def _s_params(self, f_axis=None):
        from magnelio.post.modal_sparameters import (  # noqa: PLC0415
            compute_s_parameters,
        )
        from magnelio.post.sparameter_result import (  # noqa: PLC0415
            SParameterResult,
        )

        runs = self._all_runs()
        if not runs:
            raise ValueError(f"project {self.path} has no started runs")
        general = [name for name, d in runs.items() if d.get("excited") is None]
        if general:
            raise ValueError(
                f"runs {general} are general time-domain runs (AnalysisTD) without "
                f"an excited channel; S-parameters are a scattering result — read "
                f"them with project.result(name)",
            )
        # Cache the derived S-matrix only once every run is finished — a
        # partial (live) run yields a converging-but-not-final S.
        run_index = self._run_index()
        all_done = all(info.get("state", "done") == "done" for info in run_index.values())
        use_cache = f_axis is None and all_done
        if use_cache and "default" in self._s_cache:
            return self._s_cache["default"]
        if f_axis is None:
            f_axis = next(iter(runs.values()))["f_axis"]
        band_ops = None
        cols = []
        for name, d in runs.items():
            if run_index.get(name, {}).get("port_model") == "band":
                # The band pipeline decomposes per frequency against the
                # port's own chain, so the mesh-side operators are needed
                # (DD-230).  Built once for the whole project — they are
                # functions of the grid, shared by every run and port.
                if band_ops is None:
                    band_ops = self._band_mesh_operators()
                s_dict, z_ref = _band_s_dict(d, f_axis, band_ops)
            else:
                s_dict, z_ref = compute_s_parameters(
                    recorder_signals=d["signals"],
                    port_modes=d["port_modes"],
                    excited=d["excited"],
                    reference_signal=d["reference"],
                    f_axis=f_axis,
                    taper_signals=bool(run_index.get(name, {}).get("taper_signals", False)),
                    port_normal_dx=d["port_normal_dx"],
                    port_line_params=d["port_line_params"],
                    return_reference=True,
                    port_reference_scale=d.get("port_reference_scale"),
                )
            cols.append(
                SParameterResult.from_single_excitation(
                    s_dict,
                    d["excited"],
                    f_axis,
                    reference_impedances=z_ref,
                )
            )
        res = cols[0] if len(cols) == 1 else SParameterResult.merge(cols)
        if use_cache:
            self._s_cache["default"] = res
        return res

    @property
    def s_params(self):
        """The full S-matrix, derived on read from the stored signals."""
        return self._s_params()

    def _channel_cutoffs(self) -> dict | None:
        """Per-channel cut-off frequency [Hz] from the stored modes."""
        from magnelio.analysis.scattering_td import (  # noqa: PLC0415
            _cutoffs_from_port_modes,
        )

        try:
            run = self._load_run(self._first_started_run())
        except Exception:  # noqa: BLE001 — no started run yet
            return None
        return _cutoffs_from_port_modes(run.get("port_modes"))

    def _deembed_data(self) -> tuple:
        """Port line records backing :meth:`deembed` (result contract).

        The line records are properties of the ports, not of the
        excitation, so any started run's copy serves.
        """
        run = self._load_run(self._first_started_run())
        return (
            float(run["dt"]),
            run.get("port_line_params"),
            run.get("port_normal_dx"),
            run.get("port_modes"),
            run.get("port_band") or None,
        )

    @property
    def settings(self) -> RunSettings:
        """Settings the stored run was produced with (result contract).

        Filled from the stored recipe and run data; entries the store
        does not (yet) record are ``None``.
        """
        recipe = self.setup.get("recipe", {})
        dt = None
        n_actual = None
        try:
            run = self._load_run(self._first_started_run())
            dt = float(run["dt"])
            ref = run.get("reference")
            if ref is not None:
                n_actual = int(len(ref.values))
        except Exception:  # noqa: BLE001 — no started run yet
            pass
        run_info = next(
            (info for info in self._run_index().values() if info.get("state") != "pending"),
            {},
        )
        return RunSettings(
            f_max=recipe.get("f_max"),
            f_min=recipe.get("f_min"),
            n_freq=recipe.get("n_freq"),
            dt=dt,
            n_actual_steps=n_actual,
            energy_stop_db=run_info.get("energy_stop_db"),
            port_signal_stop_db=run_info.get("port_signal_stop_db"),
            taper_signals=run_info.get("taper_signals"),
            stop_reason=run_info.get("stop_reason"),
            final_port_signal_db=run_info.get("final_port_signal_db"),
            precision=recipe.get("precision"),
            # Both pipelines stream since DD-230; the run index records
            # which one wrote this run.
            port_model_used=(run_info.get("port_model", "modal") if recipe else None),
        )

    @property
    def params(self) -> dict:
        """Free-form user parameters stored with the project.

        Whatever dict the analysis was given as ``params=`` — design
        variables, sweep coordinates, notes.  Empty when none were
        stored.
        """
        return dict(self.setup.get("params", {}))

    @property
    def channels(self) -> tuple:
        """Observed ``(port_name, mode_idx)`` pairs, in S-matrix order."""
        return self._s_params().channels

    @property
    def excitations(self) -> tuple:
        """Excited ``(port_name, mode_idx)`` pairs — the S-matrix columns stored."""
        return self._s_params().excitations

    def S(self, out_port, in_port, *, mode_out=0, mode_in=0, f_axis=None):
        """S-parameter column, derived on read (optionally on a custom ``f_axis``)."""
        return self._s_params(f_axis).S(
            out_port,
            in_port,
            mode_out=mode_out,
            mode_in=mode_in,
        )

    def db(self, out_port, in_port, *, mode_out=0, mode_in=0, floor_db=-200.0, f_axis=None):
        """S-parameter magnitude in dB, derived on read (see :meth:`S`).

        ``floor_db`` clamps the result from below so an exact zero stays
        plottable instead of becoming ``-inf``.
        """
        return self._s_params(f_axis).db(
            out_port,
            in_port,
            mode_out=mode_out,
            mode_in=mode_in,
            floor_db=floor_db,
        )

    def a(self, port, mode=0, *, excited=None, f_ref=None, destagger=True):
        """Incident power-wave time series ``a(t)`` (see ``ScatteringTDResult.a``)."""
        return self._power_wave(port, mode, excited, f_ref, +1.0, destagger)

    def b(self, port, mode=0, *, excited=None, f_ref=None, destagger=True):
        """Outgoing power-wave time series ``b(t)`` (see ``ScatteringTDResult.b``)."""
        return self._power_wave(port, mode, excited, f_ref, -1.0, destagger)

    def _power_wave(self, port, mode, excited, f_ref, sign, destagger):
        import math  # noqa: PLC0415

        from magnelio.post.modal_sparameters import (  # noqa: PLC0415
            destaggered_power_waves,
        )

        # Same contract as ScatteringTDResult.a/b (DD-057): a band run's
        # recorded channels are fixed subspace projections whose
        # incident/outgoing split is defined per frequency, so a scalar
        # (V ∓ Z·I)/2 has no calibrated Z here either.  A stored run must
        # refuse it as loudly as an in-memory one (DD-230).
        if any(info.get("port_model") == "band" for info in self._run_index().values()):
            raise ValueError(
                "time-domain power waves are not available on band-"
                "DTBC results: the band port's recorded V/I channels "
                "are fixed subspace projections whose incident/"
                "outgoing split is defined per frequency through the "
                "true-mode phasors — a scalar (V ∓ Z·I)/2 split has no "
                "calibrated Z and would show the incident wave in b.  "
                "Inspect the raw V/I via the run's signals and take "
                "S-parameters from project.S / project.s_params.",
            )
        from magnelio.signals.signal_1d import Signal1D  # noqa: PLC0415

        d = self._load_run(self._run_name_for_excited(excited))
        chan = (port, mode)
        if chan not in d["signals"]:
            raise KeyError(
                f"channel {chan!r} not recorded; available: {sorted(d['signals'].keys())}",
            )
        modes = d["port_modes"][port]
        if not 0 <= mode < len(modes):
            raise ValueError(
                f"mode index {mode} out of range for port {port!r}",
            )
        if f_ref is None:
            fa = d["f_axis"]
            f_ref = 0.5 * (float(fa[0]) + float(fa[-1]))
        Z = complex(modes[mode].z_modal(2.0 * math.pi * float(f_ref)))
        if abs(Z.imag) > 1e-9 * abs(Z):
            raise ValueError(
                f"z_modal({f_ref:.4g} Hz) = {Z:.4g} is not real; pass f_ref=",
            )
        V_sig, I_sig = d["signals"][chan]
        name = "a" if sign > 0 else "b"

        if destagger:
            a_sig, b_sig = destaggered_power_waves(
                V_sig,
                I_sig,
                modes[mode],
                z_ref=Z.real,
                normal_dx=d["port_normal_dx"].get(port),
                line_params=d["port_line_params"].get(chan),
            )
            sig = a_sig if sign > 0 else b_sig
            return Signal1D(t=sig.t, values=sig.values, dt=sig.dt, label=f"{name}({port},{mode})")

        sqrt_z = math.sqrt(Z.real)
        V = V_sig.values
        I = I_sig.values
        I_aligned = np.empty_like(I)
        I_aligned[:-1] = 0.5 * (I[:-1] + I[1:])
        I_aligned[-1] = I[-1]
        return Signal1D(
            t=V_sig.t,
            values=0.5 * (V / sqrt_z + sign * sqrt_z * I_aligned),
            dt=V_sig.dt,
            label=f"{name}({port},{mode})",
        )

    # ── how a project introduces itself (DD-254) ────────────────────

    def _summary_rows(self) -> list[tuple[str, object]]:
        """The key/value lines above the run table."""
        meta = self.meta
        rows: list[tuple[str, object]] = [
            ("analysis", meta.get("setup", {}).get("analysis") or "—"),
            ("status", self.status),
        ]
        stamp = meta.get("analysis") or {}
        started = parse_utc(stamp.get("started"))
        if started is not None:
            if stamp.get("finished"):
                rows.append(
                    (
                        "last call",
                        f"finished {fmt_value(parse_utc(stamp['finished']))} "
                        f"in {format_seconds(stamp.get('elapsed'))}",
                    )
                )
            else:
                since = (utc_now() - started).total_seconds()
                rows.append(
                    ("last call", f"started {fmt_value(started)}, {format_clock(since)} ago")
                )
        rows.append(("created", parse_utc(meta.get("created"))))
        rows.append(("runs", len(self._run_index())))
        return rows

    def __repr__(self) -> str:
        try:
            head = kv_block(f"Project {self.path}", self._summary_rows())
            table = self.runs._table_text()
        except Exception as exc:  # noqa: BLE001 — a repr must never raise
            return f"Project({str(self.path)!r}) — cannot read project.json: {exc}"
        return f"{head}\n{table}" if table else head

    def _repr_html_(self) -> str:
        try:
            head = html_kv(f"Project {self.path}", self._summary_rows())
            table = self.runs._table_html()
        except Exception as exc:  # noqa: BLE001
            return html.escape(f"Project({str(self.path)!r}) — cannot read project.json: {exc}")
        return head + table


def open_project(path: str | Path) -> Project:
    """Open a project directory for read-only post-processing.

    Parameters
    ----------
    path : str or Path
        A project directory created by :meth:`ProjectStore.create` (or,
        in later work packages, by an ``Analysis.run(project=...)``).

    Returns
    -------
    Project
    """
    return Project(path)
