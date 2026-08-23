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

import datetime
import json
import os
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

from magnelio.analysis.result_interface import RunSettings, ScatteringResultMixin
from magnelio.io._schema import (
    SCHEMA_VERSION,
    validate_schema,
)

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
    from magnelio.mesh.grid import GridLines  # noqa: PLC0415
    from magnelio.mesh.mesher import Mesh  # noqa: PLC0415

    grid = GridLines(
        x=f["grid/x"][()],
        y=f["grid/y"][()],
        z=f["grid/z"][()],
    )
    mg = f["mesh"]
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
    return Mesh(
        ports=ports,
        elements=elements,
        grid=grid,
        material_id=mg["material_id"][()],
        material_library=library,
        pec_mask_edges=mg["pec_mask_edges"][()],
        edge_material=edge_material,
        face_material=face_material,
        pec_surface=pec_surface,
        boundary_conditions=boundary_conditions,
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


def _update_meta(path: Path, fn) -> None:
    """Read ``project.json``, apply ``fn`` in place, write it back atomically."""
    with open(path / "project.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    fn(meta)
    meta["modified"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
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

    The live streaming sink (:class:`_ScatteringRunSink`) drives one
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
        excited: tuple[str, int],
        f_axis,
        channels: list,
        port_modes: dict,
        port_normal_dx: dict,
        port_line_params: dict,
        monitors=None,
        grid=None,
    ) -> None:
        import h5py  # noqa: PLC0415

        run_dir.mkdir(parents=True, exist_ok=True)
        self._f = f = h5py.File(run_dir / "results.h5", "w", libver="latest")
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["dt"] = float(dt)
        f.attrs["excited_name"] = excited[0]
        f.attrs["excited_mode"] = int(excited[1])
        f.create_dataset("f_axis", data=np.asarray(f_axis, dtype=float))

        self._ref = f.create_dataset(
            "reference",
            shape=(0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(4096,),
        )
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
            fg.attrs["plane_normal"] = mon.plane[0]
            fg.attrs["plane_position"] = float(mon.plane[1])
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

    def append(self, v_by_channel: dict, i_by_channel: dict, reference) -> None:
        """Append a block of V/I/reference samples (equal length) and flush.

        ``reference`` is grown last, so its length is the committed
        sample count a live reader observes.
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


class _ScatteringRunSink:
    """Live streaming sink attached to the FIT-TD solver (DD-070, WP-S5).

    Owns one run's :class:`_RunResultWriter` and, at every solver flush,
    pulls the newly recorded V/I tail from the
    :class:`~magnelio.ports.recorder.PortSignalRecorder`, sampling the
    reference waveform in lockstep so a separate reader process can
    derive converging S-parameters mid-run.  The solver calls
    :meth:`flush` at each energy-check interval and once at exit; the
    analysis calls :meth:`close` to flip the run to ``done`` in
    ``project.json``.
    """

    def __init__(
        self,
        writer,
        recorder,
        waveform_fn,
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
        self._recorder = recorder
        self._waveform_fn = waveform_fn
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
            ref = np.array(
                [self._waveform_fn((off + k) * self._dt) for k in range(self._n_flushed, n_rec)],
                dtype=float,
            )
            self._writer.append(
                {key: vi[0] for key, vi in tail.items()},
                {key: vi[1] for key, vi in tail.items()},
                ref,
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
                # Leading newline so the confirmation survives the live
                # "\r…" status line; ``step + 1`` == the checkpoint's
                # n_completed (the flush ran after _resume_step advanced).
                print(
                    f"\n  [checkpoint] on-demand snapshot written at step "
                    f"{step + 1} ({self._checkpoint_path})",
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
    ) -> None:
        """Drain the final tail, close the file, finalise ``project.json``.

        A ``done`` close writes a final checkpoint so the completed run
        can be resumed to run longer (DD-070).  An ``aborted`` close does
        not — the graceful-abort checkpoint was already written at the
        consistent break point by the solver.  ``stop_reason`` /
        ``final_port_signal_db`` book why the run ended (and the |V|
        envelope level below peak it reached) into the run index.
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
        excited = (str(f.attrs["excited_name"]), int(f.attrs["excited_mode"]))
        f_axis = f["f_axis"][()]

        chan_items = []  # (label, mode, V_ds, I_ds)
        lengths = [int(f["reference"].shape[0])]
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

        eg = f["energy"]
        # Same common-prefix rule as the signals: the three energy
        # sub-streams advance in lockstep but a live reader may see them
        # flushed to different lengths, so slice to their minimum.
        m = min(int(eg["step"].shape[0]), int(eg["time"].shape[0]), int(eg["energy"].shape[0]))
        energy_trace = np.empty(
            m,
            dtype=[
                ("step", int),
                ("time", float),
                ("energy", float),
            ],
        )
        if m > 0:
            energy_trace["step"] = eg["step"][:m]
            energy_trace["time"] = eg["time"][:m]
            energy_trace["energy"] = eg["energy"][:m]

        port_modes = {}
        port_normal_dx = {}
        port_line_params = {}
        for label in f["ports"]:
            p = f["ports"][label]
            port_modes[label] = [_mode_from_dict(d) for d in json.loads(p.attrs["modes"])]
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
        signals=signals,
        reference=reference,
        dt=dt,
        n_steps=n_steps,
        energy_trace=energy_trace,
        port_modes=port_modes,
        port_normal_dx=port_normal_dx,
        port_line_params=port_line_params,
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
    """The ``MonitorFarField`` subset of a list (DD-173)."""
    if not monitors:
        return []
    from magnelio.monitors.far_field import MonitorFarField  # noqa: PLC0415

    return [m for m in monitors if isinstance(m, MonitorFarField)]


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
            self.plane = (str(fg.attrs["plane_normal"]), float(fg.attrs["plane_position"]))
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

        mon = MonitorFluxTime(
            plane=self.plane,
            name=self.name,
        )
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

    def __init__(self, run_dir: Path, name: str, reference=None, grid=None) -> None:
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
        self._spectrum = None
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
        return self._squeeze_spatial(divide_by_spectrum(self._read_bins(comp), src))

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
        with h5py.File(self._run_dir / "fields_freq.h5", "r") as f:
            bg = f[self.name]["bins"]
            for comp in self._components:
                acc = DFTAccumulator(self.freqs, (nx, ny, nz))
                acc._bins[...] = bg[comp][()]
                mon._accumulators[comp] = acc
        mon._mirrors = self._mirrors
        mon._grid = self._grid
        # Carry the run's reference across, or the hydrated monitor would
        # refuse to hand out data it cannot put a unit on.
        mon._source_spectrum = self._source_spectrum()
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
    """Atomically write MonitorFarField results to *path*.

    ``dumps`` maps monitor name -> :meth:`MonitorFarField.result_dump`.
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
            for key in ("source_spectrum", "accepted_power"):
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
        for key in ("source_spectrum", "accepted_power"):
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
    """Lazy reader over one persisted ``MonitorFarField`` (DD-173).

    Hydrates the dump into a result-serving monitor on first access and
    delegates, so the reader serves exactly the in-RAM API —
    :attr:`f`, :meth:`result` and the pattern plots.
    """

    def __init__(self, run_dir: Path, name: str, reference=None):
        self._run_dir = Path(run_dir)
        self.name = name
        self._reference = reference
        self._monitor = None

    def _hydrate(self):
        if self._monitor is None:
            from magnelio.monitors.far_field import MonitorFarField  # noqa: PLC0415

            dump = _read_far_field_dump(self._run_dir, self.name)
            self._monitor = MonitorFarField.from_result_dump(dump)
            if not self._monitor.is_renormalized and self._reference is not None:
                # The streamed run renormalises after its final flush, so
                # the file carries raw bins; the run's reference waveform
                # supplies the divisor on read (the _LoadedFreqMonitor
                # pattern).
                self._monitor.renormalize(self._reference())
        return self._monitor

    @property
    def f(self) -> np.ndarray:
        """Frequency axis [Hz]."""
        return self._hydrate().f

    @property
    def freqs(self) -> np.ndarray:
        """Alias of :attr:`f` (the in-RAM monitor's attribute name)."""
        return self._hydrate().freqs

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

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
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

        ``planned`` is an iterable of ``(run_name, excited)`` pairs, one
        per excitation the caller is about to stream.  Registering them
        up front closes the status gap between sequential runs: without
        it, finishing run *k* while run *k+1* is not yet in the index
        made :meth:`_finalize_run` report ``status = "done"`` for a
        project that was still mid-analysis.  A
        ``pending`` entry counts as not-done, so the project status
        stays ``"running"`` until the last planned run finishes.

        Existing entries are left untouched (fill-in: a second analysis
        adding excitations must not clobber ``done``/``aborted`` runs).
        ``pending`` entries carry no run directory on disk; they are
        replaced wholesale when :meth:`open_scattering_run` starts the
        run.
        """

        def _upd(meta: dict) -> None:
            runs = meta.setdefault("runs", {})
            for run_name, excited in planned:
                runs.setdefault(
                    _safe_run_name(run_name),
                    {
                        "excited": [excited[0], int(excited[1])],
                        "state": "pending",
                    },
                )
            meta["status"] = "running"

        _update_meta(self.path, _upd)

    def open_scattering_run(
        self,
        run_name: str,
        *,
        excited: tuple[str, int],
        dt: float,
        f_axis,
        channels,
        port_modes: dict,
        port_normal_dx: dict,
        port_line_params: dict,
        waveform_fn,
        recorder,
        port_model: str = "modal",
        energy_stop_db: float | None = None,
        port_signal_stop_db: float | None = None,
        total_time_steps: int | None = None,
        taper_signals: bool = False,
        monitors=None,
        grid=None,
    ) -> "_ScatteringRunSink":
        """Open a live streaming sink for one scattering excitation.

        Declares the run's resizable ``results.h5`` streams (HDF5-SWMR),
        registers the run in ``project.json`` as ``running`` (replacing a
        ``pending`` pre-registration, see :meth:`register_planned_runs`),
        and returns
        a solver-attachable :class:`_ScatteringRunSink`.  The solver
        appends the V/I and energy tails during ``run()``; call
        :meth:`_ScatteringRunSink.close` when the run finishes to flip its
        state to ``done``.  ``run_name`` is sanitised for the filesystem.

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
            excited=excited,
            f_axis=f_axis,
            channels=list(channels),
            port_modes=port_modes,
            port_normal_dx=port_normal_dx,
            port_line_params=port_line_params,
            monitors=monitors,
            grid=grid,
        )

        def _upd(meta: dict) -> None:
            meta.setdefault("runs", {})[safe] = {
                "excited": [excited[0], int(excited[1])],
                "n_steps": 0,
                "dt": float(dt),
                "port_model": port_model,
                "energy_stop_db": energy_stop_db,
                "port_signal_stop_db": port_signal_stop_db,
                "total_time_steps": total_time_steps,
                "taper_signals": bool(taper_signals),
                "state": "running",
            }
            meta["status"] = "running"

        _update_meta(self.path, _upd)
        return _ScatteringRunSink(
            writer,
            recorder,
            waveform_fn,
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

    def reopen_scattering_run(
        self,
        run_name: str,
        *,
        recorder,
        waveform_fn,
        dt: float,
        n_keep: int,
        step_offset: int,
        monitors=None,
        grid=None,
        monitor_keep: dict | None = None,
        flux_keep: dict | None = None,
    ) -> "_ScatteringRunSink":
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
            meta["status"] = "running"

        _update_meta(self.path, _upd)
        return _ScatteringRunSink(
            writer,
            recorder,
            waveform_fn,
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
    ) -> None:
        """Flip a streamed run to its terminal ``state`` in project.json.

        ``stop_reason`` records why the marching ended ("energy",
        "port_signal", "port_signal_stall", "runtime_cap", "steps",
        "aborted"), ``final_port_signal_db`` the |V|-envelope level below
        peak at the stop — schema-additive, absent on older projects.
        """

        def _upd(meta: dict) -> None:
            run = meta.setdefault("runs", {}).setdefault(safe, {})
            run["state"] = state
            run["n_steps"] = int(n_steps)
            if stop_reason is not None:
                run["stop_reason"] = str(stop_reason)
            if final_port_signal_db is not None:
                run["final_port_signal_db"] = float(final_port_signal_db)
            states = [r.get("state") for r in meta["runs"].values()]
            meta["status"] = "done" if all(s == "done" for s in states) else "running"

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
        self._mesh = None
        self._geometry = None
        self._run_cache: dict = {}
        self._s_cache: dict = {}

    @property
    def meta(self) -> dict:
        """The parsed ``project.json`` contents."""
        if self._meta is None:
            with open(self.path / "project.json", encoding="utf-8") as fh:
                meta = json.load(fh)
            validate_schema(
                meta.get("schema_version"),
                f"{self.path}/project.json",
            )
            self._meta = meta
        return self._meta

    @property
    def setup(self) -> dict:
        """Analysis setup metadata stored at creation time."""
        return self.meta.get("setup", {})

    @property
    def status(self) -> str:
        """Project status: ``created`` → ``running`` → ``done``.

        ``done`` means *every planned run* is done: the analysis
        pre-registers its runs as ``pending``, so the status holds at
        ``running`` in the gaps between sequential runs — a live watcher
        may poll ``project.refresh().status`` without racing the writer.
        """
        return self.meta.get("status", "unknown")

    def refresh(self) -> "Project":
        """Re-read the metadata and drop cached run / S-parameter data.

        Call this on a *live* project to pick up newly appended steps, a
        newly added run, or a status change written by a concurrent
        solver.  The immutable model (mesh, geometry) is
        kept.  Returns ``self`` for chaining
        (``project.refresh().s_params``).
        """
        self._meta = None
        self._run_cache.clear()
        self._s_cache.clear()
        return self

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
    def runs(self) -> dict:
        """The run index from ``project.json`` (name → excited/state/…).

        Run states: ``pending`` (planned by the analysis, not started —
        no run directory on disk yet) → ``running`` → ``done`` /
        ``aborted``.  Live watchers iterating this index should skip
        ``pending`` entries; per-run readers raise on them.
        """
        return self.meta.get("runs", {})

    def _started_runs(self) -> dict:
        """Run-index entries whose run has started on disk (not pending)."""
        return {
            name: info for name, info in self.runs.items() if info.get("state", "done") != "pending"
        }

    def _require_started(self, name: str) -> str:
        """Return ``name``, raising if the run is still pending."""
        if self.runs.get(name, {}).get("state", "done") == "pending":
            raise ValueError(
                f"run {name!r} is pending (planned but not started yet); "
                f"started runs: {sorted(self._started_runs())}",
            )
        return name

    def _load_run(self, name: str) -> dict:
        # Only a finished run is safe to cache: a still-running run grows
        # on disk, so re-read it (SWMR) on every access until it is done.
        state = self.runs.get(name, {}).get("state", "done")
        if state == "done" and name in self._run_cache:
            return self._run_cache[name]
        data = _read_run_results(self.path / "runs" / name)
        if state == "done":
            self._run_cache[name] = data
        return data

    def _all_runs(self) -> dict:
        # Pending runs have no directory on disk yet — only started
        # runs are loadable.
        return {name: self._load_run(name) for name in self._started_runs()}

    def _run_name_for_excited(
        self,
        excited: str | tuple[str, int] | None,
    ) -> str:
        names = list(self.runs)
        if not names:
            raise ValueError(f"project {self.path} has no runs")
        if excited is None:
            if len(names) == 1:
                return self._require_started(names[0])
            raise ValueError(
                f"project holds {len(names)} runs; pass excited= to select one",
            )
        key = [excited, 0] if isinstance(excited, str) else [excited[0], excited[1]]
        for name, info in self.runs.items():
            if list(info["excited"]) == key:
                return self._require_started(name)
        raise KeyError(
            f"no run excited at {tuple(key)!r}; "
            f"available: {[tuple(v['excited']) for v in self.runs.values()]}",
        )

    def _first_started_run(self) -> str:
        started = self._started_runs()
        if not started:
            raise ValueError(
                f"project {self.path} has no started runs yet (pending: {sorted(self.runs)})",
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
            out[d["excited"]] = d["signals"]
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
        """Stored ``(step, time, energy)`` trace of a run [structured array]."""
        return self._load_run(self._run_name_for_excited(excited))["energy_trace"]

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
                reference=lambda rn=run_name: self._load_run(rn)["reference"],
                grid=self.grid,
            )
        for name in _list_run_wall_loss(run_dir):
            out[name] = _LoadedMonitorWallLoss(run_dir, name)
        for name in _list_run_far_field(run_dir):
            out[name] = _LoadedFarFieldMonitor(
                run_dir,
                name,
                reference=lambda rn=run_name: self._load_run(rn)["reference"],
            )
        return out

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
        names = list(self.runs)
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
    ) -> dict | None:
        """Load a run's resume checkpoint (``state_dict``), or ``None``.

        Returns the nested-dict solver state persisted at the last
        periodic / final / graceful-abort checkpoint, or
        ``None`` if the run wrote no checkpoint (streaming without resume,
        or aborted before the first interval).  ``resume()`` feeds
        this straight into ``FITTimeDomainSolver.load_state_dict``; the
        ``n_completed`` entry is the step the checkpoint corresponds to.
        """
        # Design: WP-S8 (resume from checkpoint).
        name = self._run_name_for_excited(excited)
        ckpt = self.path / "runs" / name / "checkpoint.h5"
        if not ckpt.exists():
            return None
        return _read_state_dict_h5(ckpt)

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
        # Cache the derived S-matrix only once every run is finished — a
        # partial (live) run yields a converging-but-not-final S.
        all_done = all(info.get("state", "done") == "done" for info in self.runs.values())
        use_cache = f_axis is None and all_done
        if use_cache and "default" in self._s_cache:
            return self._s_cache["default"]
        if f_axis is None:
            f_axis = next(iter(runs.values()))["f_axis"]
        run_index = self.runs
        cols = []
        for name, d in runs.items():
            s_dict = compute_s_parameters(
                recorder_signals=d["signals"],
                port_modes=d["port_modes"],
                excited=d["excited"],
                reference_signal=d["reference"],
                f_axis=f_axis,
                taper_signals=bool(run_index.get(name, {}).get("taper_signals", False)),
                port_normal_dx=d["port_normal_dx"],
                port_line_params=d["port_line_params"],
            )
            cols.append(
                SParameterResult.from_single_excitation(
                    s_dict,
                    d["excited"],
                    f_axis,
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
            (info for info in self.runs.values() if info.get("state") != "pending"),
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
            # The streamed pipeline is modal-only (band + project= is
            # unsupported), so a stored run always used "modal".
            port_model_used="modal" if recipe else None,
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

    def __repr__(self) -> str:
        return f"Project(path={str(self.path)!r}, status={self.status!r}, runs={len(self.runs)})"


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
