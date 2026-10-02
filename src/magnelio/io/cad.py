"""CAD exchange for solids and free sheets via STEP and BREP.

Reading a model that was drawn elsewhere is the normal way to get a
real device into a simulation.  Two formats are supported, and the
difference between them is what they carry *besides* the geometry:

**STEP** (``.step`` / ``.stp``) is the interchange format every CAD
system writes.  Beyond the geometry it records the file's length unit,
part names and display colours — which is exactly what the
import needs: the unit makes the geometry unambiguous, and the names
are the handle materials are assigned against.

**BREP** (``.brep``) is the geometry kernel's own dump.  It is exact
and lossless, but it carries no unit and no names at all, so
:func:`import_brep` requires the unit to be stated explicitly.

Neither format carries the parametric history of the model (both store
finished boundary representations) or the material physics, so
materials are assigned here, by name — see :func:`import_step`.
"""

from __future__ import annotations

import fnmatch
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from magnelio.geo.imported import ImportedSheet, ImportedSolid
from magnelio.geo.operations import Group
from magnelio.geo.shape import Shape

if TYPE_CHECKING:
    from magnelio.materials.material import Material

# Meters per unit, for the length units CAD systems actually emit.
_UNIT_FACTORS = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
    "in": 0.0254,
    "mil": 2.54e-5,
}


def _require_occ() -> None:
    """Fail with the install hint rather than with a bare import error."""
    try:
        import OCC  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "pythonocc-core is required to read CAD files. "
            "Install via: conda install -c conda-forge pythonocc-core"
        ) from exc


def _unit_factor(unit, what: str = "unit") -> float:
    """Meters per source unit, from a name or an explicit factor."""
    if isinstance(unit, str):
        factor = _UNIT_FACTORS.get(unit.strip().lower())
        if factor is None:
            known = ", ".join(repr(u) for u in _UNIT_FACTORS if u != "µm")
            raise ValueError(
                f"{what} must be one of {known} — or a number giving the "
                f"length of one unit in meters; got {unit!r}."
            )
        return factor
    try:
        factor = float(unit)
    except (TypeError, ValueError):
        raise TypeError(
            f"{what} must be a unit name or a number giving the length of "
            f"one unit in meters; got {unit!r}."
        ) from None
    if not (math.isfinite(factor) and factor > 0.0):
        raise ValueError(f"{what} must be finite and positive; got {unit!r}.")
    return factor


def _bodies_of(shape) -> tuple[list, list, int]:
    """Collect independent solids and free faces without duplicating solid faces."""
    from OCC.Core.TopAbs import (  # noqa: PLC0415
        TopAbs_COMPOUND,
        TopAbs_COMPSOLID,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
    )
    from OCC.Core.TopoDS import TopoDS_Iterator, topods  # noqa: PLC0415

    solids, sheets = [], []
    unsupported = 0

    def visit(node):
        nonlocal unsupported
        kind = node.ShapeType()
        if kind == TopAbs_SOLID:
            solids.append(topods.Solid(node))
        elif kind == TopAbs_FACE:
            sheets.append(topods.Face(node))
        elif kind in (TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SHELL):
            children = TopoDS_Iterator(node)
            while children.More():
                visit(children.Value())
                children.Next()
        else:
            unsupported += 1

    visit(shape)
    return solids, sheets, unsupported


def _heal_solid(solid, *, heal: bool, unify: bool):
    """Run the requested repair passes over one solid."""
    if heal:
        from OCC.Core.ShapeFix import ShapeFix_Shape  # noqa: PLC0415

        fixer = ShapeFix_Shape(solid)
        fixer.Perform()
        fixed = fixer.Shape()
        if not fixed.IsNull():
            solid = fixed
    if unify:
        from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain  # noqa: PLC0415

        unifier = ShapeUpgrade_UnifySameDomain(solid, True, True, False)
        unifier.Build()
        merged = unifier.Shape()
        if not merged.IsNull():
            solid = merged
    return solid


def _warn_if_invalid(solid, label: str, *, healed: bool) -> None:
    """Report a solid the kernel considers broken, without failing."""
    from OCC.Core.BRepCheck import BRepCheck_Analyzer  # noqa: PLC0415

    if BRepCheck_Analyzer(solid).IsValid():
        return
    remedy = (
        "Try unify=True to merge its fragmented faces"
        if healed
        else "Try heal=True to repair it on import"
    )
    warnings.warn(
        f"Imported solid {label!r} is not a valid solid according to the "
        f"geometry kernel. {remedy}; meshing it may give wrong results.",
        UserWarning,
        stacklevel=3,
    )


def _scaled_to_meters(shape, factor: float):
    """Return *shape* scaled from its source unit into meter space."""
    if factor == 1.0:
        return shape
    from magnelio.geo._occ_backend import occ_scale  # noqa: PLC0415

    return occ_scale(shape, factor, (0.0, 0.0, 0.0))


# ─────────────────────────────────────────────────────────────────────
# material assignment by name
# ─────────────────────────────────────────────────────────────────────


def _is_pattern(key: str) -> bool:
    return any(c in key for c in "*?[")


def _resolve_materials(names: list[str], materials) -> list:
    """Map each imported body name to a material (or ``None``).

    A single material broadcasts to every solid.  A dict is a name
    mapping in which a literal key beats a wildcard, two wildcards
    disagreeing over the same solid are an error rather than a silent
    first-wins, and a key that matches nothing is an error too — a
    mis-typed solid name must not pass as "no material wanted".
    """
    if materials is None:
        return [None] * len(names)
    if not isinstance(materials, dict):
        from magnelio.materials.material import (  # noqa: PLC0415
            Material,
            resolve_material,
        )

        # DD-185: a built-in name string broadcasts like the instance.
        materials = resolve_material(materials, "materials")
        if not isinstance(materials, Material):
            raise TypeError(
                "materials must be a Material (applied to every solid) or a "
                "dict mapping solid names to materials; got "
                f"{type(materials).__name__}."
            )
        return [materials] * len(names)

    for key in materials:
        if not isinstance(key, str):
            raise TypeError(f"materials keys must be solid names (strings); got {key!r}.")

    known = ", ".join(repr(n) for n in dict.fromkeys(names))
    assigned: list = [None] * len(names)
    winner: list[str | None] = [None] * len(names)
    for key, mat in materials.items():
        pattern = _is_pattern(key)
        hits = [
            i for i, n in enumerate(names) if (fnmatch.fnmatchcase(n, key) if pattern else n == key)
        ]
        if not hits:
            raise ValueError(
                f"materials key {key!r} matches none of the solids in the "
                f"file. Available names: {known}."
            )
        for i in hits:
            if winner[i] is None or (_is_pattern(winner[i]) and not pattern):
                assigned[i], winner[i] = mat, key
            elif _is_pattern(winner[i]) and pattern and assigned[i] is not mat:
                raise ValueError(
                    f"Solid {names[i]!r} is matched by two patterns with "
                    f"different materials, {winner[i]!r} and {key!r}. Name "
                    f"the solid literally to say which one wins."
                )
            elif not _is_pattern(winner[i]) and not pattern:
                # Same literal key twice cannot happen in a dict; a second
                # literal hit means duplicate solid names, which is fine.
                pass
    return assigned


# ─────────────────────────────────────────────────────────────────────
# STEP
# ─────────────────────────────────────────────────────────────────────


# Names the kernel invents for a label that carried none in the file.
# They are shape types, not part names: every unnamed solid would get
# the same one, which is useless as a key to assign materials against.
_PLACEHOLDER_NAMES = frozenset(
    {"SOLID", "SHELL", "FACE", "COMPOUND", "COMPSOLID", "WIRE", "EDGE", "VERTEX"}
)


def _label_name(label) -> str | None:
    """The part name an XCAF label carries, or ``None`` if it has none."""
    if label is None or label.IsNull():
        return None
    text = (label.GetLabelName() or "").strip()
    if not text or text in _PLACEHOLDER_NAMES:
        return None
    return text


def _shape_color(color_tool, shape, labels) -> tuple[float, float, float] | None:
    """Display colour of an imported solid, or ``None``.

    A colour can sit on the placed instance (two copies of the same part
    painted differently) or on the prototype label; the instance wins.
    Within each, the surface colour is what a viewer shows, so it is
    tried before the generic and the curve colour.
    """
    from OCC.Core.Quantity import Quantity_Color  # noqa: PLC0415
    from OCC.Core.TDF import TDF_ChildIterator  # noqa: PLC0415
    from OCC.Core.XCAFDoc import (  # noqa: PLC0415
        XCAFDoc_ColorCurv,
        XCAFDoc_ColorGen,
        XCAFDoc_ColorSurf,
        XCAFDoc_ColorTool,
    )

    kinds = (XCAFDoc_ColorSurf, XCAFDoc_ColorGen, XCAFDoc_ColorCurv)
    color = Quantity_Color()
    for kind in kinds:
        if shape is not None and color_tool.GetInstanceColor(shape, kind, color):
            return (color.Red(), color.Green(), color.Blue())
        if shape is not None and color_tool.GetColor(shape, kind, color):
            return (color.Red(), color.Green(), color.Blue())
    for label in labels:
        if label is None or label.IsNull():
            continue
        for kind in kinds:
            # GetColor(label, ...) is a static method of the tool class.
            if XCAFDoc_ColorTool.GetColor(label, kind, color):
                return (color.Red(), color.Green(), color.Blue())
    # Some STEP readers attach a standalone face's colour to its XCAF
    # sublabel. Use it only when all coloured children agree; a multicolour
    # solid has no unambiguous single display colour.
    child_colors = []
    for label in labels:
        if label is None or label.IsNull():
            continue
        children = TDF_ChildIterator(label, True)
        while children.More():
            child = children.Value()
            for kind in kinds:
                if XCAFDoc_ColorTool.GetColor(child, kind, color):
                    child_colors.append((color.Red(), color.Green(), color.Blue()))
                    break
            children.Next()
    if child_colors and all(
        all(abs(a - b) < 1e-6 for a, b in zip(child_colors[0], other)) for other in child_colors[1:]
    ):
        return child_colors[0]
    return None


def _walk_labels(shape_tool, color_tool, label, location, inherited, out) -> None:
    """Flatten the XCAF assembly tree into placed (shape, name, colour).

    Assemblies are containers, not geometry: a component's placement is
    accumulated down the tree and applied to the leaf, so what comes out
    is every solid where it actually sits in the model.
    """
    from OCC.Core.TDF import TDF_Label, TDF_LabelSequence  # noqa: PLC0415

    referred = TDF_Label()
    has_referred = shape_tool.GetReferredShape(label, referred)
    target = referred if has_referred else label
    # A component label carries the placement of that instance; every
    # other label reports the identity, so accumulating unconditionally
    # is what turns the tree into world positions.
    here = location.Multiplied(shape_tool.GetLocation(label))
    # The instance names the part in the assembly; the prototype names
    # the part itself.  Prefer the instance, which is what a CAD user
    # renamed when they placed it.
    name = _label_name(label) or (_label_name(referred) if has_referred else None) or inherited

    if shape_tool.IsAssembly(target):
        components = TDF_LabelSequence()
        shape_tool.GetComponents(target, components)
        for i in range(1, components.Length() + 1):
            _walk_labels(shape_tool, color_tool, components.Value(i), here, name, out)
        return

    shape = shape_tool.GetShape(target)
    if shape is None or shape.IsNull():
        return
    color = _shape_color(color_tool, shape, (label, referred if has_referred else None))
    out.append((shape.Moved(here), name, color))


def import_step(
    path: str | Path,
    materials=None,
    *,
    heal: bool = True,
    unify: bool = False,
    name: str | None = None,
) -> Group:
    """Import solids and free sheets from a STEP file.

    The file's length unit is read from the file itself, so a model
    drawn in millimetres arrives at its true size in meters and no
    conversion is needed on the caller's side.  Assemblies are
    flattened: every solid comes back where it sits in the assembly,
    as a member of one :class:`~magnelio.geo.Group`. Free shell faces
    become independent sheets; faces belonging to a solid remain in that
    solid and are not duplicated.

    STEP carries no material physics — only names — so materials are
    assigned here, against the name each body carries in the file.
    Names survive a re-export of the same CAD model, so the same call
    keeps working after the drawing changes.

    Parameters
    ----------
    path : str or Path
        The ``.step`` / ``.stp`` file to read.
    materials : Material or dict, optional
        A single :class:`~magnelio.Material` is given to every body.
        A dict maps body names to materials; keys may use shell
        wildcards (``"shield_*"``), a literal name beats a wildcard,
        and ``"*"`` acts as a catch-all.  Every key must match at least
        one body, so a typo is reported. Unmatched bodies carry no
        material. Sheets require a resolved thickness before meshing.
    heal : bool
        Repair each solid on import (tolerances, orientation, small
        topological defects).  Cheap and harmless on a clean file;
        turn it off only to inspect a file exactly as written.
    unify : bool
        Additionally merge adjacent faces that lie on the same surface.
        CAD kernels often split one planar face into many; merging them
        simplifies the solid, at the price of editing its topology.
        Off by default.
    name : str, optional
        Name for the returned Group.  Defaults to the file name.

    Returns
    -------
    Group
        ImportedSolid and ImportedSheet members carrying names, display
        colours and assigned materials. A Group is flattened on model
        insertion; sheets still need physical thickness to be meshable.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be read, contains no solid or free face, or a
        *materials* key matches no body.

    Examples
    --------
    Assign materials by the names the parts carry in the CAD model::

        from magnelio import GeometryModel, Material
        from magnelio.io import import_step

        parts = import_step(
            "connector.step",
            {"pin": Material.pec(), "shell": Material.pec(),
             "insulator": Material(name="PTFE", epsilon=(2.1,) * 3)},
        )
        model = GeometryModel().add(parts)

    Import first, inspect the names, assign afterwards::

        parts = import_step("connector.step")
        print([s.name for s in parts.members()])
    """
    _require_occ()

    from OCC.Core.IFSelect import IFSelect_RetDone  # noqa: PLC0415
    from OCC.Core.Interface import Interface_Static  # noqa: PLC0415
    from OCC.Core.STEPCAFControl import STEPCAFControl_Reader  # noqa: PLC0415
    from OCC.Core.TDF import TDF_LabelSequence  # noqa: PLC0415
    from OCC.Core.TDocStd import TDocStd_Document  # noqa: PLC0415
    from OCC.Core.TopLoc import TopLoc_Location  # noqa: PLC0415
    from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"STEP file not found: {path}")

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    reader.SetLayerMode(False)

    # Transfer into millimetres before converting the finished BReps to
    # meter space. Transfer directly into meters can destroy nanometre
    # solids at the kernel's absolute sewing tolerance.
    # The setting is process-global, hence the restore.
    previous = Interface_Static.CVal("xstep.cascade.unit")
    Interface_Static.SetCVal("xstep.cascade.unit", "MM")
    try:
        if reader.ReadFile(str(path)) != IFSelect_RetDone:
            raise ValueError(
                f"Could not read {path} as a STEP file. Check that it is a "
                f"STEP file (AP203/AP214/AP242) and not truncated."
            )
        doc = TDocStd_Document("XCAF")
        if not reader.Transfer(doc):
            raise ValueError(f"STEP file {path} carries no transferable geometry.")
    finally:
        if previous is not None:
            Interface_Static.SetCVal("xstep.cascade.unit", previous)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)

    placed: list = []
    for i in range(1, roots.Length() + 1):
        _walk_labels(shape_tool, color_tool, roots.Value(i), TopLoc_Location(), None, placed)

    bodies: list = []
    names: list[str] = []
    colors: list = []
    skipped: list[str] = []
    for shape, label_name, color in placed:
        solids, sheets, unsupported = _bodies_of(shape)
        parts = [(solid, "solid") for solid in solids] + [(sheet, "sheet") for sheet in sheets]
        if unsupported:
            skipped.append(label_name or "<unnamed>")
        for index, (body, category) in enumerate(parts, start=1):
            bodies.append((body, category))
            colors.append(color)
            if label_name is None:
                names.append(f"{category}_{len(bodies)}")
            elif len(parts) == 1:
                names.append(label_name)
            else:
                names.append(f"{label_name}_{index}")

    if skipped:
        warnings.warn(
            "STEP entries containing unsupported free curves or points were skipped: "
            + ", ".join(repr(s) for s in skipped)
            + ". Curves and points have no solid or sheet area.",
            UserWarning,
            stacklevel=2,
        )
    if not bodies:
        raise ValueError(f"STEP file {path} contains no solid or free face.")

    assigned = _resolve_materials(names, materials)
    members = []
    for (body, category), body_name, color, material in zip(bodies, names, colors, assigned):
        if category == "solid":
            body = _heal_solid(body, heal=heal, unify=unify)
            body = _scaled_to_meters(body, 1e-3)
            _warn_if_invalid(body, body_name, healed=heal)
            members.append(ImportedSolid(body, material, name=body_name, color=color))
        else:
            body = _scaled_to_meters(body, 1e-3)
            members.append(ImportedSheet(body, material, name=body_name, color=color))
    return Group(*members, name=name or path.stem)


# ─────────────────────────────────────────────────────────────────────
# BREP
# ─────────────────────────────────────────────────────────────────────


def import_brep(
    path: str | Path,
    *,
    unit,
    material: "Material | None" = None,
    heal: bool = False,
    name: str | None = None,
) -> Group:
    """Import solids and free sheets from a BREP file.

    BREP is the geometry kernel's native dump: exact, but it records
    nothing but the geometry — no length unit, no names, no colours.
    The unit therefore has to be stated, and it has to be right: a file
    drawn in millimetres and read as meters is a thousand times too
    big, and nothing in the file says so.  Prefer
    :func:`import_step` whenever the CAD system can write STEP.

    Parameters
    ----------
    path : str or Path
        The ``.brep`` file to read.
    unit : str or float
        Length unit the file is written in: ``"m"``, ``"cm"``,
        ``"mm"``, ``"um"``, ``"nm"``, ``"in"``, ``"mil"`` — or a number
        giving the length of one unit in meters.
    material : Material, optional
        Material for every solid or free sheet in the file. Omitted,
        bodies return as construction geometry (see :func:`import_step`).
    heal : bool
        Repair each solid on import.  Off by default: a BREP file comes
        from the same kernel and is normally already clean.
    name : str, optional
        Name for the returned Group and the base name of its bodies.
        Defaults to the file name.

    Returns
    -------
    Group
        ImportedSolid and ImportedSheet members, named ``<name>`` for
        one body and ``<name>_1``, ``<name>_2``, … for several.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be read, or contains no solid or free face.

    Examples
    --------
    ::

        from magnelio import Material
        from magnelio.io import import_brep

        horn = import_brep("horn.brep", unit="mm", material=Material.pec())
    """
    _require_occ()

    from OCC.Core.BRep import BRep_Builder  # noqa: PLC0415
    from OCC.Core.BRepTools import breptools  # noqa: PLC0415
    from OCC.Core.TopoDS import TopoDS_Shape  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BREP file not found: {path}")
    factor = _unit_factor(unit)

    shape = TopoDS_Shape()
    if not breptools.Read(shape, str(path), BRep_Builder()):
        raise ValueError(f"Could not read {path} as a BREP file.")

    solids, sheets, unsupported = _bodies_of(shape)
    if unsupported:
        warnings.warn(
            "BREP contains unsupported free curves or points; they were skipped.",
            UserWarning,
            stacklevel=2,
        )
    if not solids and not sheets:
        raise ValueError(f"BREP file {path} contains no solid or free face.")

    base = name or path.stem
    members = []
    bodies = [(solid, "solid") for solid in solids] + [(sheet, "sheet") for sheet in sheets]
    for index, (body, category) in enumerate(bodies, start=1):
        solid_name = base if len(bodies) == 1 else f"{base}_{index}"
        if category == "sheet":
            body = _scaled_to_meters(body, factor)
            members.append(ImportedSheet(body, material, name=solid_name))
            continue
        body = _heal_solid(body, heal=heal, unify=False)
        body = _scaled_to_meters(body, factor)
        _warn_if_invalid(body, solid_name, healed=heal)
        members.append(ImportedSolid(body, material, name=solid_name))
    return Group(*members, name=base)


def _export_members(geometry) -> list:
    """Return chosen solids and sheets, preserving their input order."""
    from magnelio.geo._sheet import Sheet  # noqa: PLC0415
    from magnelio.geo.shape import Solid  # noqa: PLC0415

    if isinstance(geometry, (Shape, Group)):
        geometry = (geometry,)
    try:
        inputs = tuple(geometry)
    except TypeError:
        raise TypeError("geometry must be a Solid, Sheet, Group, or iterable of them.") from None
    members = []
    for obj in inputs:
        leaves = obj.members() if isinstance(obj, Group) else (obj,)
        for leaf in leaves:
            if not isinstance(leaf, (Solid, Sheet)):
                raise TypeError(
                    f"CAD export accepts Solid and Sheet geometry; got {type(leaf).__name__}."
                )
            members.append(leaf)
    if not members:
        raise ValueError("CAD export needs at least one Solid or Sheet.")
    return members


def _export_path(path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"CAD export target already exists: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"CAD export directory does not exist: {path.parent}")
    return path


def _shape_in_units(member, factor):
    from magnelio.geo._occ_backend import occ_scale  # noqa: PLC0415
    from magnelio.geo._scaling import model_scale  # noqa: PLC0415

    scale = model_scale([member])
    shape = member._occ_shape(scale)
    conversion = 1.0 / (factor * scale)
    return shape if conversion == 1.0 else occ_scale(shape, conversion, (0.0, 0.0, 0.0))


def export_step(path: str | Path, geometry, *, unit: str = "mm", overwrite: bool = False) -> Path:
    """Write selected solids and sheets to a STEP file.

    STEP records geometry, length unit, part names and display colours.
    It does not encode Magnelio materials, named topology or construction
    history. A Group is flattened; each leaf is exported in world placement.

    Parameters
    ----------
    path : str or Path
        Destination ``.step`` or ``.stp`` file.
    geometry : Solid or Sheet or Group or iterable
        Bodies to write. Free curves cannot be exported by this function.
    unit : {'m', 'cm', 'mm', 'in'}, optional
        Length unit written into the file. Defaults to millimetres.
    overwrite : bool, optional
        Replace an existing file when True; otherwise raise FileExistsError.

    Returns
    -------
    Path
        Destination file.
    """
    _require_occ()
    from OCC.Core.IFSelect import IFSelect_RetDone  # noqa: PLC0415
    from OCC.Core.Interface import Interface_Static  # noqa: PLC0415
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB  # noqa: PLC0415
    from OCC.Core.STEPCAFControl import STEPCAFControl_Writer  # noqa: PLC0415
    from OCC.Core.TDataStd import TDataStd_Name  # noqa: PLC0415
    from OCC.Core.TDocStd import TDocStd_Document  # noqa: PLC0415
    from OCC.Core.XCAFDoc import XCAFDoc_ColorSurf, XCAFDoc_DocumentTool  # noqa: PLC0415

    code = (
        {"m": "M", "cm": "CM", "mm": "MM", "in": "INCH"}.get(unit.lower())
        if isinstance(unit, str)
        else None
    )
    if code is None:
        raise ValueError("STEP export unit must be 'm', 'cm', 'mm' or 'in'.")
    path = _export_path(path, overwrite)
    members = _export_members(geometry)
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.SetColorMode(True)
    doc = TDocStd_Document("XCAF")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())
    # XCAF's exchange workspace uses millimetres; write.step.unit controls
    # the file unit conversion from that workspace.
    factor = 1e-3
    for member in members:
        label = shape_tool.AddShape(_shape_in_units(member, factor), False)
        if member.name:
            TDataStd_Name.Set(label, member.name)
        if getattr(member, "color", None) is not None:
            color_tool.SetColor(
                label, Quantity_Color(*member.color, Quantity_TOC_RGB), XCAFDoc_ColorSurf
            )
    previous = Interface_Static.CVal("write.step.unit")
    previous_cascade = Interface_Static.CVal("xstep.cascade.unit")
    Interface_Static.SetCVal("xstep.cascade.unit", "MM")
    Interface_Static.SetCVal("write.step.unit", code)
    try:
        if not writer.Transfer(doc):
            raise IOError("STEP writer could not transfer the selected geometry.")
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".step", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            if writer.Write(str(temporary)) != IFSelect_RetDone:
                raise IOError(f"STEP writer failed for {path}")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        if previous is not None:
            Interface_Static.SetCVal("write.step.unit", previous)
        if previous_cascade is not None:
            Interface_Static.SetCVal("xstep.cascade.unit", previous_cascade)
    return path


def export_brep(path: str | Path, geometry, *, unit, overwrite: bool = False) -> Path:
    """Write selected solids and sheets to a BREP file.

    BREP contains exact geometry only. Record the chosen *unit* externally:
    :func:`import_brep` requires it when reading the file back.

    Parameters
    ----------
    path : str or Path
        Destination ``.brep`` file.
    geometry : Solid or Sheet or Group or iterable
        Bodies to write.
    unit : str or float
        Written length unit, or meters per unit as a positive number.
    overwrite : bool, optional
        Replace an existing file when True; otherwise raise FileExistsError.

    Returns
    -------
    Path
        Destination file.
    """
    _require_occ()
    from OCC.Core.BRep import BRep_Builder  # noqa: PLC0415
    from OCC.Core.BRepTools import breptools  # noqa: PLC0415
    from OCC.Core.TopoDS import TopoDS_Compound  # noqa: PLC0415

    factor = _unit_factor(unit)
    path = _export_path(path, overwrite)
    members = _export_members(geometry)
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for member in members:
        builder.Add(compound, _shape_in_units(member, factor))
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".brep", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        if not breptools.Write(compound, str(temporary)):
            raise IOError(f"BREP writer failed for {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
