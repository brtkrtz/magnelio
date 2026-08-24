"""CAD file import — STEP (primary) and BREP (secondary).

Reading a model that was drawn elsewhere is the normal way to get a
real device into a simulation.  Two formats are supported, and the
difference between them is what they carry *besides* the geometry:

**STEP** (``.step`` / ``.stp``) is the interchange format every CAD
system writes.  Beyond the solids it records the file's length unit,
a name per solid and a display colour — which is exactly what the
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
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from magnelio.geo.imported import ImportedSolid
from magnelio.geo.operations import Group

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
    if not (factor > 0.0):
        raise ValueError(f"{what} must be positive; got {unit!r}.")
    return factor


def _solids_of(shape) -> list:
    """Every ``TopoDS_Solid`` inside *shape*, in kernel order."""
    from OCC.Core.TopAbs import TopAbs_SOLID  # noqa: PLC0415
    from OCC.Core.TopExp import TopExp_Explorer  # noqa: PLC0415
    from OCC.Core.TopoDS import topods  # noqa: PLC0415

    solids = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)  # pyright: ignore[reportArgumentType]
    while explorer.More():
        solids.append(topods.Solid(explorer.Current()))
        explorer.Next()
    return solids


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
    """Map each solid name to a material (or ``None``).

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
    for label in labels:
        if label is None or label.IsNull():
            continue
        for kind in kinds:
            # GetColor(label, ...) is a static method of the tool class.
            if XCAFDoc_ColorTool.GetColor(label, kind, color):
                return (color.Red(), color.Green(), color.Blue())
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
    """Import the solids of a STEP file.

    The file's length unit is read from the file itself, so a model
    drawn in millimetres arrives at its true size in meters and no
    conversion is needed on the caller's side.  Assemblies are
    flattened: every solid comes back where it sits in the assembly,
    as a member of one :class:`~magnelio.geo.Group`.

    STEP carries no material physics — only names — so materials are
    assigned here, against the name each solid carries in the file.
    Names survive a re-export of the same CAD model, so the same call
    keeps working after the drawing changes.

    Parameters
    ----------
    path : str or Path
        The ``.step`` / ``.stp`` file to read.
    materials : Material or dict, optional
        A single :class:`~magnelio.Material` is given to every solid.
        A dict maps solid names to materials; keys may use shell
        wildcards (``"shield_*"``), a literal name beats a wildcard,
        and ``"*"`` acts as a catch-all.  Every key must match at least
        one solid, so a typo is reported instead of silently doing
        nothing.  Solids left unmatched (and every solid if *materials*
        is omitted) come back as construction bodies — usable as
        Boolean operands, rejected by
        :meth:`~magnelio.geo.GeometryModel.add` until they are given a
        material.
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
        The imported solids, each an
        :class:`~magnelio.geo.ImportedSolid` carrying its name, colour
        and assigned material.  Add it to a model like any other
        shape — a Group is flattened into its members on insertion.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be read, contains no solid, or a *materials*
        key matches no solid.

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

    # The reader converts the file's unit to this one, which is how the
    # geometry lands in meter space regardless of what the CAD system
    # drew in.  The setting is process-global, hence the restore.
    previous = Interface_Static.CVal("xstep.cascade.unit")
    Interface_Static.SetCVal("xstep.cascade.unit", "M")
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
        if previous:
            Interface_Static.SetCVal("xstep.cascade.unit", previous)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool(doc.Main())
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)

    placed: list = []
    for i in range(1, roots.Length() + 1):
        _walk_labels(shape_tool, color_tool, roots.Value(i), TopLoc_Location(), None, placed)

    solids: list = []
    names: list[str] = []
    colors: list = []
    skipped: list[str] = []
    for shape, label_name, color in placed:
        parts = _solids_of(shape)
        if not parts:
            skipped.append(label_name or "<unnamed>")
            continue
        for index, solid in enumerate(parts, start=1):
            solids.append(solid)
            colors.append(color)
            if label_name is None:
                names.append(f"solid_{len(solids)}")
            elif len(parts) == 1:
                names.append(label_name)
            else:
                names.append(f"{label_name}_{index}")

    if skipped:
        warnings.warn(
            "STEP entries without a solid body were skipped: "
            + ", ".join(repr(s) for s in skipped)
            + ". Only solids are imported; surface models (shells, free "
            "faces) have no volume to fill with a material.",
            UserWarning,
            stacklevel=2,
        )
    if not solids:
        raise ValueError(
            f"STEP file {path} contains no solid. Only solid bodies can be "
            f"imported; a surface model has to be turned into a solid in "
            f"the CAD system first."
        )

    assigned = _resolve_materials(names, materials)
    members = []
    for solid, solid_name, color, material in zip(solids, names, colors, assigned):
        body = _heal_solid(solid, heal=heal, unify=unify)
        _warn_if_invalid(body, solid_name, healed=heal)
        members.append(ImportedSolid(body, material, name=solid_name, color=color))
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
    """Import the solids of a BREP file.

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
        Material for every solid in the file.  Omitted, the solids come
        back as construction bodies (see :func:`import_step`).
    heal : bool
        Repair each solid on import.  Off by default: a BREP file comes
        from the same kernel and is normally already clean.
    name : str, optional
        Name for the returned Group and the base name of its solids.
        Defaults to the file name.

    Returns
    -------
    Group
        The imported solids as :class:`~magnelio.geo.ImportedSolid`
        members, named ``<name>`` for a single solid and ``<name>_1``,
        ``<name>_2``, … for several.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be read, or contains no solid.

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

    solids = _solids_of(_scaled_to_meters(shape, factor))
    if not solids:
        raise ValueError(
            f"BREP file {path} contains no solid. Only solid bodies can be "
            f"imported; a surface model has to be turned into a solid in "
            f"the CAD system first."
        )

    base = name or path.stem
    members = []
    for index, solid in enumerate(solids, start=1):
        solid_name = base if len(solids) == 1 else f"{base}_{index}"
        body = _heal_solid(solid, heal=heal, unify=False)
        _warn_if_invalid(body, solid_name, healed=heal)
        members.append(ImportedSolid(body, material, name=solid_name))
    return Group(*members, name=base)
