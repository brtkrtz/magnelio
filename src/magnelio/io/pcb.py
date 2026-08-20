"""Printed circuit board import — Gerber, drill data and a job file.

A board layout leaves its design tool as *fabrication data*: one Gerber
file per copper layer, one per board outline, drill files for the
holes, and a job file that records the stackup.  That set is what a
board house is sent, it is written by every layout tool, and it is
therefore what this import reads — not any one tool's project format.

Turning it into a model is mostly a matter of taking the stackup
literally.  Each copper and dielectric layer of the stackup becomes a
slab at its own height, built as an area in the plane and raised once;
a plated hole becomes a copper cylinder that exactly fills the circles
cut from the layers it passes through.  Nothing is fitted or snapped,
so the layers meet on coincident faces and the mesher sees the board
the fabricator would build.

What the fabrication data cannot say, this import does not invent.
Solder mask and silkscreen are ignored — they are coatings whose effect
is below the accuracy of everything else here.  A dielectric whose
permittivity the job file omits arrives without a material rather than
as vacuum.  And a loss tangent, which is one number at an unstated
frequency, is reported but never modelled: see :func:`import_pcb`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from magnelio.geo.imported import ImportedSolid
from magnelio.geo.operations import Group
from magnelio.io import _pcb_geom as geometry
from magnelio.io._excellon import DrillFile, Slot, parse_excellon
from magnelio.io._gbrjob import Stackup, read_gbrjob
from magnelio.io._gerber import GerberLayer, parse_gerber
from magnelio.io.cad import _require_occ, _resolve_materials

# Extensions a drill file is written with, for the case where the job
# file does not list one.
_DRILL_SUFFIXES = (".drl", ".xln")


# ─────────────────────────────────────────────────────────────────────
# reading the set
# ─────────────────────────────────────────────────────────────────────


def _read_gerber(path: Path, role: str) -> GerberLayer:
    if not path.exists():
        raise FileNotFoundError(
            f"The job file names {path.name!r} as the {role}, but that file "
            f"is not next to it in {path.parent}. Export the whole "
            f"fabrication set into one folder."
        )
    return parse_gerber(path.read_text(encoding="utf-8", errors="replace"), source=path.name)


def _check_role(drawing: GerberLayer, source: Path, layer, stackup: Stackup) -> None:
    """Refuse a Gerber file the job file assigns to the wrong layer.

    Both the job file and the Gerber file itself declare which copper
    layer a drawing belongs to.  When they disagree the set has been
    edited or assembled by hand, and building the board anyway would
    stack the layers in the wrong order — which produces a plausible
    model of a different board.
    """
    function = drawing.file_function
    if len(function) < 2 or function[0] != "Copper":
        return  # the file makes no claim; the job file decides
    token = function[1]
    if not token.startswith("L") or not token[1:].isdigit():
        return
    number = int(token[1:])
    if number != layer.number:
        raise ValueError(
            f"The job file {stackup.path.name} uses {source.name} as copper "
            f"layer L{layer.number} ({layer.name!r}), but the file itself says "
            f"it is layer L{number}. The Gerber set and the job file are not "
            f"from the same export."
        )


def _read_drills(stackup: Stackup) -> list[tuple[Path, DrillFile, bool | None, tuple | None]]:
    """Every drill file of the set, with the plating and span it has.

    The file and the job file both declare these, and either may be
    silent; what the drill file itself says wins, because it is the one
    the board house drills from.
    """
    listed = list(stackup.drill_files)
    if not listed:
        from magnelio.io._gbrjob import DrillRole  # noqa: PLC0415

        listed = [
            DrillRole(path=candidate)
            for candidate in sorted(stackup.path.parent.iterdir())
            if candidate.suffix.lower() in _DRILL_SUFFIXES
        ]

    out = []
    for role in listed:
        if not role.path.exists():
            raise FileNotFoundError(
                f"The job file names the drill file {role.path.name!r}, but "
                f"it is not next to it in {role.path.parent}."
            )
        drill = parse_excellon(
            role.path.read_text(encoding="utf-8", errors="replace"), source=role.path.name
        )
        plated = drill.plated if drill.plated is not None else role.plated
        span = drill.span or role.span
        out.append((role.path, drill, plated, span))
    return out


# ─────────────────────────────────────────────────────────────────────
# where a hole goes
# ─────────────────────────────────────────────────────────────────────


def _copper_index(stackup: Stackup, number: int) -> int:
    for index, layer in enumerate(stackup.layers):
        if layer.is_copper and layer.number == number:
            return index
    raise ValueError(
        f"The drill data runs to copper layer {number}, but the stackup in "
        f"{stackup.path.name} has only {len(stackup.copper_layers)} copper "
        f"layers. The Gerber set and the drill files are not from the same "
        f"export."
    )


def _crossed(stackup: Stackup, span: tuple[int, int] | None) -> range:
    """Indices of the layers a hole with *span* passes through."""
    coppers = stackup.copper_layers
    first, last = coppers[0].number, coppers[-1].number
    if span is None:
        return range(len(stackup.layers))
    start, end = span
    low, high = _copper_index(stackup, start), _copper_index(stackup, end)
    # A hole reaching an outer copper layer leaves the board there, so
    # it also passes through anything the stackup puts beyond it.
    return range(0 if start == first else low, len(stackup.layers) if end == last else high + 1)


def _barrel_extent(stackup: Stackup, span: tuple[int, int] | None) -> tuple[float, float]:
    """``(z_bottom, z_top)`` of the copper barrel of a plated hole."""
    elevations = stackup.elevations()
    coppers = stackup.copper_layers
    start, end = span if span is not None else (coppers[0].number, coppers[-1].number)
    top = elevations[_copper_index(stackup, start)][1]
    bottom = elevations[_copper_index(stackup, end)][0]
    return bottom, top


def _hole_face(hole, scale: float):
    if isinstance(hole, Slot):
        return geometry.slot_face(hole.start, hole.end, hole.diameter, scale)
    return geometry.circle_face(hole.at, hole.diameter, scale)


# ─────────────────────────────────────────────────────────────────────
# materials
# ─────────────────────────────────────────────────────────────────────


def _auto_dielectric(layer, job: Path):
    """The material a dielectric layer of the stackup describes."""
    from magnelio.materials.material import Material  # noqa: PLC0415

    if layer.epsilon is None:
        warnings.warn(
            f"The stackup in {job.name} gives no dielectric constant for "
            f"{layer.name!r}, so it was imported without a material. Assign "
            f"one with materials={{{layer.name!r}: ...}} before meshing; a "
            f"substrate left as vacuum would look like a working model.",
            UserWarning,
            stacklevel=3,
        )
        return None
    epsilon = float(layer.epsilon)
    return Material(name=layer.material or layer.name, epsilon=(epsilon, epsilon, epsilon))


def _report_loss_tangents(stackup: Stackup) -> None:
    """Say that the substrate arrived lossless, and what it would take.

    A loss tangent is a single number measured at a frequency the job
    file does not record, and a loss tangent that does not vary with
    frequency is not causal, so there is nothing here to turn it into.
    Modelling it needs a dispersive material, which needs the data
    sheet — a decision for the caller, not for the import.
    """
    lossy = [
        (layer.name, layer.loss_tangent)
        for layer in stackup.layers
        if layer.loss_tangent not in (None, 0.0)
    ]
    if not lossy:
        return
    listed = ", ".join(f"{name} (tan d = {value:g})" for name, value in lossy)
    warnings.warn(
        f"The stackup states a loss tangent for {listed}, which was not "
        f"modelled: the imported dielectric is lossless. A loss tangent is "
        f"one number, and the job file does not record the frequency it "
        f"was measured at. Supply that frequency and pass the result "
        f"through materials=, for example Material.dispersive(name=..., "
        f"model=DispersionModel.djordjevic_sarkar(eps_r=..., "
        f"tan_delta=..., f_ref=...)).",
        UserWarning,
        stacklevel=3,
    )


# ─────────────────────────────────────────────────────────────────────
# the import
# ─────────────────────────────────────────────────────────────────────


def _outline_bounds(layer: GerberLayer):
    """A box containing the profile, from its coordinates alone."""
    from magnelio.io._gerber import ArcStroke, Flash, Region, Stroke  # noqa: PLC0415

    points: list[tuple[float, float]] = []
    for _, obj in layer.objects:
        if isinstance(obj, (Stroke, ArcStroke)):
            points.extend((obj.start, obj.end))
        elif isinstance(obj, Flash):
            points.append(obj.at)
        elif isinstance(obj, Region):
            for contour in obj.contours:
                points.extend(segment.start for segment in contour)
    if not points:
        raise ValueError("The board outline layer draws nothing.")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), 0.0), (max(xs), max(ys), 0.0)


def _to_meters(shape, scale: float):
    from magnelio.geo._occ_backend import occ_scale  # noqa: PLC0415

    if scale == 1.0:
        return shape
    return occ_scale(shape, 1.0 / scale, (0.0, 0.0, 0.0))


def import_pcb(
    path: str | Path,
    materials=None,
    *,
    copper=None,
    name: str | None = None,
) -> Group:
    """Import a printed circuit board from its fabrication data.

    The set is the one a layout tool writes for a board house: Gerber
    files for the copper layers and the board outline, Excellon drill
    files, and the ``.gbrjob`` job file that records the stackup.  The
    job file is required — the Gerber files are flat drawings and say
    nothing about layer thicknesses or the dielectric, so without it
    there is no third dimension to build.

    Each layer of the stackup arrives as one solid: a copper layer with
    its real thickness, a dielectric filling the board outline.  Plated
    holes arrive as solid copper barrels joining the layers they run
    between; unplated holes and slots are cut out of everything they
    pass through.  Solder mask and silkscreen are ignored.

    Parameters
    ----------
    path : str or Path
        The ``.gbrjob`` job file, or the folder holding the fabrication
        set (which must contain exactly one job file).
    materials : Material or dict, optional
        A single :class:`~magnelio.Material` is given to every solid.
        A dict maps solid names to materials; keys may use shell
        wildcards (``"via_*"``), a literal name beats a wildcard, and
        every key must match at least one solid, so a typo is reported
        instead of silently doing nothing.  The names are the layer
        names of the stackup (``"F.Cu"``, ``"dielectric_1"``) and
        ``"via_1"``, ``"via_2"``, … for the barrels.
    copper : Material, optional
        Material for every copper layer and every plated barrel.
        Defaults to a perfect electric conductor.  A copper layer is
        thin against any usable cell size, which the mesher resolves
        below the cell only for a perfect conductor — give a finite
        conductivity here only with a mesh fine enough to hold the
        metal thickness.
    name : str, optional
        Name for the returned Group.  Defaults to the project name in
        the job file, or the job file's name.

    Returns
    -------
    Group
        The layers and barrels, each an
        :class:`~magnelio.geo.ImportedSolid` carrying its name and
        material.  Add it to a model like any other shape.

    Raises
    ------
    FileNotFoundError
        If the job file, or a file it names, is missing.
    ValueError
        If the fabrication data is malformed, states no layer
        thicknesses, or a *materials* key matches no solid.

    Warns
    -----
    UserWarning
        If the stackup states a loss tangent (which is reported, never
        modelled), or omits a dielectric constant.

    Notes
    -----
    Copper is placed where the stackup puts it, so the top of the
    topmost dielectric is at ``z = 0``, the top copper occupies
    ``0`` to its thickness, and the stack grows downwards.

    A plated hole is modelled as a solid cylinder rather than as a
    plated wall around a void.  The wall is a closed conductor, so the
    space it encloses carries no field either way.

    A copper layer is thin against any usable cell size.  The mesher
    resolves it below the cell — one grid plane, thickness carried in
    the sub-cell material fractions — for a perfect conductor and only
    when :class:`~magnelio.MeshControl` carries a ``min_cell_size``
    larger than the metal thickness.

    Examples
    --------
    Import a board and give the substrate a material of its own::

        from magnelio import GeometryModel, Material
        from magnelio.io import import_pcb

        board = import_pcb(
            "fabrication/",
            {"dielectric_1": Material(name="RO4350B", epsilon=(3.66,) * 3)},
        )
        model = GeometryModel().add(board)

    Inspect the names before assigning anything::

        board = import_pcb("fabrication/")
        print([solid.name for solid in board.members()])
    """
    _require_occ()
    from magnelio.geo._scaling import fine_detail_scale  # noqa: PLC0415
    from magnelio.materials.material import Material  # noqa: PLC0415

    if copper is not None and not isinstance(copper, Material):
        raise TypeError(
            f"copper must be a Material for the copper layers and barrels; "
            f"got {type(copper).__name__}."
        )
    stackup = read_gbrjob(path)
    conductor = copper if copper is not None else Material.pec()

    if stackup.outline_file is None:
        raise ValueError(
            f"The job file {stackup.path.name} names no board outline, so the "
            f"extent of the board is unknown. Re-export the fabrication data "
            f"with the profile (board edge) layer included."
        )
    profile = _read_gerber(stackup.outline_file, "board outline")
    coppers = {}
    for layer in stackup.copper_layers:
        source = stackup.copper_files.get(layer.number)
        if source is None:
            raise ValueError(
                f"The stackup has a copper layer {layer.name!r}, but the job "
                f"file names no Gerber file for it. Re-export the fabrication "
                f"data with every copper layer plotted."
            )
        drawing = _read_gerber(source, f"copper layer {layer.name!r}")
        _check_role(drawing, source, layer, stackup)
        coppers[layer.number] = drawing

    drills = _read_drills(stackup)
    _report_loss_tangents(stackup)

    scale = fine_detail_scale(*_outline_bounds(profile))
    board = geometry.merge_faces(geometry.outline_faces(profile, scale), 0.0)
    elevations = stackup.elevations()

    # Holes first: every layer needs to know which circles to cut, and
    # the barrels need the same circles to fill.
    cuts: dict[int, list] = {index: [] for index in range(len(stackup.layers))}
    barrels: list[tuple[tuple[float, float], object]] = []
    for source, drill, plated, span in drills:
        if plated is None and drill.holes:
            warnings.warn(
                f"Neither {source.name} nor the job file says whether its "
                f"holes are plated, so they were taken as unplated and left "
                f"as holes. A plated hole would be a copper barrel joining "
                f"the layers it runs through — re-export the drill data with "
                f"the file attributes included.",
                UserWarning,
                stacklevel=2,
            )
        if plated and span is None:
            warnings.warn(
                f"The drill file {source.name} does not say which copper "
                f"layers its plated holes run between; they were taken to go "
                f"through the whole board. Re-export the drill data with the "
                f"file attributes included if it holds blind or buried vias.",
                UserWarning,
                stacklevel=2,
            )
        crossed = _crossed(stackup, span if plated else None)
        for hole in drill.holes:
            face = _hole_face(hole, scale)
            for index in crossed:
                cuts[index].append(face)
            if plated:
                bottom, top = _barrel_extent(stackup, span)
                barrels.append((hole.at, geometry.extrude(face, bottom, top, scale)))

    members: list[ImportedSolid] = []
    names: list[str] = []
    defaults: list = []

    for index, layer in enumerate(stackup.layers):
        bottom, top = elevations[index]
        if layer.is_copper:
            drawn = geometry.layer_shape(coppers[layer.number], scale)
            if drawn is None:
                raise ValueError(
                    f"The Gerber file for copper layer {layer.name!r} draws no "
                    f"copper at all. Check that the fabrication export is "
                    f"complete."
                )
            area = geometry.clip(drawn, board)
            default = conductor
        else:
            area = board
            default = _auto_dielectric(layer, stackup.path)
        solid = geometry.extrude(geometry.cut(area, cuts[index]), bottom, top, scale)
        members.append(solid)
        names.append(layer.name)
        defaults.append(default)

    for number, (_, barrel) in enumerate(sorted(barrels, key=lambda item: item[0]), start=1):
        members.append(barrel)
        names.append(f"via_{number}")
        defaults.append(conductor)

    assigned = _resolve_materials(names, materials)
    solids = [
        ImportedSolid(
            _to_meters(shape, scale),
            given if given is not None else default,
            name=solid_name,
        )
        for shape, solid_name, given, default in zip(members, names, assigned, defaults)
    ]
    return Group(*solids, name=name or stackup.project or stackup.path.stem)
