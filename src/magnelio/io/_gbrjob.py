"""Gerber job file (``.gbrjob``) — the board's stackup and file roles.

A folder of Gerber files is a set of 2D drawings.  Each one says where
copper sits on one layer, but none of them says how thick that copper
is, how far apart two layers are, or what the dielectric in between is
made of — and all of that is needed to turn a layout into a 3D model.

The job file is where a fabrication export records it: a JSON document
that travels with the Gerber set (the format is Ucamco's, not one
vendor's) and carries the two things this import cannot invent — the
ordered material stackup, and the role every file in the folder plays.

This module is pure data handling: it reads the job file and hands out
a :class:`Stackup`.  Nothing here touches the geometry kernel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The job file format states all dimensions in millimeters.
_MM = 1e-3

_COPPER = "copper"
_DIELECTRIC = "dielectric"

# Stackup entry types that carry no simulated volume.  They are part of
# a fabrication stackup but not of the electromagnetic model: the mask
# and legend layers are thin coatings whose effect on a board's fields
# is below the accuracy of everything else in the import.
_IGNORED_TYPES = frozenset({"legend", "solderpaste", "soldermask", "surfacefinish"})


@dataclass(frozen=True)
class StackLayer:
    """One physical layer of the board, top to bottom.

    Attributes
    ----------
    kind : str
        ``"copper"`` or ``"dielectric"``.
    name : str
        Unique name of the layer; the key materials are assigned
        against, and the name the imported solid carries.
    thickness : float
        Layer thickness [meters].
    number : int or None
        Copper layer number as the file functions count it (``L1`` is
        the top copper), or ``None`` for a dielectric.
    epsilon : float or None
        Relative permittivity of a dielectric, if the job file states
        one.
    loss_tangent : float or None
        Loss tangent of a dielectric, if stated.  Reported to the
        caller, never modelled: a single number carries no reference
        frequency, and a frequency-independent loss tangent is not
        causal.
    material : str or None
        Material name from the job file (``"FR4"``), for reporting.
    """

    kind: str
    name: str
    thickness: float
    number: int | None = None
    epsilon: float | None = None
    loss_tangent: float | None = None
    material: str | None = None

    @property
    def is_copper(self) -> bool:
        return self.kind == _COPPER


@dataclass(frozen=True)
class DrillRole:
    """A drill file as the job file lists it.

    Attributes
    ----------
    path : Path
        The drill file.
    plated : bool or None
        Whether the holes are plated through, if the job file says.
    span : tuple of int or None
        Copper layer numbers the holes run between (``(1, 2)`` on a
        two-layer board), if the job file says.
    """

    path: Path
    plated: bool | None = None
    span: tuple[int, int] | None = None


@dataclass(frozen=True)
class Stackup:
    """Everything the job file says about one board.

    Attributes
    ----------
    layers : tuple of StackLayer
        Copper and dielectric layers, top to bottom.
    copper_files : dict
        Copper layer number (``1`` is top) to Gerber file path.
    outline_file : Path or None
        The profile (board outline) Gerber, if the job file names one.
    drill_files : tuple of DrillRole
        Drill files the job file names, in the order it names them,
        with the plating and layer span it declares for each.
    project : str or None
        Project name from the job file.
    path : Path
        The job file itself, for error messages.
    """

    layers: tuple[StackLayer, ...]
    copper_files: dict[int, Path]
    outline_file: Path | None
    drill_files: tuple[DrillRole, ...]
    project: str | None
    path: Path

    @property
    def copper_layers(self) -> tuple[StackLayer, ...]:
        """The copper layers, top to bottom."""
        return tuple(layer for layer in self.layers if layer.is_copper)

    def copper_number(self, layer: StackLayer) -> int:
        """Copper layer number of *layer* (``1`` is the top copper)."""
        if layer.number is None:
            raise ValueError(f"{layer.name!r} is not a copper layer.")
        return layer.number

    def elevations(self) -> tuple[tuple[float, float], ...]:
        """``(z_bottom, z_top)`` [meters] of every layer, in stack order.

        The top face of the topmost dielectric is the origin, and the
        stack grows downwards from there, so the top copper occupies
        ``[0, t]`` and everything below the substrate surface has
        negative *z*.  Fixing the origin to the substrate rather than to
        the outermost copper keeps the reference plane where the fields
        are: adding an outer layer to the stack does not move the board.
        """
        above = 0.0
        for layer in self.layers:
            if layer.kind == _DIELECTRIC:
                break
            above += layer.thickness

        out: list[tuple[float, float]] = []
        z = above
        for layer in self.layers:
            out.append((z - layer.thickness, z))
            z -= layer.thickness
        return tuple(out)

    def thickness(self) -> float:
        """Total board thickness [meters]."""
        return sum(layer.thickness for layer in self.layers)


# ─────────────────────────────────────────────────────────────────────
# locating the job file
# ─────────────────────────────────────────────────────────────────────


def find_job_file(path: str | Path) -> Path:
    """The job file at *path*, or the only one in the directory *path*.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist, or names a directory without a job
        file in it.
    ValueError
        If *path* is a directory holding more than one job file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No such file or directory: {path}. Give the .gbrjob file of a "
            f"fabrication export, or the directory holding it."
        )
    if path.is_file():
        return path

    jobs = sorted(path.glob("*.gbrjob"))
    if not jobs:
        raise FileNotFoundError(
            f"No .gbrjob job file in {path}. A Gerber set alone does not say "
            f"how thick its layers are or what the dielectric is, so the job "
            f"file is required. Re-export the fabrication data with the job "
            f"file enabled."
        )
    if len(jobs) > 1:
        names = ", ".join(repr(job.name) for job in jobs)
        raise ValueError(
            f"{path} holds more than one job file ({names}). Pass the one to "
            f"import instead of the directory."
        )
    return jobs[0]


# ─────────────────────────────────────────────────────────────────────
# reading
# ─────────────────────────────────────────────────────────────────────


def _number(value, what: str, job: Path) -> float:
    """A job file number, which the format also permits as a string."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} in job file {job} is not a number: {value!r}.") from None


def _file_function(entry: dict) -> tuple[str, ...]:
    """The comma-separated file function of a ``FilesAttributes`` entry."""
    text = entry.get("FileFunction")
    if not isinstance(text, str):
        return ()
    return tuple(part.strip() for part in text.split(","))


def _copper_number(function: tuple[str, ...]) -> int | None:
    """Copper layer number of a file function, or ``None``.

    A copper file function reads ``Copper,L<n>,<position>`` — the layer
    number is what ties a drawing to a layer of the stackup.
    """
    if len(function) < 2 or function[0] != "Copper":
        return None
    token = function[1]
    if not token.startswith("L") or not token[1:].isdigit():
        return None
    return int(token[1:])


def _drill_span(function: tuple[str, ...]) -> tuple[int, int] | None:
    """Copper layers a drill file function runs between, if it says.

    The function reads ``(Non)Plated,<from>,<to>,<kind>`` — the two
    layer numbers are what separates a through hole from a blind or
    buried one, and therefore how deep the barrel goes.
    """
    if len(function) < 3:
        return None
    try:
        start, end = int(function[1]), int(function[2])
    except ValueError:
        return None
    return (min(start, end), max(start, end))


def _stack_layers(entries: list, job: Path) -> tuple[StackLayer, ...]:
    """The copper and dielectric layers of a ``MaterialStackup``."""
    layers: list[StackLayer] = []
    copper_index = 0
    dielectric_index = 0

    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {position} of the material stackup in {job} is not an object.")
        kind_text = str(entry.get("Type", "")).strip()
        folded = kind_text.replace(" ", "").lower()
        if folded in _IGNORED_TYPES:
            continue
        if folded == "copper":
            kind = _COPPER
            copper_index += 1
            number = copper_index
            default_name = f"copper_L{copper_index}"
        elif folded == "dielectric":
            kind = _DIELECTRIC
            dielectric_index += 1
            number = None
            default_name = f"dielectric_{dielectric_index}"
        else:
            raise ValueError(
                f"Entry {position} of the material stackup in {job} has an "
                f"unknown type {kind_text!r}. Expected 'Copper', 'Dielectric', "
                f"or one of the coating types that carry no volume "
                f"(Legend, SolderMask, SolderPaste, SurfaceFinish)."
            )

        if "Thickness" not in entry:
            label = entry.get("Name") or default_name
            raise ValueError(
                f"The material stackup in job file {job} gives no thickness "
                f"for {kind} layer {label!r}. A layout without layer "
                f"thicknesses has no 3D shape. Fill in the physical stackup "
                f"in the layout tool (thicknesses of every copper and "
                f"dielectric layer) and export the fabrication data again."
            )
        thickness = _number(entry["Thickness"], f"Thickness of layer {position}", job) * _MM
        if thickness <= 0.0:
            label = entry.get("Name") or default_name
            raise ValueError(
                f"Layer {label!r} in job file {job} has a thickness of "
                f"{thickness / _MM} mm. Every layer needs a positive thickness."
            )

        epsilon = None
        loss_tangent = None
        material = None
        if kind == _DIELECTRIC:
            if entry.get("DielectricConstant") is not None:
                epsilon = _number(
                    entry["DielectricConstant"], f"DielectricConstant of layer {position}", job
                )
            if entry.get("LossTangent") is not None:
                loss_tangent = _number(
                    entry["LossTangent"], f"LossTangent of layer {position}", job
                )
            if entry.get("Material") is not None:
                material = str(entry["Material"])

        name = str(entry.get("Name") or "").strip() or default_name
        # Dielectric names repeat in practice ("FR4" for every core), and
        # names are the key materials are assigned against, so a
        # dielectric is named by its position in the stack.  Copper layer
        # names are unique by construction and are kept as drawn.
        if kind == _DIELECTRIC:
            name = default_name
        layers.append(
            StackLayer(
                kind=kind,
                name=name,
                thickness=thickness,
                number=number,
                epsilon=epsilon,
                loss_tangent=loss_tangent,
                material=material,
            )
        )

    if not layers:
        raise ValueError(
            f"The material stackup in job file {job} holds no copper or dielectric layer."
        )
    return _uniquified(layers)


def _uniquified(layers: list[StackLayer]) -> tuple[StackLayer, ...]:
    """Force layer names apart, since they are the material keys."""
    seen: dict[str, int] = {}
    out: list[StackLayer] = []
    for layer in layers:
        name = layer.name
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(layer if name == layer.name else StackLayer(**{**layer.__dict__, "name": name}))
    return tuple(out)


def read_gbrjob(path: str | Path) -> Stackup:
    """Read a Gerber job file into a :class:`Stackup`.

    Parameters
    ----------
    path : str or Path
        The ``.gbrjob`` file, or a directory holding exactly one.

    Returns
    -------
    Stackup
        Layers top to bottom, with the Gerber and drill files of the
        set resolved relative to the job file.
    """
    job = find_job_file(path)
    try:
        text = job.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"Could not read job file {job}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Job file {job} is not valid JSON (line {exc.lineno}, column "
            f"{exc.colno}: {exc.msg}). Check that it is a Gerber job file and "
            f"not truncated."
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"Job file {job} does not hold a JSON object.")

    entries = data.get("MaterialStackup")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Job file {job} carries no material stackup, so it says nothing "
            f"about layer thicknesses or the dielectric. Fill in the physical "
            f"stackup in the layout tool and export the fabrication data again."
        )
    layers = _stack_layers(entries, job)

    folder = job.parent
    copper_files: dict[int, Path] = {}
    outline: Path | None = None
    drills: list[DrillRole] = []
    attributes = data.get("FilesAttributes")
    for entry in attributes if isinstance(attributes, list) else []:
        if not isinstance(entry, dict):
            continue
        raw_path = entry.get("Path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        # Paths are relative to the job file; some writers quote a bare
        # name, others a path with separators.  Both are joined the same.
        target = folder / raw_path
        function = _file_function(entry)
        number = _copper_number(function)
        if number is not None:
            copper_files.setdefault(number, target)
        elif function[:1] == ("Profile",):
            outline = outline or target
        elif function[:1] in {("Plated",), ("NonPlated",)}:
            drills.append(
                DrillRole(path=target, plated=function[0] == "Plated", span=_drill_span(function))
            )

    project = None
    specs = data.get("GeneralSpecs")
    if isinstance(specs, dict):
        identity = specs.get("ProjectId")
        if isinstance(identity, dict) and isinstance(identity.get("Name"), str):
            project = identity["Name"].strip() or None

    return Stackup(
        layers=layers,
        copper_files=copper_files,
        outline_file=outline,
        drill_files=tuple(drills),
        project=project,
        path=job,
    )
