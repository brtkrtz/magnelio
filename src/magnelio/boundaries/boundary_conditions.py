"""Boundary-condition specification — string-keyed thin facade.

A user-facing dataclass that names the BC type for each of the six
domain faces (``"PEC"``, ``"PMC"``, ``"CPML"``, ``"Periodic"``).  The
:meth:`BoundaryConditions.to_objects` method materialises the strings
into the concrete runtime BC instances the solver expects.

A face may also be declared a *symmetry plane* — physically
identical to a PEC/PMC wall, plus the semantic "the mirror image of
the model exists beyond this wall".  Declared as the type strings
``"SymmetryPEC"`` / ``"SymmetryPMC"`` or as a :class:`Symmetry`
instance carrying an explicit plane position.  The dataclass fields
always hold the *physical* type ("PEC"/"PMC") after normalisation, so
every consumer that dispatches on the wall type keeps working
unchanged; the symmetry semantics live in the separate
:attr:`BoundaryConditions.symmetry` map, read through
:func:`symmetry_entries`.

The boundary closure is declared on the *model* — it is
passed to :class:`~magnelio.geo.GeometryModel` (or to
:meth:`~magnelio.mesh.mesher.Mesh.from_grid` on the OCC-free path),
carried by the :class:`~magnelio.mesh.mesher.Mesh`, and read from there
by the analyses.  The declaration drives *all four* of its
consequences from one place: the CPML grid extension, the PMC
grid-line pull-in, the PEC wall mask, and the runtime BC objects.

Advanced users who need finer control (e.g. custom BC subclasses) can
skip this dataclass and pass a ``dict[str, BoundaryProtocol]`` in the
same places — :func:`bc_type_entries` reduces either form to the
canonical ``{face: type_str}`` map every consumer dispatches on.
"""

# Design: DD-103 (boundary closure declared on the model), DD-154 (symmetry
# planes).

from __future__ import annotations

from dataclasses import dataclass, field

FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


@dataclass(frozen=True)
class Symmetry:
    """Symmetry-plane declaration for one domain face.

    Physically the face is closed with the wall named by *kind*; the
    declaration additionally records that the model continues as its
    mirror image beyond the plane, so symmetry-aware readers (port
    impedance reports, field plots, exports) can restore full-model
    semantics.

    Parameters
    ----------
    kind : str
        Wall type of the symmetry plane: ``"PEC"`` (electric symmetry,
        tangential E vanishes on the plane) or ``"PMC"`` (magnetic
        symmetry, tangential H vanishes).
    position : float, optional
        World coordinate of the symmetry plane on the face's axis [m].
        When given, the mesher clips the computational domain to the
        kept half-space at exactly this plane — the full geometry may
        be modelled and the discarded half is simply never meshed.
        When *None* (default), the domain is taken as built: the
        geometry itself already ends at the symmetry plane.
    """

    kind: str
    position: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("PEC", "PMC"):
            raise ValueError(
                f"Symmetry kind must be 'PEC' or 'PMC'; got {self.kind!r}",
            )
        if self.position is not None:
            object.__setattr__(self, "position", float(self.position))


# String shorthand for a Symmetry declaration without a position.
_SYMMETRY_STRINGS = {"SymmetryPEC": "PEC", "SymmetryPMC": "PMC"}


@dataclass
class BoundaryConditions:
    """Boundary-condition specification for all six domain faces.

    Parameters
    ----------
    xmin, xmax, ymin, ymax, zmin, zmax : str or Symmetry
        BC type for each face.  One of ``"PEC"``, ``"PMC"``, ``"CPML"``,
        ``"Periodic"``, ``"SymmetryPEC"``, ``"SymmetryPMC"``, or a
        :class:`Symmetry` instance.  Default ``"PEC"`` everywhere.
        Symmetry declarations are normalised on construction: the field
        keeps the physical wall type (``"PEC"``/``"PMC"``) and the
        symmetry semantics move into :attr:`symmetry`.
    symmetry : dict, optional
        Canonical symmetry map ``{face: position_or_None}``.
        Filled automatically from Symmetry-typed face values; may also
        be passed directly.  At most one face per axis can be a
        symmetry plane — two parallel mirror planes would describe an
        infinite image chain, not a finite full model.
    cpml_thickness_cells : int, default 8
        Layer depth for every face declared ``"CPML"``, measured in
        bulk (``h_max``-sized) cells.  One number for both consequences
        of that depth: the mesher extends the grid by this many cells
        on each CPML face, and the runtime layer grades its profile
        over the same span (previously the two were separate
        settings on ``MeshControl`` and the analysis and could disagree).
    """

    xmin: str = "PEC"
    xmax: str = "PEC"
    ymin: str = "PEC"
    ymax: str = "PEC"
    zmin: str = "PEC"
    zmax: str = "PEC"
    cpml_thickness_cells: int = 8
    symmetry: dict = field(default_factory=dict)

    _VALID = ("PEC", "PMC", "CPML", "Periodic")

    def __post_init__(self) -> None:
        # Normalise symmetry declarations (DD-154): the face field keeps
        # the physical wall type, the symmetry map keeps the semantics.
        sym = dict(self.symmetry)
        for face in FACES:
            val = getattr(self, face)
            if isinstance(val, Symmetry):
                sym[face] = val.position
                setattr(self, face, val.kind)
            elif val in _SYMMETRY_STRINGS:
                sym.setdefault(face, None)
                setattr(self, face, _SYMMETRY_STRINGS[val])
        for face in FACES:
            val = getattr(self, face)
            if val not in self._VALID:
                raise ValueError(
                    f"BoundaryConditions.{face}: {val!r} is not valid. "
                    f"Choose from "
                    f"{list(self._VALID) + list(_SYMMETRY_STRINGS)!r}.",
                )
        for face, pos in sym.items():
            if face not in FACES:
                raise ValueError(
                    f"unknown symmetry face {face!r}; choose from {list(FACES)!r}.",
                )
            if getattr(self, face) not in ("PEC", "PMC"):
                raise ValueError(
                    f"symmetry face {face!r} must carry a PEC or PMC "
                    f"wall; got {getattr(self, face)!r}.",
                )
            if pos is not None:
                sym[face] = float(pos)
        for axis in "xyz":
            if f"{axis}min" in sym and f"{axis}max" in sym:
                raise ValueError(
                    f"both {axis}min and {axis}max are declared symmetry "
                    f"planes; two parallel mirror planes describe an "
                    f"infinite image chain, not a finite full model.",
                )
        self.symmetry = sym
        if self.cpml_thickness_cells < 1:
            raise ValueError(
                f"cpml_thickness_cells must be >= 1; got {self.cpml_thickness_cells}",
            )

    def to_dict(self) -> dict[str, str]:
        return {face: getattr(self, face) for face in FACES}

    def to_objects(self, grid) -> dict:
        """Materialise BC strings into runtime BC objects.

        Parameters
        ----------
        grid : GridLines
            The simulation grid.  Required for PMC, CPML, and Periodic
            instances which need it for their internal index maps and
            polynomial profiles.

        Returns
        -------
        dict[str, BoundaryProtocol]
            Mapping ``face -> BC instance``, ready to be passed as
            ``boundary_conditions=`` to ``FITTimeDomainSolver``.
        """
        return {
            face: materialize_boundary(
                face,
                bc_type,
                grid,
                cpml_thickness_cells=self.cpml_thickness_cells,
            )
            for face, bc_type in self.to_dict().items()
        }


def bc_type_entries(boundary_conditions) -> dict[str, str]:
    """Reduce any accepted BC declaration to ``{face: type_str}``.

    The canonical reader for every consumer that dispatches on the BC
    *type* rather than on runtime behaviour — the mesher (which faces
    extend the grid, pull their grid line in, or carry a wall mask),
    the port mode-path detection, the SIBC wall enumeration, and the
    resume recipe.

    Accepts a :class:`BoundaryConditions` or a ``dict`` whose values
    are type strings and/or runtime BC instances; faces absent from a
    partial dict default to ``"PEC"``, matching the FIT update's
    behaviour on an undeclared face.

    Raises
    ------
    TypeError
        On an unknown declaration type, or a dict value that is neither
        a type string nor a recognised BC instance.
    ValueError
        On an unknown face key or an invalid type string.
    """
    from magnelio.boundaries.cpml import CPMLBoundary  # noqa: PLC0415
    from magnelio.boundaries.pec import PECBoundary  # noqa: PLC0415
    from magnelio.boundaries.periodic import PeriodicBoundary  # noqa: PLC0415
    from magnelio.boundaries.pmc import PMCBoundary  # noqa: PLC0415

    if isinstance(boundary_conditions, BoundaryConditions):
        return boundary_conditions.to_dict()
    if not isinstance(boundary_conditions, dict):
        raise TypeError(
            f"boundary_conditions must be BoundaryConditions or dict; "
            f"got {type(boundary_conditions).__name__}",
        )

    object_type = (
        (PECBoundary, "PEC"),
        (PMCBoundary, "PMC"),
        (CPMLBoundary, "CPML"),
        (PeriodicBoundary, "Periodic"),
    )
    out = dict.fromkeys(FACES, "PEC")
    for face, value in boundary_conditions.items():
        if face not in out:
            raise ValueError(
                f"unknown boundary face {face!r}; choose from {list(FACES)!r}.",
            )
        if isinstance(value, Symmetry):
            out[face] = value.kind
            continue
        if isinstance(value, str):
            value = _SYMMETRY_STRINGS.get(value, value)
            if value not in BoundaryConditions._VALID:
                raise ValueError(
                    f"boundary condition {value!r} on face {face!r} is "
                    f"not valid. Choose from "
                    f"{list(BoundaryConditions._VALID) + list(_SYMMETRY_STRINGS)!r}.",
                )
            out[face] = value
            continue
        for cls, name in object_type:
            if isinstance(value, cls):
                out[face] = name
                break
        else:
            raise TypeError(
                f"cannot read the boundary type of {value!r} on face "
                f"{face!r}; use a type string or a PEC/PMC/CPML/"
                f"Periodic instance",
            )
    return out


def resolve_boundary_conditions(boundary_conditions):
    """Normalise a user-supplied BC declaration for storage.

    ``None`` becomes the all-PEC default (a closed electric chamber —
    the conventional closure, and the safe one for a user who has not
    thought about the boundary yet).  A plain type-string dict, whether
    partial or complete, is canonicalised into a
    :class:`BoundaryConditions` so the stored closure always shows all
    six faces.

    A dict carrying runtime BC *instances* is returned unchanged: a
    ``PECBoundary`` may hold this face's wall material and a
    ``CPMLBoundary`` its own profile, neither of which a type string
    can express.
    """
    # Wall material on PECBoundary: DD-099.
    if boundary_conditions is None:
        return BoundaryConditions()
    entries = bc_type_entries(boundary_conditions)  # validates
    if isinstance(boundary_conditions, BoundaryConditions):
        return boundary_conditions
    if all(isinstance(v, (str, Symmetry)) for v in boundary_conditions.values()):
        return BoundaryConditions(
            **entries,
            symmetry=symmetry_entries(boundary_conditions),
        )
    return boundary_conditions


def symmetry_entries(boundary_conditions) -> dict[str, float | None]:
    """Symmetry-plane declarations of *boundary_conditions*.

    The canonical reader for every symmetry-aware consumer — the
    mesher's domain clip, and the result-side readers that restore
    full-model semantics.  Returns ``{face: position_or_None}``:
    *None* means the geometry itself ends at the symmetry plane (the
    domain wall as built), a float is the declared world coordinate
    the mesher clipped the domain at.

    Accepts every declaration form :func:`bc_type_entries` accepts;
    forms that cannot express symmetry (``None``, runtime BC
    instances) yield an empty map.
    """
    # Symmetry-plane semantics: DD-154.
    if isinstance(boundary_conditions, BoundaryConditions):
        return dict(boundary_conditions.symmetry)
    if not isinstance(boundary_conditions, dict):
        return {}
    out: dict[str, float | None] = {}
    for face, value in boundary_conditions.items():
        if isinstance(value, Symmetry):
            out[face] = value.position
        elif isinstance(value, str) and value in _SYMMETRY_STRINGS:
            out[face] = None
    return out


def cpml_thickness_of(boundary_conditions) -> int:
    """CPML layer depth [cells] declared by *boundary_conditions*.

    Reads the :class:`BoundaryConditions` field; on the dict path the
    depth of the first ``CPMLBoundary`` instance found (a plain
    ``{"zmax": "CPML"}`` string dict has no place to carry one and
    yields the default 8).
    """
    from magnelio.boundaries.cpml import CPMLBoundary  # noqa: PLC0415

    if isinstance(boundary_conditions, BoundaryConditions):
        return int(boundary_conditions.cpml_thickness_cells)
    if isinstance(boundary_conditions, dict):
        for value in boundary_conditions.values():
            if isinstance(value, CPMLBoundary):
                return int(getattr(value, "thickness_cells", 8))
    return 8


def materialize_boundary(
    face: str,
    bc_type: str,
    grid,
    cpml_thickness_cells: int = 8,
):
    """Materialise one BC-type string into a runtime BC instance.

    Shared by :meth:`BoundaryConditions.to_objects` and the high-level
    ``AnalysisScatteringTD``, which accepts string-valued
    ``boundary_conditions`` dicts (``{"ymin": "PEC", ...}``).  The
    solver only dispatches on ``apply_E`` / ``apply_H`` attributes, so
    a raw string in its BC dict would be a silent no-op — every string
    entry must pass through here first.

    Parameters
    ----------
    face : str
        Bbox face name (``"xmin"`` … ``"zmax"``).
    bc_type : str
        One of ``"PEC"``, ``"PMC"``, ``"CPML"``, ``"Periodic"``.
    grid : GridLines
        The simulation grid (needed by PMC, CPML, Periodic).
    cpml_thickness_cells : int, default 8
        Layer thickness when ``bc_type == "CPML"``.

    Returns
    -------
    BoundaryProtocol
        The runtime BC instance for this face.
    """
    from magnelio.boundaries.cpml import CPMLBoundary
    from magnelio.boundaries.pec import PECBoundary
    from magnelio.boundaries.periodic import PeriodicBoundary
    from magnelio.boundaries.pmc import PMCBoundary

    if bc_type == "PEC":
        return PECBoundary(face)
    if bc_type == "PMC":
        return PMCBoundary(face, grid)
    if bc_type == "CPML":
        return CPMLBoundary(
            face,
            grid,
            thickness_cells=cpml_thickness_cells,
        )
    if bc_type == "Periodic":
        axis = face[0]  # 'x', 'y', or 'z'
        return PeriodicBoundary(axis, grid)
    raise ValueError(
        f"boundary condition {bc_type!r} on face {face!r} is not "
        f"valid. Choose from {list(BoundaryConditions._VALID)!r}.",
    )
