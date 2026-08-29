"""
CSG geometry subsystem.

- Base class: ``Shape`` — the Boolean operators and the chainable verbs
  every geometry object shares; the documented home of both.
- Primitives: ``Brick``, ``Sphere``, ``Cylinder``, ``Cone``, ``Torus``,
  ``Face``; curves: ``Curve`` (polyline / arc / spline / helix);
  curved sheets: ``Surface`` (``Surface.parametric`` samples a map
  ``(u, v) -> (x, y, z)`` into a B-spline sheet — a reflector dish before
  it is extruded into metal).
- Profiles: ``Path`` draws a chained curve segment by segment;
  ``Curve.joined()`` chains existing curves, ``Curve.covered()`` turns a
  closed one into a planar sheet and ``Curve.traced()`` into a
  conductor track.
- Operations: ``Union``, ``Intersection``, ``Difference`` — or the
  operators ``a + b`` / ``a - b`` / ``a & b`` on any shape — and
  ``Loft`` through a series of cross-sections.
- Containers: ``Group`` (material-preserving bundle), ``GeometryModel``.
- Imported geometry: ``ImportedSolid`` — a solid read from a CAD file
  (``magnelio.io.import_step``) or from a project store.
- Verbs: chainable shape methods — ``.translated()``, ``.rotated()``,
  ``.scaled()``, ``.mirrored()``, ``.chamfered()``, ``.filleted()``,
  ``.extruded()``, ``.lofted()``, ``.revolved()``, ``.swept()``,
  ``.shelled()``, ``.thickened()``.  They are documented on
  :class:`Shape`, the base class every geometry object inherits from.

``GeometryModel`` lives in the core ``magnelio`` namespace; every
other geometry name is public here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magnelio.geo._validate import operand
from magnelio.geo.curves import Curve
from magnelio.geo.imported import ImportedSolid
from magnelio.geo.modifications import Loft
from magnelio.geo.operations import Difference, Group, Intersection, Union
from magnelio.geo.path import Path
from magnelio.geo.primitives import Brick, Cone, Cylinder, Face, Sphere, Torus
from magnelio.geo.shape import Shape
from magnelio.geo.surfaces import Surface
from magnelio.geo.wire import ThinWire

if TYPE_CHECKING:
    from magnelio.materials.material import Material


class GeometryOverlapError(ValueError):
    """Raised when shapes in a GeometryModel overlap volumetrically."""

    pass


def _require_material(shape) -> None:
    """Reject a construction body on its way into a GeometryModel.

    Shapes may be built without a material — those are construction
    bodies whose only purpose is to shape other bodies (Boolean tools,
    extrusion profiles).  Boolean results inherit the material of their
    base/first operand, so a result that is still material-less means
    *that* operand was a construction body too.
    """
    # DD-127: the single door into a model, so every downstream
    # consumer keeps its unconditional shape.material access.
    if getattr(shape, "material", None) is not None:
        return
    label = getattr(shape, "name", None)
    what = f"{type(shape).__name__} {label!r}" if label else type(shape).__name__
    raise ValueError(
        f"{what} carries no material and cannot be added to a "
        f"GeometryModel. A shape without a material is a construction "
        f"body — usable as a Boolean operand or an extrusion profile, "
        f"but not a physical object. Give the shape a material, or (for "
        f"a Boolean result) give one to its base operand: a Difference "
        f"takes the material of its base, a Union that of its first "
        f"shape."
    )


def _disjoint_by_construction(solids: list) -> set[tuple[int, int]]:
    """Index pairs a model's own construction proves disjoint.

    A ``Difference`` cannot overlap a shape that was cut out of it:
    the tool itself, or — when the tool is a ``Union`` — any of the
    union's operands.  Matching is by object identity, so only the
    very shapes the user cut with count (a copy is a different shape).
    """
    index = {id(s): k for k, s in enumerate(solids)}
    pairs: set[tuple[int, int]] = set()
    for k, shape in enumerate(solids):
        if not isinstance(shape, Difference):
            continue
        tools = list(shape.tools)
        for tool in shape.tools:
            if isinstance(tool, Union):
                tools.extend(tool.shapes)
        for tool in tools:
            j = index.get(id(tool))
            if j is not None and j != k:
                pairs.add((min(k, j), max(k, j)))
    return pairs


class GeometryModel:
    """Container for a CSG geometry model (ordered list of shapes).

    Shapes are stored in insertion order.  Each spatial point should be
    covered by at most one shape; use CSG operations (``Difference``,
    ``Union``, ``Intersection``) to partition overlapping volumes.

    Parameters
    ----------
    background : Material or str or None
        Material for cells not covered by any shape — an instance or a
        built-in name (``"air"``, ``"vacuum"``, ``"pec"``).  Defaults to air
        (eps=mu=1, sigma=0).  The background fills the *volume* outside
        every shape; the boundary closure of the domain is declared
        separately via *boundary_conditions* and wins over it on the
        bbox faces.
    boundary_conditions : BoundaryConditions or dict or None
        Closure of the six domain faces — ``"PEC"``, ``"PMC"``,
        ``"CPML"``, ``"Periodic"``, or a symmetry declaration
        (``"SymmetryPEC"``/``"SymmetryPMC"``, optionally as a
        ``("SymmetryPEC", position)`` tuple, or
        ``"ForceSymmetryPEC"``/``"ForceSymmetryPMC"``) per face.
        Declared here because it is a property of the modelled domain,
        not of the analysis run on it: a symmetry face is a mirror
        plane, a CPML face is an opening.
        :meth:`~magnelio.mesh.mesher.Mesh.from_geometry` derives all
        mesh-time consequences from it (CPML grid extension, PMC
        grid-line pull-in, PEC wall mask, symmetry domain clip), and
        the analyses read it back off the mesh.  ``None`` (default)
        closes every face with PEC.
    allow_overlaps : bool
        If *False* (default), :meth:`~magnelio.mesh.mesher.Mesh.from_geometry`
        raises :class:`GeometryOverlapError` when shapes overlap.  Set to
        *True* to restore legacy last-wins semantics.

    Examples
    --------
    A cavity carved out of a metal block::

        from magnelio import GeometryModel, Material
        from magnelio.geo import Brick, Difference

        pec  = Material.pec()
        air  = Material.air()

        outer = Brick(origin=(0,0,0), size=(10e-3,10e-3,10e-3), material=pec)
        cavity = Brick(origin=(2e-3,2e-3,2e-3), size=(6e-3,6e-3,6e-3), material=air)

        model = GeometryModel()
        model.add(Difference(outer, cavity))
        model.add(cavity)

    A magnetic symmetry plane at ``x = 0`` — the full geometry may be
    modelled; the declared plane clips the mesh to ``x >= 0`` and the
    mirror half is never meshed::

        model = GeometryModel(
            background=pec,
            boundary_conditions={"xmin": "SymmetryPMC"},
        )

    For a plane away from the origin pass the position explicitly:
    ``{"xmin": ("SymmetryPMC", 1.5e-3)}``.  If the geometry itself
    already ends at the symmetry plane, declare it without clipping:
    ``{"xmin": "ForceSymmetryPMC"}``.
    """

    def __init__(
        self,
        *,
        background: "Material | None" = None,
        boundary_conditions=None,
        allow_overlaps: bool = False,
    ) -> None:
        from magnelio.boundaries.boundary_conditions import (  # noqa: PLC0415
            resolve_boundary_conditions,
        )
        from magnelio.materials.material import (  # noqa: PLC0415
            Material,
            resolve_material,
        )

        self.shapes: list = []
        background = resolve_material(background, "GeometryModel(background=...)")
        self.background: Material = background if background is not None else Material.air()
        self.boundary_conditions = resolve_boundary_conditions(
            boundary_conditions,
        )
        self.allow_overlaps: bool = allow_overlaps
        self.ports: list = []
        self.elements: list = []
        self.sources: list = []

    def add(self, shape) -> "GeometryModel":
        """Add a CSG shape (or list of shapes) to the model.

        A :class:`~magnelio.geo.Group` is flattened into its member
        shapes on insertion (recursively for nested Groups), so the
        mesher, material filling and overlap layers only ever see leaf
        shapes.  Lists/tuples are added element-wise, and may themselves
        contain Groups.

        Every shape entering the model must carry a material: a
        material-less shape is a construction body (a Boolean operand or
        an extrusion profile), not a physical object, and is rejected
        here rather than at mesh time.

        Returns
        -------
        GeometryModel
            ``self``, to allow chaining.

        Raises
        ------
        ValueError
            If *shape* carries no material.
        """
        if isinstance(shape, (list, tuple)):
            for s in shape:
                self.add(s)
        elif isinstance(shape, Group):
            for s in shape.members():
                self.add(s)
        else:
            operand(shape, "The object added to a GeometryModel")
            _require_material(shape)
            self.shapes.append(shape)
        return self

    def add_port(self, port) -> "GeometryModel":
        """Declare a port on the model, before meshing.

        Accepts the declarative port objects
        (:class:`~magnelio.ports.declarative.PortWaveguide`,
        :class:`~magnelio.ports.declarative.PortAnalytical`) — the
        geometric declaration only; the mode physics is resolved by the
        analysis against the finished mesh, exactly as before.

        Declaring ports here lets :meth:`~magnelio.mesh.mesher.Mesh.from_geometry`
        see which domain faces carry one: the equidistant-cell buffer
        the modal operators require is then generated only on those
        faces instead of on all six (the port-blind fallback).  The
        mesh carries the declarations to the analysis, so
        ``AnalysisScatteringTD(mesh=mesh, ...)`` needs no ``ports=`` of
        its own.

        Parameters
        ----------
        port : PortWaveguide, PortAnalytical or PortLumped
            Declarative port; its name must be unique on this model.

        Returns
        -------
        GeometryModel
            ``self``, for chaining.
        """
        from magnelio.ports.declarative import (  # noqa: PLC0415
            PortAnalytical,
            PortLumped,
            PortWaveguide,
        )

        if not isinstance(port, (PortWaveguide, PortAnalytical, PortLumped)):
            raise TypeError(
                f"add_port() takes a declarative port (PortWaveguide / "
                f"PortAnalytical / PortLumped), got {type(port).__name__}. Spec-level "
                f"ports carry solver detail the model must not depend "
                f"on — pass those to the analysis via ports=."
            )
        self._check_unique_name(port.name, "port")
        self.ports.append(port)
        return self

    def add_element(self, element) -> "GeometryModel":
        """Declare a passive lumped circuit element on the model.

        Accepts a declarative :class:`magnelio.circuit.LumpedElement`
        — a straight interior edge path carrying a trapezoidal RLC
        companion model as a pure passive load (no excitation, no
        S-matrix column).  Like ports, elements travel with the mesh
        to the analysis; ports and elements share one name namespace
        because the solver keys per-operator checkpoint state by name.

        Parameters
        ----------
        element : magnelio.circuit.LumpedElement
            Declarative element; its name must be unique among the
            ports *and* elements of this model.

        Returns
        -------
        GeometryModel
            ``self``, for chaining.
        """
        from magnelio.circuit import LumpedElement  # noqa: PLC0415

        if not isinstance(element, LumpedElement):
            raise TypeError(
                f"add_element() takes a magnelio.circuit.LumpedElement, "
                f"got {type(element).__name__}."
            )
        self._check_unique_name(element.name, "element")
        self.elements.append(element)
        return self

    def add_source(self, source) -> "GeometryModel":
        """Declare a field source on the model, before meshing.

        Accepts a :class:`magnelio.sources.SourceFieldIncident` (such
        as :class:`~magnelio.sources.SourcePlaneWave`).  Like ports and
        elements, sources travel with the mesh to the analysis, and an
        :class:`~magnelio.Excitation` drives one by name — so ports,
        elements and sources share one name namespace.

        Parameters
        ----------
        source : magnelio.sources.Source
            Declarative source; its name must be unique among the
            ports, elements and sources of this model.

        Returns
        -------
        GeometryModel
            ``self``, for chaining.
        """
        from magnelio.sources.base import Source  # noqa: PLC0415

        if not isinstance(source, Source):
            raise TypeError(
                f"add_source() takes a magnelio.sources source (SourcePlaneWave, ...), "
                f"got {type(source).__name__}."
            )
        self._check_unique_name(source.name, "source")
        self.sources.append(source)
        return self

    def _check_unique_name(self, name: str, kind: str) -> None:
        """Ports, elements and sources share one name namespace (DD-123, DD-224)."""
        taken = [p.name for p in self.ports] + [e.name for e in self.elements]
        taken += [s.name for s in self.sources]
        if name in taken:
            raise ValueError(f"duplicate {kind} name {name!r} on this model")

    def bounding_box(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return the axis-aligned bounding box of all shapes combined.

        Returns
        -------
        tuple
            ``(min_corner, max_corner)`` in meters.  Requires OCC.
        """
        if not self.shapes:
            raise ValueError("GeometryModel is empty — no shapes added.")
        from magnelio.geo._scaling import model_scale  # noqa: PLC0415

        geo_scale = model_scale(self.shapes)
        corners_min = []
        corners_max = []
        for shape in self.shapes:
            lo, hi = shape.bounding_box(geo_scale)
            corners_min.append(lo)
            corners_max.append(hi)
        lo = tuple(min(c[i] for c in corners_min) for i in range(3))
        hi = tuple(max(c[i] for c in corners_max) for i in range(3))
        return lo, hi  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_cross_section(
        self,
        normal: str,
        position: float,
        **kwargs,
    ):
        """Plot a 2D cross-section at an axis-aligned plane.

        Thin wrapper around
        :func:`~magnelio.post.plot_geometry.plot_cross_section`.
        All keyword arguments are forwarded.

        Parameters
        ----------
        normal : str
            ``'x'``, ``'y'``, or ``'z'``.
        position : float
            Position along the normal axis [m].
        **kwargs
            Forwarded (``mesh``, ``scale_mm``, ``flip``, ``ax``, ``title``,
            ``deflection``).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        from magnelio.post.plot_geometry import (  # noqa: PLC0415
            plot_cross_section as _plot_cross_section,
        )

        return _plot_cross_section(self, normal, position, **kwargs)

    def plot(self, mesh=None, **kwargs):
        """Interactive 3D view of the model.

        Thin wrapper around
        :func:`~magnelio.post.plot_3d.show_geometry`.  In a notebook the
        view is a widget with an axis-aligned cutting plane driven from
        its toolbar; in a script it opens a window.

        Parameters
        ----------
        mesh : Mesh, optional
            Show this mesh's grid with the geometry: the grid cells —
            coloured by assigned material — on the cutting plane.
        **kwargs
            Forwarded (``cut``, ``flip``, ``show_ports``, ``show_wires``,
            ``show_grid``, ``mode``, ``size``, ``render_edges``,
            ``edge_color``, ``quality``, ``scale_mm``, ``camera``).

        Returns
        -------
        pyvista.Plotter or None
            The plotter for ``mode="none"``; otherwise the view is
            displayed and ``None`` is returned.
        """
        from magnelio.post.plot_3d import show_geometry as _show_geometry  # noqa: PLC0415

        return _show_geometry(self, mesh=mesh, **kwargs)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Check that no two shapes overlap volumetrically.

        Overlaps between shapes of the **same material** are allowed and not
        reported: the overlap region is unambiguous (it gets that material
        either way).  "Same material" is decided by value equality
        (:meth:`Material.__eq__`), so two distinct ``Material.pec()``
        instances count as the same.  Overlaps between shapes of *different*
        materials are ambiguous and still raise.

        Raises
        ------
        GeometryOverlapError
            If any two shapes of different materials have a nonzero
            volumetric intersection.
        """
        # Thin wires (DD-080) are 1D sub-cell objects: a wire endpoint
        # on/inside a PEC solid is a legal, documented topology (the
        # monopole-on-ground connection), so wires are excluded from the
        # volumetric overlap check.
        from magnelio.geo.wire import ThinWire as _ThinWire  # noqa: PLC0415

        solids = [s for s in self.shapes if not isinstance(s, _ThinWire)]
        if len(solids) < 2:
            return

        from magnelio.geo._occ_backend import check_pairwise_overlaps  # noqa: PLC0415
        from magnelio.geo._scaling import model_scale  # noqa: PLC0415

        materials = [getattr(s, "material", None) for s in solids]
        overlaps = check_pairwise_overlaps(
            solids,
            materials=materials,
            scale=model_scale(solids),
            disjoint=_disjoint_by_construction(solids),
        )
        if overlaps:
            lines = []
            for i, j, vol in overlaps:
                name_i = getattr(solids[i], "name", None) or f"shape_{i}"
                name_j = getattr(solids[j], "name", None) or f"shape_{j}"
                lines.append(f"  {name_i} & {name_j}: {vol:.3e} m³")
            msg = (
                f"{len(overlaps)} overlapping shape pair(s) detected:\n"
                + "\n".join(lines)
                + "\n\nUse CSG operations (Difference, Union, Intersection) to "
                "partition overlapping volumes, or pass allow_overlaps=True."
            )
            raise GeometryOverlapError(msg)

    def __len__(self) -> int:
        return len(self.shapes)

    def __iter__(self):
        return iter(self.shapes)

    def __repr__(self) -> str:
        return f"GeometryModel({len(self.shapes)} shapes, background={self.background.name})"


__all__ = [
    "Shape",
    "Brick",
    "Sphere",
    "Cylinder",
    "Cone",
    "Torus",
    "Face",
    "Surface",
    "Curve",
    "Path",
    "Union",
    "Intersection",
    "Difference",
    "Loft",
    "Group",
    "ThinWire",
    "ImportedSolid",
    "GeometryOverlapError",
]
