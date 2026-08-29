"""
CSG Boolean operations: Union, Intersection, Difference — plus Group.

Boolean ops wrap two (or more) shapes and build the OCC BRep shape lazily.
The material of a boolean operation is derived from the constituent shapes;
the result shape uses the material of the base operand for Difference,
and requires explicit material assignment for Union/Intersection.

:class:`Group` is the odd one out: a *heterogeneous* container that keeps
each member's own material, so it deliberately has no single material or
OCC solid.  Boolean ops therefore reject a Group operand (see
:func:`_reject_group`); a Group is instead flattened into its members when
added to a :class:`~magnelio.geo.GeometryModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magnelio.materials.material import Material


from magnelio.geo._cache import cached_occ_shape
from magnelio.geo._validate import operand
from magnelio.geo.shape import Shape
from magnelio.materials.material import resolve_material


def _reject_group(operands, op_name: str) -> None:
    """Raise if any operand is a :class:`Group`.

    A Group has no single material or OCC solid, so it cannot be an
    operand of a Boolean operation.  This is the one hard boundary of the
    Group abstraction (see the module docstring).
    """
    for s in operands:
        if isinstance(s, Group):
            raise TypeError(
                f"{op_name} cannot take a Group as an operand: a Group is a "
                "heterogeneous bundle of shapes with no single material or "
                "solid. Apply the Boolean operation to the individual "
                "members, or add the Group directly to a GeometryModel."
            )


def _check_operands(operands, op_name: str, *, minimum: int) -> None:
    """Reject a wrong operand count, a Group, or a non-geometry operand.

    Runs before anything reads ``.material`` off an operand, so passing
    a list of shapes where the shapes themselves belong — the natural
    misreading of the ``*shapes`` signature — is named for what it is
    instead of failing as a missing attribute.
    """
    if len(operands) < minimum:
        raise ValueError(
            f"{op_name} needs at least {minimum} operand"
            f"{'s' if minimum > 1 else ''}; got {len(operands)}."
        )
    _reject_group(operands, op_name)
    for index, s in enumerate(operands):
        operand(s, f"{op_name} operand {index}")


@dataclass
class Union(Shape):
    """Boolean union of two or more shapes.

    Args:
        shapes:   Two or more CSG shapes to unite.
        material: Material for the resulting volume. If None, uses the
                  material of the first shape.
        name:     Optional label.
    """

    shapes: tuple
    material: "Material | None" = None
    name: str | None = None

    def __init__(self, *shapes, material=None, name=None):
        _check_operands(shapes, "Union", minimum=1)
        self.shapes = shapes
        material = resolve_material(material, "Union(material=...)")
        self.material = material if material is not None else shapes[0].material
        self.name = name

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import boolean_union

        return boolean_union([s._occ_shape(scale) for s in self.shapes])

    def _analytic_bbox(self):
        from magnelio.geo._scaling import union_boxes

        return union_boxes([s._analytic_bbox() for s in self.shapes])


@dataclass
class Intersection(Shape):
    """Boolean intersection of two shapes.

    Args:
        shape_a:  Base shape.
        shape_b:  Tool shape.
        material: Material for the resulting volume. Defaults to shape_a's material.
        name:     Optional label.
    """

    shape_a: object
    shape_b: object
    material: "Material | None" = None
    name: str | None = None

    def __post_init__(self):
        _check_operands((self.shape_a, self.shape_b), "Intersection", minimum=2)
        self.material = resolve_material(self.material, "Intersection(material=...)")
        if self.material is None:
            self.material = self.shape_a.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import boolean_intersection

        return boolean_intersection(self.shape_a._occ_shape(scale), self.shape_b._occ_shape(scale))

    def _analytic_bbox(self):
        from magnelio.geo._scaling import intersect_boxes

        return intersect_boxes(self.shape_a._analytic_bbox(), self.shape_b._analytic_bbox())


@dataclass
class Difference(Shape):
    """Boolean difference: base minus one or more tools.

    Parameters
    ----------
    base : shape
        Shape to subtract from.
    *tools : shape
        One or more shapes to subtract.  When multiple tools are given
        they are fused first and then cut from the base in a single
        Boolean operation.
    material : Material or None
        Material for the resulting volume.  Defaults to base's material.
    name : str or None
        Optional label.
    """

    base: object
    tools: tuple
    material: "Material | None" = None
    name: str | None = None

    def __init__(self, base, *tools, material=None, name=None):
        _check_operands((base, *tools), "Difference", minimum=2)
        self.base = base
        self.tools = tools
        material = resolve_material(material, "Difference(material=...)")
        self.material = material if material is not None else base.material
        self.name = name

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import boolean_difference

        return boolean_difference(self.base._occ_shape(scale), self._occ_tools(scale))

    def _occ_tools(self, scale=1.0):
        """The tools fused into one kernel shape at *scale*, cached per instance.

        Kept beside the result because the mesher reads it in place of
        the result wherever the tools lie inside a box-shaped base:
        the effective PEC solid of a housing (the housing brick minus
        this Difference is the brick minus the base plus the tools),
        the section contours, face and edge planes and edge convexity
        of the body (the base's and the tools' union's) — the cut
        itself is then never built.
        """
        cache = self.__dict__.setdefault("_occ_tools_cache", {})
        key = float(scale)
        if key not in cache:
            from magnelio.geo._occ_backend import boolean_union  # noqa: PLC0415

            shapes = [t._occ_shape(key) for t in self.tools]
            cache[key] = shapes[0] if len(shapes) == 1 else boolean_union(shapes)
        return cache[key]

    def _analytic_bbox(self):
        return self.base._analytic_bbox()


@dataclass
class Group(Shape):
    """A logical bundle of shapes that preserves each member's material.

    Unlike :class:`Union` — which *fuses* its operands into a single solid
    carrying one material — a Group is a **heterogeneous container**: every
    member keeps its own material and its own OCC solid.  It exists so a
    multi-material assembly (e.g. an SMA connector: PEC pin + PTFE
    dielectric + PEC shell) can be positioned and added to a model as one
    unit, without collapsing the materials.

    A Group therefore has **no single** ``material`` or ``_occ_shape`` — it
    is a *compound* node exposing :meth:`members` and :meth:`bounding_box`.
    Consequences:

    - **Transforms distribute over members.**
      :meth:`~magnelio.geo.Shape.translated` on a Group returns a new
      Group whose members are each translated; likewise
      :meth:`~magnelio.geo.Shape.rotated`,
      :meth:`~magnelio.geo.Shape.scaled` and
      :meth:`~magnelio.geo.Shape.mirrored`.  The repeat helpers grow
      a ``group=True`` sibling of ``unite=True`` that aggregates copies
      into a Group instead of fusing them.
    - **Nesting is allowed** (a Group of Groups); :meth:`members` flattens
      recursively.
    - **Flattened at** :meth:`~magnelio.geo.GeometryModel.add`, so the
      mesher, material filling and overlap layers never see a Group.
    - **CSG Boolean ops reject a Group** (:class:`Union`,
      :class:`Intersection`, :class:`Difference`), since it has no single
      material or solid.

    Parameters
    ----------
    *shapes : shape or Group
        Leaf shapes and/or nested Groups to bundle.
    name : str, optional
        Optional label.
    """

    shapes: tuple
    name: str | None = None

    def __init__(self, *shapes, name=None):
        for index, s in enumerate(shapes):
            if not isinstance(s, Group):
                operand(s, f"Group member {index}")
        self.shapes = shapes
        self.name = name

    def members(self):
        """Yield the leaf shapes, recursively flattening nested Groups.

        Any transform applied to the Group has already been baked into
        each member (transforms distribute on application), so the yielded
        leaves are ready to hand to the mesher.
        """
        for s in self.shapes:
            if isinstance(s, Group):
                yield from s.members()
            else:
                yield s

    def volume(self, scale: float | None = None) -> float:
        """Total volume of every member [cubic meters].

        A Group is a bundle of separate solids, so the volumes simply
        add; members are not fused first, and a Group whose members
        overlapped would count the overlap twice — which a
        :class:`~magnelio.geo.GeometryModel` does not allow in the first
        place.
        """
        return sum(s.volume(scale) for s in self.members())

    def bounding_box(self):
        """Axis-aligned bounding box enclosing every member [meters]."""
        mins: list = []
        maxs: list = []
        for s in self.members():
            lo, hi = s.bounding_box()
            mins.append(lo)
            maxs.append(hi)
        if not mins:
            raise ValueError("Group is empty — no members to bound.")
        lo = tuple(min(c[i] for c in mins) for i in range(3))
        hi = tuple(max(c[i] for c in maxs) for i in range(3))
        return lo, hi

    def _analytic_bbox(self):
        from magnelio.geo._scaling import union_boxes

        boxes = [s._analytic_bbox() for s in self.members()]
        if not boxes:
            raise ValueError("Group is empty — no members to bound.")
        return union_boxes(boxes)
