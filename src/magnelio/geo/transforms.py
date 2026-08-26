"""
Geometric transforms: translate, rotate, scale, mirror.

These functions wrap a CSG shape in a transformed copy.
Transforms are applied via OCC BRepBuilderAPI_Transform.
"""

from __future__ import annotations

from dataclasses import dataclass

from magnelio.geo._cache import cached_occ_shape
from magnelio.geo._validate import count, finite, nonzero, point3, vector3
from magnelio.geo.shape import Shape

_SHEET_VARIANTS: dict = {}


def _wrapper(cls, inner):
    """The wrapper class to instantiate for *inner*: *cls* itself, or
    its sheet-preserving variant when *inner* is a sheet.

    A moved, turned, scaled or mirrored sheet is still a sheet — still
    a profile for ``extruded()``/``thickened()``, and still planar if it
    was — so the wrapper inherits the marker base of what it wraps.
    """
    from magnelio.geo._sheet import PlanarSheet, Sheet  # noqa: PLC0415

    if isinstance(inner, PlanarSheet):
        marker = PlanarSheet
    elif isinstance(inner, Sheet):
        marker = Sheet
    else:
        return cls
    key = (cls, marker)
    if key not in _SHEET_VARIANTS:
        _SHEET_VARIANTS[key] = type(
            f"{cls.__name__}{marker.__name__}",
            (cls, marker),
            {"__doc__": f"{cls.__name__} of a {marker.__name__}.", "__module__": __name__},
        )
    return _SHEET_VARIANTS[key]


def _apply_repeat(shape, make_one, repeat, copy, unite, group=False):
    """Shared logic for repeated transforms.

    Parameters
    ----------
    shape : CSG shape
        The original (untransformed) shape.
    make_one : callable(int) -> shape
        Returns the *i*-th transformed copy (i = 1 … repeat).
    repeat : int
        Number of transformed copies.
    copy : bool
        If True, include the untransformed original in the result.
    unite : bool
        If True, fuse the copies into a single :class:`Union` (one solid,
        one material).
    group : bool
        If True, bundle the copies into a :class:`Group` (each copy keeps
        its own material).  The material-preserving sibling of *unite*;
        mutually exclusive with it.

    Returns
    -------
    shape or list or Union or Group
    """
    if unite and group:
        raise ValueError("Pass either unite=True or group=True, not both.")
    repeat = count(repeat, "repeat", minimum=1)

    if repeat == 1 and not copy:
        return make_one(1)

    copies = []
    if copy:
        copies.append(shape)
    for i in range(1, repeat + 1):
        copies.append(make_one(i))

    if unite:
        from magnelio.geo.operations import Union  # noqa: PLC0415

        return Union(*copies)
    if group:
        from magnelio.geo.operations import Group  # noqa: PLC0415

        return Group(*copies)
    return copies


def _distribute(group, transform_one):
    """Apply a per-member transform to a Group, preserving nesting/material.

    Parameters
    ----------
    group : Group
        The bundle to transform.
    transform_one : callable(member) -> member
        Transforms a single member.  A nested Group member recurses
        (``transform_one`` on a Group re-enters this helper), so the tree
        structure and every member's material are preserved.

    Returns
    -------
    Group
        A new Group with the same *name* and each member transformed.
    """
    from magnelio.geo.operations import Group  # noqa: PLC0415

    return Group(*[transform_one(m) for m in group.shapes], name=group.name)


def translate(
    shape,
    vector: tuple[float, float, float],
    *,
    repeat: int = 1,
    copy: bool = False,
    unite: bool = False,
    group: bool = False,
):
    """Implementation of :meth:`magnelio.geo.Shape.translated`.

    A :class:`Group` is translated member-by-member; the result is again
    a Group.
    """
    from magnelio.geo.operations import Group  # noqa: PLC0415

    vector = vector3(vector, "translated(vector)")
    if isinstance(shape, Group):

        def make_one(i):
            v = tuple(c * i for c in vector)
            return _distribute(shape, lambda m: translate(m, v))

        return _apply_repeat(shape, make_one, repeat, copy, unite, group)

    def make_one(i):
        v = tuple(c * i for c in vector)
        return _wrapper(_TranslatedShape, shape)(shape, v)

    return _apply_repeat(shape, make_one, repeat, copy, unite, group)


def rotate(
    shape,
    axis: tuple[float, float, float],
    angle_deg: float,
    origin=(0.0, 0.0, 0.0),
    *,
    repeat: int = 1,
    copy: bool = False,
    unite: bool = False,
    group: bool = False,
):
    """Implementation of :meth:`magnelio.geo.Shape.rotated`.

    A :class:`Group` is rotated member-by-member; the result is again a
    Group.
    """
    from magnelio.geo._axes import normalize_axis  # noqa: PLC0415
    from magnelio.geo.operations import Group  # noqa: PLC0415

    axis = normalize_axis(axis, "rotated(axis)")
    angle_deg = finite(angle_deg, "rotated(angle_deg)")
    origin = point3(origin, "rotated(origin)")
    if isinstance(shape, Group):

        def make_one(i):
            return _distribute(shape, lambda m: rotate(m, axis, angle_deg * i, origin))

        return _apply_repeat(shape, make_one, repeat, copy, unite, group)

    def make_one(i):
        return _wrapper(_RotatedShape, shape)(shape, axis, angle_deg * i, origin)

    return _apply_repeat(shape, make_one, repeat, copy, unite, group)


def scale(shape, factor: float, center=(0.0, 0.0, 0.0)):
    """Implementation of :meth:`magnelio.geo.Shape.scaled`.

    A :class:`Group` is scaled member-by-member about the common
    *center*; the result is again a Group.
    """
    from magnelio.geo.operations import Group  # noqa: PLC0415

    factor = nonzero(factor, "scaled(factor)")
    center = point3(center, "scaled(center)")
    if isinstance(shape, Group):
        return _distribute(shape, lambda m: scale(m, factor, center))
    return _wrapper(_ScaledShape, shape)(shape, factor, center)


def mirror(
    shape,
    *,
    normal,
    position: float = 0.0,
    copy: bool = False,
    unite: bool = False,
    group: bool = False,
):
    """Implementation of :meth:`magnelio.geo.Shape.mirrored`.

    A :class:`Group` is mirrored member-by-member; the result is again a
    Group.
    """
    from magnelio.geo._axes import normalize_axis  # noqa: PLC0415
    from magnelio.geo.operations import Group  # noqa: PLC0415

    normal = normalize_axis(normal, "mirrored(normal)")
    position = finite(position, "mirrored(position)")
    if (unite or group) and not copy:
        # DD-126: mirror has no repeat, so without copy there is a
        # single shape to bundle.  Silently honouring unite/group would
        # hand back a bare mirror image where the caller asked for the
        # symmetric whole — a wrong geometry that meshes and solves.
        raise ValueError(
            "mirror(unite=True) / mirror(group=True) needs copy=True — "
            "there is nothing to combine the mirror image with otherwise."
        )

    if isinstance(shape, Group):

        def make_one(i):
            return _distribute(shape, lambda m: mirror(m, normal=normal, position=position))

        return _apply_repeat(shape, make_one, 1, copy, unite, group)

    def make_one(i):
        return _wrapper(_MirroredShape, shape)(shape, normal, position)

    return _apply_repeat(shape, make_one, 1, copy, unite, group)


@dataclass
class _TranslatedShape(Shape):
    _inner: object
    _vector: tuple[float, float, float]

    @property
    def material(self):
        return self._inner.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import occ_translate

        return occ_translate(self._inner._occ_shape(scale), self._vector, scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import translate_box

        return translate_box(self._inner._analytic_bbox(), self._vector)


@dataclass
class _RotatedShape(Shape):
    _inner: object
    _axis: tuple[float, float, float]
    _angle_deg: float
    _origin: tuple[float, float, float]

    @property
    def material(self):
        return self._inner.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import occ_rotate

        return occ_rotate(
            self._inner._occ_shape(scale), self._axis, self._angle_deg, self._origin, scale=scale
        )

    def _analytic_bbox(self):
        from magnelio.geo._scaling import rotate_box

        return rotate_box(self._inner._analytic_bbox(), self._axis, self._angle_deg, self._origin)


@dataclass
class _ScaledShape(Shape):
    _inner: object
    _factor: float
    _center: tuple[float, float, float]

    @property
    def material(self):
        return self._inner.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import occ_scale

        return occ_scale(self._inner._occ_shape(scale), self._factor, self._center, scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import scale_box

        return scale_box(self._inner._analytic_bbox(), self._factor, self._center)


@dataclass
class _MirroredShape(Shape):
    _inner: object
    _normal: tuple[float, float, float]
    _position: float

    @property
    def material(self):
        return self._inner.material

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import occ_mirror

        return occ_mirror(self._inner._occ_shape(scale), self._normal, self._position, scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import mirror_box

        return mirror_box(self._inner._analytic_bbox(), self._normal, self._position)
