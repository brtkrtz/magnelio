"""Immutable affine transforms shared by every geometry category."""

from __future__ import annotations

import math
from dataclasses import dataclass

from magnelio.geo._cache import cached_occ_shape
from magnelio.geo._topology_history import finish
from magnelio.geo._validate import count, finite, nonzero, point3, vector3
from magnelio.geo.shape import Shape, Solid

_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _matrix(value) -> tuple[tuple[float, float, float, float], ...]:
    try:
        rows = tuple(tuple(float(c) for c in row) for row in value)
    except (TypeError, ValueError):
        raise TypeError("Transform matrix must be a 4 x 4 sequence of finite numbers.") from None
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("Transform matrix must have exactly 4 rows and 4 columns.")
    if not all(math.isfinite(c) for row in rows for c in row):
        raise ValueError("Transform matrix entries must be finite.")
    if rows[3] != (0.0, 0.0, 0.0, 1.0):
        raise ValueError("Transform matrix must be affine: its last row must be (0, 0, 0, 1).")

    columns = tuple(tuple(rows[i][j] for i in range(3)) for j in range(3))
    lengths = tuple(math.hypot(*column) for column in columns)
    if min(lengths) <= 0.0:
        raise ValueError("Transform matrix must be invertible.")
    scale = lengths[0]
    if any(not math.isclose(length, scale, rel_tol=1e-12) for length in lengths[1:]):
        raise ValueError("Transform supports uniform scale only; its three axis scales must match.")
    unit_columns = tuple(
        tuple(component / length for component in column)
        for column, length in zip(columns, lengths)
    )
    for i in range(3):
        for j in range(i + 1, 3):
            dot = sum(a * b for a, b in zip(unit_columns[i], unit_columns[j]))
            if abs(dot) > 1e-12:
                raise ValueError(
                    "Transform axes must remain perpendicular (no shear is supported)."
                )
    return rows


def _multiply(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)) for i in range(4)
    )


@dataclass(frozen=True)
class Transform:
    """An immutable affine placement value.

    Matrices act on column-vector points.  Consequently ``A @ B @ shape``
    applies ``B`` first and then ``A``.  Only rigid transforms, reflections
    and uniform scales are accepted; shear and non-uniform scale would change
    the public analytic geometry categories.

    Parameters
    ----------
    matrix : sequence of sequence of float
        Homogeneous 4 x 4 affine matrix.  Prefer :class:`Translation`,
        :class:`Rotation`, :class:`Mirror`, :class:`Scale`, and composition.
    """

    matrix: tuple

    def __post_init__(self):
        object.__setattr__(self, "matrix", _matrix(self.matrix))

    @classmethod
    def identity(cls) -> "Transform":
        """Return the identity transform."""
        return Transform(_IDENTITY)

    def __matmul__(self, other):
        if isinstance(other, Transform):
            return Transform(_multiply(self.matrix, other.matrix))
        from magnelio.geo.operations import Group  # noqa: PLC0415

        if isinstance(other, Group):
            return Group(*(self @ member for member in other.shapes), name=other.name)
        if isinstance(other, Shape):
            return _apply(self, other)
        raise TypeError(
            "A Transform can be composed with another Transform or applied "
            f"to a Shape or Group; got {type(other).__name__}."
        )

    def point(self, point) -> tuple[float, float, float]:
        """Return *point* transformed by this affine value."""
        x, y, z = point3(point, "Transform.point(point)")
        p = (x, y, z, 1.0)
        return tuple(sum(self.matrix[i][j] * p[j] for j in range(4)) for i in range(3))


@dataclass(frozen=True, init=False)
class Translation(Transform):
    """Translation by a three-dimensional vector.

    Parameters
    ----------
    vector : tuple of float
        ``(dx, dy, dz)`` translation [meters].
    """

    def __init__(self, vector):
        x, y, z = vector3(vector, "Translation(vector)")
        super().__init__(((1.0, 0.0, 0.0, x), (0.0, 1.0, 0.0, y), (0.0, 0.0, 1.0, z), _IDENTITY[3]))


@dataclass(frozen=True, init=False)
class Rotation(Transform):
    """Right-handed rotation about an axis through an origin.

    Parameters
    ----------
    axis : str or tuple of float
        ``"x"``, ``"y"``, ``"z"`` or an arbitrary direction vector.
    angle_deg : float
        Right-handed rotation angle [degrees].
    origin : tuple of float
        Point on the rotation axis [meters].
    """

    def __init__(self, axis, angle_deg, origin=(0.0, 0.0, 0.0)):
        from magnelio.geo._axes import normalize_axis  # noqa: PLC0415

        x, y, z = normalize_axis(axis, "Rotation(axis)")
        angle = math.radians(finite(angle_deg, "Rotation(angle_deg)"))
        origin = point3(origin, "Rotation(origin)")
        c = math.cos(angle)
        s = math.sin(angle)
        one_c = 1.0 - c
        linear = (
            (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
            (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
            (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
        )
        moved = tuple(origin[i] - sum(linear[i][j] * origin[j] for j in range(3)) for i in range(3))
        super().__init__(tuple((*linear[i], moved[i]) for i in range(3)) + (_IDENTITY[3],))


@dataclass(frozen=True, init=False)
class Mirror(Transform):
    """Reflection across the plane ``point . normal == position``.

    Parameters
    ----------
    normal : str or tuple of float
        ``"x"``, ``"y"``, ``"z"`` or the plane's normal vector.
    position : float
        Signed plane position along the unit normal [meters].
    """

    def __init__(self, normal, position=0.0):
        from magnelio.geo._axes import normalize_axis  # noqa: PLC0415

        n = normalize_axis(normal, "Mirror(normal)")
        position = finite(position, "Mirror(position)")
        linear = tuple(
            tuple((1.0 if i == j else 0.0) - 2.0 * n[i] * n[j] for j in range(3)) for i in range(3)
        )
        moved = tuple(2.0 * position * n[i] for i in range(3))
        super().__init__(tuple((*linear[i], moved[i]) for i in range(3)) + (_IDENTITY[3],))


@dataclass(frozen=True, init=False)
class Scale(Transform):
    """Uniform scale about a fixed centre.

    Parameters
    ----------
    factor : float
        Non-zero uniform scale factor.
    center : tuple of float
        Fixed point of the scale [meters].
    """

    def __init__(self, factor, center=(0.0, 0.0, 0.0)):
        factor = nonzero(factor, "Scale(factor)")
        center = point3(center, "Scale(center)")
        moved = tuple((1.0 - factor) * c for c in center)
        super().__init__(
            (
                (factor, 0.0, 0.0, moved[0]),
                (0.0, factor, 0.0, moved[1]),
                (0.0, 0.0, factor, moved[2]),
                _IDENTITY[3],
            )
        )


_CATEGORY_WRAPPERS: dict[type, type] = {}


def _category_wrapper(shape):
    from magnelio.geo._sheet import Profile, Sheet  # noqa: PLC0415
    from magnelio.geo.surfaces import Surface  # noqa: PLC0415

    if isinstance(shape, Profile):
        category = Profile
    elif isinstance(shape, Surface):
        category = Surface
    elif isinstance(shape, Sheet):
        category = Sheet
    elif isinstance(shape, Solid):
        category = Solid
    else:
        category = Shape
    if category not in _CATEGORY_WRAPPERS:
        _CATEGORY_WRAPPERS[category] = type(
            f"Transformed{category.__name__}",
            (_TransformedShape, category),
            {
                "__doc__": f"Affine transform of a {category.__name__}.",
                "__module__": __name__,
            },
        )
    return _CATEGORY_WRAPPERS[category]


def _apply(transform: Transform, shape: Shape):
    from magnelio.geo.curves import Curve  # noqa: PLC0415

    if isinstance(shape, Curve):
        return _transform_curve(transform, shape)
    if isinstance(shape, _TransformedShape):
        transform = transform @ shape._transform
        shape = shape._inner
    return finish(_category_wrapper(shape)(shape, transform))


def _transform_curve(transform: Transform, curve):
    from magnelio.geo.curves import Curve  # noqa: PLC0415

    root = getattr(curve, "_transform_root", curve)
    previous = getattr(curve, "_transform_value", Transform.identity())
    combined = transform @ previous

    def build(scale):
        from magnelio.geo._occ_backend import occ_transform  # noqa: PLC0415

        return occ_transform(root._occ_shape(scale), combined.matrix, scale=scale)

    bounds = _transform_box(root._analytic_bbox(), combined)
    ends = None
    if root._ends is not None:
        ends = tuple(combined.point(point) for point in root._ends)
    result = Curve(_build=build, name=root.name, _bounds=bounds, _ends=ends)
    result._transform_root = root
    result._transform_value = combined
    return result


@dataclass
class _TransformedShape(Shape):
    _inner: object
    _transform: Transform

    @property
    def material(self):
        return self._inner.material

    @property
    def name(self):
        return getattr(self._inner, "name", None)

    @property
    def color(self):
        return getattr(self._inner, "color", None)

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import occ_transform  # noqa: PLC0415

        return occ_transform(self._inner._occ_shape(scale), self._transform.matrix, scale=scale)

    def _analytic_bbox(self):
        return _transform_box(self._inner._analytic_bbox(), self._transform)


def _transform_box(box, transform):
    from magnelio.geo._scaling import box_of_points  # noqa: PLC0415

    lo, hi = box
    corners = tuple(
        (x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])
    )
    return box_of_points([transform.point(point) for point in corners])


def _apply_repeat(shape, make_one, repeat, copy, unite, group):
    """Place independent copies, then apply the requested result category."""
    if unite and group:
        raise ValueError("Pass either unite=True or group=True, not both.")
    repeat = count(repeat, "repeat", minimum=1)
    if unite and not isinstance(shape, Solid):
        raise TypeError(
            "unite=True accepts Solid geometry only; "
            f"got {type(shape).__name__}. Use group=True to preserve separate members."
        )
    if repeat == 1 and not copy and not unite and not group:
        return make_one(1)
    copies = [shape] if copy else []
    copies.extend(make_one(i) for i in range(1, repeat + 1))
    if unite:
        from magnelio.geo.operations import Union  # noqa: PLC0415

        return Union(*copies)
    if group:
        from magnelio.geo.operations import Group  # noqa: PLC0415

        return Group(*copies)
    return copies


def translate(shape, vector, *, repeat=1, copy=False, unite=False, group=False):
    """Place a linear array through the common affine backend."""
    vector = vector3(vector, "Translation(vector)")
    return _apply_repeat(
        shape,
        lambda i: Translation(tuple(i * x for x in vector)) @ shape,
        repeat,
        copy,
        unite,
        group,
    )


def rotate(
    shape,
    axis,
    angle_deg,
    origin=(0.0, 0.0, 0.0),
    *,
    repeat=1,
    copy=False,
    unite=False,
    group=False,
):
    """Place a circular array through the common affine backend."""
    from magnelio.geo._axes import normalize_axis  # noqa: PLC0415

    axis = normalize_axis(axis, "Rotation(axis)")
    angle_deg = finite(angle_deg, "Rotation(angle_deg)")
    origin = point3(origin, "Rotation(origin)")
    return _apply_repeat(
        shape,
        lambda i: Rotation(axis, i * angle_deg, origin) @ shape,
        repeat,
        copy,
        unite,
        group,
    )


def scale(shape, factor, center=(0.0, 0.0, 0.0)):
    """Apply :class:`Scale`; retained as an internal helper."""
    return Scale(factor, center) @ shape


def mirror(shape, *, normal, position=0.0, copy=False, unite=False, group=False):
    """Place one mirror image, optionally combined with its original."""
    transform = Mirror(normal, position)
    if (unite or group) and not copy:
        raise ValueError("mirrored(unite=True) or mirrored(group=True) requires copy=True.")
    return _apply_repeat(shape, lambda i: transform @ shape, 1, copy, unite, group)
