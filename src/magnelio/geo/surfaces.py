"""Curved sheets from parametric maps.

A :class:`Surface` is a zero-thickness curved sheet — the free-form
counterpart of the planar :class:`~magnelio.geo.Face`.  It is built
from a parametric map ``(u, v) -> (x, y, z)`` sampled on a grid and
interpolated by a B-spline surface, and it serves as the profile that
:meth:`~magnelio.geo.Shape.extruded` or
:meth:`~magnelio.geo.Shape.thickened` turn into a solid: a reflector
dish, a shaped sub-reflector, a lens surface.

The class stores the sampled points, not the map: a shape is a value
(hashable, cacheable, serialisable like every other shape), and the
parametric history is by design not part of a model — the same rule
that governs imported CAD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnelio.geo._cache import cached_occ_shape
from magnelio.geo._sheet import Sheet
from magnelio.geo._validate import count, finite
from magnelio.materials.material import resolve_material


@dataclass
class Surface(Sheet):
    """A curved zero-thickness sheet through a grid of sample points.

    Build one with :meth:`parametric`; the constructor itself takes the
    point grid, which is what a stored or hand-sampled surface provides.
    The sheet is the B-spline surface interpolating the samples exactly
    (a degree-3 interpolant between them), bounded by the grid's four
    edge rows.

    Like a :class:`~magnelio.geo.Face`, a Surface carries an
    **optional** material: none (default) makes it a *construction
    profile* — the input to :meth:`~magnelio.geo.Shape.extruded` or
    :meth:`~magnelio.geo.Shape.thickened`, which turn it into a solid;
    a material makes it a *thin sheet*, whose physics is not wired
    yet, so it cannot be added to a model on its own.

    Parameters
    ----------
    points : sequence of sequence of (float, float, float)
        The sample grid, ``nu`` rows of ``nv`` points [meters], at
        least 2 × 2.  Rows follow the first parameter, columns the
        second.  A row may collapse onto one point (the pole of a polar
        parametrisation) — the surface closes there.
    material : Material or str, optional
        Material of the thin sheet.  ``None`` (default) = construction
        profile.
    name : str, optional
        Optional label.
    """

    points: tuple
    material: object = None
    name: str | None = None

    def __post_init__(self):
        self.material = resolve_material(self.material, "Surface.material")
        self.points = _point_grid(self.points, "Surface.points")

    @classmethod
    def parametric(cls, fn, *, u, v, samples=(32, 32), material=None, name=None):
        """Sample a parametric map ``(u, v) -> (x, y, z)`` into a Surface.

        Parameters
        ----------
        fn : callable
            The map.  Called once with two NumPy arrays of parameter
            values (shape ``(nu, nv)``) and expected to return the three
            coordinates ``(x, y, z)`` [meters] — each an array of the
            same shape or a scalar.  A function written for scalars is
            accepted too: if the array call fails, it is evaluated
            point by point.
        u, v : (float, float)
            Parameter intervals ``(start, end)`` of the two parameters.
        samples : (int, int)
            Number of sample points along ``u`` and ``v`` (default
            ``(32, 32)``, at least 2 each).  The interpolant is exact at
            the samples and degree-3 between them, so the count sets
            how closely the sheet follows the map between samples —
            32 × 32 places a 240 mm paraboloid dish to a few micrometres,
            32 × 64 to 1e-8 m.
        material : Material or str, optional
            Material of the thin sheet; ``None`` (default) makes a
            construction profile.
        name : str, optional
            Optional label.

        Returns
        -------
        Surface
            The sampled sheet.

        Examples
        --------
        An offset paraboloid dish of focal length ``F`` and aperture
        ``D``, centred ``x_c`` off the axis, parametrised in polar
        coordinates about the aperture centre so that its rim is a
        circle::

            def dish(r, phi):
                x = x_c + r * np.cos(phi)
                y = r * np.sin(phi)
                return x, y, (x**2 + y**2) / (4 * F)

            sheet = geo.Surface.parametric(dish, u=(0.0, D / 2), v=(0.0, 2 * np.pi))
            reflector = sheet.extruded(vector=(0, 0, -5e-3), material="pec")
        """
        if not callable(fn):
            raise TypeError(f"Surface.parametric() needs a callable map; got {type(fn).__name__}")
        u0, u1 = _interval(u, "Surface.parametric(u)")
        v0, v1 = _interval(v, "Surface.parametric(v)")
        try:
            nu, nv = samples
        except (TypeError, ValueError):
            raise ValueError(
                f"Surface.parametric(samples) must be a pair of integers; got {samples!r}"
            ) from None
        nu = count(nu, "Surface.parametric(samples[0])", minimum=2)
        nv = count(nv, "Surface.parametric(samples[1])", minimum=2)
        uu, vv = np.meshgrid(np.linspace(u0, u1, nu), np.linspace(v0, v1, nv), indexing="ij")
        xyz = _evaluate(fn, uu, vv)
        points = tuple(tuple(tuple(float(c) for c in p) for p in row) for row in xyz)
        return cls(points=points, material=material, name=name)

    @cached_occ_shape
    def _occ_shape(self, scale=1.0):
        from magnelio.geo._occ_backend import make_bspline_surface  # noqa: PLC0415

        return make_bspline_surface(self.points, scale=scale)

    def _analytic_bbox(self):
        from magnelio.geo._scaling import box_diagonal, box_of_points, pad_box  # noqa: PLC0415

        # An interpolating spline may overshoot the hull of its samples;
        # pad by a quarter of the hull diagonal (the Loft's allowance
        # for a spline blend) — conservative, kernel-free.
        box = box_of_points([p for row in self.points for p in row])
        return pad_box(box, 0.25 * box_diagonal(box))


def _interval(value, what):
    try:
        a, b = value
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a pair (start, end); got {value!r}") from None
    a = finite(a, f"{what}[0]")
    b = finite(b, f"{what}[1]")
    if a == b:
        raise ValueError(f"{what} must span a non-empty interval; got ({a}, {b})")
    return a, b


def _evaluate(fn, uu, vv):
    """Evaluate the map on the sample grid; returns an (nu, nv, 3) array."""
    shape = uu.shape
    try:
        out = fn(uu, vv)
        coords = [np.broadcast_to(np.asarray(c, dtype=float), shape) for c in _three(out)]
        xyz = np.stack(coords, axis=-1)
    except Exception:  # noqa: BLE001 — a scalar-only map is legitimate
        xyz = np.empty(shape + (3,), dtype=float)
        for idx in np.ndindex(*shape):
            xyz[idx] = [float(c) for c in _three(fn(float(uu[idx]), float(vv[idx])))]
    if not np.all(np.isfinite(xyz)):
        raise ValueError("Surface.parametric(): the map returned a non-finite coordinate.")
    return xyz


def _three(out):
    try:
        x, y, z = out
    except (TypeError, ValueError):
        raise ValueError(
            "Surface.parametric(): the map must return three coordinates (x, y, z)."
        ) from None
    return x, y, z


def _point_grid(value, what):
    try:
        rows = [list(row) for row in value]
    except TypeError:
        raise ValueError(f"{what} must be a grid (sequence of rows of points)") from None
    if len(rows) < 2 or any(len(r) < 2 for r in rows):
        raise ValueError(f"{what} needs at least 2 x 2 sample points")
    nv = len(rows[0])
    if any(len(r) != nv for r in rows):
        raise ValueError(f"{what} rows must all have the same length")
    grid = []
    for i, row in enumerate(rows):
        pts = []
        for j, p in enumerate(row):
            try:
                x, y, z = p
            except (TypeError, ValueError):
                raise ValueError(f"{what}[{i}][{j}] must be a 3D point (x, y, z)") from None
            pts.append(
                (
                    finite(x, f"{what}[{i}][{j}][0]"),
                    finite(y, f"{what}[{i}][{j}][1]"),
                    finite(z, f"{what}[{i}][{j}][2]"),
                )
            )
        grid.append(tuple(pts))
    return tuple(grid)
