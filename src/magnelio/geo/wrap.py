"""Explicit chart mapping of planar components onto curved surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from magnelio.geo._sheet import Sheet
from magnelio.geo._topology_history import finish
from magnelio.geo.bend import Bend, _BentBase
from magnelio.geo.shape import Solid


@dataclass(frozen=True)
class Wrap(Bend):
    """Wrap a flat source component onto a target surface chart.

    Source points are measured from *origin* along orthonormal *along* and
    *across* directions, with their cross product as the layer direction.
    Coordinates ``(u, v, w)`` map to ``S(u, v) + w*n(u, v)`` over the whole
    source. The target's bounded parameter intervals correspond linearly to
    the declared source intervals. All members of a Group share the map.

    Parameters
    ----------
    target : Sheet
        One regular, hole-free target surface face.
    origin : sequence of float
        World origin of the flat source chart [m].
    along, across : str or sequence of float
        Perpendicular world directions of the source chart.
    u, v : pair of float
        Source intervals covered by the target surface [m].
    max_strain : float
        Required upper bound on sampled principal in-plane stretch or
        compression of the neutral surface.
    tolerance : float, optional
        Absolute sampled CAD approximation budget [m]. Defaults to one
        millionth of the source/target extent, subject to CAD resolution.

    Notes
    -----
    The mapping can change lengths and volume. The strain, fold and CAD fit
    checks are sampled, not global certificates. A source must lie completely
    within the declared chart. Periodic targets need one explicitly cut chart;
    crossing its seam is not inferred.
    """

    def _context(self, source, scale, *, finite=False):
        return super()._context(source, scale, finite=False)

    def __matmul__(self, other):
        """Wrap a Solid, Sheet or material-preserving Group independently."""
        from magnelio.geo.operations import Group

        if isinstance(other, Group):
            return Group(*(self @ member for member in other.shapes), name=other.name)
        if isinstance(other, Solid):
            return finish(_WrappedSolid(other, self))
        if isinstance(other, Sheet):
            return finish(_WrappedSheet(other, self))
        raise TypeError(f"Wrap accepts Solid, Sheet or Group; got {type(other).__name__}.")


class _WrappedBase(_BentBase):
    pass


class _WrappedSolid(_WrappedBase, Solid):
    pass


class _WrappedSheet(_WrappedBase, Sheet):
    pass
