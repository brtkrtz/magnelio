"""SourceCurrentPath — an impressed current along a curve.

A current filament: the current ``I(t)`` is *prescribed* on a path
through the grid, and the fields it radiates follow.  This is the
source behind a Hertzian dipole, a small loop standing in for a coil,
an injection probe on a cable harness, or a lightning channel — every
case where the current distribution is known and the wire itself need
not be resolved.

The path is any :class:`~magnelio.geo.Curve`, rasterised onto the
primary grid edges by the shared rasteriser
(:func:`magnelio.circuit.rasterize_curve`) that lumped elements and
voltage integration already use, so a filament and a voltage probe can
never disagree about which edges a curve occupies.

Discretisation
--------------
Ampère's law in the FIT form ``M_ε de/dt = C̃ᵀ ĥ − ĵ`` puts an
impressed current straight on the right-hand side: the edge state
``e`` is a *voltage* and ``β = dt / M_ε`` is capacitance-like, so the
per-step contribution of an edge current ``ĵ = I·sign`` is

    e ← e − sign · β · I(t^{n+1/2})

in volts, with no cell-size factor anywhere.  The current is
sampled half a step behind the E level being written, the time level
where ``C̃ᵀ ĥ`` stands.

Two properties follow from the discrete operators rather than from
anything this class does.  The staircase walk from the start node to
the end node is monotone, so ``Σ sign·dl·â`` is exactly the vector
between the snapped endpoints: the electric dipole moment of a short
filament is exact, and only higher multipoles see the staircase.  And
because the discrete divergence of a curl vanishes identically
(``S·C̃ᵀ = 0``), the update yields ``d/dt(S d̂) = −S ĵ`` — an open
filament accumulates exactly ``∓∫I dt`` on its two end nodes and
nothing anywhere else.  An open current path *is* a consistent
oscillating dipole; there is no charge to clean up.
"""

# Design: DD-227 (impressed current filament), DD-076 (the shared curve
# rasteriser), DD-085 (grid-quantity states: β·I is already a voltage),
# DD-224 (sources on the model, waveform on the excitation).

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from magnelio.signals.waveforms import Waveform
from magnelio.sources.base import _WaveformDriven


def _to_host(a):
    return a.get() if type(a).__module__.partition(".")[0] == "cupy" else np.asarray(a)


@dataclass
class SourceCurrentPath(_WaveformDriven):
    """An impressed current along a path through the model.

    Declared on the model with :meth:`~magnelio.GeometryModel.add_source`
    and driven by an :class:`~magnelio.Excitation` naming it: the
    excitation's ``amplitude`` is the peak current in amperes
    (:attr:`amplitude_unit`) and its ``waveform`` the time function.
    The current is prescribed, not solved for — the path is a source,
    not a wire.  A closed path is a current loop (a magnetic dipole);
    an open one accumulates the charge its ends demand and radiates as
    an electric dipole.

    Parameters
    ----------
    name : str
        Source name — the handle an :class:`~magnelio.Excitation` uses.
    path : Curve or sequence of points
        The filament.  Either a :class:`~magnelio.geo.Curve` (a
        polyline, arc, spline, helix, or any chain of them) or a
        sequence of at least two ``(x, y, z)`` points [m], which is
        taken as the vertices of a polyline.  The current flows from
        the first point towards the last.  The path must lie inside
        the meshed domain; under a symmetry declaration that is the
        meshed half, and the images on the other side are supplied by
        the symmetry wall.
    samples_per_cell : int, default 4
        Curve samples per smallest cell while rasterising.  Higher
        values only refine which edges a strongly curved path picks up;
        the dipole moment does not depend on it.

    Notes
    -----
    A path may cross itself or double back; every edge is impressed
    once with the net signed current it carries, so a segment traversed
    in both directions cancels, as it physically does.  Edges the
    solver holds at zero — inside a perfect conductor, or tangential to
    a PEC wall — cannot take a current, and the source reports how many
    of its edges were swallowed that way rather than silently radiating
    less than asked.

    Examples
    --------
    A short vertical filament at the origin — a Hertzian dipole of
    1 mA peak:

    >>> src = SourceCurrentPath(name="dip", path=[(0, 0, -1e-3), (0, 0, 1e-3)])
    >>> model.add_source(src)                                  # doctest: +SKIP
    >>> exc = magnelio.Excitation("dip", waveform=wf, amplitude=1e-3)  # doctest: +SKIP

    A one-turn loop of radius 5 mm in the xy-plane — a magnetic dipole:

    >>> from magnelio.geo import Curve
    >>> loop = Curve.arc((5e-3, 0, 0), (-5e-3, 0, 0), (0, -5e-3, 0)).joined(
    ...     Curve.arc((0, -5e-3, 0), (5e-3, 0, 0), (0, 5e-3, 0))
    ... )                                                       # doctest: +SKIP
    >>> src = SourceCurrentPath(name="loop", path=loop)          # doctest: +SKIP
    """

    name: str
    path: object = None
    samples_per_cell: int = 4

    amplitude_unit = "A"

    # --- excitation binding (_WaveformDriven) ---
    _waveform: Waveform | None = dc_field(default=None, repr=False, init=False)
    _amplitude: float = dc_field(default=1.0, repr=False, init=False)
    _delay: float = dc_field(default=0.0, repr=False, init=False)
    # --- solver state (attach) ---
    _curve: object = dc_field(default=None, repr=False, init=False)
    _points: tuple | None = dc_field(default=None, repr=False, init=False)
    _dt: float = dc_field(default=0.0, repr=False, init=False)
    _xp: object = dc_field(default=None, repr=False, init=False)
    _scalar: object = dc_field(default=None, repr=False, init=False)
    # Unique edges and their folded coefficients ``−sign·β``.
    _idx: object = dc_field(default=None, repr=False, init=False)
    _coef: object = dc_field(default=None, repr=False, init=False)
    _n_dead: int = dc_field(default=0, repr=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"source name must be a non-empty string; got {self.name!r}")
        if int(self.samples_per_cell) < 2:
            raise ValueError(
                f"source {self.name!r}: samples_per_cell must be >= 2 so the "
                f"rasteriser cannot skip a grid node; got {self.samples_per_cell!r}",
            )
        self.samples_per_cell = int(self.samples_per_cell)
        self._resolve_path()

    def _resolve_path(self) -> None:
        """Normalise ``path`` into a Curve, keeping given points verbatim."""
        from magnelio.geo import Curve  # noqa: PLC0415

        if self.path is None:
            raise TypeError(
                f"source {self.name!r}: path= is required — a magnelio.geo.Curve "
                f"or a sequence of at least two (x, y, z) points [m]",
            )
        if isinstance(self.path, Curve):
            self._curve, self._points = self.path, None
            return
        try:
            pts = tuple(tuple(float(c) for c in p) for p in self.path)
        except (TypeError, ValueError):
            raise TypeError(
                f"source {self.name!r}: path= must be a magnelio.geo.Curve or a "
                f"sequence of (x, y, z) points [m]; got {type(self.path).__name__}",
            ) from None
        if len(pts) < 2 or any(len(p) != 3 for p in pts):
            raise ValueError(
                f"source {self.name!r}: a point path needs at least two "
                f"(x, y, z) triples [m]; got {len(pts)} point(s)",
            )
        self._points = pts
        self._curve = Curve.polyline(pts)

    @property
    def curve(self):
        """The path as a :class:`~magnelio.geo.Curve`."""
        return self._curve

    # ── solver hooks ─────────────────────────────────────────────────────

    def attach(self, solver) -> None:
        """Rasterise the path and fold ``−sign·β`` into one coefficient per edge."""
        from magnelio.circuit.rasterize import rasterize_curve  # noqa: PLC0415

        grid = solver.mesh.grid
        self._dt = solver.dt
        self._xp = solver._xp
        self._scalar = np.dtype(solver._real_dtype).type

        self._check_inside(grid)
        path = rasterize_curve(self._curve, grid, samples_per_cell=self.samples_per_cell)

        # One entry per *distinct* edge, carrying the net signed current
        # it sees.  Fancy-index ``+=`` does not accumulate over repeated
        # indices, so a self-crossing path has to be folded here — and
        # folding is also what makes a doubled-back segment cancel.
        flat = np.asarray(path.flat_indices, dtype=np.int64)
        sign = np.asarray(path.signs, dtype=np.float64)
        uniq, inverse = np.unique(flat, return_inverse=True)
        net = np.bincount(inverse, weights=sign, minlength=uniq.size)
        keep = net != 0.0
        uniq, net = uniq[keep], net[keep]

        beta = _to_host(solver._beta_E)[uniq]
        self._n_dead = int(np.count_nonzero(beta == 0.0))
        if self._n_dead:
            warnings.warn(
                f"source {self.name!r}: {self._n_dead} of {uniq.size} path edges are "
                f"held at zero by the solver (inside a perfect conductor, or "
                f"tangential to a PEC wall) and carry no impressed current. "
                f"Move the path off the conductor, or drive the conductor with a "
                f"port instead.",
                UserWarning,
                stacklevel=2,
            )

        xp = self._xp
        self._idx = xp.asarray(uniq)
        self._coef = xp.asarray(-net * beta, dtype=solver._real_dtype)

    def _check_inside(self, grid) -> None:
        """Refuse a path that leaves the meshed domain."""
        (lo, hi) = self._curve.bounding_box()
        nodes = (np.asarray(grid.x), np.asarray(grid.y), np.asarray(grid.z))
        tol = 0.5 * min(grid.dx_min, grid.dy_min, grid.dz_min)
        for axis, n in enumerate(nodes):
            if lo[axis] < n[0] - tol or hi[axis] > n[-1] + tol:
                raise ValueError(
                    f"source {self.name!r}: the path leaves the meshed domain along "
                    f"{'xyz'[axis]} (path {lo[axis]:.6g} … {hi[axis]:.6g} m, grid "
                    f"{n[0]:.6g} … {n[-1]:.6g} m).  Under a symmetry declaration the "
                    f"mesh is the kept half — give the path in that half and let the "
                    f"symmetry wall supply its image.",
                )

    def inject_E(self, fields, t_E: float) -> None:
        """Add ``−sign·β·I`` on every path edge after the E update.

        The current is taken at ``t_E − dt/2``, the level of the
        ``C̃ᵀ ĥ`` term it stands beside.
        """
        if self._idx is None:
            return
        current = self._scalar(self._drive(t_E - self._dt / 2))
        fields.e_flat[self._idx] += self._coef * current

    def inject_H(self, fields, t_H: float) -> None:
        """Nothing — an electric current enters the E update only."""

    def __repr__(self) -> str:
        where = (
            f"{len(self._points)} points" if self._points is not None else f"curve={self._curve!r}"
        )
        return f"SourceCurrentPath(name={self.name!r}, path={where})"


__all__ = ["SourceCurrentPath"]
