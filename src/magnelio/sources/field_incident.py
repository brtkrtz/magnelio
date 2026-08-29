"""SourceFieldIncident — an incident field injected on a total-field/scattered-field box.

The domain is split by a virtual box: inside it the fields are the
*total* field (incident plus scattered), outside only the *scattered*
field remains.  Consistency corrections on the six box faces inject
the incident wave into the total-field region after every E and H
update.  This module holds what every incident-field source shares —
the box, its snapping to the grid, and the binding of the waveform an
excitation supplies; the concrete incident field (a plane wave, …) is
the subclass's business.
"""

# Design: DD-013 (TF/SF), DD-153 (corners vocabulary), DD-224 (sources
# on the model, waveform on the excitation).

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np

from magnelio.signals.waveforms import Waveform
from magnelio.sources.base import Source


@dataclass
class SourceFieldIncident(Source):
    """Incident field on a total-field/scattered-field box (abstract).

    Parameters
    ----------
    name : str
        Source name — the handle an :class:`~magnelio.Excitation` uses.
    corners : tuple of tuple, optional
        Two opposite corners ``((x0, y0, z0), (x1, y1, z1))`` of the
        total-field region [m] — the same form as
        :meth:`~magnelio.geo.Brick.from_corners`.  Corner order does
        not matter, and a component may be ``None`` (or ``±math.inf``)
        to fall back to the default extent on that side (two bulk
        cells inside the domain boundary — the scattered-field shell
        the TF/SF split needs).  Snapped to the nearest grid nodes.
        ``None`` (default) uses the default extent on all six sides.

    Notes
    -----
    The amplitude of the incident field is the excitation's
    ``amplitude`` in ``V/m`` (:attr:`amplitude_unit`); the waveform is
    the excitation's ``waveform``.  Neither is part of the source.
    """

    name: str
    corners: tuple[tuple, tuple] | None = None

    amplitude_unit = "V/m"

    # --- excitation binding (set_excitation) ---
    _waveform: Waveform | None = dc_field(default=None, repr=False, init=False)
    _amplitude: float = dc_field(default=1.0, repr=False, init=False)
    _delay: float = dc_field(default=0.0, repr=False, init=False)
    # --- solver state (attach) ---
    _attached: bool = dc_field(default=False, repr=False, init=False)
    _Nx: int = dc_field(default=0, repr=False, init=False)
    _Ny: int = dc_field(default=0, repr=False, init=False)
    _Nz: int = dc_field(default=0, repr=False, init=False)
    # TF/SF box node indices: (ix0, ix1, iy0, iy1, iz0, iz1)
    _box: tuple[int, ...] | None = dc_field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"source name must be a non-empty string; got {self.name!r}")

    # ── construction helpers ─────────────────────────────────────────────

    @classmethod
    def from_ranges(
        cls,
        *,
        x1=None,
        x2=None,
        dx=None,
        y1=None,
        y2=None,
        dy=None,
        z1=None,
        z2=None,
        dz=None,
        **kwargs,
    ):
        """Build the same source from one coordinate range per axis.

        The range spelling of ``corners=``, as in
        :meth:`~magnelio.geo.Brick.from_ranges`: each axis takes up to
        two of its three keywords — the two bounds (``x1``, ``x2``) or
        a bound and an extent (``x1``, ``dx`` / ``x2``, ``dx``).  Here
        an axis may also be open: give nothing for the whole domain
        extent, or a single bound to reach the domain boundary on the
        other side.  All remaining keyword arguments are forwarded to
        the constructor.

        Examples
        --------
        >>> src = SourcePlaneWave.from_ranges(name="pw", x1=1e-3, x2=9e-3, z1=1e-3, z2=19e-3)
        """
        from magnelio.geo._ranges import corners_from_ranges  # noqa: PLC0415

        return cls(
            corners=corners_from_ranges(x1, x2, dx, y1, y2, dy, z1, z2, dz),
            **kwargs,
        )

    # ── excitation binding ───────────────────────────────────────────────

    def set_excitation(
        self,
        waveform: Waveform,
        *,
        amplitude: float = 1.0,
        delay: float = 0.0,
    ) -> None:
        """Bind the waveform the source injects, with its weight.

        Parameters
        ----------
        waveform : Waveform
            Unit-peak time function of the incident field.
        amplitude : float, default 1.0
            Peak incident field [V/m].
        delay : float, default 0.0
            Time offset [s] of the waveform.
        """
        if not isinstance(waveform, Waveform):
            raise TypeError(
                f"{type(self).__name__}.set_excitation takes a magnelio.signals.Waveform; "
                f"got {type(waveform).__name__}",
            )
        amplitude = float(amplitude)
        delay = float(delay)
        if not math.isfinite(amplitude):
            raise ValueError(f"amplitude must be finite; got {amplitude!r}")
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError(f"delay must be a non-negative finite time [s]; got {delay!r}")
        self._waveform = waveform
        self._amplitude = amplitude
        self._delay = delay

    def clear_excitation(self) -> None:
        self._waveform = None
        self._amplitude = 1.0
        self._delay = 0.0

    @property
    def waveform(self) -> Waveform | None:
        """The bound waveform, or ``None`` before :meth:`set_excitation`."""
        return self._waveform

    def _require_waveform(self) -> Waveform:
        if self._waveform is None:
            raise ValueError(
                f"source {self.name!r} has no waveform: bind one with "
                f"set_excitation(waveform, amplitude=..., delay=...) before the run",
            )
        return self._waveform

    def _drive(self, t):
        """Incident amplitude ``A · w(t − delay)`` at time(s) *t* [s]."""
        w = self._require_waveform()
        return self._amplitude * w(t - self._delay)

    # ── TF/SF box ────────────────────────────────────────────────────────

    def _snap_box(self, grid) -> tuple[int, int, int, int, int, int]:
        """Snap the TF-region corners to nearest grid node indices.

        Returns (ix0, ix1, iy0, iy1, iz0, iz1) such that the TF region
        spans cells [ix0, ix1) × [iy0, iy1) × [iz0, iz1) in node indexing.

        The corners are normalised per axis (order does not matter, the
        shared ``corners=`` contract), and ``None``/``±inf`` components
        fall back to the default extent on that side.  A box that leaves
        no scattered-field shell is clamped inward — that is a property
        of the TF/SF split, not a silent reinterpretation of the input.
        """
        if self.corners is None:
            # Default: 2 cells inset from each face
            return (2, self._Nx - 1, 2, self._Ny - 1, 2, self._Nz - 1)

        p, q = self.corners
        lo, hi = [], []
        for a, b, default_lo, default_hi in zip(
            p, q, (2, 2, 2), (self._Nx - 1, self._Ny - 1, self._Nz - 1)
        ):
            aa = None if a is None or not np.isfinite(a) else float(a)
            bb = None if b is None or not np.isfinite(b) else float(b)
            if aa is not None and bb is not None and bb < aa:
                aa, bb = bb, aa
            lo.append((aa, default_lo))
            hi.append((bb, default_hi))

        x, y, z = np.asarray(grid.x), np.asarray(grid.y), np.asarray(grid.z)
        n_hi = (self._Nx, self._Ny, self._Nz)
        idx = []
        for axis, (nodes, n) in enumerate(zip((x, y, z), n_hi)):
            a, default_lo = lo[axis]
            b, default_hi = hi[axis]
            i0 = default_lo if a is None else int(np.searchsorted(nodes, a).clip(1, n - 1))
            i1 = default_hi if b is None else int(np.searchsorted(nodes, b).clip(i0 + 1, n))
            idx += [i0, i1]

        return tuple(idx)


__all__ = ["SourceFieldIncident"]
