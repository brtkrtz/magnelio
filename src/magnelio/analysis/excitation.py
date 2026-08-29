"""Excitation — one port or source bound to a waveform and a weight.

Run vocabulary (core namespace), like :class:`~magnelio.BoundaryConditions`:
an :class:`Excitation` names *what* is driven (a port channel or a
model source, by name), *with what* (a :class:`~magnelio.signals.Waveform`)
and *how much* (amplitude in the source's natural unit, a delay and,
on carrier waveforms, a phase).  A time-domain run driven by a list of
excitations applies them simultaneously.
"""

# Design: DD-224 (the excitation triad Source / Waveform / Excitation).

from __future__ import annotations

import math
from dataclasses import dataclass

from magnelio.signals.waveforms import Waveform


@dataclass(frozen=True)
class Excitation:
    """One port channel or model source, bound to a waveform and a weight.

    Parameters
    ----------
    source : str
        Name of the port or source to drive — a port declared with
        :meth:`~magnelio.GeometryModel.add_port` or a source declared
        with :meth:`~magnelio.GeometryModel.add_source`.
    mode : int, default 0
        Mode index on a port; ignored by sources.
    waveform : Waveform, optional
        The time function.  ``None`` (default) lets the run derive it:
        a Gaussian pulse over the analysis band, band-limited above a
        port mode's cut-off frequency.
    amplitude : float, default 1.0
        Peak amplitude in the natural unit of the source — ``sqrt(W)``
        (incident power wave) for ports, ``V/m`` for a plane wave; each
        source publishes it as ``amplitude_unit``.
    delay : float, default 0.0
        Time offset [s] of the waveform; must not be negative.
    phase : float, default 0.0
        Phase [degrees].  On a carrier waveform (one with ``f_center``)
        it is applied as a delay of ``phase / (360 · f_center)``; on a
        baseband waveform it is rejected, because a baseband pulse has
        no phase.  Two modes of one port at 90° make a circularly
        polarised feed.

    Notes
    -----
    Bare names and ``(name, mode)`` pairs are accepted wherever a list
    of excitations is: ``"port1"`` means ``Excitation("port1")`` and
    ``("port1", 1)`` means ``Excitation("port1", mode=1)``.

    Examples
    --------
    >>> from magnelio import Excitation, signals
    >>> exc = Excitation("port1", waveform=signals.WaveformGaussianModulated(8e9, 12e9))
    >>> exc.source, exc.mode, exc.waveform.f_center
    ('port1', 0, 10000000000.0)
    """

    source: str
    mode: int = 0
    waveform: Waveform | None = None
    amplitude: float = 1.0
    delay: float = 0.0
    phase: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise TypeError(
                f"Excitation.source must be the name of a port or source (a non-empty "
                f"string); got {self.source!r}",
            )
        if isinstance(self.mode, bool) or not isinstance(self.mode, int) or self.mode < 0:
            raise ValueError(f"Excitation.mode must be a non-negative integer; got {self.mode!r}")
        if self.waveform is not None and not isinstance(self.waveform, Waveform):
            raise TypeError(
                f"Excitation.waveform must be a magnelio.signals.Waveform (or None); "
                f"got {type(self.waveform).__name__}",
            )
        amplitude = float(self.amplitude)
        if not math.isfinite(amplitude):
            raise ValueError(f"Excitation.amplitude must be finite; got {self.amplitude!r}")
        delay = float(self.delay)
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError(
                f"Excitation.delay must be a non-negative finite time [s]; got {self.delay!r}",
            )
        phase = float(self.phase)
        if not math.isfinite(phase):
            raise ValueError(f"Excitation.phase must be finite [degrees]; got {self.phase!r}")
        if phase != 0.0 and self.waveform is not None and self.waveform.f_center is None:
            raise ValueError(
                f"Excitation({self.source!r}): phase = {phase:g}° needs a carrier "
                f"waveform (one with f_center), got {type(self.waveform).__name__}; "
                f"use delay= to shift a baseband pulse in time",
            )
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "delay", delay)
        object.__setattr__(self, "phase", phase)

    @classmethod
    def coerce(cls, spec) -> "Excitation":
        """Turn a shorthand into an :class:`Excitation`.

        ``"port1"`` → ``Excitation("port1")``; ``("port1", 1)`` →
        ``Excitation("port1", mode=1)``; an :class:`Excitation` is
        returned as is.
        """
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, str):
            return cls(spec)
        if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
            return cls(spec[0], mode=int(spec[1]))
        raise TypeError(
            f"an excitation is an Excitation, a port/source name or a (name, mode) "
            f"pair; got {spec!r}",
        )

    def effective_delay(self) -> float:
        """The delay [s] including the phase, ``delay + phase / (360 · f_center)``.

        Raises when a phase is set without a waveform to take it from.
        """
        if self.phase == 0.0:
            return self.delay
        if self.waveform is None or self.waveform.f_center is None:
            raise ValueError(
                f"Excitation({self.source!r}): phase = {self.phase:g}° needs a carrier "
                f"waveform (one with f_center) to resolve to a delay",
            )
        return self.delay + self.phase / (360.0 * self.waveform.f_center)


__all__ = ["Excitation"]
