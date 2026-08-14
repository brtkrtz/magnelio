"""Trapezoidal companion models for lumped RLC elements.

A lumped 2-terminal element carries a current ``I`` and a voltage ``V``; a
time-stepping solver needs its constitutive relation as a per-step **Thévenin
companion** ``V^{n+1} = R_eq · I^{n+1} + V_hist`` (a constant equivalent
resistance plus a history EMF from the previous step).  This plugs directly
into the discrete-port update, which already solves
``i = (v_src − v_total) / (Σβ + Z0)`` — generalise ``Z0 → R_eq`` and fold the
history into the source (``v_src → v_src − V_hist``).  The discrete port is
then the special case :class:`SeriesRLC` with only ``R = Z0``.

Trapezoidal (bilinear) integration is used throughout: 2nd-order accurate
like the FIT leapfrog and **energy-conserving for L/C** — backward-Euler's
artificial damping would corrupt the high-Q resonance a lumped element
usually exists to model.

Derivations (element voltage/current with the trapezoidal rule)::

    inductor   V = L dI/dt   → V^{n+1} = (2L/dt)(I^{n+1} − I^n) − V^n
    capacitor  I = C dV/dt   → V^{n+1} = V^n + (dt/2C)(I^{n+1} + I^n)

so an inductor has ``R_eq = 2L/dt`` and a capacitor ``R_eq = dt/2C``; a
series string adds the ``R_eq`` and the history voltages, a parallel bundle
adds the Norton conductances/history currents and is converted back to a
Thévenin ``(R_eq, V_hist)``.
"""

# Design: DD-077 (trapezoidal companion models; integrator choice).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _validate_rlc(R, L, C, kind: str) -> None:
    if R is None and L is None and C is None:
        raise ValueError(f"{kind}: at least one of R, L, C must be given.")
    for name, val in (("R", R), ("L", L), ("C", C)):
        if val is not None and not (val > 0.0):
            raise ValueError(f"{kind}: {name} must be positive; got {val}.")


@dataclass
class SeriesRLC:
    """Series R–L–C companion (shared current), any subset present.

    Parameters
    ----------
    R, L, C : float, optional
        Resistance [Ω], inductance [H], capacitance [F].  ``None`` omits that
        element; at least one must be given.  ``SeriesRLC(R=Z0)`` reproduces a
        plain resistor (the discrete-port internal impedance).
    """

    R: Optional[float] = None
    L: Optional[float] = None
    C: Optional[float] = None

    _i: float = field(default=0.0, repr=False, init=False)  # I^n (shared)
    _vL: float = field(default=0.0, repr=False, init=False)  # V_L^n
    _vC: float = field(default=0.0, repr=False, init=False)  # V_C^n

    def __post_init__(self) -> None:
        _validate_rlc(self.R, self.L, self.C, "SeriesRLC")

    def r_eq(self, dt: float) -> float:
        """Equivalent series resistance [Ω] at time step *dt*."""
        r = self.R if self.R is not None else 0.0
        if self.L is not None:
            r += 2.0 * self.L / dt
        if self.C is not None:
            r += dt / (2.0 * self.C)
        return r

    def v_hist(self, dt: float) -> float:
        """History EMF [V]: the ``V_hist`` of ``V^{n+1} = R_eq·I^{n+1}+V_hist``."""
        vh = 0.0
        if self.L is not None:
            vh += -(2.0 * self.L / dt) * self._i - self._vL
        if self.C is not None:
            vh += self._vC + (dt / (2.0 * self.C)) * self._i
        return vh

    def advance(self, i: float, v: float, dt: float) -> None:
        """Advance the internal state to step ``n+1`` given the solved *i*.

        *v* (the total element voltage) is unused for a series string — the
        per-element voltages follow from the shared current.
        """
        del v
        if self.L is not None:
            self._vL = (2.0 * self.L / dt) * (i - self._i) - self._vL
        if self.C is not None:
            self._vC = self._vC + (dt / (2.0 * self.C)) * (i + self._i)
        self._i = i

    def reset(self) -> None:
        """Zero the internal state (reuse across excitations)."""
        self._i = self._vL = self._vC = 0.0

    def state_dict(self) -> dict:
        return {"i": float(self._i), "vL": float(self._vL), "vC": float(self._vC)}

    def load_state_dict(self, sd: dict) -> None:
        self._i = float(sd["i"])
        self._vL = float(sd["vL"])
        self._vC = float(sd["vC"])


@dataclass
class ParallelRLC:
    """Parallel R–L–C companion (shared voltage), any subset present.

    Parameters
    ----------
    R, L, C : float, optional
        Resistance [Ω], inductance [H], capacitance [F].  ``None`` omits that
        element; at least one must be given.
    """

    R: Optional[float] = None
    L: Optional[float] = None
    C: Optional[float] = None

    _v: float = field(default=0.0, repr=False, init=False)  # V^n (shared)
    _iL: float = field(default=0.0, repr=False, init=False)  # I_L^n
    _iC: float = field(default=0.0, repr=False, init=False)  # I_C^n

    def __post_init__(self) -> None:
        _validate_rlc(self.R, self.L, self.C, "ParallelRLC")

    def _g_eq(self, dt: float) -> float:
        g = 1.0 / self.R if self.R is not None else 0.0
        if self.L is not None:
            g += dt / (2.0 * self.L)
        if self.C is not None:
            g += 2.0 * self.C / dt
        return g

    def _i_hist(self, dt: float) -> float:
        ih = 0.0
        if self.L is not None:
            ih += self._iL + (dt / (2.0 * self.L)) * self._v
        if self.C is not None:
            ih += -(2.0 * self.C / dt) * self._v - self._iC
        return ih

    def r_eq(self, dt: float) -> float:
        """Equivalent resistance [Ω] = ``1 / G_eq`` at time step *dt*."""
        return 1.0 / self._g_eq(dt)

    def v_hist(self, dt: float) -> float:
        """History EMF [V] from the Norton→Thévenin conversion ``−I_hist/G_eq``."""
        return -self._i_hist(dt) / self._g_eq(dt)

    def advance(self, i: float, v: float, dt: float) -> None:
        """Advance the internal state given the solved element voltage *v*.

        *i* (the total element current) is unused for a parallel bundle — the
        per-element currents follow from the shared voltage.
        """
        del i
        if self.L is not None:
            self._iL = self._iL + (dt / (2.0 * self.L)) * (v + self._v)
        if self.C is not None:
            self._iC = (2.0 * self.C / dt) * (v - self._v) - self._iC
        self._v = v

    def reset(self) -> None:
        """Zero the internal state (reuse across excitations)."""
        self._v = self._iL = self._iC = 0.0

    def state_dict(self) -> dict:
        return {"v": float(self._v), "iL": float(self._iL), "iC": float(self._iC)}

    def load_state_dict(self, sd: dict) -> None:
        self._v = float(sd["v"])
        self._iL = float(sd["iL"])
        self._iC = float(sd["iC"])
