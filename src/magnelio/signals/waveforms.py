"""Waveform functions for port excitation.

Consolidated from the identical implementations in Port2D._v_src and
DiscretePort._v_src.  All waveforms return scalar or array values with
peak amplitude = 1.

Convention:
    bandwidth = f_max - f_min   (or f_max when f_min = 0)
    sigma     = 2 / (pi * bandwidth)
    t0        = 4 / bandwidth

Choosing bandwidth from the actual passband [f_min, f_max] keeps the
Gaussian envelope spectrally tight: a Coax → rectangular-waveguide
junction with WR-90 cutoff (6.56 GHz) and excitation [8.2, 12.4] GHz
stays comfortably above the cutoff, while the legacy [0, f_max]
formulation leaks ~50 % of the pulse energy below the cutoff →
total reflection → slow Mur-ABC ringing.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def gaussian(t: float | np.ndarray, f_max: float) -> float | np.ndarray:
    """Plain Gaussian pulse, peak = 1 at t0 = 4/f_max.

    Suitable for TEM modes (DC-inclusive).

    Parameters
    ----------
    t : float or np.ndarray
        Time [s].
    f_max : float
        Bandwidth [Hz].
    """
    sigma = 2.0 / (math.pi * f_max)
    t0 = 4.0 / f_max
    x = (t - t0) / sigma
    if isinstance(t, np.ndarray):
        return np.exp(-x * x)
    return math.exp(-x * x)


def modulated_gaussian(
    t: float | np.ndarray,
    f_max: float,
    f_min: float,
) -> float | np.ndarray:
    """Gaussian envelope modulated at the band centre (f_min + f_max) / 2.

    The envelope sigma scales with the passband bandwidth ``f_max -
    f_min`` rather than ``f_max`` alone, so the spectrum is tightly
    confined to [f_min, f_max] and almost nothing leaks below f_min.
    This is what one wants when the lower edge is constrained — by a
    waveguide cut-off frequency or by an explicit user-specified band.

    Parameters
    ----------
    t : float or np.ndarray
        Time [s].
    f_max : float
        Upper passband edge [Hz].
    f_min : float
        Lower passband edge [Hz].  Either an explicit user value or
        the mode's cut-off frequency.
    """
    bandwidth = f_max - f_min
    sigma = 2.0 / (math.pi * bandwidth)
    t0 = 4.0 / bandwidth
    x = (t - t0) / sigma
    f_center = 0.5 * (f_min + f_max)

    if isinstance(t, np.ndarray):
        env = np.exp(-x * x)
        return env * np.cos(2.0 * math.pi * f_center * (t - t0))
    env = math.exp(-x * x)
    return env * math.cos(2.0 * math.pi * f_center * (t - t0))


def waveform_for_mode(
    f_max: float,
    omega_c: float = 0.0,
    f_min: float = 0.0,
) -> Callable[[float], float]:
    """Factory: select Gaussian or modulated Gaussian based on band edges.

    Picks a modulated Gaussian whenever a positive lower edge exists —
    either implicitly via the mode cut-off (``omega_c > 0``, e.g. TE/TM
    waveguide modes) or explicitly via ``f_min > 0`` (caller-specified
    bandpass even for TEM modes).  The carrier sits at the midpoint
    of ``[max(f_cutoff, f_min), f_max]``.  Falls back to a plain
    DC-inclusive Gaussian when both edges are zero.

    Parameters
    ----------
    f_max : float
        Upper bandwidth [Hz].
    omega_c : float, default 0.0
        Angular cut-off frequency [rad/s] of the mode (TE/TM > 0,
        TEM = 0).
    f_min : float, default 0.0
        Caller-specified lower band edge [Hz].  Set this when you want a
        bandpass excitation on a TEM mode (e.g. WR-90 measurement band).

    Returns
    -------
    Callable[[float], float]
        Waveform function ``v(t) -> float``.
    """
    eff_f_min = max(omega_c / (2.0 * math.pi), f_min)
    if eff_f_min <= 0.0:

        def _waveform(t: float) -> float:
            return float(gaussian(t, f_max))

        return _waveform

    def _waveform_mod(t: float) -> float:
        return float(modulated_gaussian(t, f_max, eff_f_min))

    return _waveform_mod
