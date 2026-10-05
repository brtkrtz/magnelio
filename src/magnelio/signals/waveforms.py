"""Excitation waveforms — pure time functions with a bandwidth.

A :class:`Waveform` is what an :class:`~magnelio.Excitation` binds to a
port or source: a unit-peak function of time that also knows its
spectral occupancy (``f_min``, ``f_max``, ``f_center``) and its
duration (``t_end``, infinite for continuous-wave forms).  The
amplitude, delay and phase of a drive are *not* part of the waveform —
they live on the excitation, so one waveform object can drive several
ports or sources.

Convention of the Gaussian family (unit peak):

    envelope  = exp(-((t - peak_time) / tau)**2)
    peak_time = 4.5 * tau
    t_end     = 9 * tau

The width follows the requested upper-edge spectral attenuation,
25 dB by default, relative to DC or the carrier-centre spectrum.
The modulated pulse includes both mirrored spectral lobes in this
definition. Its lower edge can be more strongly excited near DC.
Choosing the actual passband reduces unwanted below-cutoff content;
Gaussian pulses have spectral tails rather than a strict band limit.

The module-level functions ``gaussian``, ``modulated_gaussian`` and
``waveform_for_mode`` are internal default-pulse helpers. Production
drives retain waveform objects with cached width and timing; the
helpers are not part of the public surface.
"""

# Design: DD-224 (the excitation triad), DD-281 (Gaussian spectral edges).

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from magnelio.signals.signal_1d import Signal1D

# ─────────────────────────────────────────────────────────────────────
# Closed forms (internal)
# ─────────────────────────────────────────────────────────────────────


def _is_array(t) -> bool:
    """True for a (NumPy or CuPy) array of one or more dimensions.

    The waveforms are evaluated on whatever array module the solver
    runs — a TF/SF face's retardation array lives on the device — so
    the array paths use NumPy ufuncs, which dispatch to the array's
    own module, and never ``np.asarray`` (a device array refuses the
    implicit copy).
    """
    return getattr(t, "ndim", 0) > 0


def _gaussian_value(t, tau: float, peak_time: float, f_center: float | None = None):
    """Evaluate cached pulse parameters, preserving NumPy/CuPy dispatch."""
    x = (t - peak_time) / tau
    if _is_array(t):
        value = np.exp(-x * x)
        return (
            value
            if f_center is None
            else value * np.cos(2.0 * math.pi * f_center * (t - peak_time))
        )
    value = math.exp(-x * x)
    return (
        value if f_center is None else value * math.cos(2.0 * math.pi * f_center * (t - peak_time))
    )


def _modulated_edge_log(k: float, m: float) -> float:
    """Log attenuation of the full real-pulse spectrum at its upper edge.

    k = pi*tau*bandwidth/2, m = (f_max+f_min)/bandwidth.
    expm1 avoids cancellation at small requested attenuation.
    """
    square = k * k
    mirror = math.exp(-4.0 * m * m * square)
    correction = mirror / (1.0 + mirror) * math.expm1(-4.0 * m * square)
    return square - math.log1p(correction)


def _modulated_tau(f_min: float, f_max: float, attenuation_db: float) -> float:
    width = f_max - f_min
    m = f_max / width + f_min / width
    log_attenuation = attenuation_db * (math.log(10.0) / 20.0)
    scale = math.sqrt(log_attenuation)
    # Solve once on construction; normalised roots also cover tiny dB values.
    root = brentq(
        lambda u: _modulated_edge_log(u * scale, m) / log_attenuation - 1.0,
        0.0,
        1.01,
        xtol=1e-14,
        rtol=1e-14,
    )
    return 2.0 * scale * root / (math.pi * width)


def gaussian(t: float | np.ndarray, f_max: float) -> float | np.ndarray:
    """Evaluate the default DC-inclusive Gaussian pulse."""
    return WaveformGaussian(f_max)(t)


def modulated_gaussian(
    t: float | np.ndarray,
    f_max: float,
    f_min: float,
) -> float | np.ndarray:
    """Evaluate the default Gaussian pulse on the band-centre carrier."""
    return WaveformGaussianModulated(f_min, f_max)(t)


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
        return WaveformGaussian(f_max)
    return WaveformGaussianModulated(eff_f_min, f_max)


# ─────────────────────────────────────────────────────────────────────
# Waveform classes
# ─────────────────────────────────────────────────────────────────────


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number; got {value!r}")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a non-negative finite number; got {value!r}")
    return value


class Waveform(ABC):
    """Excitation waveform: a unit-peak time function with a bandwidth.

    Every waveform is callable — ``w(t)`` for a float or an array of
    times [s] — and describes its own spectral occupancy and duration:

    ``f_max``
        Upper band edge [Hz]; sizes the time step and the run-length
        estimate.
    ``f_min``
        Lower band edge [Hz]; ``0`` for baseband forms.
    ``f_center``
        Carrier frequency [Hz], or ``None`` for baseband forms.  An
        :class:`~magnelio.Excitation` may carry a ``phase`` only on a
        waveform with a carrier.
    ``t_end``
        Time [s] after which the waveform is (effectively) zero;
        ``inf`` for continuous-wave forms, which need an explicit run
        duration.

    The amplitude, delay and phase of a drive live on the
    :class:`~magnelio.Excitation`, not here, so a single waveform can
    drive several ports and sources.

    See Also
    --------
    WaveformGaussian, WaveformGaussianModulated, WaveformSine,
    WaveformStep, WaveformTable, WaveformFunction
    """

    @abstractmethod
    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        """Waveform value at time *t* [s] (float in, float out; array in, array out)."""

    def sample(self, dt: float, n: int, label: str = "") -> Signal1D:
        """Sample the waveform on ``t = arange(n) · dt``.

        Parameters
        ----------
        dt : float
            Time step [s].
        n : int
            Number of samples.
        label : str, optional
            Label of the returned signal.

        Returns
        -------
        Signal1D
            The sampled waveform.
        """
        dt = _positive("dt", dt)
        n = int(n)
        if n < 1:
            raise ValueError(f"n must be >= 1; got {n}")
        t = np.arange(n) * dt
        return Signal1D(t=t, values=np.asarray(self(t), dtype=float), dt=dt, label=label)

    def spectrum(self, f: np.ndarray) -> np.ndarray:
        """Continuous-time spectrum ``∫ w(t) e^{-2πj f t} dt`` at frequencies *f* [Hz].

        The same sign convention as :meth:`Signal1D.at_frequencies`.
        Closed-form where the waveform has one; otherwise the waveform
        is sampled to ``t_end`` at twenty points per ``1/f_max`` and
        integrated numerically.  Continuous-wave forms (``t_end =
        inf``) have no finite-energy spectrum and raise.
        """
        f = np.asarray(f, dtype=float)
        if not math.isfinite(self.t_end):
            raise ValueError(
                f"{type(self).__name__} is a continuous-wave form (t_end = inf) "
                f"and has no finite-energy spectrum",
            )
        dt = 1.0 / (20.0 * self.f_max)
        n = max(int(math.ceil(self.t_end / dt)) + 1, 2)
        sig = self.sample(dt, n)
        return sig.at_frequencies(f) * dt

    def plot(self, ax=None, *, n: int = 2000, t_max: float | None = None, **kwargs):
        """Plot the waveform against time.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure otherwise.
        n : int, default 2000
            Number of samples.
        t_max : float, optional
            End of the time axis [s].  Default: ``t_end`` for finite
            forms, ten carrier periods (or ten rise times) for
            continuous-wave forms.
        **kwargs
            Forwarded to ``ax.plot``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        if t_max is None:
            t_max = self.t_end if math.isfinite(self.t_end) else 10.0 / self.f_max
        t = np.linspace(0.0, float(t_max), int(n))
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(t * 1e9, self(t), **kwargs)
        ax.set_xlabel("t [ns]")
        ax.set_ylabel("w(t)")
        ax.set_title(repr(self))
        return ax


@dataclass(frozen=True)
class WaveformGaussian(Waveform):
    """Baseband Gaussian pulse (DC-inclusive), with a specified spectral edge.

    The pulse for TEM and lumped ports and for any source that may
    carry DC. The envelope is ``exp(-((t-peak_time)/tau)**2)``;
    its peak is at ``4.5*tau`` and its effective duration at ``9*tau``.

    Parameters
    ----------
    f_max : float
        Upper band edge [Hz].
    edge_attenuation_db : float, default 25.0
        Positive spectral amplitude attenuation at ``f_max`` relative
        to DC: ``|G(f_max)| / |G(0)| = 10**(-edge_attenuation_db/20)``.
        Must be finite and strictly positive. Increasing it widens the
        pulse in time and reduces excitation at the band edge.

    Notes
    -----
    Waveforms restored from older projects retain their original width
    and peak time. Read ``peak_time`` and ``t_end`` for their timing.

    Examples
    --------
    >>> from magnelio import signals
    >>> w = signals.WaveformGaussian(f_max=10e9)
    >>> w(w.peak_time)
    1.0
    """

    f_max: float
    edge_attenuation_db: float = dc_field(default=25.0, kw_only=True)
    _tau: float = dc_field(init=False, repr=False)
    _peak_time: float = dc_field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "f_max", _positive("f_max", self.f_max))
        attenuation = _positive("edge_attenuation_db", self.edge_attenuation_db)
        object.__setattr__(self, "edge_attenuation_db", attenuation)
        tau = math.sqrt(attenuation * (math.log(10.0) / 20.0)) / (math.pi * self.f_max)
        object.__setattr__(self, "_tau", _positive("tau", tau))
        object.__setattr__(self, "_peak_time", _positive("peak_time", 4.5 * tau))

    @property
    def tau(self) -> float:
        """Envelope time constant [s]; statistical standard deviation is tau/sqrt(2)."""
        return self._tau

    @property
    def peak_time(self) -> float:
        """Time of the unit peak [s]."""
        return self._peak_time

    @property
    def f_min(self) -> float:
        return 0.0

    @property
    def f_center(self) -> None:
        return None

    @property
    def t_end(self) -> float:
        """Effective duration [s]; the default endpoint envelope is about 1.6e-9."""
        return 2.0 * self.peak_time

    def __call__(self, t):
        return _gaussian_value(t, self.tau, self.peak_time)

    def spectrum(self, f):
        f = np.asarray(f, dtype=float)
        env = self.tau * math.sqrt(math.pi) * np.exp(-((math.pi * self.tau * f) ** 2))
        return env * np.exp(-2j * math.pi * f * self.peak_time)


@dataclass(frozen=True)
class WaveformGaussianModulated(Waveform):
    """Gaussian envelope on a carrier at the band center, unit peak.

    The pulse for TE/TM modes and any drive whose lower band edge
    matters. The envelope is ``exp(-((t-peak_time)/tau)**2)``;
    its peak is at ``4.5*tau`` and its effective duration at ``9*tau``.
    The carrier sits at ``(f_min + f_max) / 2``. Gaussian spectral tails
    extend beyond the requested band, including below ``f_min``.

    Parameters
    ----------
    f_min : float
        Lower band edge [Hz].
    f_max : float
        Upper band edge [Hz]; must exceed ``f_min``.
    edge_attenuation_db : float, default 25.0
        Positive spectral amplitude attenuation at ``f_max`` relative
        to the carrier-centre spectrum. Includes both mirrored spectral
        lobes; the lower edge is more strongly excited near DC. Must be
        finite and strictly positive.

    Notes
    -----
    Waveforms restored from older projects retain their original width
    and peak time. Read ``peak_time`` and ``t_end`` for their timing.

    Examples
    --------
    >>> from magnelio import signals
    >>> w = signals.WaveformGaussianModulated(f_min=8.2e9, f_max=12.4e9)
    >>> w.f_center
    10300000000.0
    """

    f_min: float
    f_max: float
    edge_attenuation_db: float = dc_field(default=25.0, kw_only=True)
    _tau: float = dc_field(init=False, repr=False)
    _peak_time: float = dc_field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "f_min", _nonnegative("f_min", self.f_min))
        object.__setattr__(self, "f_max", _positive("f_max", self.f_max))
        if self.f_max <= self.f_min:
            raise ValueError(
                f"f_max ({self.f_max:.4g} Hz) must exceed f_min ({self.f_min:.4g} Hz)",
            )
        attenuation = _positive("edge_attenuation_db", self.edge_attenuation_db)
        object.__setattr__(self, "edge_attenuation_db", attenuation)
        tau = _modulated_tau(self.f_min, self.f_max, attenuation)
        object.__setattr__(self, "_tau", _positive("tau", tau))
        object.__setattr__(self, "_peak_time", _positive("peak_time", 4.5 * tau))

    @property
    def tau(self) -> float:
        """Envelope time constant [s]; statistical standard deviation is tau/sqrt(2)."""
        return self._tau

    @property
    def peak_time(self) -> float:
        """Time of the unit peak [s]."""
        return self._peak_time

    @property
    def f_center(self) -> float:
        """Carrier frequency [Hz]: the center of ``[f_min, f_max]``."""
        return 0.5 * self.f_min + 0.5 * self.f_max

    @property
    def t_end(self) -> float:
        """Effective duration [s]; the default endpoint envelope is about 1.6e-9."""
        return 2.0 * self.peak_time

    def __call__(self, t):
        return _gaussian_value(t, self.tau, self.peak_time, self.f_center)

    def spectrum(self, f):
        f = np.asarray(f, dtype=float)
        fc = self.f_center
        env = 0.5 * self.tau * math.sqrt(math.pi)
        lobes = np.exp(-((math.pi * self.tau * (f - fc)) ** 2)) + np.exp(
            -((math.pi * self.tau * (f + fc)) ** 2)
        )
        return env * lobes * np.exp(-2j * math.pi * f * self.peak_time)


def _raised_cosine_ramp(t, rise_time: float):
    """0 → 1 over ``[0, rise_time]`` with a raised-cosine profile (scalar or array)."""
    x = np.clip(t / rise_time, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(math.pi * x))


@dataclass(frozen=True)
class WaveformSine(Waveform):
    """Continuous-wave sinusoid ``sin(2π f t + phase_deg)``, unit amplitude.

    A single-frequency drive; zero for ``t < 0``.  Its duration is
    infinite (``t_end = inf``), so a run driven by it needs an explicit
    duration and cannot stop on energy decay.  With ``rise_time`` the
    amplitude ramps in with a raised-cosine envelope, which keeps the
    switch-on from exciting the whole band.

    Parameters
    ----------
    f : float
        Frequency [Hz].
    phase_deg : float, default 0.0
        Phase [degrees].
    rise_time : float, optional
        Length of the raised-cosine switch-on [s].  ``None`` (default)
        switches on hard at ``t = 0``.
    """

    f: float
    phase_deg: float = 0.0
    rise_time: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "f", _positive("f", self.f))
        object.__setattr__(self, "phase_deg", float(self.phase_deg))
        if self.rise_time is not None:
            object.__setattr__(self, "rise_time", _positive("rise_time", self.rise_time))

    @property
    def f_max(self) -> float:
        return self.f

    @property
    def f_min(self) -> float:
        return self.f

    @property
    def f_center(self) -> float:
        return self.f

    @property
    def t_end(self) -> float:
        return math.inf

    def __call__(self, t):
        scalar = not _is_array(t)
        tt = float(t) if scalar else t
        out = np.sin(2.0 * math.pi * self.f * tt + math.radians(self.phase_deg))
        if self.rise_time is not None:
            out = out * _raised_cosine_ramp(tt, self.rise_time)
        out = np.where(tt < 0.0, 0.0, out)
        return float(out) if scalar else out


@dataclass(frozen=True)
class WaveformStep(Waveform):
    """Raised-cosine step (or pulse), unit plateau.

    Rises from 0 to 1 over ``rise_time``; with ``hold`` it stays at 1
    for that long and falls back over ``fall_time`` — a smooth
    rectangular pulse for time-domain reflectometry.  Without ``hold``
    the plateau lasts forever (``t_end = inf``), so the run needs an
    explicit duration.  Zero for ``t < 0``.

    Parameters
    ----------
    rise_time : float
        Length of the raised-cosine rise [s].  ``f_max`` is
        ``1 / rise_time``, the bandwidth the edge occupies.
    hold : float, optional
        Plateau duration [s].  ``None`` (default) never falls.
    fall_time : float, optional
        Length of the fall [s]; defaults to ``rise_time`` when ``hold``
        is given, ignored otherwise.
    """

    rise_time: float
    hold: float | None = None
    fall_time: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rise_time", _positive("rise_time", self.rise_time))
        if self.hold is not None:
            object.__setattr__(self, "hold", _nonnegative("hold", self.hold))
            fall = self.rise_time if self.fall_time is None else self.fall_time
            object.__setattr__(self, "fall_time", _positive("fall_time", fall))
        elif self.fall_time is not None:
            raise ValueError("fall_time needs a hold duration; give hold= as well")

    @property
    def f_max(self) -> float:
        return 1.0 / self.rise_time

    @property
    def f_min(self) -> float:
        return 0.0

    @property
    def f_center(self) -> None:
        return None

    @property
    def t_end(self) -> float:
        if self.hold is None:
            return math.inf
        return self.rise_time + self.hold + self.fall_time

    def __call__(self, t):
        scalar = not _is_array(t)
        tt = float(t) if scalar else t
        out = _raised_cosine_ramp(tt, self.rise_time)
        if self.hold is not None:
            t_fall = self.rise_time + self.hold
            out = np.where(
                tt >= t_fall,
                1.0 - _raised_cosine_ramp(tt - t_fall, self.fall_time),
                out,
            )
        out = np.where(tt < 0.0, 0.0, out)
        return float(out) if scalar else out


@dataclass(frozen=True, eq=False)
class WaveformTable(Waveform):
    """Tabulated waveform, linearly interpolated between its samples.

    Zero outside ``[t[0], t[-1]]``.  The band edges default to what
    the table's own spectrum shows: ``f_max`` is the highest frequency
    at which the magnitude is still within 40 dB of its peak; give it
    explicitly when the table is short or noisy.

    Parameters
    ----------
    t : array_like
        Sample times [s], strictly increasing, starting at or after 0.
    values : array_like
        Sample values, same length as ``t``.
    f_max : float, optional
        Upper band edge [Hz]; estimated from the samples by default.
    f_min : float, default 0.0
        Lower band edge [Hz].
    f_center : float, optional
        Carrier frequency [Hz] when the table holds a modulated pulse.
    """

    t: np.ndarray
    values: np.ndarray
    f_max: float | None = None
    f_min: float = 0.0
    f_center: float | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=float).ravel()
        v = np.asarray(self.values, dtype=float).ravel()
        if t.size < 2 or t.shape != v.shape:
            raise ValueError(
                f"t and values must be 1-D arrays of the same length >= 2; "
                f"got {t.shape} and {v.shape}",
            )
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("t must be strictly increasing")
        if t[0] < 0.0:
            raise ValueError(f"t must start at or after 0; got t[0] = {t[0]!r}")
        t.setflags(write=False)
        v.setflags(write=False)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "f_min", _nonnegative("f_min", self.f_min))
        if self.f_center is not None:
            object.__setattr__(self, "f_center", _positive("f_center", self.f_center))
        f_max = self._estimate_f_max() if self.f_max is None else _positive("f_max", self.f_max)
        object.__setattr__(self, "f_max", f_max)
        if self.f_max <= self.f_min:
            raise ValueError(
                f"f_max ({self.f_max:.4g} Hz) must exceed f_min ({self.f_min:.4g} Hz)",
            )

    def _estimate_f_max(self) -> float:
        """Highest frequency within 40 dB of the spectral peak (resampled uniformly)."""
        dt = float(np.min(np.diff(self.t)))
        n = int(math.ceil((self.t[-1] - self.t[0]) / dt)) + 1
        tt = self.t[0] + np.arange(n) * dt
        mag = np.abs(np.fft.rfft(np.interp(tt, self.t, self.values)))
        freqs = np.fft.rfftfreq(n, d=dt)
        peak = float(mag.max())
        if peak <= 0.0:
            raise ValueError("values are all zero; give f_max explicitly")
        above = np.nonzero(mag >= 1e-2 * peak)[0]
        f_max = float(freqs[above[-1]])
        if f_max <= 0.0:
            raise ValueError("the table's spectrum is all at DC; give f_max explicitly")
        return f_max

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def __call__(self, t):
        scalar = not _is_array(t)
        tt = float(t) if scalar else t
        out = np.interp(tt, self.t, self.values, left=0.0, right=0.0)
        return float(out) if scalar else out

    def __repr__(self) -> str:
        return (
            f"WaveformTable(n={self.t.size}, t_end={self.t_end:.3e}, "
            f"f_max={self.f_max:.3e}, f_min={self.f_min:.3e}, f_center={self.f_center})"
        )


@dataclass(frozen=True, eq=False)
class WaveformFunction(Waveform):
    """Waveform from a user function ``fn(t)``.

    The band edges cannot be read off a Python function, so ``f_max``
    is required — it sizes the run-length estimate and the warning
    against exceeding the mesh's design frequency.  A function
    waveform cannot be stored in a project recipe, so a run driven by
    it cannot be resumed.

    Parameters
    ----------
    fn : callable
        ``fn(t) -> value`` for a float ``t`` [s]; may accept arrays.
    f_max : float
        Upper band edge [Hz].
    f_min : float, default 0.0
        Lower band edge [Hz].
    f_center : float, optional
        Carrier frequency [Hz], if the function is a modulated form.
    t_end : float, default inf
        Time [s] after which ``fn`` is effectively zero; ``inf`` marks
        a continuous-wave form.
    """

    fn: Callable = dc_field(repr=False)
    f_max: float
    f_min: float = 0.0
    f_center: float | None = None
    t_end: float = math.inf

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError(f"fn must be callable; got {type(self.fn).__name__}")
        object.__setattr__(self, "f_max", _positive("f_max", self.f_max))
        object.__setattr__(self, "f_min", _nonnegative("f_min", self.f_min))
        if self.f_center is not None:
            object.__setattr__(self, "f_center", _positive("f_center", self.f_center))
        t_end = float(self.t_end)
        if t_end <= 0.0 or math.isnan(t_end):
            raise ValueError(f"t_end must be positive (inf for CW forms); got {t_end!r}")
        object.__setattr__(self, "t_end", t_end)
        if self.f_max <= self.f_min:
            raise ValueError(
                f"f_max ({self.f_max:.4g} Hz) must exceed f_min ({self.f_min:.4g} Hz)",
            )

    def __call__(self, t):
        if not _is_array(t):
            return float(self.fn(float(t)))
        try:
            out = self.fn(t)
            if getattr(out, "shape", None) == t.shape:
                return out
        except (TypeError, ValueError):
            pass
        return np.array([float(self.fn(float(x))) for x in t.ravel()], dtype=float).reshape(t.shape)


__all__ = [
    "Waveform",
    "WaveformGaussian",
    "WaveformGaussianModulated",
    "WaveformSine",
    "WaveformStep",
    "WaveformTable",
    "WaveformFunction",
]
