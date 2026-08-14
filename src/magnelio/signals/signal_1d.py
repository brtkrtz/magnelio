"""Signal1D — immutable time-domain signal container with lazy FFT.

Provides a lightweight wrapper around (t, values) pairs with frequency-domain
operations needed for S-parameter extraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Signal1D:
    """Immutable time-domain signal.

    Parameters
    ----------
    t : np.ndarray
        Time axis [s], shape (N,).
    values : np.ndarray
        Signal values, shape (N,).
    dt : float
        Time step [s].
    label : str
        Optional label for identification.
    """

    t: np.ndarray
    values: np.ndarray
    dt: float
    label: str = ""

    def __post_init__(self) -> None:
        # Ensure arrays are numpy arrays (frozen=True requires object.__setattr__)
        object.__setattr__(self, "t", np.asarray(self.t, dtype=float))
        object.__setattr__(self, "values", np.asarray(self.values, dtype=float))

    # ------------------------------------------------------------------
    # Frequency-domain properties (lazy, cached via object.__setattr__)
    # ------------------------------------------------------------------

    @property
    def f(self) -> np.ndarray:
        """Frequency axis [Hz]."""
        return np.fft.rfftfreq(len(self.values), d=self.dt)

    @property
    def spectrum(self) -> np.ndarray:
        """Complex FFT spectrum (cached)."""
        key = "_spectrum_cache"
        cache = self.__dict__.get(key)
        if cache is None:
            cache = np.fft.rfft(self.values)
            object.__setattr__(self, key, cache)
        return cache

    def at_frequencies(self, f_target: np.ndarray) -> np.ndarray:
        """Evaluate spectrum at arbitrary frequency points.

        Two paths:

        - **Direct DFT** (default for small ``Nf · N``): evaluates
          ``Σ_n x_n · e^{-2π j f t_n} · dt`` exactly at every requested
          frequency.  Cost: ``O(Nf · N)``.  Equivalent in scale to
          ``np.fft.rfft`` (``rfft`` returns ``Σ_n x_n · e^{-2π j k n / N}``
          without a ``dt`` factor; the direct DFT here returns the same
          magnitude after dividing the Riemann-sum form by ``dt``,
          i.e. cancels the explicit ``dt`` factor).
        - **Zero-padded rFFT + linear interp** (fallback for large
          ``Nf · N``): the historical path; pads so the FFT bin
          spacing is at most ``df_target / 2`` and linear-interpolates
          ``real`` / ``imag``.  Faster for very dense ``f_target``,
          but introduces a 1–3 % magnitude error when the inter-bin
          phase rotates significantly (~30° per bin) — manifests as a
          spurious ``|S|² < 1`` floor in the modal-port S-parameter
          pipeline.  Switching to the direct DFT for small ``Nf · N``
          eliminates that floor down to floating-point precision.

        Parameters
        ----------
        f_target : np.ndarray
            Target frequencies [Hz].

        Returns
        -------
        np.ndarray
            Complex spectrum values at ``f_target``.
        """
        f_target = np.asarray(f_target, dtype=float).ravel()
        N = len(self.values)

        # Direct DFT cutoff: O(Nf · N) ≤ 1e8 keeps the cost under ~1 s on
        # a single core for typical post-processing frequency sweeps.
        if f_target.size * N <= int(1e8):
            n = np.arange(N)
            phase = -2j * math.pi * np.outer(f_target, n) * self.dt
            return np.exp(phase) @ self.values

        # Fallback: zero-padded rFFT + linear interp on real/imag.
        if f_target.size > 1:
            df_target = np.min(np.diff(np.sort(f_target[f_target > 0])))
            n_pad = max(N, int(np.ceil(1.0 / (self.dt * df_target / 2))))
        else:
            n_pad = N
        n_pad = int(2 ** np.ceil(np.log2(n_pad)))

        spec = np.fft.rfft(self.values, n=n_pad)
        f_src = np.fft.rfftfreq(n_pad, d=self.dt)
        real = np.interp(f_target, f_src, spec.real)
        imag = np.interp(f_target, f_src, spec.imag)
        return real + 1j * imag

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: Signal1D) -> Signal1D:
        if not isinstance(other, Signal1D):
            return NotImplemented
        return Signal1D(
            t=self.t,
            values=self.values + other.values,
            dt=self.dt,
            label=self.label,
        )

    def __sub__(self, other: Signal1D) -> Signal1D:
        if not isinstance(other, Signal1D):
            return NotImplemented
        return Signal1D(
            t=self.t,
            values=self.values - other.values,
            dt=self.dt,
            label=self.label,
        )

    def __mul__(self, scalar: float) -> Signal1D:
        if isinstance(scalar, Signal1D):
            return NotImplemented
        return Signal1D(
            t=self.t,
            values=self.values * scalar,
            dt=self.dt,
            label=self.label,
        )

    def __rmul__(self, scalar: float) -> Signal1D:
        return self.__mul__(scalar)

    def __neg__(self) -> Signal1D:
        return Signal1D(
            t=self.t,
            values=-self.values,
            dt=self.dt,
            label=self.label,
        )

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return (
            f"Signal1D(N={len(self.values)}, dt={self.dt:.3e}s"
            f"{', label=' + repr(self.label) if self.label else ''})"
        )
