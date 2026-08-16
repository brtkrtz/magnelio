"""
Running Discrete Fourier Transform accumulator for frequency-domain monitors.

At each time step *n* the DFT bins are updated:

    F_k[x,y,z] += field[n][x,y,z] * exp(+j * 2π * f_k * t_n) * dt

where *t_n* is the physical time of the field value (E at n·dt, H at
(n+0.5)·dt).  The half-step stagger is automatically handled by passing
the correct *t* for each field type.
"""

from __future__ import annotations

import numpy as np


class DFTAccumulator:
    """Running DFT accumulator for a set of spatial arrays.

    Parameters
    ----------
    freqs : np.ndarray
        Target frequencies [Hz], shape ``(Nf,)``.
    shape : tuple[int, ...]
        Spatial shape of the data being accumulated.
    """

    def __init__(self, freqs: np.ndarray, shape: tuple[int, ...]) -> None:
        self._freqs = np.asarray(freqs, dtype=float)
        self._shape = shape
        nf = len(self._freqs)
        # DFT bins: (Nf, *shape), complex128
        self._bins: np.ndarray = np.zeros((nf, *shape), dtype=complex)
        self._omega = 2.0 * np.pi * self._freqs  # (Nf,)

    def accumulate(self, data: np.ndarray, t: float, dt: float) -> None:
        """Add the contribution of the current time step.

        Parameters
        ----------
        data : np.ndarray
            Field values at time *t*, shape must match ``self._shape``.
        t : float
            Physical time [s] of *data*.
        dt : float
            Simulation time step [s] (used as integration weight).
        """
        # Phase factors for all frequencies: shape (Nf,)
        phase = np.exp(1j * self._omega * t) * dt
        # Outer product: (Nf, *shape) += (Nf, 1, 1, ...) * (*shape,)
        self._bins += phase.reshape(-1, *([1] * len(self._shape))) * data

    @property
    def result(self) -> np.ndarray:
        """DFT result array, shape ``(Nf, *shape)``, complex128."""
        return self._bins

    @property
    def freqs(self) -> np.ndarray:
        return self._freqs


def source_spectrum(values, dt: float, freqs) -> np.ndarray:
    """Transform a source waveform in the accumulator's own convention.

    ``Σ v[n] exp(+jω t_n) dt`` — the same sum :meth:`DFTAccumulator.
    accumulate` forms, so dividing recorded bins by this cancels the
    excitation exactly rather than approximately.
    """
    values = np.asarray(values, dtype=float)
    t = np.arange(len(values)) * dt
    omega = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    return (np.exp(1j * omega[:, np.newaxis] * t[np.newaxis, :]) * dt) @ values


def divide_by_spectrum(arr: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Divide bins (frequency axis 0) by *spectrum*, broadcasting spatially.

    Bins whose source amplitude underflows carry no information about
    the structure, only about the pulse's own null — they are returned
    as zero rather than as a quotient of two vanishing numbers.
    """
    src = np.asarray(spectrum).reshape(-1, *([1] * (arr.ndim - 1)))
    live = np.abs(src) > 1e-300
    return np.where(live, arr / np.where(live, src, 1.0 + 0j), 0.0 + 0j)
