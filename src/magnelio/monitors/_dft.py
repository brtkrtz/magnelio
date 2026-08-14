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
