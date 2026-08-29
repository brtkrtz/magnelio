"""Signals — excitation waveforms and sampled time series.

A :class:`Waveform` is the time function an :class:`~magnelio.Excitation`
binds to a port or source (unit peak, known bandwidth and duration);
:class:`Signal1D` is a *sampled* series on the result side, with its
spectrum.
"""

from magnelio.signals.signal_1d import Signal1D
from magnelio.signals.waveforms import (
    Waveform,
    WaveformFunction,
    WaveformGaussian,
    WaveformGaussianModulated,
    WaveformSine,
    WaveformStep,
    WaveformTable,
)

__all__ = [
    "Signal1D",
    "Waveform",
    "WaveformGaussian",
    "WaveformGaussianModulated",
    "WaveformSine",
    "WaveformStep",
    "WaveformTable",
    "WaveformFunction",
]
