"""Signal components — time series and excitation waveforms."""

from magnelio.signals.signal_1d import Signal1D
from magnelio.signals.waveforms import (
    gaussian,
    modulated_gaussian,
    waveform_for_mode,
)

__all__ = [
    "Signal1D",
    "gaussian",
    "modulated_gaussian",
    "waveform_for_mode",
]
