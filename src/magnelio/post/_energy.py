"""The one number a running simulation is judged by: dB below the energy peak."""

from __future__ import annotations

import numpy as np


def db_below_peak(trace) -> float | None:
    """The last sample of an energy trace in dB below the trace's peak.

    ``None`` when the trace is empty or not positive — a run that has
    not started, or whose energy has not risen above zero yet.  Ten
    times the log: the trace is an energy, not an amplitude.
    """
    if trace is None:
        return None
    trace = np.asarray(trace)
    if trace.size == 0:
        return None
    energy = np.asarray(trace["energy"], dtype=float)
    peak = float(energy.max())
    last = float(energy[-1])
    if not (peak > 0.0 and last > 0.0):
        return None
    return float(10.0 * np.log10(last / peak))
