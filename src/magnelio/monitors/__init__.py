"""Monitors — record fields, fluxes and wall losses during a run.
Monitors are attached to the solver via the analysis ``monitors=``
parameter: ``attach`` once, ``record`` every step, ``finalize`` after
the run.
"""

from magnelio.monitors.base import MonitorRegion
from magnelio.monitors.far_field import MonitorFarField
from magnelio.monitors.field_frequency import MonitorFieldFrequency
from magnelio.monitors.field_time import MonitorFieldTime
from magnelio.monitors.flux import MonitorFluxTime
from magnelio.monitors.wall_loss import MonitorWallLoss

__all__ = [
    "MonitorFieldTime",
    "MonitorFieldFrequency",
    "MonitorFluxTime",
    "MonitorFarField",
    "MonitorWallLoss",
]
