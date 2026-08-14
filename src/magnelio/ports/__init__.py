"""Ports — declarative port classes, specs, conductors and reports.
``PortWaveguide``, ``PortAnalytical`` and ``PortLumped`` are declared
on the :class:`~magnelio.GeometryModel` before meshing; the
``PortSpec*`` family covers custom setups passed to an analysis via
``ports=``.  The builders and runtime operators behind them are
internal.
"""

from magnelio.ports._lumped import (
    PortOperatorLumped,
    PortSpecLumped,
    build_lumped_port,
)
from magnelio.ports._modal import (
    BboxLateralConductor,
    ConductorSpec,
    ExcitationSpec,
    LevelResult,
    Mode,
    ModeRefinementReport,
    ModeReport,
    ModeType,
    PortOperatorBandDTBC,
    PortOperatorModal,
    PortOperatorReport,
    PortPlane,
    PortReport,
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
    RegionConductor,
    WallConductor,
    build_band_dtbc_port,
    build_modal_port,
    solve_modes_refined,
)
from magnelio.ports.base import Port
from magnelio.ports.declarative import PortAnalytical, PortLumped, PortWaveguide
from magnelio.ports.recorder import PortSignalRecorder

__all__ = [
    "PortWaveguide",
    "PortAnalytical",
    "PortLumped",
    "PortSpecCoax",
    "PortSpecRectWG",
    "PortSpecNumerical",
    "PortSpecMultiConductor",
    "PortSpecLumped",
    "ExcitationSpec",
    "ConductorSpec",
    "WallConductor",
    "RegionConductor",
    "BboxLateralConductor",
    "Mode",
    "ModeType",
    "ModeReport",
    "PortReport",
    "PortOperatorReport",
    "ModeRefinementReport",
]
