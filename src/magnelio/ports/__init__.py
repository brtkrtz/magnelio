"""Ports — declarative port classes, specs, conductors and reports.
``PortWaveguide``, ``PortAnalytical`` and ``PortLumped`` are declared
on the :class:`~magnelio.GeometryModel` before meshing; the
``PortSpec*`` family covers custom setups passed to an analysis via
``ports=``; :func:`refine_port_modes` converges a port's mode parameters
on its own plane.  The builders and runtime operators behind them are
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
    Mode,
    ModeReport,
    ModeType,
    PortDispersionReport,
    PortOperatorBandDTBC,
    PortOperatorModal,
    PortOperatorReport,
    PortPlane,
    PortRefinementReport,
    PortReport,
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
    RefinementLevel,
    RegionConductor,
    WallConductor,
    build_band_dtbc_port,
    build_modal_port,
    refine_port_modes,
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
    "ConductorSpec",
    "WallConductor",
    "RegionConductor",
    "BboxLateralConductor",
    "Mode",
    "ModeType",
    "ModeReport",
    "PortReport",
    "PortDispersionReport",
    "PortRefinementReport",
    "RefinementLevel",
    "refine_port_modes",
    "PortOperatorReport",
]
