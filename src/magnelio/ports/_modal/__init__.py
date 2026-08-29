"""Modal waveguide-port pipeline (Phase 1).

Public exports:

- ``Mode``, ``ModeType`` — passive modal data class.
- ``ModeSolver`` — Protocol for analytical and numerical solvers.
- ``CoaxAnalyticalModeSolver`` — closed-form TEM solver for coaxial lines.
- ``PortSpecCoax`` / ``PortSpecRectWG`` /
  ``build_modal_port`` — public, face-agnostic port factory.

Constants ``EPS0``, ``MU0``, ``C0``, ``ETA0`` are re-exported from
``mode`` for convenience.
"""

from magnelio.constants import C0, EPS0, ETA0, MU0
from magnelio.ports._modal.auto_conductors import (
    extract_conductor_groups_from_mesh,
)
from magnelio.ports._modal.band_dtbc import (
    BandPortData,
    PortOperatorBandDTBC,
)
from magnelio.ports._modal.coax import CoaxAnalyticalModeSolver
from magnelio.ports._modal.discrete import DiscreteMode, discretize_modes, gram_matrix
from magnelio.ports._modal.factory import (
    BboxLateralConductor,
    ConductorSpec,
    PortSpecCoax,
    PortSpecMultiConductor,
    PortSpecNumerical,
    PortSpecRectWG,
    RegionConductor,
    WallConductor,
    build_band_dtbc_port,
    build_cw_true_mode_port,
    build_modal_port,
)
from magnelio.ports._modal.mode import (
    Mode,
    ModeType,
)
from magnelio.ports._modal.mode_report import ModeReport, PortReport
from magnelio.ports._modal.numerical_2d import Numerical2DModeSolver
from magnelio.ports._modal.operator import PortOperatorModal
from magnelio.ports._modal.port_plane import BoxFace, PortPlane
from magnelio.ports._modal.port_report import PortOperatorReport
from magnelio.ports._modal.rect import RectWGAnalyticalModeSolver
from magnelio.ports._modal.refinement import (
    LevelResult,
    ModeRefinementReport,
    solve_modes_refined,
)
from magnelio.ports._modal.solver import ModeSolver
from magnelio.ports._modal.tem_laplace import (
    solve_qtem_laplace,
    solve_tem_laplace,
)
from magnelio.ports._modal.zeta_pencil import (
    CWChannel,
    CWPortData,
    cw_decompose,
    cw_lockin_phasors,
)

__all__ = [
    "BboxLateralConductor",
    "BoxFace",
    "C0",
    "CoaxAnalyticalModeSolver",
    "PortSpecCoax",
    "ConductorSpec",
    "DiscreteMode",
    "EPS0",
    "ETA0",
    "MU0",
    "PortOperatorModal",
    "Mode",
    "ModeReport",
    "ModeSolver",
    "ModeType",
    "PortReport",
    "PortSpecMultiConductor",
    "Numerical2DModeSolver",
    "PortSpecNumerical",
    "PortPlane",
    "PortOperatorReport",
    "RectWGAnalyticalModeSolver",
    "PortSpecRectWG",
    "RegionConductor",
    "WallConductor",
    "LevelResult",
    "ModeRefinementReport",
    "PortOperatorBandDTBC",
    "BandPortData",
    "CWChannel",
    "CWPortData",
    "build_band_dtbc_port",
    "build_cw_true_mode_port",
    "build_modal_port",
    "cw_decompose",
    "cw_lockin_phasors",
    "discretize_modes",
    "extract_conductor_groups_from_mesh",
    "gram_matrix",
    "solve_modes_refined",
    "solve_qtem_laplace",
    "solve_tem_laplace",
]
