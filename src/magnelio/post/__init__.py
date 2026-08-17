"""Post-processing components — S-parameter computation helpers.
The result objects themselves come back from the analyses; this
component holds the free functions and containers for custom pipelines.
"""

from magnelio.post.far_field import FarFieldResult, ntff_transform
from magnelio.post.modal_sparameters import (
    compute_band_s_parameters,
    compute_s_parameters,
    destaggered_power_waves,
)
from magnelio.post.sparameter_result import SParameterResult
from magnelio.post.wall_loss import wall_loss_Q

__all__ = [
    "compute_s_parameters",
    "compute_band_s_parameters",
    "FarFieldResult",
    "ntff_transform",
    "SParameterResult",
    "wall_loss_Q",
]
