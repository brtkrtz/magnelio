"""Material components — dispersion, roughness and surface-impedance fits.
``Material`` itself lives in the core ``magnelio`` namespace.
"""

from magnelio.materials.dispersion import DispersionModel
from magnelio.materials.material import Material
from magnelio.materials.roughness import Hammerstad, Huray, SurfaceRoughness
from magnelio.materials.surface_impedance import (
    SurfaceImpedanceFit,
    fit_surface_impedance,
    fit_wall_impedances,
)

__all__ = [
    "DispersionModel",
    "SurfaceRoughness",
    "Hammerstad",
    "Huray",
    "SurfaceImpedanceFit",
    "fit_surface_impedance",
    "fit_wall_impedances",
]
