"""Analytical TEM mode solver for a coaxial waveguide cross-section.

For a homogeneously-filled coax with inner radius ``r_i`` and outer radius
``r_o``, the TEM mode has the closed form

    E_r(r)   =  C / r
    H_φ(r)   =  C / (η · r)

with η = η₀ / √ε_r the wave impedance of the dielectric.  The Poynting
normalisation that yields 1 W of axial power is

    C  =  √( η / (2π · ln(r_o / r_i)) )

and the characteristic line impedance is

    Z₀  =  (η / 2π) · ln(r_o / r_i).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from magnelio.ports._modal.mode import ETA0, Mode, ModeType


@dataclass
class CoaxAnalyticalModeSolver:
    """Closed-form TEM mode solver for a coaxial waveguide.

    Parameters
    ----------
    inner_radius : float
        Inner-conductor radius [m].
    outer_radius : float
        Outer-conductor (shield) radius [m].  Must be > ``inner_radius``.
    epsilon_r : float, default 1.0
        Relative permittivity of the (homogeneous) dielectric between the
        conductors.
    center : tuple[float, float], default (0.0, 0.0)
        Coordinates of the coax axis in the port-plane local (u, v) frame.

    Notes
    -----
    Phase 1 supports only the fundamental TEM mode (``n_modes = 1``).
    Higher-order coax modes (TE, TM with cylindrical Bessel-function
    cross-sections) are out of scope until Phase 2.
    """

    inner_radius: float
    outer_radius: float
    epsilon_r: float = 1.0
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.inner_radius <= 0.0:
            raise ValueError("inner_radius must be positive")
        if self.outer_radius <= self.inner_radius:
            raise ValueError("outer_radius must exceed inner_radius")
        if self.epsilon_r <= 0.0:
            raise ValueError("epsilon_r must be positive")

    def solve(self, n_modes: int = 1, f_calc: float = 0.0) -> list[Mode]:
        # f_calc is unused: TEM is non-dispersive, profile is identical at
        # every frequency.  Argument retained to match the ModeSolver
        # Protocol.
        del f_calc
        if n_modes != 1:
            raise NotImplementedError(
                "CoaxAnalyticalModeSolver supports only n_modes=1 (TEM).  "
                "Higher-order coax modes are deferred to Phase 2."
            )

        r_i = self.inner_radius
        r_o = self.outer_radius
        eps_r = self.epsilon_r
        u_c, v_c = self.center

        eta = ETA0 / math.sqrt(eps_r)
        c_norm = math.sqrt(eta / (2.0 * math.pi * math.log(r_o / r_i)))
        c_h = c_norm / eta

        r_i_sq = r_i * r_i
        r_o_sq = r_o * r_o

        def evaluator(u: np.ndarray, v: np.ndarray):
            du = np.asarray(u, dtype=float) - u_c
            dv = np.asarray(v, dtype=float) - v_c
            r_sq = du * du + dv * dv
            in_ring = (r_sq >= r_i_sq) & (r_sq <= r_o_sq)
            r_sq_safe = np.where(r_sq > 0.0, r_sq, 1.0)
            scale_e = np.where(in_ring, c_norm / r_sq_safe, 0.0)
            scale_h = np.where(in_ring, c_h / r_sq_safe, 0.0)
            E_u = scale_e * du
            E_v = scale_e * dv
            H_u = -scale_h * dv
            H_v = scale_h * du
            return E_u, E_v, H_u, H_v

        z_line = (eta / (2.0 * math.pi)) * math.log(r_o / r_i)

        return [
            Mode(
                name="TEM",
                mode_type=ModeType.TEM,
                omega_c=0.0,
                epsilon_r=eps_r,
                field_evaluator=evaluator,
                z_line=z_line,
            )
        ]
