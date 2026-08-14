"""Analytical TE/TM mode solver for a rectangular waveguide cross-section.

For a homogeneously filled rectangular waveguide of width ``a`` (along
``u``) and height ``b`` (along ``v``), the modes split into:

- ``TE_mn``  with ``(m, n)`` not both zero (m = 0, 1, ...; n = 0, 1, ...)
- ``TM_mn``  with both ``m >= 1`` and ``n >= 1``

The cut-off (angular) frequency is shared between same-(m, n) TE and TM:

    omega_c  =  c₀ / √ε_r · π · √( (m/a)² + (n/b)² ).

The Poynting-normalised transverse field profiles at mode-calc frequency
``omega_calc`` are (with ``K_x = k_x/k_c``, ``K_y = k_y/k_c``,
``A_TE = √(4·Z_TE/(a·b·ε_m·ε_n))``, ``A_TM = √(4·Z_TM/(a·b))``):

    TE_mn:
      E_u =  K_y · A_TE · cos(k_x u) · sin(k_y v)
      E_v = -K_x · A_TE · sin(k_x u) · cos(k_y v)
      H_u =  K_x · A_TE / Z_TE · sin(k_x u) · cos(k_y v)
      H_v =  K_y · A_TE / Z_TE · cos(k_x u) · sin(k_y v)

    TM_mn:
      E_u =  K_x · A_TM · cos(k_x u) · sin(k_y v)
      E_v =  K_y · A_TM · sin(k_x u) · cos(k_y v)
      H_u = -K_y · A_TM / Z_TM · sin(k_x u) · cos(k_y v)
      H_v =  K_x · A_TM / Z_TM · cos(k_x u) · sin(k_y v)

with the Neumann factor ``ε_m = 2 if m == 0 else 1`` (similarly ``ε_n``)
present only in the TE normalisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from magnelio.constants import C0, EPS0, MU0
from magnelio.ports._modal.mode import (
    Mode,
    ModeType,
)


@dataclass
class RectWGAnalyticalModeSolver:
    """Closed-form TE/TM mode solver for a rectangular waveguide.

    Parameters
    ----------
    width_a : float
        Cross-section width along the local-``u`` axis [m].  Conventionally
        the broader dimension.
    height_b : float
        Cross-section height along the local-``v`` axis [m].
    epsilon_r : float, default 1.0
        Relative permittivity of the (homogeneous) dielectric filling.
    center : tuple[float, float], default (0.0, 0.0)
        ``(u, v)`` coordinates of the lower-left corner of the waveguide
        in the port-plane local frame.

    Notes
    -----
    The field evaluators use a real-amplitude time-domain convention:
    ``E_t`` and ``H_t`` are returned as real arrays whose product gives
    the Poynting axial component directly (no factor of 1/2, no complex
    conjugation).  For modes that are evanescent at ``f_calc`` the
    bake-in uses ``|Z(f_calc)|``; the actual frequency-dependent phase is
    recovered at run time via ``Mode.z_modal(omega)`` in the operator.
    """

    width_a: float
    height_b: float
    epsilon_r: float = 1.0
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.width_a <= 0.0:
            raise ValueError("width_a must be positive")
        if self.height_b <= 0.0:
            raise ValueError("height_b must be positive")
        if self.epsilon_r <= 0.0:
            raise ValueError("epsilon_r must be positive")

    def solve(self, n_modes: int, f_calc: float) -> list[Mode]:
        if n_modes <= 0:
            raise ValueError("n_modes must be positive")
        if f_calc <= 0.0:
            raise ValueError("f_calc must be positive")

        # Enumerate enough (mode_type, m, n) candidates to cover n_modes
        # with margin.  Heuristic upper bound on m and n.
        m_max = max(2, int(math.ceil(math.sqrt(2.0 * n_modes))) + 2)
        n_max = m_max
        candidates: list[tuple[str, int, int, float]] = []
        for m in range(m_max + 1):
            for n in range(n_max + 1):
                if m == 0 and n == 0:
                    continue
                kx = m * math.pi / self.width_a
                ky = n * math.pi / self.height_b
                kc = math.sqrt(kx * kx + ky * ky)
                omega_c = kc * C0 / math.sqrt(self.epsilon_r)
                candidates.append(("TE", m, n, omega_c))
                if m >= 1 and n >= 1:
                    candidates.append(("TM", m, n, omega_c))

        candidates.sort(key=lambda c: (c[3], 0 if c[0] == "TE" else 1, c[1], c[2]))

        if len(candidates) < n_modes:
            raise RuntimeError(
                f"Only {len(candidates)} candidate modes generated, "
                f"requested {n_modes}.  Widen the m_max/n_max heuristic."
            )

        chosen = candidates[:n_modes]
        omega_calc = 2.0 * math.pi * f_calc
        return [self._build_mode(mt, m, n, omega_c, omega_calc) for (mt, m, n, omega_c) in chosen]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mode(
        self,
        mode_type_str: str,
        m: int,
        n: int,
        omega_c: float,
        omega_calc: float,
    ) -> Mode:
        mode_type = ModeType.TE if mode_type_str == "TE" else ModeType.TM
        label = f"{mode_type_str}{m},{n}" if max(m, n) >= 10 else f"{mode_type_str}{m}{n}"
        evaluator = self._make_evaluator(mode_type, m, n, omega_calc)
        return Mode(
            name=label,
            mode_type=mode_type,
            omega_c=omega_c,
            epsilon_r=self.epsilon_r,
            field_evaluator=evaluator,
            z_line=None,
        )

    def _make_evaluator(
        self,
        mode_type: ModeType,
        m: int,
        n: int,
        omega_calc: float,
    ):
        a = self.width_a
        b = self.height_b
        eps_r = self.epsilon_r
        u_c, v_c = self.center
        kx = m * math.pi / a
        ky = n * math.pi / b
        kc = math.sqrt(kx * kx + ky * ky)
        omega_c = kc * C0 / math.sqrt(eps_r)

        z_real = self._z_real_at(mode_type, omega_calc, omega_c, eps_r)

        eps_m = 2.0 if m == 0 else 1.0
        eps_n = 2.0 if n == 0 else 1.0

        if mode_type is ModeType.TE:
            A = math.sqrt(4.0 * z_real / (a * b * eps_m * eps_n))
        else:  # TM (m, n both >= 1, no Neumann factors)
            A = math.sqrt(4.0 * z_real / (a * b))

        K_x = kx / kc
        K_y = ky / kc
        inv_z = A / z_real

        def evaluator(u: np.ndarray, v: np.ndarray):
            du = np.asarray(u, dtype=float) - u_c
            dv = np.asarray(v, dtype=float) - v_c
            inside = (du >= 0.0) & (du <= a) & (dv >= 0.0) & (dv <= b)
            cos_x = np.cos(kx * du)
            sin_x = np.sin(kx * du)
            cos_y = np.cos(ky * dv)
            sin_y = np.sin(ky * dv)
            if mode_type is ModeType.TE:
                E_u = (K_y * A) * cos_x * sin_y
                E_v = -(K_x * A) * sin_x * cos_y
                H_u = (K_x * inv_z) * sin_x * cos_y
                H_v = (K_y * inv_z) * cos_x * sin_y
            else:  # TM
                E_u = (K_x * A) * cos_x * sin_y
                E_v = (K_y * A) * sin_x * cos_y
                H_u = -(K_y * inv_z) * sin_x * cos_y
                H_v = (K_x * inv_z) * cos_x * sin_y
            mask = inside.astype(float)
            return E_u * mask, E_v * mask, H_u * mask, H_v * mask

        return evaluator

    @staticmethod
    def _z_real_at(
        mode_type: ModeType,
        omega: float,
        omega_c: float,
        eps_r: float,
    ) -> float:
        """Real-valued ``|Z|`` at ``omega`` for profile bake-in.

        Above cut-off, ``Z`` is real positive; below cut-off, ``Z`` is
        purely imaginary, in which case we return its magnitude.  At
        cut-off the impedance is singular (TE) or zero (TM) — caller is
        expected to choose a meaningful ``f_calc``.
        """
        k_sq = (omega**2) * MU0 * EPS0 * eps_r
        kc_sq = (omega_c**2) * MU0 * EPS0 * eps_r
        diff = kc_sq - k_sq
        if abs(diff) < 1e-30:
            raise ValueError(
                "f_calc coincides with a mode cut-off; the profile "
                "bake-in is singular.  Choose f_calc strictly different "
                "from any included mode's cutoff."
            )
        sqrt_abs = math.sqrt(abs(diff))
        if mode_type is ModeType.TE:
            return omega * MU0 / sqrt_abs
        # TM
        return sqrt_abs / (omega * EPS0 * eps_r)
