"""Modal data class for waveguide ports — Phase 1.

A ``Mode`` carries everything the modal port operator needs to know about
one waveguide eigenmode: its type (TEM/TE/TM), cut-off frequency, the
cross-section permittivity it lives in, and a closure that evaluates the
Poynting-normalised transverse field profile at any (u, v) point of the
port plane.

Modal wave impedance ``Z_wave(omega)`` and propagation constant
``gamma(omega)`` are derived from the textbook closed-form relations and
exposed as methods so callers can evaluate them at any frequency without
holding an external table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Tuple

import numpy as np

from magnelio.constants import EPS0, ETA0, MU0  # noqa: E402


class ModeType(Enum):
    """Mode classification for homogeneously-filled waveguides."""

    TEM = "TEM"
    TE = "TE"
    TM = "TM"


# Field evaluator signature.  Given (u, v) coordinate arrays in the
# port-plane local frame, returns ``(E_u, E_v, H_u, H_v)`` arrays of the
# same shape.  The profile is Poynting-normalised so the integral of
# ``Re(E × H*) · n̂`` over the cross-section equals 1 W.
FieldEvaluator = Callable[
    [np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]


@dataclass(frozen=True)
class Mode:
    """A single waveguide eigenmode with closed-form frequency dependence.

    A ``Mode`` carries its transverse field information in **one** of two
    mutually exclusive forms:

    - **Analytical (Phase 1)** — ``field_evaluator`` is set to a closure
      that samples the Poynting-normalised profile at any (u, v) point.
      All four ``discrete_*_profile`` fields are ``None``.  Used by
      :class:`CoaxAnalyticalModeSolver` and
      :class:`RectWGAnalyticalModeSolver`; resampled and B-orthonormalised
      onto the FIT grid by :func:`discretize_modes`.
    - **Numerical (Phase 2)** — all four ``discrete_*_profile`` arrays
      are set to M_ε-orthonormal edge vectors built natively on the
      port-plane FIT grid; ``field_evaluator`` is ``None``.  Used by
      ``Numerical2DModeSolver``; :func:`discretize_modes` passes these
      through unchanged (no resampling, no Gram-Schmidt).

    The validity invariant is enforced at construction.

    Parameters
    ----------
    name : str
        Human-readable mode identifier, e.g. ``"TEM"`` or ``"TE10"``.
    mode_type : ModeType
        Mode classification.  Drives the formulas for ``z_wave`` and
        ``gamma``.
    omega_c : float
        Cut-off angular frequency [rad/s].  Zero for TEM.
    epsilon_r : float
        Relative permittivity of the (homogeneous) cross-section
        filling.  Required by the wave-impedance and propagation-
        constant relations.
    field_evaluator : FieldEvaluator or None
        Analytical-path closure that returns the Poynting-normalised
        transverse field profile at user-supplied (u, v) sample points.
        ``None`` on the numerical path.
    z_line : float or None, default None
        Frequency-independent line impedance ``Z₀ = 2P/I·I*`` for
        multi-conductor ports (TEM/QTEM).  ``None`` for hollow-pipe
        modes where no inner-conductor current is available.
        ``z_modal()`` returns this value when set; otherwise falls back
        to ``z_wave(omega)``.
    discrete_e_u_profile : np.ndarray or None, default None
        Numerical-path ``E_u`` profile, sampled at the port-plane
        u-edge midpoints.  Shape ``(N_u,)``.  M_ε-orthonormal by
        construction.
    discrete_e_v_profile : np.ndarray or None, default None
        Numerical-path ``E_v`` profile, shape ``(N_v,)``.
    discrete_h_u_profile : np.ndarray or None, default None
        Numerical-path ``H_u`` profile co-located with v-edges, shape
        ``(N_v,)``.
    discrete_h_v_profile : np.ndarray or None, default None
        Numerical-path ``H_v`` profile co-located with u-edges, shape
        ``(N_u,)``.

    Notes
    -----
    The Phase-1 architecture assumes a homogeneously filled cross-
    section (single ``epsilon_r``).  Phase 2 (QTEM) will extend this to
    a frequency-dependent effective permittivity.
    """

    name: str
    mode_type: ModeType
    omega_c: float
    epsilon_r: float
    field_evaluator: FieldEvaluator | None
    z_line: float | None = None
    discrete_e_u_profile: np.ndarray | None = None
    discrete_e_v_profile: np.ndarray | None = None
    discrete_h_u_profile: np.ndarray | None = None
    discrete_h_v_profile: np.ndarray | None = None

    def __post_init__(self) -> None:
        has_evaluator = self.field_evaluator is not None
        profiles = (
            self.discrete_e_u_profile,
            self.discrete_e_v_profile,
            self.discrete_h_u_profile,
            self.discrete_h_v_profile,
        )
        n_set = sum(p is not None for p in profiles)
        has_all_profiles = n_set == 4
        has_any_profile = n_set > 0

        if has_evaluator and has_any_profile:
            raise ValueError(
                f"Mode '{self.name}': field_evaluator and "
                f"discrete_*_profile fields are mutually exclusive (both "
                f"are set)."
            )
        if not has_evaluator and not has_all_profiles:
            if has_any_profile:
                raise ValueError(
                    f"Mode '{self.name}': numerical path requires all "
                    f"four discrete_*_profile fields to be set "
                    f"(got {n_set} of 4)."
                )
            raise ValueError(
                f"Mode '{self.name}': either field_evaluator or all four "
                f"discrete_*_profile fields must be set."
            )

        if has_all_profiles:
            n_u_e = len(self.discrete_e_u_profile)
            n_u_h = len(self.discrete_h_v_profile)
            n_v_e = len(self.discrete_e_v_profile)
            n_v_h = len(self.discrete_h_u_profile)
            if n_u_e != n_u_h:
                raise ValueError(
                    f"Mode '{self.name}': discrete_e_u_profile and "
                    f"discrete_h_v_profile must share length (got "
                    f"{n_u_e} vs {n_u_h})."
                )
            if n_v_e != n_v_h:
                raise ValueError(
                    f"Mode '{self.name}': discrete_e_v_profile and "
                    f"discrete_h_u_profile must share length (got "
                    f"{n_v_e} vs {n_v_h})."
                )

    def gamma(self, omega: float) -> complex:
        """Propagation constant ``γ = α + jβ`` at angular frequency ``omega``.

        Above cut-off, the mode is propagating and ``γ = j β``.  Below
        cut-off it is evanescent and ``γ = α`` (real).

        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        """
        k_sq = (omega**2) * MU0 * EPS0 * self.epsilon_r
        kc_sq = (self.omega_c**2) * MU0 * EPS0 * self.epsilon_r
        diff = kc_sq - k_sq
        if diff <= 0.0:
            return 1j * math.sqrt(-diff)
        return complex(math.sqrt(diff))

    def z_wave(self, omega: float) -> complex:
        """Modal wave impedance at angular frequency ``omega``.

        - TEM: ``Z = η₀ / √ε_r`` (frequency-independent).
        - TE:  ``Z = j ω μ / γ`` (real positive above cut-off, reactive below).
        - TM:  ``Z = γ / (j ω ε)`` (real positive above cut-off, reactive below).

        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].  Must be > 0.
        """
        if omega <= 0.0:
            raise ValueError("z_wave requires omega > 0")

        eta = ETA0 / math.sqrt(self.epsilon_r)

        if self.mode_type is ModeType.TEM:
            return complex(eta)

        gamma = self.gamma(omega)

        if self.mode_type is ModeType.TE:
            return 1j * omega * MU0 / gamma
        if self.mode_type is ModeType.TM:
            return gamma / (1j * omega * EPS0 * self.epsilon_r)

        raise ValueError(f"Unknown mode_type: {self.mode_type}")

    def z_modal(self, omega: float) -> complex:
        """Reference impedance for power-wave decomposition.

        For multi-conductor modes (TEM/QTEM with ``z_line`` set), returns
        the line impedance ``Z₀ = 2P/I·I*``.  For hollow-pipe modes,
        falls back to ``z_wave(omega)``.

        This is the impedance to use in
        ``V_m^± = (V_m ± Z_modal · I_m) / 2`` for the modal-load
        absorption update.

        Parameters
        ----------
        omega : float
            Angular frequency [rad/s].
        """
        if self.z_line is not None:
            return complex(self.z_line)
        return self.z_wave(omega)
