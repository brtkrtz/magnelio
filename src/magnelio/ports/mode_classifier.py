"""
Waveguide mode classification: TEM, TE, TM, or hybrid.

Classification is based on the relative magnitudes of the longitudinal
(Ez, Hz) versus transverse (Ex, Ey, Hx, Hy) components in the 2D eigenmode.

See spec.md for the threshold definition.
"""

from __future__ import annotations

import numpy as np

_THRESHOLD = 1e-6  # Relative threshold for TE/TM classification


def classify_mode(E_t: np.ndarray, H_t: np.ndarray) -> str:
    """Classify a waveguide mode as TEM, TE, TM, or hybrid.

    Args:
        E_t: Transverse (tangential) E-field mode profile, concatenated as
             [Ex_t, Ey_t, Ez_longitudinal].  The longitudinal component Ez
             is the last portion of the array.
        H_t: Transverse H-field mode profile, similarly structured.

    Returns:
        One of ``'TEM'``, ``'TE'``, ``'TM'``, or ``'hybrid'``.

    Note:
        In the 2D port eigenvalue formulation, the longitudinal (z-directed)
        components Ez and Hz appear as separate DOFs. The split point between
        transverse and longitudinal DOFs is determined by the 2D grid size.
        This function uses a simplified 50/50 split as a placeholder; the
        full implementation passes explicit ``n_transverse`` counts.
    """
    n = len(E_t)
    n_trans = (2 * n) // 3  # approximate: 2/3 transverse, 1/3 longitudinal

    E_trans = E_t[:n_trans]
    E_long = E_t[n_trans:]
    H_trans = H_t[:n_trans]
    H_long = H_t[n_trans:]

    norm_E_trans = float(np.linalg.norm(E_trans))
    norm_E_long = float(np.linalg.norm(E_long))
    norm_H_trans = float(np.linalg.norm(H_trans))
    norm_H_long = float(np.linalg.norm(H_long))

    E_total = norm_E_trans + norm_E_long
    H_total = norm_H_trans + norm_H_long

    rel_Ez = norm_E_long / E_total if E_total > 0 else 0.0
    rel_Hz = norm_H_long / H_total if H_total > 0 else 0.0

    is_TE = rel_Ez < _THRESHOLD  # Ez ≈ 0
    is_TM = rel_Hz < _THRESHOLD  # Hz ≈ 0

    if is_TE and is_TM:
        return "TEM"
    elif is_TE:
        return "TE"
    elif is_TM:
        return "TM"
    else:
        return "hybrid"
