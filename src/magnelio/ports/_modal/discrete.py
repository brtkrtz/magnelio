"""Discretisation of analytical modes onto a PortPlane.

Samples each :class:`~magnelio.ports._modal.mode.Mode`'s analytical
``field_evaluator`` at the port-plane edge midpoints and re-orthonormalises
the resulting discrete vectors in the FIT ``M_eps`` inner product
(modified Gram-Schmidt in cut-off-ascending order).

B-orthonormalisation convention:

- Profiles store raw analytical field values (V/m for E, A/m for H) at
  edge midpoints, not edge-voltage form.
- Inner product weight is the flat ``M_eps[edge_index]`` directly.
- After Gram-Schmidt, ``Σ_p M_eps[p] · ê_i,p · ê_j,p ≈ δ_ij`` for the
  combined u-edges + v-edges concatenation.
- The same Gram-Schmidt operations are applied to the H profiles, so
  the modal impedance ``E/H`` ratio is preserved (no separate
  orthogonalisation in ``M_mu``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from magnelio.ports._modal.mode import Mode
from magnelio.ports._modal.port_plane import PortPlane


@dataclass(frozen=True)
class DiscreteMode:
    """A :class:`Mode` resampled and B-orthonormalised onto a PortPlane.

    Attributes
    ----------
    mode : Mode
        The original analytical mode.  Carries ``omega_c``, ``epsilon_r``,
        ``mode_type``, and the impedance methods ``z_modal(omega)``,
        ``z_wave(omega)``, ``gamma(omega)``.
    e_u_profile : np.ndarray, shape (N_u,)
        ``E_u`` profile at u-edge midpoints, in V/m.  After
        orthonormalisation the discrete field still carries no
        physical V/m meaning — it is the basis vector in the
        ``M_eps``-weighted Hilbert space.
    e_v_profile : np.ndarray, shape (N_v,)
        ``E_v`` profile at v-edge midpoints.
    h_u_profile : np.ndarray, shape (N_v,)
        ``H_u`` profile at the dual edges co-located with the v-edges.
    h_v_profile : np.ndarray, shape (N_u,)
        ``H_v`` profile at duals co-located with the u-edges.
    """

    mode: Mode
    e_u_profile: np.ndarray
    e_v_profile: np.ndarray
    h_u_profile: np.ndarray
    h_v_profile: np.ndarray


def discretize_modes(
    modes: list[Mode],
    plane: PortPlane,
    m_eps_flat: np.ndarray,
) -> list[DiscreteMode]:
    """Sample analytical modes at port-plane edges and B-orthonormalise.

    Dispatches on the Phase-2 ``Mode`` API extension (Variant B,
    architecture document §2.5):

    - If every mode carries a ``field_evaluator``, runs the Phase-1
      analytical path: sample at edge midpoints, modified Gram-Schmidt
      in the ``M_eps`` inner product.
    - If every mode carries the four ``discrete_*_profile`` arrays,
      runs the Phase-2 numerical pass-through: the profiles are already
      M_ε-orthonormal by construction (the numerical 2D solver builds
      them in this metric natively, Reference §2.2), so they are packed
      into ``DiscreteMode`` without resampling or re-orthonormalisation.

    Mixed lists are not supported and raise ``ValueError``.

    Parameters
    ----------
    modes : list[Mode]
        Modes returned by a :class:`ModeSolver`, sorted by ascending
        ``omega_c``.  This order is preserved during Gram-Schmidt
        (analytical path): the lowest-cut-off mode is unchanged, higher
        modes lose their projection onto lower modes.
    plane : PortPlane
        Port-plane geometry; provides edge midpoints, lengths, and flat
        E/H indices.
    m_eps_flat : np.ndarray, shape (N_edges_global,)
        Diagonal of the FIT ``M_eps`` matrix, indexed by the flat
        E-vector layout.  Only ``m_eps_flat[plane.e_u_indices]`` and
        ``m_eps_flat[plane.e_v_indices]`` are used (analytical path).

    Returns
    -------
    list[DiscreteMode]
        One ``DiscreteMode`` per input mode, in the same order.

    Raises
    ------
    ValueError
        If the input list mixes analytical and numerical modes; if a
        numerical mode's profile shape does not match the port plane's
        edge counts; or, in the analytical path, if a mode produces a
        zero-norm discrete vector after Gram-Schmidt (linearly
        dependent on earlier modes).
    """
    if not modes:
        return []

    is_analytical = [m.field_evaluator is not None for m in modes]
    if all(is_analytical):
        return _discretize_analytical(modes, plane, m_eps_flat)
    if not any(is_analytical):
        return _discretize_numerical(modes, plane)
    raise ValueError(
        "discretize_modes: mode list mixes analytical (field_evaluator) "
        "and numerical (discrete_*_profile) modes.  All modes must come "
        "from the same solver path."
    )


def _discretize_numerical(
    modes: list[Mode],
    plane: PortPlane,
) -> list[DiscreteMode]:
    """Pass-through for modes with discrete_*_profile arrays already filled.

    The numerical 2D mode solver constructs profiles that are
    M_ε-orthonormal in the FIT inner product by construction, so this
    routine validates shapes and packs them into ``DiscreteMode`` without
    any further processing.
    """
    n_u = int(plane.e_u_indices.size)
    n_v = int(plane.e_v_indices.size)
    out: list[DiscreteMode] = []
    for i, m in enumerate(modes):
        e_u = np.asarray(m.discrete_e_u_profile, dtype=float)
        e_v = np.asarray(m.discrete_e_v_profile, dtype=float)
        h_u = np.asarray(m.discrete_h_u_profile, dtype=float)
        h_v = np.asarray(m.discrete_h_v_profile, dtype=float)
        if e_u.shape != (n_u,) or h_v.shape != (n_u,):
            raise ValueError(
                f"Mode {i} ('{m.name}'): u-edge profiles must have shape "
                f"({n_u},); got e_u={e_u.shape}, h_v={h_v.shape}."
            )
        if e_v.shape != (n_v,) or h_u.shape != (n_v,):
            raise ValueError(
                f"Mode {i} ('{m.name}'): v-edge profiles must have shape "
                f"({n_v},); got e_v={e_v.shape}, h_u={h_u.shape}."
            )
        out.append(
            DiscreteMode(
                mode=m,
                e_u_profile=e_u.copy(),
                e_v_profile=e_v.copy(),
                h_u_profile=h_u.copy(),
                h_v_profile=h_v.copy(),
            )
        )
    return out


def _discretize_analytical(
    modes: list[Mode],
    plane: PortPlane,
    m_eps_flat: np.ndarray,
) -> list[DiscreteMode]:
    """Phase-1 path: sample analytical evaluators and B-orthonormalise."""
    # u_edge_uv / v_edge_uv are stored in (u, v) column order.
    u_uv = plane.u_edge_uv
    v_uv = plane.v_edge_uv

    # Sample raw profiles for every mode at the port-plane midpoints.
    # u-edges hold E_u and the co-located dual holds H_v.
    # v-edges hold E_v and the co-located dual holds H_u.
    e_u_list: list[np.ndarray] = []
    e_v_list: list[np.ndarray] = []
    h_u_list: list[np.ndarray] = []
    h_v_list: list[np.ndarray] = []
    for m in modes:
        E_u_at_u, _, _, H_v_at_u = m.field_evaluator(u_uv[:, 0], u_uv[:, 1])
        _, E_v_at_v, H_u_at_v, _ = m.field_evaluator(v_uv[:, 0], v_uv[:, 1])
        e_u_list.append(np.asarray(E_u_at_u, dtype=float).copy())
        e_v_list.append(np.asarray(E_v_at_v, dtype=float).copy())
        h_u_list.append(np.asarray(H_u_at_v, dtype=float).copy())
        h_v_list.append(np.asarray(H_v_at_u, dtype=float).copy())

    me_u = m_eps_flat[plane.e_u_indices]
    me_v = m_eps_flat[plane.e_v_indices]

    # Modified Gram-Schmidt in M_eps inner product on the combined
    # (e_u, e_v) vector.  Apply identical scalings to (h_u, h_v).
    n_modes = len(modes)
    for i in range(n_modes):
        e_u_i = e_u_list[i]
        e_v_i = e_v_list[i]
        h_u_i = h_u_list[i]
        h_v_i = h_v_list[i]
        for j in range(i):
            e_u_j = e_u_list[j]
            e_v_j = e_v_list[j]
            proj = float(np.dot(me_u, e_u_i * e_u_j)) + float(np.dot(me_v, e_v_i * e_v_j))
            if proj != 0.0:
                e_u_i = e_u_i - proj * e_u_j
                e_v_i = e_v_i - proj * e_v_j
                h_u_i = h_u_i - proj * h_u_list[j]
                h_v_i = h_v_i - proj * h_v_list[j]
        norm_sq = float(np.dot(me_u, e_u_i**2)) + float(np.dot(me_v, e_v_i**2))
        if norm_sq <= 0.0:
            raise ValueError(
                f"Mode {i} ({modes[i].name}) is linearly dependent on "
                f"earlier modes (norm² = {norm_sq:.3e}).  Check the mode "
                f"list for duplicates or near-degeneracies."
            )
        scale = 1.0 / math.sqrt(norm_sq)
        e_u_list[i] = e_u_i * scale
        e_v_list[i] = e_v_i * scale
        h_u_list[i] = h_u_i * scale
        h_v_list[i] = h_v_i * scale

    return [
        DiscreteMode(
            mode=modes[i],
            e_u_profile=e_u_list[i],
            e_v_profile=e_v_list[i],
            h_u_profile=h_u_list[i],
            h_v_profile=h_v_list[i],
        )
        for i in range(n_modes)
    ]


def gram_matrix(
    discrete_modes: list[DiscreteMode],
    plane: PortPlane,
    m_eps_flat: np.ndarray,
) -> np.ndarray:
    """Compute the Gram matrix ``G_ij = ⟨ê_i, ê_j⟩_Mε`` for a diagnostic check.

    Should be the identity matrix to within numerical roundoff after a
    successful :func:`discretize_modes` call.
    """
    n = len(discrete_modes)
    me_u = m_eps_flat[plane.e_u_indices]
    me_v = m_eps_flat[plane.e_v_indices]
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            G[i, j] = float(
                np.dot(me_u, discrete_modes[i].e_u_profile * discrete_modes[j].e_u_profile)
            ) + float(np.dot(me_v, discrete_modes[i].e_v_profile * discrete_modes[j].e_v_profile))
    return G
