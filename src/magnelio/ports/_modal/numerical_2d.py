"""Numerical 2D mode solver — Phase 2a step 3 (TE/TM via curl-curl eigsh)
plus Phase 2b steps 8 + 9 (TEM and QTEM via Laplace dispatch).

Solves the generalised eigenvalue problem ``K · ê = ω_c² · M · ê`` on the
port plane for TE/TM modes (`reference_architecture_phase2_mode_solver.md`
§2.1 / §3.1), or dispatches to ``solve_tem_laplace`` for the TEM Laplace
shortcut (§3.3, homogeneous filling) or ``solve_qtem_laplace`` for the
QTEM dual-Laplace solve (§3.4 / §7.1, inhomogeneous filling) when
multi-conductor inputs are supplied.

Phase-2a scope: hollow, homogeneously-filled cross-section with PEC
lateral walls (TE/TM eigsh path).  Phase 2b adds two dispatches:
- step 8 (TEM): ``conductor_node_groups`` + ``g_2d`` → routes to
  ``solve_tem_laplace`` (homogeneous filling, single ``epsilon_r``).
- step 9 (QTEM): also supply ``m_eps_2d_vacuum`` → routes to
  ``solve_qtem_laplace`` (inhomogeneous filling; the actual
  ``M`` carries the real ε distribution, the vacuum mass anchors the
  ε_eff ratio).

The K, M operators are produced upstream by ``build_2d_curl_curl``
(`curl_curl_2d.py`, Phase 2a step 1) — they are *the* 3D FIT operators
restricted to the port plane, so the resulting modes are exact
eigenvectors of the FIT 2D operator on the 3D-consistent inner product
(Reference §2.2).  The solver therefore returns Mode objects on the
Phase-2 numerical path: ``field_evaluator = None`` and the four
``discrete_*_profile`` arrays carry the M_ε-orthonormal edge vectors
(`mode.py` §2.5 contract; `discretize_modes` pass-through).

H-profiles carry the dual voltages of the discrete travelling wave
(WP7.2, :func:`~magnelio.ports._modal.tem_laplace.
travelling_wave_h_profiles`):

    h_u = - e_v · (μ₀·normal_dx / Z_modal(f_calc)) / M_μ[h_u face]
    h_v = + e_u · (μ₀·normal_dx / Z_modal(f_calc)) / M_μ[h_v face]

This sets the Poynting forward-orientation ``(E × H) · n̂ > 0``
automatically (Reference §2.4 / §8.1) and, unlike the former pointwise
``±E/Z`` field convention, stays a faithful travelling-wave probe on
non-uniform transversal grids.  The overall scale is later corrected
to the M-weighted ``V_m / I_m = Z_modal`` invariant by
``PortOperatorModal._calibrate_v_i`` (DD-042 / WP7.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from magnelio.constants import C0, EPS0, MU0
from magnelio.mesh.grid import GridLines
from magnelio.ports._modal.mode import (
    Mode,
    ModeType,
)
from magnelio.ports._modal.port_plane import PortPlane
from magnelio.ports._modal.tem_laplace import (
    solve_qtem_laplace,
    solve_tem_laplace,
    travelling_wave_h_profiles,
)


def _arpack_v0(n: int) -> np.ndarray:
    """Deterministic ARPACK start vector for an ``n``-dimensional solve.

    Left to itself ARPACK starts from a random vector, so the converged
    eigenvectors carry a residual that differs run to run.  On a
    degenerate pair that residual is what the cross-projection gates
    measure: the TE/TM-vs-TEM crosstalk on the coax fixture wandered
    over 3.1e-16 … 1.1e-13 across rebuilds of the *same* port (KB-010),
    which is physically zero either way but occasionally crossed a
    1e-12 assertion.  A fixed generic start makes every rebuild
    reproduce the same numbers; the direction is arbitrary but must not
    be structured (a vector of ones can sit orthogonal to a mode of
    interest and starve it).
    """
    return np.random.default_rng(0).standard_normal(n)


@dataclass
class Numerical2DModeSolver:
    """TE/TM mode solver via 2D generalised eigenvalue problem.

    Parameters
    ----------
    plane : PortPlane
        Port-plane geometry (already built via ``PortPlane.from_mesh``).
        Provides the u-/v-edge counts used to split the eigenvector
        and the (u, v)-extent used to estimate the shift-invert sigma.
    K, M : scipy.sparse.csr_matrix
        2D curl-curl stiffness and ε-mass on the primal 2D edges,
        produced by ``build_2d_curl_curl``.  Both must be square of
        size ``primal_2d_indices.size``.
    primal_2d_indices : np.ndarray of int, shape (n_2d,)
        Concatenation of ``plane.e_u_indices`` and ``plane.e_v_indices``,
        as returned by ``build_2d_curl_curl``.  This is the basis
        ordering of the eigenvector ``ê`` (first the u-block, then the
        v-block).
    pec_edge_mask : np.ndarray of bool, optional
        Length ``primal_2d_indices.size``.  ``True`` for primal 2D edges
        that are tangential to a PEC lateral wall and must be
        Dirichlet-eliminated before the eigsh solve.  ``None`` (default)
        means no walls.  PEC edges receive zero in the scattered
        eigenvector.
    epsilon_r : float, default 1.0
        Permittivity of the (homogeneous) cross-section filling.  Used
        to bake the modal impedance and the sigma heuristic.
    mode_type : ModeType, default ModeType.TE
        Stamped on the resulting ``Mode`` objects on the TE/TM eigsh
        path.  ``ModeType.TEM`` is accepted only on the TEM Laplace
        path (i.e. when ``conductor_node_groups`` is set); on the
        TE/TM path it must be ``TE`` or ``TM``.  The K, M pair is the
        same for both; the modal Ohm's law sign-convention differs
        only via Z_TE/Z_TM(omega).
    g_2d : scipy.sparse.csr_matrix, optional
        2D discrete gradient ``(n_2d_edges, n_2d_nodes)`` from
        :func:`build_2d_gradient`.  Required for the TEM/QTEM
        Laplace dispatch (Phase 2b steps 8 + 9); ignored on the
        TE/TM path.
    conductor_node_groups : list of np.ndarray, optional
        ``[ground, signal_1, signal_2, ...]`` lists of *local* 2D
        primal-node indices.  When provided (with at least 2 groups
        and a non-None ``g_2d``), :meth:`solve` dispatches to
        :func:`solve_tem_laplace` (homogeneous filling, §3.3) or
        :func:`solve_qtem_laplace` (inhomogeneous filling, §3.4)
        depending on whether ``m_eps_2d_vacuum`` is also set.
        When ``None``, the TE/TM curl-curl eigsh path runs (Phase 2a
        behaviour, §3.1, unchanged).
    m_eps_2d_vacuum : scipy.sparse.csr_matrix, optional
        Vacuum 2D ε-mass — same geometry as ``M`` but with all
        dielectrics replaced by ``ε ≡ 1``.  When set (alongside
        ``conductor_node_groups`` and ``g_2d``), the QTEM dispatch
        kicks in: :meth:`solve` calls :func:`solve_qtem_laplace`
        which performs a dual Laplace solve to extract ``ε_eff`` and
        ``Z_0``.  When ``None`` and ``conductor_node_groups`` is set,
        the homogeneous TEM dispatch runs.
    m_mu_flat : np.ndarray
        Diagonal of the FIT ``M_μ`` matrix in the flat H-vector
        layout.  Required on every path: the H profiles carry the
        per-face ``1/M_μ`` weights of the discrete travelling wave
        (WP7.2, :func:`~magnelio.ports._modal.tem_laplace.
        travelling_wave_h_profiles`).

    Notes
    -----
    Path dispatch is auto-detected from the inputs (see
    :meth:`_classify`):

    - **QTEM dual-Laplace** path: requires ``conductor_node_groups``
      (≥ 2 groups), ``g_2d``, and ``m_eps_2d_vacuum``.  Runs
      :func:`solve_qtem_laplace`, yielding ``K-1`` modes with
      ``mode_type=ModeType.TEM``, ``epsilon_r=ε_eff``, and
      ``z_line=Z_0``.  Per Reference §7.1 "Standard adequate"; the
      single-frequency error grows linearly with distance from
      ``f_calc`` (out-of-scope IPAE refinement is Phase 3).
    - **TEM Laplace** path: requires ``conductor_node_groups`` (≥ 2
      groups) and ``g_2d``.  Runs :func:`solve_tem_laplace`,
      yielding ``K-1`` modes (``K`` = number of conductor groups);
      ``f_calc`` is unused (TEM is non-dispersive).  ``pec_edge_mask``
      is unused on this path (conductor walls enter via the
      Dirichlet boundary in the Laplace solve).
    - **TE/TM eigsh** path: requires neither ``conductor_node_groups``
      nor ``g_2d``.  Runs the shift-invert ``eigsh`` from §3.1.
      The shift-invert is bracketed below the lowest expected
      physical eigenvalue via a heuristic from the plane dimensions:
      ``sigma ≈ 0.95 · (π · c_eff / L_max)²`` where
      ``L_max = max(L_u, L_v)`` and ``c_eff = c₀ / √ε_r``.  The
      eigvals closest to sigma are the physical TE/TM eigenmodes;
      the gradient null-space at λ=0 is filtered out by a positive
      threshold.  The returned eigenvectors are ``v^T M v = 1``-
      normalised (eigsh's default for the generalised problem),
      matching the M_ε-orthonormal convention of ``DiscreteMode`` —
      no Gram-Schmidt or rescaling needed downstream.
    """

    plane: PortPlane
    K: sp.csr_matrix
    M: sp.csr_matrix
    primal_2d_indices: np.ndarray
    pec_edge_mask: np.ndarray | None = None
    epsilon_r: float = 1.0
    mode_type: ModeType = ModeType.TE
    g_2d: sp.csr_matrix | None = None
    conductor_node_groups: list[np.ndarray] | None = None
    m_eps_2d_vacuum: sp.csr_matrix | None = None
    grid: GridLines | None = None
    L_node: sp.csr_matrix | None = None
    M_node: sp.csr_matrix | None = None
    pec_node_mask: np.ndarray | None = None
    m_mu_flat: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.m_mu_flat is None:
            raise ValueError(
                "m_mu_flat (flat 3D M_μ diagonal) is required: the H "
                "profiles carry per-face 1/M_μ weights so that they "
                "match the discrete travelling wave on non-uniform "
                "transversal grids (WP7.2)."
            )
        n = int(self.primal_2d_indices.size)
        if self.K.shape != (n, n):
            raise ValueError(
                f"K shape {self.K.shape} does not match primal_2d_indices size ({n}, {n})."
            )
        if self.M.shape != (n, n):
            raise ValueError(
                f"M shape {self.M.shape} does not match primal_2d_indices size ({n}, {n})."
            )
        if self.pec_edge_mask is not None:
            mask = np.asarray(self.pec_edge_mask)
            if mask.shape != (n,):
                raise ValueError(
                    f"pec_edge_mask shape {mask.shape} does not match "
                    f"primal_2d_indices size ({n},)."
                )
            if mask.dtype != np.bool_:
                raise ValueError(f"pec_edge_mask must be a boolean array (got {mask.dtype}).")
        if self.epsilon_r <= 0.0:
            raise ValueError("epsilon_r must be positive.")

        # Phase 2b step 8: validate the TEM-path inputs together.  ``g_2d``
        # is shared between the TEM path (which also needs
        # ``conductor_node_groups``) and the TM path (which also needs
        # the node-Laplace inputs).  Reject the case where neither
        # consumer is set up — that pairing is unreachable in :meth:`_classify`.
        has_g = self.g_2d is not None
        has_cg = self.conductor_node_groups is not None
        wants_tm = self.mode_type is ModeType.TM and not has_cg and self.L_node is not None
        if has_g and not has_cg and not wants_tm:
            raise ValueError(
                "g_2d set without a downstream consumer: provide "
                "conductor_node_groups (TEM/QTEM Laplace path) or "
                "L_node + M_node + pec_node_mask (TM eigsh path)."
            )
        if has_cg and not has_g:
            raise ValueError(
                "conductor_node_groups requires g_2d (TEM/QTEM Laplace "
                "path uses g_2d.T · M_eps · g_2d as the stiffness)."
            )
        if has_cg:
            if len(self.conductor_node_groups) < 2:
                raise ValueError(
                    "conductor_node_groups must have at least 2 groups "
                    "(groups[0] = ground, groups[1:] = signal "
                    "conductors) for the TEM/QTEM Laplace path."
                )
            if self.g_2d.shape[0] != n:
                raise ValueError(
                    f"g_2d row count {self.g_2d.shape[0]} does not match "
                    f"primal_2d_indices size ({n})."
                )
            if self.m_eps_2d_vacuum is not None:
                if self.m_eps_2d_vacuum.shape != (n, n):
                    raise ValueError(
                        f"m_eps_2d_vacuum shape "
                        f"{self.m_eps_2d_vacuum.shape} does not match "
                        f"({n}, {n})."
                    )
        elif self.m_eps_2d_vacuum is not None:
            raise ValueError(
                "m_eps_2d_vacuum requires conductor_node_groups (it is "
                "only used by the QTEM Laplace dispatch)."
            )

        # ``mode_type`` validation depends on the dispatched path.  TEM
        # is accepted only on the TEM Laplace path (where it also gets
        # stamped on the resulting Mode by ``solve_tem_laplace``); TE/TM
        # are valid on the eigsh path.
        if self.mode_type is ModeType.TEM:
            if not has_cg:
                raise ValueError(
                    "mode_type=TEM requires conductor_node_groups "
                    "(TEM modes are computed via the Laplace path)."
                )
        elif self.mode_type not in (ModeType.TE, ModeType.TM):
            raise ValueError(f"mode_type must be TE, TM, or TEM (got {self.mode_type}).")

        # TM path: the curl-curl operator (K) only resolves TE modes
        # (TM-mode E_t is a gradient field and lives in K's
        # null-space).  TM mode_type therefore requires the separate
        # normal-E eigenproblem inputs from ``build_2d_tm_curl_curl``.
        if self.mode_type is ModeType.TM and not has_cg:
            missing = [
                name
                for name, val in [
                    ("L_node", self.L_node),
                    ("M_node", self.M_node),
                    ("pec_node_mask", self.pec_node_mask),
                    ("g_2d", self.g_2d),
                ]
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"mode_type=TM on the TE/TM path requires the TM "
                    f"eigenproblem inputs from build_2d_tm_curl_curl; "
                    f"missing: {', '.join(missing)}.  TM modes cannot be "
                    f"solved on the curl-curl operator (their E_t lies in "
                    f"its null-space)."
                )
            n_nodes = int(self.L_node.shape[0])
            if self.M_node.shape != (n_nodes, n_nodes):
                raise ValueError(
                    f"M_node shape {self.M_node.shape} does not match L_node {self.L_node.shape}."
                )
            if self.pec_node_mask.shape != (n_nodes,):
                raise ValueError(
                    f"pec_node_mask shape {self.pec_node_mask.shape} "
                    f"does not match L_node node count ({n_nodes},)."
                )
            if self.g_2d.shape != (n, n_nodes):
                raise ValueError(
                    f"g_2d shape {self.g_2d.shape} does not match "
                    f"({n}, {n_nodes}) implied by primal_2d_indices and "
                    f"L_node."
                )

    def _classify(self) -> str:
        """Return the dispatched mode-class identifier (``"qtem"``,
        ``"tem"``, or ``"te_tm"``).

        Phase 2b steps 8 + 9:
        - QTEM if ``conductor_node_groups`` *and* ``m_eps_2d_vacuum``
          are set (inhomogeneous filling — caller signals it via the
          dual mass matrices).
        - TEM if only ``conductor_node_groups`` is set (homogeneous,
          single ``epsilon_r``).
        - TE/TM otherwise (curl-curl eigsh).
        """
        if self.conductor_node_groups is not None:
            if self.m_eps_2d_vacuum is not None:
                return "qtem"
            return "tem"
        if self.mode_type is ModeType.TM:
            return "tm"
        return "te"

    def solve(
        self,
        n_modes: int,
        f_calc: float = 0.0,
        *,
        sigma: float | None = None,
    ) -> list[Mode]:
        """Compute the lowest ``n_modes`` modes; dispatches on input
        topology (TEM vs. TE/TM).

        Parameters
        ----------
        n_modes : int
            Number of modes to return.  TE/TM path: sorted by ascending
            ``omega_c``.  TEM path: returned in the order of the
            non-ground groups in ``conductor_node_groups``.
        f_calc : float
            Mode-calculation frequency [Hz].  TE/TM path: must be > 0;
            used to bake the H-profile amplitude via
            ``Z_modal(2π f_calc)``.  TEM path: unused (TEM is non-
            dispersive — ``Z_TEM`` is frequency-independent).
        sigma : float, optional
            Shift-invert center for ``eigsh`` [(rad/s)²]; TE/TM path
            only.  Default: heuristic at ``0.95 · (π · c_eff / L_max)²``.

        Returns
        -------
        list[Mode]
            ``n_modes`` modes on the Phase-2 numerical path
            (``field_evaluator=None``, ``discrete_*_profile`` filled).

        Raises
        ------
        ValueError
            If ``n_modes <= 0``; if ``f_calc <= 0`` on the TE/TM path.
        RuntimeError
            If the TEM path is selected and ``n_modes`` exceeds the
            number of independent TEM modes (``K - 1`` for ``K``
            conductor groups).
        """
        if n_modes <= 0:
            raise ValueError("n_modes must be positive.")

        path = self._classify()
        if path == "qtem":
            return self._solve_qtem(n_modes)
        if path == "tem":
            return self._solve_tem(n_modes)
        if path == "tm":
            return self._solve_tm(n_modes, f_calc, sigma=sigma)
        return self._solve_te_tm(n_modes, f_calc, sigma=sigma)

    def _solve_tem(self, n_modes: int) -> list[Mode]:
        """Dispatch to :func:`solve_tem_laplace`; truncate to n_modes.

        ``solve_tem_laplace`` returns ``K - 1`` modes for ``K``
        conductor groups.  Requesting more is a RuntimeError; requesting
        fewer returns a leading prefix in input-group order.
        """
        if self.grid is None:
            raise ValueError(
                "TEM dispatch requires the underlying GridLines (for "
                "tangential-boundary mass correction); pass `grid=` to "
                "Numerical2DModeSolver."
            )
        modes = solve_tem_laplace(
            self.plane,
            self.g_2d,
            self.M,
            self.conductor_node_groups,
            self.epsilon_r,
            grid=self.grid,
            m_mu_flat=self.m_mu_flat,
        )
        if n_modes > len(modes):
            raise RuntimeError(
                f"TEM dispatch: requested {n_modes} modes but the "
                f"conductor setup ({len(self.conductor_node_groups)} "
                f"groups) only supports {len(modes)} mode(s) (= K - 1)."
            )
        return modes[:n_modes]

    def _solve_qtem(self, n_modes: int) -> list[Mode]:
        """Dispatch to :func:`solve_qtem_laplace`; truncate to n_modes.

        ``solve_qtem_laplace`` runs the dual Laplace solve (actual ε
        + vacuum) and returns ``K - 1`` modes with
        ``epsilon_r = ε_eff`` and ``z_line = Z_0``.
        """
        if self.grid is None:
            raise ValueError(
                "QTEM dispatch requires the underlying GridLines (for "
                "tangential-boundary mass correction); pass `grid=` to "
                "Numerical2DModeSolver."
            )
        modes = solve_qtem_laplace(
            self.plane,
            self.g_2d,
            self.M,
            self.m_eps_2d_vacuum,
            self.conductor_node_groups,
            grid=self.grid,
            m_mu_flat=self.m_mu_flat,
        )
        if n_modes > len(modes):
            raise RuntimeError(
                f"QTEM dispatch: requested {n_modes} modes but the "
                f"conductor setup ({len(self.conductor_node_groups)} "
                f"groups) only supports {len(modes)} mode(s) (= K - 1)."
            )
        return modes[:n_modes]

    def _solve_te_tm(
        self,
        n_modes: int,
        f_calc: float,
        *,
        sigma: float | None = None,
    ) -> list[Mode]:
        """TE/TM path: shift-invert ``eigsh`` of the curl-curl operator
        (Phase 2a step 3 — this is the original ``solve`` body).
        """
        if f_calc <= 0.0:
            raise ValueError("f_calc must be positive.")

        n = int(self.primal_2d_indices.size)
        if self.pec_edge_mask is not None:
            free = np.where(~np.asarray(self.pec_edge_mask))[0]
        else:
            free = np.arange(n)

        if free.size == 0:
            raise RuntimeError("All edges masked as PEC — no free degrees of freedom.")

        K_f = self.K.tocsr()[free, :][:, free]
        M_f = self.M.tocsr()[free, :][:, free]

        if sigma is None:
            sigma = self._sigma_heuristic()

        # Ask for more eigenvalues than needed to cover (a) the gradient
        # null-space at λ=0 and (b) any nearly-degenerate physical modes
        # close to sigma.  ARPACK requires k < n - 1 in shift-invert mode.
        # How many of the k pairs land in the null-space depends on
        # ARPACK's start vector, so retry with doubled k before
        # giving up (observed: n_modes=5 on WR-90 occasionally converges
        # only 4 physical pairs at the first k).
        n_total = min(max(n_modes + 8, 2 * n_modes), K_f.shape[0] - 2)
        if n_total < n_modes:
            raise RuntimeError(
                f"Free DoF count ({K_f.shape[0]}) too small to request {n_modes} modes via eigsh."
            )

        # Filter the gradient null-space.  Threshold at 1e-3 of the
        # heuristic sigma — physical eigenvalues are O(sigma); null-space
        # entries are O(numerical-eps · ||K||).
        threshold = 1e-3 * sigma
        k_max = K_f.shape[0] - 2
        while True:
            eigvals, eigvecs = eigsh(
                K_f, k=n_total, M=M_f, sigma=sigma, v0=_arpack_v0(K_f.shape[0])
            )
            physical = eigvals > threshold
            eigvals_p = eigvals[physical]
            eigvecs_p = eigvecs[:, physical]
            if eigvals_p.size >= n_modes or n_total >= k_max:
                break
            n_total = min(2 * n_total, k_max)

        # Sort by eigenvalue and pick the n_modes lowest physical modes.
        order = np.argsort(eigvals_p)
        eigvals_p = eigvals_p[order]
        eigvecs_p = eigvecs_p[:, order]

        if eigvals_p.size < n_modes:
            raise RuntimeError(
                f"Numerical2DModeSolver: only {eigvals_p.size} physical "
                f"modes converged above the null-space threshold "
                f"({threshold:.3e}); requested {n_modes}.  Increase "
                f"n_total or override sigma=."
            )

        eigvals_p = eigvals_p[:n_modes]
        eigvecs_p = eigvecs_p[:, :n_modes]

        # Scatter free-DoF eigenvectors back to the full 2D basis.
        eigvecs_full = np.zeros((n, n_modes), dtype=float)
        eigvecs_full[free, :] = eigvecs_p

        # Port-symmetric sign-flip via global tangential coordinates
        # (see ``_resolve_sign`` for why ``argmax(|v|)`` is unsafe for
        # modes with two equal-magnitude antiphase peaks like TE20).
        for i in range(n_modes):
            eigvecs_full[:, i] = self._resolve_sign(eigvecs_full[:, i])

        # Split eigenvector into u- and v-blocks (the order of
        # primal_2d_indices is [e_u_indices, e_v_indices]).
        n_u = int(self.plane.e_u_indices.size)
        n_v = int(self.plane.e_v_indices.size)
        if n_u + n_v != n:
            raise RuntimeError(
                f"Primal 2D index count mismatch: e_u({n_u}) + e_v({n_v}) "
                f"!= primal_2d_indices size ({n})."
            )

        omega_calc = 2.0 * math.pi * f_calc
        modes: list[Mode] = []
        for i in range(n_modes):
            e_u = eigvecs_full[:n_u, i].copy()
            e_v = eigvecs_full[n_u : n_u + n_v, i].copy()
            omega_c = float(math.sqrt(eigvals_p[i]))
            z_real = self._z_real_at(
                self.mode_type,
                omega_calc,
                omega_c,
                self.epsilon_r,
            )
            # Travelling-wave dual voltages (WP7.2): modal Ohm's law
            # H_t = (1/Z) · ẑ × E_t in the (u, v, n̂) frame with
            # per-face 1/M_μ weights.
            h_u, h_v = travelling_wave_h_profiles(
                e_u,
                e_v,
                self.plane,
                self.m_mu_flat,
                z_real,
            )
            modes.append(
                Mode(
                    name=self._label(self.mode_type, i),
                    mode_type=self.mode_type,
                    omega_c=omega_c,
                    epsilon_r=self.epsilon_r,
                    field_evaluator=None,
                    z_line=None,
                    discrete_e_u_profile=e_u,
                    discrete_e_v_profile=e_v,
                    discrete_h_u_profile=h_u,
                    discrete_h_v_profile=h_v,
                )
            )
        return modes

    def _solve_tm(
        self,
        n_modes: int,
        f_calc: float,
        *,
        sigma: float | None = None,
    ) -> list[Mode]:
        """TM path: shift-invert ``eigsh`` of the normal-E eigenproblem.

        TM modes are characterised by a non-zero longitudinal ``E_z``;
        their transverse ``E_t ∝ ∇_t E_z`` is a gradient field — i.e.
        exactly in the null-space of the curl-curl operator used by
        ``_solve_te_tm`` — which is why TM modes need their own scalar
        eigenproblem to be discoverable at all.  ``L_node`` / ``M_node``
        carry the *exact* discrete form from
        :func:`~magnelio.ports._modal.curl_curl_2d.build_2d_tm_curl_curl`
        (the index-sliced restriction of the 3D FIT operators onto the
        port slab's normal-E edges, 1:1 with the plane's primal
        nodes): the eigenvalue is the discrete cut-off ``ω̂_c²`` of the
        3D update — the Klein-Gordon mass of the separated
        longitudinal chain (WP-R3) — and the eigenvector's topological
        gradient is the exact transversal profile.

        Parameters and conventions match ``_solve_te_tm``: the result
        is a list of ``Mode`` objects with the four ``discrete_*_profile``
        arrays filled and ``M_eps_2d``-orthonormal, ready for
        ``discretize_modes`` pass-through.
        """
        if f_calc <= 0.0:
            raise ValueError("f_calc must be positive.")

        n_nodes = int(self.L_node.shape[0])
        free = np.where(~np.asarray(self.pec_node_mask))[0]
        if free.size == 0:
            raise RuntimeError(
                "All 2D primal nodes masked as PEC — no free degrees of "
                "freedom for the TM eigenproblem."
            )

        L_ff = self.L_node.tocsr()[free, :][:, free]
        M_ff = self.M_node.tocsr()[free, :][:, free]

        if sigma is None:
            sigma = self._sigma_heuristic_tm()

        n_total = min(max(n_modes + 4, 2 * n_modes), L_ff.shape[0] - 2)
        if n_total < n_modes:
            raise RuntimeError(
                f"Free TM-node count ({L_ff.shape[0]}) too small to "
                f"request {n_modes} modes via eigsh."
            )

        eigvals, eigvecs = eigsh(L_ff, k=n_total, M=M_ff, sigma=sigma, v0=_arpack_v0(L_ff.shape[0]))

        # The Dirichlet TM eigenproblem is positive-definite — no
        # gradient null-space here.  Sort eigenvalues ascending and
        # keep the n_modes lowest.  A defensive non-positive-eigenvalue
        # guard catches catastrophic numerical breakdown.
        if np.any(eigvals <= 0.0):
            raise RuntimeError(
                f"TM eigsh returned non-positive eigenvalues "
                f"({eigvals[eigvals <= 0.0]}); the Dirichlet "
                f"node-Laplace is mathematically positive-definite, so "
                f"this indicates a catastrophic numerical breakdown."
            )

        order = np.argsort(eigvals)
        eigvals = eigvals[order][:n_modes]
        eigvecs = eigvecs[:, order][:, :n_modes]

        # Scatter free-DoF eigenvectors back to the full 2D-node basis.
        phi_full = np.zeros((n_nodes, n_modes), dtype=float)
        phi_full[free, :] = eigvecs

        # Reconstruct transverse E-field via E_t = G_2d · φ (continuous
        # equivalent: E_t ∝ -∇_t E_z — the constant prefactor falls out
        # under M_eps-orthonormalisation).  Normalise directly against
        # the curl-curl-path M_eps_2d so the resulting profile lives in
        # the same metric as the TE modes (per ``DiscreteMode``'s
        # M_eps-orthonormal contract).
        e_t_full = self.g_2d @ phi_full  # (n_2d_edges, n_modes)
        omega_c_arr = np.sqrt(eigvals)
        m_eps_2d_diag = self.M.diagonal()
        norm_sq = np.einsum(
            "ij,i,ij->j",
            e_t_full,
            m_eps_2d_diag,
            e_t_full,
        )
        if np.any(norm_sq <= 0.0):
            raise RuntimeError(
                f"TM mode reconstruction yielded a non-positive M_eps "
                f"norm ({norm_sq[norm_sq <= 0.0]}); the gradient field "
                f"of a non-trivial Dirichlet eigenmode should be "
                f"positive-definite under M_eps."
            )
        e_t_full = e_t_full / np.sqrt(norm_sq)[np.newaxis, :]

        # Port-symmetric sign-flip via global tangential coordinates.
        for i in range(n_modes):
            e_t_full[:, i] = self._resolve_sign(e_t_full[:, i])

        n_u = int(self.plane.e_u_indices.size)
        n_v = int(self.plane.e_v_indices.size)
        n_2d_edges = int(self.primal_2d_indices.size)
        if n_u + n_v != n_2d_edges:
            raise RuntimeError(
                f"Primal 2D edge split mismatch: e_u({n_u}) + e_v({n_v}) "
                f"!= primal_2d_indices size ({n_2d_edges})."
            )

        omega_calc = 2.0 * math.pi * f_calc
        modes: list[Mode] = []
        for i in range(n_modes):
            e_u = e_t_full[:n_u, i].copy()
            e_v = e_t_full[n_u : n_u + n_v, i].copy()
            omega_c = float(omega_c_arr[i])
            z_real = self._z_real_at(
                ModeType.TM,
                omega_calc,
                omega_c,
                self.epsilon_r,
            )
            h_u, h_v = travelling_wave_h_profiles(
                e_u,
                e_v,
                self.plane,
                self.m_mu_flat,
                z_real,
            )
            modes.append(
                Mode(
                    name=self._label(ModeType.TM, i),
                    mode_type=ModeType.TM,
                    omega_c=omega_c,
                    epsilon_r=self.epsilon_r,
                    field_evaluator=None,
                    z_line=None,
                    discrete_e_u_profile=e_u,
                    discrete_e_v_profile=e_v,
                    discrete_h_u_profile=h_u,
                    discrete_h_v_profile=h_v,
                )
            )
        return modes

    def _sigma_heuristic_tm(self) -> float:
        """Sigma estimate just below the lowest TM mode of a hollow WG.

        For a hollow rectangular WG with PEC walls, the lowest TM
        cut-off is TM11 with
        ``ω_c² = (π·c_eff/L_u)² + (π·c_eff/L_v)²``.  We bracket sigma
        at 0.95² of that — well clear of the (positive-definite)
        spectrum's lower edge but close enough for fast eigsh
        convergence to TM11 / TM21 / ...
        """
        u_coords = self.plane.u_edge_uv
        v_coords = self.plane.v_edge_uv
        L_u = float(np.max(u_coords[:, 0]) - np.min(u_coords[:, 0]))
        L_v = float(np.max(v_coords[:, 1]) - np.min(v_coords[:, 1]))
        if L_u <= 0.0 or L_v <= 0.0:
            raise RuntimeError(
                "PortPlane has zero u/v extent — TM sigma heuristic "
                "cannot estimate a cut-off scale."
            )
        c_eff = C0 / math.sqrt(self.epsilon_r)
        omega_c_sq = (math.pi * c_eff / L_u) ** 2 + (math.pi * c_eff / L_v) ** 2
        return 0.95 * 0.95 * omega_c_sq

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_sign(self, v: np.ndarray) -> np.ndarray:
        """Port-symmetric sign convention for an eigenvector.

        The previous ``argmax(|v|)``-based flip is unstable for modes
        with two equal-magnitude antiphase peaks (TE20 has
        sin(2πu/a) — two extrema at ±1).  Port 1 and Port 2 with
        swapped local (u, v) ordering pick *different* local indices,
        so the sign decision differs between the two ports even for
        the same physical mode → reciprocity fails by ±2 (≈+6 dB).

        The replacement uses a port-symmetric criterion built from
        edge-midpoint coordinates in the **global** tangential frame:
        ``(coord_lo, coord_hi)`` where ``lo`` / ``hi`` are the
        lower-numbered / higher-numbered global axes the port plane
        spans (e.g. (y, z) on X_MIN and X_MAX alike).  A short cascade
        of test functions (linear, bilinear, quadratic) is tried in
        order; the first one with non-trivial overlap fixes the sign.
        """
        if self.plane.face.u_axis < self.plane.face.v_axis:
            u_lo = self.plane.u_edge_uv[:, 0]
            u_hi = self.plane.u_edge_uv[:, 1]
            v_lo = self.plane.v_edge_uv[:, 0]
            v_hi = self.plane.v_edge_uv[:, 1]
        else:
            u_lo = self.plane.u_edge_uv[:, 1]
            u_hi = self.plane.u_edge_uv[:, 0]
            v_lo = self.plane.v_edge_uv[:, 1]
            v_hi = self.plane.v_edge_uv[:, 0]
        coord_lo = np.concatenate([u_lo, v_lo])
        coord_hi = np.concatenate([u_hi, v_hi])
        lo_mid = 0.5 * (coord_lo.max() + coord_lo.min())
        hi_mid = 0.5 * (coord_hi.max() + coord_hi.min())
        dlo = coord_lo - lo_mid
        dhi = coord_hi - hi_mid

        v_max = float(np.max(np.abs(v))) if v.size else 1.0
        for f in (dlo, dhi, dlo * dhi, dlo * dlo + dhi * dhi):
            f_max = float(np.max(np.abs(f))) if f.size else 1.0
            overlap = float(np.sum(v * f))
            tol = 1e-9 * v_max * f_max * max(v.size, 1)
            if abs(overlap) > tol:
                return -v if overlap < 0.0 else v
        # Final fallback: original argmax convention.
        j_max = int(np.argmax(np.abs(v)))
        return -v if v[j_max] < 0.0 else v

    def _sigma_heuristic(self) -> float:
        """Sigma estimate just below the lowest TE/TM mode of a hollow WG.

        For a hollow rectangular WG with PEC walls, the lowest cut-off
        is ``omega_c_min = π · c_eff / L_max`` with
        ``c_eff = c₀ / √ε_r``.  We bracket sigma at 0.95² of that to
        keep ARPACK's shift-invert close to the physical band but
        clear of λ=0.
        """
        u_coords = self.plane.u_edge_uv
        v_coords = self.plane.v_edge_uv
        L_u = float(np.max(u_coords[:, 0]) - np.min(u_coords[:, 0]))
        L_v = float(np.max(v_coords[:, 1]) - np.min(v_coords[:, 1]))
        L_max = max(L_u, L_v)
        if L_max <= 0.0:
            raise RuntimeError(
                "PortPlane has zero extent in both u and v — sigma "
                "heuristic cannot estimate a cut-off scale."
            )
        c_eff = C0 / math.sqrt(self.epsilon_r)
        omega_c_est = math.pi * c_eff / L_max
        return 0.95 * 0.95 * omega_c_est * omega_c_est

    @staticmethod
    def _z_real_at(
        mode_type: ModeType,
        omega: float,
        omega_c: float,
        eps_r: float,
    ) -> float:
        """Real-valued ``|Z|`` at omega for H-profile bake-in.

        Above cut-off, ``Z`` is real positive; below cut-off, ``Z`` is
        purely imaginary, in which case we return its magnitude.  Same
        convention as ``RectWGAnalyticalModeSolver._z_real_at``.
        """
        k_sq = (omega**2) * MU0 * EPS0 * eps_r
        kc_sq = (omega_c**2) * MU0 * EPS0 * eps_r
        diff = kc_sq - k_sq
        if abs(diff) < 1e-30:
            raise ValueError(
                "f_calc coincides with a mode cut-off; H-profile bake-in "
                "is singular.  Choose f_calc strictly different from any "
                "included mode's cut-off."
            )
        sqrt_abs = math.sqrt(abs(diff))
        if mode_type is ModeType.TE:
            return omega * MU0 / sqrt_abs
        return sqrt_abs / (omega * EPS0 * eps_r)

    @staticmethod
    def _label(mode_type: ModeType, i: int) -> str:
        """Generic mode label.  The eigsh order is not (m, n)-mappable.

        Phase-2b will introduce mode-classification heuristics that map
        eigenvectors to (m, n) labels; for Phase-2a we tag them with
        the eigsh ordering only.
        """
        prefix = "TE" if mode_type is ModeType.TE else "TM"
        return f"{prefix}_num{i:02d}"
