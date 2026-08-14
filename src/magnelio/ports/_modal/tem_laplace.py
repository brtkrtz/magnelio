"""TEM and QTEM mode solvers via 2D Laplace + conductor-potential BCs.

Phase 2b steps 7 + 9 (`reference_architecture_phase2_mode_solver.md`
§3.3 / §3.4 / §5).

For a multi-conductor cross-section, TEM has ``ω_c = 0`` and the
curl-curl eigenvalue solve is degenerate at the lowest eigenvalue (the
gradient null-space).  The Laplace formulation skips the eigenvalue
problem entirely:

    ∇·(ε ∇φ_k) = 0       in the cross-section interior
    φ_k = +1 V           on conductor k
    φ_k =  0 V           on all other conductors

For ``K`` conductor groups (groups[0] = ground, groups[1..K-1] = signal
conductors) both functions return ``K - 1`` independent modes, one per
non-ground conductor, with ``ê_k = -∇φ_k`` on the primal 2D edges.

FIT discrete form: with ``G_2d`` the 2D primal-gradient (nodes → edges,
±1) and ``M_ε,2D`` the 2D ε-mass on primal 2D edges, the Laplace
stiffness is ``L = G_2d^T · M_ε,2D · G_2d``.  Conductor nodes are
Dirichlet-eliminated; the free-DoF problem is solved via
``scipy.sparse.linalg.spsolve``.  The shared per-signal-mode solve and
sign convention are factored into the private
:func:`_solve_signal_modes_laplace` helper.

Two public entry points:

- :func:`solve_tem_laplace` (Phase 2b step 7) — homogeneous filling,
  single ``epsilon_r`` scalar.  Returns Mode objects with
  ``z_line = √(με) / C'`` (per-mode line impedance, frequency-
  independent) and travelling-wave H profiles at
  ``Z_TEM = η₀/√ε_r``.  Labels ``"TEM_lap{i:02d}"``.
- :func:`solve_qtem_laplace` (Phase 2b step 9) — inhomogeneous filling
  ("Standard adequate" QTEM, Reference §7.1).  Caller supplies two
  mass matrices: the actual ``m_eps_2d`` and the vacuum
  ``m_eps_2d_vacuum`` (same geometry, all dielectrics replaced by
  vacuum).  Returns Mode objects with ``epsilon_r = ε_eff`` and
  ``z_line = Z_0 = 1 / (c · √(C' · C'_0))``, where ``ε_eff = C' / C'_0``
  and the H profiles use the effective wave impedance
  ``Z_TEM_eff = η₀/√ε_eff``.  Labels ``"QTEM_lap{i:02d}"``.

Output convention matches :class:`Numerical2DModeSolver` (Phase 2a step
3) so the downstream pipeline (``discretize_modes`` pass-through →
``PortOperatorModal._calibrate_v_i`` → ``compute_s_parameters``) needs
no changes:

- ``Mode.field_evaluator = None``;
- the four ``discrete_*_profile`` arrays carry M_ε-orthonormal edge
  vectors (``ê^T · M_ε,2D · ê = 1``);
- the H-profile carries the dual voltages of the exact discrete
  travelling wave (WP7.2), built by
  :func:`travelling_wave_h_profiles` in the (u, v, n̂) frame:
  ``h_u = -e_v · (μ₀·normal_dx/Z_TEM_*) / M_μ[h_u face]``,
  ``h_v = +e_u · (μ₀·normal_dx/Z_TEM_*) / M_μ[h_v face]``;
- ``Mode.z_line`` carries the DD-025 power-current line impedance
  ``Z_line,k = 2 P_k / |I_k|²``.

Z_line derivation (TEM, homogeneous): with the gauge ``V_k = +1 V`` and
``ê_raw = -G φ`` (unnormalised), the per-length capacitance reads
``C'_k = ê_raw^T M_ε,2D ê_raw / normal_dx``.  Note the FIT mass at a
port-plane Ey-edge is ``M_ε[i=0,j,k] = ε · dx_avg[0] · dz_avg[k] / dy[j]``
with ``dx_avg[0] = dx[0]`` (full boundary cell, *not* half — see
``operators/material_matrices.py::_build_avg_d``), so
``ê^T M_ε,2D ê ≈ ε · normal_dx · ∫∫|E|² dA`` without a 1/2 prefactor.
Then for homogeneous TEM,
``Z_line,k = √(με) / C'_k = √(ε_r) · normal_dx /
(c · ê_raw^T M_ε,2D ê_raw)``.  The Coax check
((η/2π)·ln(r_o/r_i)) recovers analytically up to the discretisation
error of the staircased radial geometry.

Z_0 / ε_eff derivation (QTEM, inhomogeneous): solve the Laplace
problem twice — once with the actual ε distribution (yielding
``C' = norm_sq_actual / normal_dx``), once with all materials replaced
by vacuum (yielding ``C'_0 = norm_sq_vacuum / normal_dx``).  Then
``ε_eff = C' / C'_0`` and ``Z_0 = 1 / (c · √(C' · C'_0))``, the standard
QTEM closed-form (Pozar §11.7, Hammerstad-Jensen reference for
microstrip).  Phase velocity ``v_eff = 1/√(L'C')`` with non-magnetic
``L' = L'_0 = μ_0 · ε_0 / C'_0`` recovers ``v_eff = c/√ε_eff``.

Sign convention: per architecture §2.4 / §7, the integral of ``ê·n̂``
across signal conductor k is required positive ("+ pin" convention).
With the natural gauge ``V_k = +1 V``, the natural ``ê = -∇φ_k``
already satisfies this — the conductor-flux check below would only
flip in pathological gauges and serves as a robustness rail.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from magnelio.constants import C0, ETA0, MU0
from magnelio.mesh.grid import GridLines
from magnelio.ports._modal.mode import Mode, ModeType
from magnelio.ports._modal.port_plane import PortPlane


def travelling_wave_h_profiles(
    e_u: np.ndarray,
    e_v: np.ndarray,
    plane: PortPlane,
    m_mu_flat: np.ndarray,
    z_wave: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Dual-voltage H profiles of the exact discrete travelling wave.

    The pointwise field convention ``H = ±E / Z`` is exact for *fields*
    but wrong for FIT face *voltages* on non-uniform transversal grids
    (WP7.1 spike, session 80): the discrete travelling wave's dual
    voltages obey, per co-located (e, h) pair,

        h_u = -e_v · (μ₀ · normal_dx / Z) / M_μ[h_u face]
        h_v = +e_u · (μ₀ · normal_dx / Z) / M_μ[h_v face]

    verified to machine precision against the exact leapfrog travelling
    wave on uniform *and* graded transversal grids (TEM and QTEM).  On
    a grid where ``M_μ[face] = μ₀ · normal_dx`` this reduces to the
    pointwise form, which is why the mismatch never showed on the
    historical (transversally uniform) benchmarks.

    Parameters
    ----------
    e_u, e_v : np.ndarray
        Transversal E profile on the plane's u-/v-edges (any scaling).
    plane : PortPlane
        Port-plane geometry; supplies ``normal_dx`` and the co-located
        dual-face index arrays (``e_u[k]`` pairs with ``h_v[k]``,
        ``e_v[k]`` with ``h_u[k]``).
    m_mu_flat : np.ndarray
        Diagonal of the FIT ``M_μ`` matrix in the flat H-vector layout.
    z_wave : float
        Real wave impedance of the mode: ``η₀/√ε_eff`` for TEM/QTEM,
        ``Z_TE/Z_TM(ω_calc)`` magnitude for hollow-pipe modes.  For
        TEM this makes the scale ``normal_dx·√ε_eff/c`` — frequency-
        independent and exact for the leapfrog wave at *every*
        frequency; for TE/TM the residual is the O(dispersion) gap
        between the continuous and discrete ``β(ω_calc)``.

    Returns
    -------
    (h_u, h_v) : tuple of np.ndarray
        H profiles co-located with the v-/u-edges respectively.
    """
    if z_wave <= 0.0:
        raise ValueError("z_wave must be positive.")
    m_mu_flat = np.asarray(m_mu_flat, dtype=float)
    scale = MU0 * float(plane.normal_dx) / float(z_wave)
    # M_μ = 0 marks enlarged-cell-donated faces (WP-R5, ~fully inside
    # PEC): the solver freezes h there exactly, so the travelling-wave
    # profile is 0 on those faces, not the 1/M_μ limit.
    mu_u = m_mu_flat[plane.h_u_indices]
    mu_v = m_mu_flat[plane.h_v_indices]
    h_u = np.where(
        mu_u > 0,
        -e_v * (scale / np.where(mu_u > 0, mu_u, 1.0)),
        0.0,
    )
    h_v = np.where(
        mu_v > 0,
        +e_u * (scale / np.where(mu_v > 0, mu_v, 1.0)),
        0.0,
    )
    return h_u, h_v


def _window_boundary_factor(
    deltas: np.ndarray,
    node_window: tuple[int, int],
    at_lo: bool,
    *,
    magnetic_wall: bool = False,
) -> float:
    """Mass-correction factor for one window boundary along one axis.

    ``deltas`` are the primal cell sizes of the axis, ``node_window``
    the plane's inclusive node-index window.  At a bbox boundary the 3D
    dual is the full boundary cell; the 2D capacitance window owns

    - ``d_in/2`` up to the outermost grid line → factor ``0.5`` for a
      wall ON that line (PEC: the tangential edges there are masked
      anyway, so the factor is inert), but
    - the full boundary cell when the face is a magnetic wall → factor
      ``1.0``: the natural magnetic wall of the staggered grid sits half
      the outer dual cell BEYOND the outermost line (both declaration
      paths — the mesher's pulled-in line places that wall on the bbox
      face, a post-meshing declaration leaves it half a cell outside),
      and the mirror-symmetric field continuation up to the wall carries
      exactly the outer half-cell of energy.  Without this the reported
      ``C'`` misses one half-cell strip per magnetic window edge and
      ``z_line`` is biased O(h) — measured ``η0·b/(a - 2d/3)`` instead
      of ``η0·b/a`` on the parallel-plate line, error ``2/(3·Nx)``.

    At an interior window boundary the 3D dual is ``(d_out + d_in)/2``
    while the 2D window owns ``d_in/2`` → factor
    ``d_in / (d_out + d_in)``.
    """
    lo, hi = node_window
    if at_lo:
        if lo == 0:
            return 1.0 if magnetic_wall else 0.5
        d_out, d_in = float(deltas[lo - 1]), float(deltas[lo])
    else:
        if hi == deltas.size:
            return 1.0 if magnetic_wall else 0.5
        d_out, d_in = float(deltas[hi]), float(deltas[hi - 1])
    return d_in / (d_out + d_in)


_FACE_NAMES = (("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax"))


def _is_magnetic_face(boundary_conditions, axis: int, at_lo: bool) -> bool:
    """True when the bbox face of ``axis`` (lo/hi side) is a PMC wall."""
    if boundary_conditions is None:
        return False
    face = _FACE_NAMES[axis][0 if at_lo else 1]
    return getattr(boundary_conditions, face, None) == "PMC"


def _tangential_boundary_factor(
    plane: PortPlane,
    grid: GridLines,
    n_2d_edges: int,
    boundary_conditions=None,
) -> np.ndarray:
    """Multiplicative correction for the 2D Mass at tangential bbox edges.

    The 3D ``_build_avg_d`` convention sets ``d[0]`` (full cell) at every
    bbox boundary index — correct for the port-plane *normal* axis (the
    modal-port operator's V/I projection relies on this), but **wrong**
    for the two *tangential* bbox axes, where the capacitance window
    only owns part of that cell.  Without this correction a free
    (non-Dirichlet) tangential boundary edge contributes twice its true
    energy to ``ê^T M_ε,2D ê``, biasing every line integral by a factor
    of ``(N+1)/N``.

    Returns an array of shape ``(n_2d_edges,)``: ``1.0`` for interior
    edges, and a boundary factor for edges whose endpoints span a
    plane-boundary node.  At a bbox-boundary edge the factor is ``0.5``
    (half the boundary cell) — except on a declared PMC face, where the
    magnetic wall sits half the outer dual cell beyond the outermost
    grid line and the full boundary cell belongs to the window (factor
    ``1.0``); see :func:`_window_boundary_factor`.  At an *interior*
    window boundary of a sub-face plane the 3D value is the full
    interior dual ``(d_out + d_in)/2`` while the window's 2D problem
    only owns the inner half ``d_in/2`` — factor
    ``d_in / (d_out + d_in)`` (``0.5`` on uniform grids).  Multiply
    onto the diagonal of the sliced ``M_ε,2D`` before building the
    Laplace stiffness.
    """
    deltas = (grid.dx, grid.dy, grid.dz)
    u_axis = plane.face.u_axis
    v_axis = plane.face.v_axis

    # Raster of e_u edges (per ``_build_uv_edges`` in port_plane.py):
    # (N_cells_u, N_nodes_v) with meshgrid("ij") → flat[p, s] = p*N_s + s.
    # Tangential boundary on the v-axis: s ∈ {0, N_nodes_v - 1}.
    n_cells_u = plane.n_cells_u
    n_nodes_v = plane.n_nodes_v
    fac_u = np.ones((n_cells_u, n_nodes_v))
    fac_u[:, 0] = _window_boundary_factor(
        deltas[v_axis],
        plane.v_node_window,
        at_lo=True,
        magnetic_wall=_is_magnetic_face(boundary_conditions, v_axis, at_lo=True),
    )
    fac_u[:, -1] = _window_boundary_factor(
        deltas[v_axis],
        plane.v_node_window,
        at_lo=False,
        magnetic_wall=_is_magnetic_face(boundary_conditions, v_axis, at_lo=False),
    )

    # Raster of e_v edges: (N_cells_v, N_nodes_u).
    # Tangential boundary on the u-axis: s ∈ {0, N_nodes_u - 1}.
    n_cells_v = plane.n_cells_v
    n_nodes_u = plane.n_nodes_u
    fac_v = np.ones((n_cells_v, n_nodes_u))
    fac_v[:, 0] = _window_boundary_factor(
        deltas[u_axis],
        plane.u_node_window,
        at_lo=True,
        magnetic_wall=_is_magnetic_face(boundary_conditions, u_axis, at_lo=True),
    )
    fac_v[:, -1] = _window_boundary_factor(
        deltas[u_axis],
        plane.u_node_window,
        at_lo=False,
        magnetic_wall=_is_magnetic_face(boundary_conditions, u_axis, at_lo=False),
    )

    n_u = n_cells_u * n_nodes_v
    if n_u + n_cells_v * n_nodes_u != n_2d_edges:
        raise ValueError(
            f"port-plane edge basis ({n_u} + {n_cells_v * n_nodes_u}) does "
            f"not match n_2d_edges = {n_2d_edges}."
        )

    out = np.empty(n_2d_edges)
    out[:n_u] = fac_u.ravel()
    out[n_u:] = fac_v.ravel()
    return out


def _correct_boundary_mass(
    m_eps_2d: sp.csr_matrix,
    plane: PortPlane,
    grid: GridLines,
    boundary_conditions=None,
) -> sp.csr_matrix:
    """Return ``m_eps_2d`` with corrected tangential window-boundary entries."""
    n_edges = int(m_eps_2d.shape[0])
    factor = _tangential_boundary_factor(plane, grid, n_edges, boundary_conditions)
    return sp.diags(m_eps_2d.diagonal() * factor, format="csr")


def solve_tem_laplace(
    plane: PortPlane,
    g_2d: sp.csr_matrix,
    m_eps_2d: sp.csr_matrix,
    conductor_node_groups: list[np.ndarray],
    epsilon_r: float,
    grid: GridLines,
    *,
    m_mu_flat: np.ndarray,
    boundary_conditions=None,
) -> list[Mode]:
    """Compute K-1 TEM modes via 2D Laplace with conductor-potential BCs.

    Parameters
    ----------
    plane : PortPlane
        Port-plane geometry.  Used for ``e_u_indices`` / ``e_v_indices``
        sizes (to split the edge eigenvector into u/v blocks) and for
        ``normal_dx`` (Z_line formula).
    g_2d : scipy.sparse.csr_matrix, shape (n_2d_edges, n_2d_nodes)
        2D discrete gradient (primal 2D nodes → primal 2D edges) from
        :func:`build_2d_gradient`.  Topological ±1 entries.
    m_eps_2d : scipy.sparse.csr_matrix, shape (n_2d_edges, n_2d_edges)
        Diagonal 2D ε-mass on the primal 2D edges, identical to the
        ``M`` returned by :func:`build_2d_curl_curl`.
    conductor_node_groups : list of np.ndarray
        ``[ground, signal_1, signal_2, ..., signal_{K-1}]``.  Local
        2D node indices into the ``primal_2d_node_indices`` basis from
        :func:`build_2d_gradient`.  At least 2 groups; disjoint.
    epsilon_r : float
        Uniform relative permittivity of the cross-section dielectric.
    m_mu_flat : np.ndarray
        Diagonal of the FIT ``M_μ`` matrix in the flat H-vector layout.
        Required for the travelling-wave-consistent H profiles (WP7.2),
        see :func:`travelling_wave_h_profiles`.
    boundary_conditions : BoundaryConditions, optional
        The mesh's declared boundary closure.  Needed to book the full
        boundary dual cell into ``C'`` at tangential PMC window edges
        (see :func:`_window_boundary_factor`); omitting it falls back
        to the wall-on-window-edge convention (``z_line`` biased O(h)
        when the mode reaches a magnetic lateral wall).

    Returns
    -------
    list[Mode]
        ``K - 1`` TEM modes (one per signal conductor).  Labels
        ``"TEM_lap00"``, ``"TEM_lap01"``, ...

    Raises
    ------
    ValueError
        Per :func:`_solve_signal_modes_laplace`'s argument validation
        plus ``epsilon_r > 0``.
    RuntimeError
        Free-DoF set empty, or a TEM mode collapses to zero M-norm
        (electrically indistinguishable conductor groups).
    """
    if epsilon_r <= 0.0:
        raise ValueError("epsilon_r must be positive.")

    if m_eps_2d.shape != (g_2d.shape[0], g_2d.shape[0]):
        raise ValueError(
            f"m_eps_2d shape {m_eps_2d.shape} does not match "
            f"({g_2d.shape[0]}, {g_2d.shape[0]}) implied by g_2d."
        )
    # Mode lives in the 3D-FIT mass metric (so the FIT-TD operator stays
    # numerically consistent), but the per-length capacitance integral
    # ``C' = ê^T M ê / normal_dx`` uses a corrected mass whose tangential
    # window-boundary entries book exactly the dual width the physical
    # cross-section owns there: half a cell at a wall-on-line boundary,
    # the full boundary cell at a PMC face (wall half a cell beyond the
    # line) — see :func:`_tangential_boundary_factor`.
    m_eps_2d_cap = _correct_boundary_mass(m_eps_2d, plane, grid, boundary_conditions)
    raw_modes = _solve_signal_modes_laplace(
        plane,
        g_2d,
        m_eps_2d,
        m_eps_2d_cap,
        conductor_node_groups,
    )

    z_tem = ETA0 / math.sqrt(epsilon_r)
    normal_dx = float(plane.normal_dx)
    n_u = int(plane.e_u_indices.size)
    n_v = int(plane.e_v_indices.size)

    if len(raw_modes) == 1:
        # Single signal conductor: the historical path, bit-identical.
        channels = [
            (e_normalised, raw_norm_sq / normal_dx)
            for (e_normalised, raw_norm_sq, _e_raw) in raw_modes
        ]
    else:
        # K > 2: the per-conductor Laplace solutions (gauge
        # ``V_k = +1 V``) are each M-normalised but mutually
        # NON-orthogonal (measured 32 % on the symmetric two-wire),
        # while everything downstream — the ``discretize_modes``
        # pass-through, the operator projections, the TF/SF injection
        # and the per-channel DTBC — assumes an M_ε-orthonormal
        # channel basis; the DTBC feedback loop between overlapping
        # channels is unstable (measured blow-up to 1e64 on the
        # two-wire fixture).  The port channels are therefore the
        # eigenvectors of the raw-field Gram matrix in the FIT metric
        # — the capacitance-matrix eigenmodes of the line (the exact
        # odd/even pair on the symmetric two-wire), M_ε-orthogonal by
        # construction.  All TEM channels of a homogeneous line are
        # degenerate (same velocity, ``f_c = 0``), so any orthogonal
        # basis of the TEM subspace is an equally valid channel set.
        e_raw_mat = np.stack([raw for (_, _, raw) in raw_modes], axis=1)
        gram_fit = e_raw_mat.T @ (m_eps_2d @ e_raw_mat)
        gram_cap = e_raw_mat.T @ (m_eps_2d_cap @ e_raw_mat)
        eigvals, W = np.linalg.eigh(gram_fit)
        order = np.argsort(eigvals)[::-1]  # descending, deterministic
        eigvals, W = eigvals[order], W[:, order]
        channels = []
        for m in range(W.shape[1]):
            w = W[:, m]
            # Deterministic gauge: the leading near-maximal entry of the
            # conductor-voltage pattern is positive.  On symmetric lines
            # the degenerate pair has |w_i| equal up to rounding noise,
            # so a plain argmax tie-breaks on that noise and can mirror
            # the whole channel; the tolerance keeps the gauge stable.
            aw = np.abs(w)
            lead = int(np.flatnonzero(aw >= (1.0 - 1e-9) * aw.max())[0])
            if w[lead] < 0.0:
                w = -w
            e_field = e_raw_mat @ w
            e_normalised = e_field / math.sqrt(float(eigvals[m]))
            # C' of the channel's voltage pattern (‖w‖ = 1): the
            # quadratic form of the capacitance-correct Gram.
            c_per_length = float(w @ gram_cap @ w) / normal_dx
            channels.append((e_normalised, c_per_length))

    modes: list[Mode] = []
    for i, (e_normalised, c_per_length) in enumerate(channels):
        # Z_line via per-length capacitance: C' = ê_raw^T M_ε ê_raw /
        # normal_dx, then Z_line = √(ε_r)/(c · C').
        z_line = math.sqrt(epsilon_r) / (C0 * c_per_length)

        e_u = e_normalised[:n_u].copy()
        e_v = e_normalised[n_u : n_u + n_v].copy()
        # Travelling-wave-consistent dual voltages (WP7.2) — the field-
        # level Ohm's law with per-face 1/M_μ weights.
        h_u, h_v = travelling_wave_h_profiles(
            e_u,
            e_v,
            plane,
            m_mu_flat,
            z_tem,
        )

        modes.append(
            Mode(
                name=_tem_label(i),
                mode_type=ModeType.TEM,
                omega_c=0.0,
                epsilon_r=epsilon_r,
                field_evaluator=None,
                z_line=z_line,
                discrete_e_u_profile=e_u,
                discrete_e_v_profile=e_v,
                discrete_h_u_profile=h_u,
                discrete_h_v_profile=h_v,
            )
        )
    return modes


def solve_qtem_laplace(
    plane: PortPlane,
    g_2d: sp.csr_matrix,
    m_eps_2d: sp.csr_matrix,
    m_eps_2d_vacuum: sp.csr_matrix,
    conductor_node_groups: list[np.ndarray],
    grid: GridLines,
    *,
    m_mu_flat: np.ndarray,
    boundary_conditions=None,
) -> list[Mode]:
    """Compute K-1 QTEM modes via dual 2D Laplace ("Standard adequate").

    Implements Reference §7.1: solve the 2D Laplace problem twice —
    once with the actual ε distribution (yielding the actual line
    capacitance ``C'``), once with all dielectrics replaced by vacuum
    (yielding ``C'_0``).  Then ``ε_eff = C'/C'_0`` and
    ``Z_0 = 1/(c·√(C'·C'_0))``.  The actual ê (from the first solve)
    is used as the field profile; H is baked in via
    ``Z_TEM_eff = η₀/√ε_eff``.

    Parameters
    ----------
    plane, g_2d, conductor_node_groups
        As in :func:`solve_tem_laplace`.
    m_eps_2d : scipy.sparse.csr_matrix
        Actual 2D ε-mass — encodes the real (inhomogeneous) ε
        distribution of the cross-section.
    m_eps_2d_vacuum : scipy.sparse.csr_matrix
        Vacuum 2D ε-mass — same geometry but all materials replaced
        by ε ≡ 1.  Built independently by the caller (typically by
        running ``build_M_eps`` on a vacuum-only mesh of the same
        grid, then slicing to the same primal 2D edges).
    m_mu_flat : np.ndarray
        Diagonal of the FIT ``M_μ`` matrix in the flat H-vector layout.
        Required for the travelling-wave-consistent H profiles (WP7.2),
        see :func:`travelling_wave_h_profiles`.
    boundary_conditions : BoundaryConditions, optional
        As in :func:`solve_tem_laplace` — corrects both capacitance
        integrals (``C'`` and ``C'_0``) at tangential PMC window edges.

    Returns
    -------
    list[Mode]
        ``K - 1`` QTEM modes.  Each carries:
        - ``mode_type = ModeType.TEM`` (QTEM is treated as TEM with
          effective parameters; magnelio's ``ModeType`` does not have a
          dedicated QTEM value),
        - ``omega_c = 0.0``,
        - ``epsilon_r = ε_eff,k``,
        - ``z_line = Z_0,k``,
        - ``field_evaluator = None``,
        - ``discrete_*_profile`` arrays from the *actual*-ε solve,
          M_ε-orthonormalised against the actual mass.
        Labels: ``"QTEM_lap00"``, ``"QTEM_lap01"``, ...

    Raises
    ------
    ValueError
        If ``m_eps_2d_vacuum`` shape does not match ``m_eps_2d``;
        plus the validation from
        :func:`_solve_signal_modes_laplace`.
    RuntimeError
        Free-DoF set empty, or a QTEM mode collapses to zero M-norm.
    """
    if m_eps_2d_vacuum.shape != m_eps_2d.shape:
        raise ValueError(
            f"m_eps_2d_vacuum shape {m_eps_2d_vacuum.shape} does not "
            f"match m_eps_2d shape {m_eps_2d.shape}."
        )

    m_eps_2d_cap = _correct_boundary_mass(m_eps_2d, plane, grid, boundary_conditions)
    m_eps_2d_vacuum_cap = _correct_boundary_mass(m_eps_2d_vacuum, plane, grid, boundary_conditions)
    raw_actual = _solve_signal_modes_laplace(
        plane,
        g_2d,
        m_eps_2d,
        m_eps_2d_cap,
        conductor_node_groups,
    )
    raw_vacuum = _solve_signal_modes_laplace(
        plane,
        g_2d,
        m_eps_2d_vacuum,
        m_eps_2d_vacuum_cap,
        conductor_node_groups,
    )

    normal_dx = float(plane.normal_dx)
    n_u = int(plane.e_u_indices.size)
    n_v = int(plane.e_v_indices.size)

    # NOTE (WP-U2 finding): for K > 2 the per-conductor QTEM modes are
    # mutually non-orthogonal in M_ε, like the TEM ones — but each
    # carries its own ε_eff, so the TEM Gram-eigenbasis mixing does
    # not transfer.  Multi-signal QTEM channel bases are WP-U6
    # territory (ζ-pencil true modes); until then a K > 2 QTEM port
    # inherits the non-orthogonal basis and its modal-port caveats.
    modes: list[Mode] = []
    for i, ((e_actual, raw_norm_actual, _raw_a), (_e_vacuum, raw_norm_vacuum, _raw_v)) in enumerate(
        zip(raw_actual, raw_vacuum)
    ):
        c_per_length = raw_norm_actual / normal_dx
        c_per_length_vacuum = raw_norm_vacuum / normal_dx
        if c_per_length_vacuum <= 0.0:
            raise RuntimeError(
                f"QTEM mode {i}: vacuum capacitance is non-positive ({c_per_length_vacuum:.3e})."
            )
        eps_eff = c_per_length / c_per_length_vacuum
        if eps_eff <= 0.0:
            raise RuntimeError(
                f"QTEM mode {i}: ε_eff is non-positive ({eps_eff:.3e}); "
                f"check that the actual mass dominates the vacuum mass."
            )
        z_0 = 1.0 / (C0 * math.sqrt(c_per_length * c_per_length_vacuum))

        # Travelling-wave-consistent dual voltages (WP7.2) at the
        # effective wave impedance.
        z_tem_eff = ETA0 / math.sqrt(eps_eff)
        e_u = e_actual[:n_u].copy()
        e_v = e_actual[n_u : n_u + n_v].copy()
        h_u, h_v = travelling_wave_h_profiles(
            e_u,
            e_v,
            plane,
            m_mu_flat,
            z_tem_eff,
        )

        modes.append(
            Mode(
                name=_qtem_label(i),
                mode_type=ModeType.TEM,
                omega_c=0.0,
                epsilon_r=eps_eff,
                field_evaluator=None,
                z_line=z_0,
                discrete_e_u_profile=e_u,
                discrete_e_v_profile=e_v,
                discrete_h_u_profile=h_u,
                discrete_h_v_profile=h_v,
            )
        )
    return modes


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _solve_signal_modes_laplace(
    plane: PortPlane,
    g_2d: sp.csr_matrix,
    m_eps_2d: sp.csr_matrix,
    m_eps_2d_capacitance: sp.csr_matrix,
    conductor_node_groups: list[np.ndarray],
) -> list[tuple[np.ndarray, float]]:
    """Solve K-1 signal modes via the FIT 2D Laplace.

    Internal helper shared by :func:`solve_tem_laplace` (Phase 2b
    step 7) and :func:`solve_qtem_laplace` (Phase 2b step 9).
    Validates inputs, builds the Laplace stiffness, sets Dirichlet BCs
    on conductor nodes, runs ``spsolve`` per signal conductor,
    normalises ê in the *3D-FIT* metric (``m_eps_2d``), and applies the
    conductor-flux sign convention.  The capacitance is integrated
    against ``m_eps_2d_capacitance`` — typically the same as
    ``m_eps_2d`` except with tangential bbox-boundary entries halved.

    Returns
    -------
    list of (e_normalised, capacitance_norm_sq, e_raw)
        For each non-ground conductor k = 1..K-1:
        - ``e_normalised`` : M-normalised field on primal 2D edges in
          the **3D-FIT** mass metric (so it stays compatible with the
          time-domain solver's mass-matrix conventions).  NOTE the
          fields of *different* signal conductors are mutually
          non-orthogonal — the caller owns any cross-family
          orthogonalisation (see :func:`solve_tem_laplace`).
        - ``capacitance_norm_sq`` : ``ê_raw^T M_ε,cap ê_raw`` evaluated
          against the **capacitance-correct** mass.  The line
          capacitance per length is ``C'_k = capacitance_norm_sq /
          normal_dx``.
        - ``e_raw`` : the un-normalised ``-∇φ_k`` field of the
          ``V_k = +1 V`` gauge (same sign convention as
          ``e_normalised``).
    """
    if len(conductor_node_groups) < 2:
        raise ValueError(
            "solve_*_laplace requires at least 2 conductor groups "
            "(groups[0] = ground, groups[1:] = signal conductors); "
            f"got {len(conductor_node_groups)}."
        )

    n_2d_edges = int(g_2d.shape[0])
    n_2d_nodes = int(g_2d.shape[1])
    if m_eps_2d.shape != (n_2d_edges, n_2d_edges):
        raise ValueError(
            f"m_eps_2d shape {m_eps_2d.shape} does not match "
            f"({n_2d_edges}, {n_2d_edges}) implied by g_2d."
        )
    n_u = int(plane.e_u_indices.size)
    n_v = int(plane.e_v_indices.size)
    if n_u + n_v != n_2d_edges:
        raise ValueError(
            f"plane edge basis split mismatch: e_u({n_u}) + e_v({n_v}) != n_2d_edges({n_2d_edges})."
        )

    for grp_idx, grp in enumerate(conductor_node_groups):
        grp = np.asarray(grp, dtype=np.int64)
        if grp.size == 0:
            raise ValueError(f"conductor_node_groups[{grp_idx}] is empty.")
        if grp.min() < 0 or grp.max() >= n_2d_nodes:
            raise ValueError(
                f"conductor_node_groups[{grp_idx}] indices out of range "
                f"[0, {n_2d_nodes}); got "
                f"[{int(grp.min())}, {int(grp.max())}]."
            )
        conductor_node_groups[grp_idx] = grp

    all_conductor_nodes = np.concatenate(conductor_node_groups)
    if np.unique(all_conductor_nodes).size != all_conductor_nodes.size:
        raise ValueError(
            "conductor_node_groups must be disjoint (a node may appear in at most one group)."
        )

    L = (g_2d.T @ m_eps_2d @ g_2d).tocsr()

    free_mask = np.ones(n_2d_nodes, dtype=bool)
    free_mask[all_conductor_nodes] = False
    free = np.where(free_mask)[0]
    if free.size == 0:
        raise RuntimeError(
            "All 2D primal nodes are on conductors — no free DoFs to solve the Laplace problem."
        )

    L_csc = L.tocsc()
    fixed = all_conductor_nodes
    L_ff = L_csc[free, :][:, free]
    L_fb = L_csc[free, :][:, fixed]

    group_of_fixed = np.empty(fixed.size, dtype=np.int64)
    cursor = 0
    for grp_idx, grp in enumerate(conductor_node_groups):
        gn = grp.size
        group_of_fixed[cursor : cursor + gn] = grp_idx
        cursor += gn

    n_signals = len(conductor_node_groups) - 1
    out: list[tuple[np.ndarray, float]] = []
    for k in range(1, n_signals + 1):
        phi_fixed = np.where(group_of_fixed == k, 1.0, 0.0)
        rhs = -L_fb @ phi_fixed
        phi_free = spsolve(L_ff, rhs)
        phi = np.zeros(n_2d_nodes, dtype=float)
        phi[free] = phi_free
        phi[fixed] = phi_fixed

        e_raw = -(g_2d @ phi)
        norm_sq = float(e_raw @ (m_eps_2d @ e_raw))
        if norm_sq <= 0.0:
            raise RuntimeError(
                f"signal mode {k}: degenerate (M-norm² = {norm_sq:.3e}). "
                f"Conductor group {k} may be electrically "
                f"indistinguishable from the ground group."
            )

        e_normalised = e_raw / math.sqrt(norm_sq)

        # Conductor-flux sign rule: total D-flux out of conductor k must
        # be positive.  See module docstring.  Applied to the raw field
        # too, so ``e_raw`` and ``e_normalised`` stay parallel (the
        # multi-signal orthogonalisation combines the raw fields).
        d_flux = m_eps_2d @ e_normalised
        node_div = g_2d.T @ d_flux
        outward_flux = -float(node_div[conductor_node_groups[k]].sum())
        if outward_flux < 0.0:
            e_normalised = -e_normalised
            e_raw = -e_raw

        cap_norm_sq = float(e_raw @ (m_eps_2d_capacitance @ e_raw))
        out.append((e_normalised, cap_norm_sq, e_raw))
    return out


def _tem_label(i: int) -> str:
    """TEM mode label (matches Numerical2DModeSolver._label style)."""
    return f"TEM_lap{i:02d}"


def _qtem_label(i: int) -> str:
    """QTEM mode label."""
    return f"QTEM_lap{i:02d}"
