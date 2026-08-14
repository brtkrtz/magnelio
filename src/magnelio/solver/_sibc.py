"""TD surface-impedance boundary operator (SIBC) on the H update (WP-D4).

Realises the Leontovich wall impedance ``E_tan = Z_s(omega) (n x H)`` in
the FIT-TD leapfrog as the additive Faraday-update term derived in the
internal dossier ``investigations/sibc/DERIVATION.md`` §3/§4: the
PEC-masked wall edge
contributed ``e = 0`` to the face circulation; the SIBC restores its
physical voltage from the face's own state, giving per booked wall face
the damping term

    T_f = Z_s * G_f * h_f,        G_f = A_f / l_dual_f^2,

with ``G_f`` and the driving-state indices supplied by the WP-D3
enumeration (``mesh.surfaces.enumerate_sibc_surfaces``) and ``Z_s`` as
the WP-D2 Foster/Stieltjes ladder ``c0 + sum_p c_p s/(s + b_p)``
(``materials.surface_impedance.SurfaceImpedanceFit``).

Discrete form (grid quantities, trapezoidal per branch, midpoint drive
``h_mid = (h^{n+3/2} + h^{n+1/2}) / 2``)::

    u_p^+ = k_p u_p^- + q_p h_mid
    k_p = (1 - b_p dt/2)/(1 + b_p dt/2),  q_p = c_p b_p dt/(1 + b_p dt/2)
    T_f  = G_f [ R_inst h_mid - sum_p ((1 + k_p)/2) u_p^- ]
    R_inst = c0 + sum_p c_p / (1 + b_p dt/2)

``G_f R_inst`` multiplies the midpoint exactly like a magnetic surface
conductivity, so the solver folds :attr:`W` as a plain addition to its
``M_sigma_m`` diagonal (simpler than the ADE's two-sided W folding —
DD-081 semi-implicit form, unconditionally stable); the history part is
added ``beta_H``-weighted right after the H kernel.  Structurally the
operator mirrors :class:`~magnelio.solver._dispersion.DispersionOperator`
(same two-phase ``save_field``/``update_field`` hooks, block-per-tag
layout, ``state_dict`` checkpoint pattern) with two differences: the
drive is the midpoint (not the difference), and ``update_field`` itself
is two-phase over blocks — a bimetal seam books one face from two tags,
and every history correction must land before any branch reads the
completed ``h`` (the corrections are mutually independent, they depend
only on ``u^-``, so phase 1 is exact for shared faces too).

Every branch has ``c_p >= 0, b_p > 0`` by NNLS construction, hence the
exact per-branch dissipation identity of DERIVATION.md §5: stability is
unconditional at the UNCHANGED lossless CFL, independent of fit
accuracy.  Cost is ``O(N_wall_faces * n_branches)`` scalar states —
surface-scaling, marginal against the volume curls, which is why no
fused Numba kernel exists here (the DD-084 fused-ADE threshold of 65536
states is a volume phenomenon; the generic array path below runs on
NumPy and CuPy alike through ``xp``).

``sigma -> inf`` reduces exactly to PEC (the SIBC_PLAN mandatory gate):
an empty surface set or an identically-zero fit makes
:meth:`SIBCOperator.from_spec` return ``None`` and the solver path stays
bit-identical to the master PEC path (the DD-084 rule — the folding
adds *nothing*, not ``+0.0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnelio._backend.array_api import copy_into


@dataclass(frozen=True)
class SIBCSpec:
    """Wall-impedance update data handed to the solver (WP-D4).

    Built by the analysis (WP-D5) or directly from the WP-D3/WP-D2 chain::

        surfaces = enumerate_sibc_surfaces(mesh, bc_pec_faces=...)
        resolved = resolve_wall_conductors(mesh, surfaces, sigma=...)
        fits     = fit_wall_impedances(resolved, f_lo, f_hi)
        spec     = SIBCSpec(surfaces=tuple(surfaces), fits=fits)

    Parameters
    ----------
    surfaces : tuple of SIBCSurface
        WP-D3 update topology, one entry per wall tag.
    fits : dict
        ``tag -> SurfaceImpedanceFit`` covering every surface tag.
    """

    surfaces: tuple
    fits: dict


class _Branch:
    """One Foster ladder branch ``c s/(s + b)`` on a block's faces."""

    __slots__ = ("k", "q", "u")

    def __init__(self, b: float, c: float, dt: float, n_faces: int):
        self.k = (1.0 - b * dt / 2.0) / (1.0 + b * dt / 2.0)
        self.q = c * b * dt / (1.0 + b * dt / 2.0)
        self.u = np.zeros(n_faces, dtype=np.float64)


class _Block:
    """All branch states of one wall tag on its driving faces."""

    __slots__ = ("tag", "idx", "g", "r_inst", "branches", "beta", "h_prev")

    def __init__(self, tag, idx: np.ndarray, g: np.ndarray, fit, dt: float):
        self.tag = tag
        self.idx = idx
        self.g = g
        self.r_inst = fit.c0 + sum(c_p / (1.0 + b_p * dt / 2.0) for b_p, c_p in fit.branches)
        self.branches = [_Branch(b_p, c_p, dt, idx.size) for b_p, c_p in fit.branches]
        self.beta = None  # bound to beta_H[idx] after setup
        self.h_prev = np.zeros(idx.size, dtype=np.float64)


class SIBCOperator:
    """Broadband wall-impedance update over all SIBC walls of a run.

    Built by :meth:`from_spec` (returns ``None`` when no wall carries a
    nonzero impedance, keeping the PEC solver path bit-identical).  The
    solver adds :attr:`W` to its ``M_sigma_m`` diagonal before building
    ``alpha_H``/``beta_H``, calls :meth:`bind` once the final ``beta_H``
    exists, then :meth:`save_field` at the top of every iteration and
    :meth:`update_field` right after the H kernel (after the DD-089
    mu-dispersion hook).
    """

    def __init__(self, blocks: list[_Block], n_states: int):
        self.blocks = blocks
        # Full-length instantaneous coefficient diagonal (zeros off the
        # subsets): the solver adds it to M_sigma_m — it multiplies
        # h_mid exactly like sigma* (DERIVATION.md §3).
        W = np.zeros(n_states, dtype=np.float64)
        for b in blocks:
            W[b.idx] += b.g * b.r_inst
        self.W = W

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(
        cls,
        spec: SIBCSpec,
        grid,
        dt: float,
        frozen: np.ndarray | None = None,
    ) -> "SIBCOperator | None":
        """Build the operator, or ``None`` when it would be a no-op.

        Parameters
        ----------
        spec : SIBCSpec
            Enumerated wall surfaces + per-tag impedance fits.
        grid : GridLines
            Solver grid — fixes the H-state layout the face indices
            address.
        dt : float
            Time step [s] — branch coefficients are fixed at build time.
        frozen : ndarray of bool, optional
            H states to exclude.  The solver passes ``M_mu <= 0`` — the
            WP-R5 donor faces, whose exact ``beta_H = 0`` freezes them,
            so a wall term there would never act.
        """
        blocks: list[_Block] = []
        for surf in spec.surfaces:
            if surf.tag not in spec.fits:
                raise KeyError(
                    f"SIBCSpec: no surface-impedance fit for wall tag "
                    f"{surf.tag!r}; fits cover {sorted(map(str, spec.fits))}"
                )
            fit = spec.fits[surf.tag]
            if fit.c0 == 0.0 and not fit.branches:
                # Z identically zero == sigma -> inf: exact PEC, no block
                # (Gate A, DERIVATION.md §6).
                continue
            idx = surf.state_indices(grid)
            g = np.asarray(surf.g, dtype=np.float64)
            if frozen is not None:
                keep = ~np.asarray(frozen, dtype=bool)[idx]
                idx, g = idx[keep], g[keep]
            if idx.size == 0:
                continue
            blocks.append(_Block(surf.tag, idx, g, fit, dt))
        if not blocks:
            return None
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        n_states = (Nx + 1) * Ny * Nz + Nx * (Ny + 1) * Nz + Nx * Ny * (Nz + 1)
        return cls(blocks, n_states)

    # ------------------------------------------------------------------
    # Solver hooks
    # ------------------------------------------------------------------

    def bind(self, beta, xp) -> None:
        """Cache per-block ``beta_H`` slices; move state to the backend.

        The branch states ``u``, the geometry factor ``g`` and the field
        stash ``h_prev`` follow the field precision (DD-094) — ``beta_H``
        was cast to the real field dtype by the solver, so its dtype
        drives them.  Every Foster branch has ``|k| < 1`` (``b_p > 0``),
        so the state is a decaying accumulator with no √N growth; the
        per-branch scalar coefficients (``k``/``q``/``r_inst``) stay
        double (single-store / double-op)."""
        real_dtype = beta.dtype
        for b in self.blocks:
            b.idx = xp.asarray(b.idx)
            b.g = xp.asarray(b.g.astype(real_dtype, copy=False))
            b.h_prev = xp.asarray(b.h_prev.astype(real_dtype, copy=False))
            b.beta = beta[b.idx]
            for br in b.branches:
                br.u = xp.asarray(br.u.astype(real_dtype, copy=False))

    def save_field(self, h) -> None:
        """Stash ``h^{n+1/2}`` on the subsets (top of the marching
        iteration, so it is the FINAL field of the previous step — all
        BC/port/source corrections included)."""
        for b in self.blocks:
            b.h_prev[:] = h[b.idx]

    def update_field(self, h) -> None:
        """Complete the update with the branch history, advance branches.

        Runs right after the H kernel: at that point ``h[idx]`` holds
        ``alpha_H h^- - beta_H (C e)`` with ``W`` already folded into
        ``alpha_H``/``beta_H`` — adding ``beta_H G_f sum_p ((1+k_p)/2)
        u_p^-`` completes the implicit solution exactly (DERIVATION.md
        §3).  Phase 1 applies ALL corrections (they depend only on the
        pre-step ``u``, so they are additive even on a seam face booked
        by two tags); phase 2 then advances every branch on the final
        midpoint.
        """
        for b in self.blocks:
            hist = None
            for br in b.branches:
                term = (0.5 * (1.0 + br.k)) * br.u
                hist = term if hist is None else hist + term
            if hist is not None:
                h[b.idx] = h[b.idx] + b.beta * (b.g * hist)
        for b in self.blocks:
            if not b.branches:
                continue
            h_mid = 0.5 * (h[b.idx] + b.h_prev)
            for br in b.branches:
                br.u *= br.k
                br.u += br.q * h_mid

    # ------------------------------------------------------------------
    # Checkpoint (DD-070, WP-S6 pattern)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Branch states keyed by wall tag (``h_prev`` is not state: it
        is rewritten by ``save_field`` before every use)."""
        return {
            str(b.tag): {f"u{i}": br.u.copy() for i, br in enumerate(b.branches)}
            for b in self.blocks
        }

    def load_state_dict(self, state: dict) -> None:
        blocks_by_tag = {str(b.tag): b for b in self.blocks}
        for key, branch_states in state.items():
            b = blocks_by_tag[key]
            for i, br in enumerate(b.branches):
                copy_into(br.u, branch_states[f"u{i}"])
