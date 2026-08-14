"""Trapezoidal ADE update for pole-residue dispersive materials.

Realises ``eps(omega) = eps_inf + sum_p r_p/(j*omega - a_p)`` (DD-083) in
the FIT-TD leapfrog by the auxiliary-differential-equation method: per
pole one polarisation-current state on the E-edges of the dispersive
region only (never a full-field allocation), advanced with the
trapezoidal rule — the DD-077 discretisation convention shared with the
lumped RLC companions.

Discrete form on a dispersive edge (grid quantities, ``e = E*l``,
``J = current through the dual facet``)::

    M_eps_inf de/dt + M_sigma e + sum_p J_p|_{n+1/2} = (C^T h)^{n+1/2}
    J_p^{n+1} = k_p J_p^n + c_p g (e^{n+1} - e^n)

with ``k_p = (1 + a_p dt/2)/(1 - a_p dt/2)``,
``c_p = r_p/(1 - a_p dt/2)``, ``g = eps0 * A_dual/l_primal`` (the M_eps
geometry factor) and the midpoint ``J|_{n+1/2} = (J^{n+1}+J^n)/2``.
Substituting the recursion splits the pole current into a coefficient
part — ``W = g * sum_p w_p Re(c_p)`` enters BOTH sides of the
semi-implicit E-update exactly like ``M_sigma`` (so ``alpha_E``/
``beta_E`` absorb it and the update kernels stay untouched) — and a
history part ``j_hist = sum_p w_p Re((1+k_p)/2 J_p^n)`` subtracted from
``e`` right after the curl kernel.  ``w_p`` is 1 for a real pole and 2
for a conjugate pair stored once (the partner contributes the complex
conjugate, so the physical current is twice the real part).

Properties inherited from the trapezoidal rule: A-stable for every
passive pole (``Re(a_p) < 0``) at any dt — the CFL limit stays the
``eps_inf`` one; the Drude DC pole (``a_p = 0``, ``k_p = 1``) reduces
bit-exactly to the standard semi-implicit conductor update with
``sigma = eps0 r_p``.

Edge subsets (WP-C2, DD-093): bulk edges use the clamped one-sided
cell lookup that drives ``build_M_sigma``'s bulk sampling; conformal
boundary edges join every dispersive material's block with the
classifier's per-material area share (WP-C1) as the weight on the
edge's own M_eps geometry factor — ``eps_eff(omega) = sum_i f_i
eps_i(omega)`` exactly, the arithmetic mixing rule of the static
``eps_avg`` (blocks may share edges; the two-phase update below makes
the shared-state completion the joint implicit solve).  PEC-masked
edges are excluded at build time — they never carry field, so their
pole states would stay zero anyway.

mu(omega) — the H side (DD-089)
-------------------------------
The magnetic mirror is the SAME operator, not a twin: substituting
``M_eps -> M_mu``, ``M_sigma -> M_sigma_m``, ``g -> g_m = mu0 *
A_primal/l_dual`` (the M_mu geometry factor) and ``(C^T h) -> -(C e)``
leaves the algebra above character for character, because the curl term
enters the derivation only as an opaque right-hand side ``R``: the
kernel produces ``alpha f^n + beta R`` whatever the sign of ``R``, and
``- beta * j_hist`` completes the implicit solution either way.  Hence
one class with a ``side`` parameter — the DD-084 factor-2 bug lived in
exactly these coefficients, and a copy would be a second place to get
them wrong.

``DispersionModel`` is reused verbatim: it is a relative-units
pole-residue form, so its ``eps_inf`` field carries ``mu_inf`` and its
passivity condition reads ``mu'' >= 0``.  The H-side exclusions are the
PEC-mask analogue: WP-R5 donor faces (``M_mu == 0``, exact
``beta_H = 0`` freezes them — their inertia lives on the donor face), so
a pole state there would never be integrated.
"""

from __future__ import annotations

import numpy as np

from magnelio._backend.array_api import copy_into
from magnelio._operators.material_matrices import (
    EPS0,
    MU0,
    _build_geom_E,
    _build_geom_H,
)
from magnelio.mesh.mesher import Mesh

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Fused-kernel dispatch threshold, see the kernel comment below.
_PARALLEL_MIN_STATES = 65536


if HAS_NUMBA:
    # Fused elementwise ADE kernels (NumPy-path replacement on the CPU
    # backend only — bind() keeps the generic array path for CuPy).
    # Arithmetic-order parity with the NumPy branch is load-bearing:
    # every state runs the identical FP operations in the identical
    # order — j_hist is INITIALISED from the first pole's term (the
    # NumPy branch has no 0.0 seed), accumulated in pole order via the
    # kind/row indirection (real and conjugate-pair poles may
    # interleave in a vector-fitted model), and the pole advance is
    # J*k + c*gdelta exactly like the in-place `J *= k; J += c*gdelta`.
    # Verified contract (TestFusedADEKernels): REAL poles are
    # bit-identical to the NumPy branch; CONJUGATE-PAIR poles are
    # bit-identical to the strict-IEEE scalar reference — NumPy's own
    # complex-multiply ufunc is FMA-contracted on SIMD builds and
    # deviates from that reference by 1 ULP, so it is the
    # machine-dependent side of that comparison, not this kernel.
    # Elements are independent (idx is unique per block), so prange
    # does not affect the per-state arithmetic.

    # The fused kernels are PARALLEL-ONLY, and _build_fused() only
    # equips blocks with >= _PARALLEL_MIN_STATES states — small blocks
    # keep the NumPy branch.  Both cut-offs are measured, not guessed:
    # below ~64k states the parallel win drowns in the per-region
    # thread wake/join (the solver's workers park between parallel
    # regions — unlike a hot back-to-back benchmark loop), and a
    # SERIAL fused variant was built, measured and REJECTED: the
    # per-pole kind/row branch costs ~6.5 ns/state serially, losing to
    # NumPy's vectorized subset path at every size below the parallel
    # window (55k-state case: 0.44 ms/step serial-fused vs 0.31 NumPy
    # vs 0.13 parallel-if-it-were-hot).  Above the threshold the
    # parallel kernel wins outright: 452k states, 0.62 vs 2.7 ms/step.
    #
    # If a serial variant is ever reintroduced, it must be a SEPARATE
    # function definition: two njit flag-sets compiled from one
    # function object collide in Numba's on-disk cache (the cache key
    # does not include the parallel flag — measured: the parallel
    # dispatcher silently loaded the serial binary in every fresh
    # process, costing the entire parallel win).

    @njit(parallel=True, cache=True)
    def _fused_save(f, idx, f_prev):
        for i in prange(idx.size):
            f_prev[i] = f[idx[i]]

    # WP-C2 (DD-093): the update is split into two kernels — subtract
    # every block's history current first, then advance every block's
    # pole states on the completed field — so blocks may SHARE states
    # (two dispersive materials on one conformal-boundary edge).  The
    # per-state arithmetic and float-op order match the former fused
    # single pass exactly (subtract reads f, writes f; advance re-reads
    # the identical written value), so disjoint meshes stay
    # bit-identical.

    @njit(parallel=True, cache=True)
    def _fused_subtract(f, idx, beta, kind, row, hw, opk_r, opk_c, Jr, Jc):
        nq = kind.size
        for i in prange(idx.size):
            r0 = row[0]
            if kind[0] == 0:
                jh = hw[0] * (opk_r[0] * Jr[r0, i])
            else:
                jh = hw[0] * ((opk_c[0] * Jc[r0, i]).real)
            for q in range(1, nq):
                r = row[q]
                if kind[q] == 0:
                    jh = jh + hw[q] * (opk_r[q] * Jr[r, i])
                else:
                    jh = jh + hw[q] * ((opk_c[q] * Jc[r, i]).real)
            f[idx[i]] = f[idx[i]] - beta[i] * jh

    @njit(parallel=True, cache=True)
    def _fused_advance(f, idx, g, f_prev, kind, row, k_r, c_r, k_c, c_c, Jr, Jc):
        nq = kind.size
        for i in prange(idx.size):
            gd = g[i] * (f[idx[i]] - f_prev[i])
            for q in range(nq):
                r = row[q]
                if kind[q] == 0:
                    Jr[r, i] = Jr[r, i] * k_r[q] + c_r[q] * gd
                else:
                    Jc[r, i] = Jc[r, i] * k_c[q] + c_c[q] * gd


class _Pole:
    """One pole recursion on a block's edge subset."""

    __slots__ = ("k", "c", "weight", "J")

    def __init__(self, a: complex, r: complex, dt: float, n_edges: int):
        k = (1.0 + a * dt / 2.0) / (1.0 - a * dt / 2.0)
        c = r / (1.0 - a * dt / 2.0)
        if a.imag == 0.0:  # real pole: real state, half the memory
            self.k, self.c = k.real, c.real
            self.weight = 1.0
            self.J = np.zeros(n_edges, dtype=np.float64)
        else:  # conjugate pair stored once
            self.k, self.c = k, c
            self.weight = 2.0
            self.J = np.zeros(n_edges, dtype=np.complex128)

    def w_coeff(self) -> float:
        """Real effective coefficient of ``(e^{n+1}-e^n)`` in the pole
        current ``J|_{n+1/2}``; the solver supplies the midpoint 1/2 via
        its ``0.5*dt*W`` folding."""
        return self.weight * (self.c.real if isinstance(self.c, complex) else self.c)


class _Block:
    """All pole states of one dispersive material on its state subset."""

    __slots__ = ("mat_id", "idx", "g", "beta", "f_prev", "poles", "fused")

    def __init__(self, mat_id: int, idx: np.ndarray, g: np.ndarray, poles: list[_Pole]):
        self.mat_id = mat_id
        self.idx = idx
        self.g = g
        self.poles = poles
        self.beta = None  # bound to beta_E/beta_H[idx] after setup
        self.f_prev = np.zeros(idx.size, dtype=np.float64)
        self.fused = None  # Numba kernel args, built by bind()


class DispersionOperator:
    """ADE pole-current update over all dispersive materials of a mesh.

    One class serves both sides (``side="E"`` for eps(omega), DD-084;
    ``side="H"`` for mu(omega), DD-089) — the recursion is identical, see
    the module docstring.

    Built by :meth:`from_mesh` (returns ``None`` when no material is
    dispersive on that side, keeping the no-dispersion solver path
    bit-identical).  The solver adds :attr:`W` (times ``dt/2``) to both
    the numerator and denominator of its ``alpha``/``beta``
    coefficients, calls :meth:`bind` once the final ``beta`` exists,
    then :meth:`save_field` at the top of every iteration and
    :meth:`update_field` right after that side's curl kernel.
    """

    def __init__(self, blocks: list[_Block], n_states: int, side: str = "E"):
        self.blocks = blocks
        self.side = side
        # Full-length coefficient diagonal (zeros off the subsets):
        # the solver folds (dt/2) * W into alpha / beta.
        W = np.zeros(n_states, dtype=np.float64)
        for b in blocks:
            W[b.idx] += b.g * sum(p.w_coeff() for p in b.poles)
        self.W = W

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        dt: float,
        side: str = "E",
        frozen: np.ndarray | None = None,
    ) -> "DispersionOperator | None":
        """Build the operator, or ``None`` if no material is dispersive.

        Parameters
        ----------
        mesh : Mesh
        dt : float
            Time step [s] — the pole recursion coefficients are fixed at
            build time.
        side : {"E", "H"}
            ``"E"``: eps(omega) on the E-edges (DD-084).  ``"H"``:
            mu(omega) on the H-faces (DD-089).
        frozen : ndarray of bool, optional
            States to exclude on top of the side's intrinsic exclusion.
            The solver passes ``M_mu == 0`` on the H side — the WP-R5
            donor faces, whose exact ``beta_H = 0`` freezes them (their
            inertia lives on the donor face), so a pole state there
            would never be integrated.
        """
        if side not in ("E", "H"):
            raise ValueError(f"side must be 'E' or 'H', got {side!r}")
        attr = "dispersion" if side == "E" else "dispersion_mu"
        dispersive = [
            (mid, mat)
            for mid, mat in sorted(mesh.material_library.items())
            if getattr(mat, attr, None) is not None
        ]
        if not dispersive:
            return None

        grid = mesh.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        mat_id = mesh.material_id
        i_clamp = np.clip(np.arange(Nx + 1), 0, max(Nx - 1, 0))
        j_clamp = np.clip(np.arange(Ny + 1), 0, max(Ny - 1, 0))
        k_clamp = np.clip(np.arange(Nz + 1), 0, max(Nz - 1, 0))

        if side == "E":
            # Per-edge owning cell — the identical clamped one-sided
            # lookup build_M_sigma uses for its bulk (cat-0) sampling.
            own = np.concatenate(
                [
                    mat_id[:, j_clamp][:, :, k_clamp].ravel(),  # Ex
                    mat_id[i_clamp][:, :, k_clamp].ravel(),  # Ey
                    mat_id[i_clamp][:, j_clamp].ravel(),  # Ez
                ]
            )
            n_Ex = Nx * (Ny + 1) * (Nz + 1)
            n_Ey = (Nx + 1) * Ny * (Nz + 1)
            n_Ez = (Nx + 1) * (Ny + 1) * Nz
            pec = mesh.pec_mask_edges
            excluded = np.concatenate(
                [
                    pec[0, :n_Ex],
                    pec[1, :n_Ey],
                    pec[2, :n_Ez],
                ]
            )
            const, geom = EPS0, _build_geom_E(grid)
        else:
            # Per-face owning cell — the lookup build_M_sigma_m uses.
            own = np.concatenate(
                [
                    mat_id[i_clamp].ravel(),  # Hx: (Nx+1, Ny, Nz)
                    mat_id[:, j_clamp].ravel(),  # Hy: (Nx, Ny+1, Nz)
                    mat_id[:, :, k_clamp].ravel(),  # Hz: (Nx, Ny, Nz+1)
                ]
            )
            excluded = np.zeros(own.size, dtype=bool)
            const, geom = MU0, _build_geom_H(grid)
        if frozen is not None:
            excluded = excluded | np.asarray(frozen, dtype=bool)

        # Conformal boundary membership (WP-C2 E side / WP-C3 H side,
        # DD-093): where the classifier recorded per-material area
        # fractions (WP-C1), a boundary entity joins the ADE block of
        # EVERY dispersive material with a positive share, weighted by
        # that share on the SAME geometry factor its own mass-matrix
        # entry uses (E cat 2: ``A_dual / L_free``; H cat 2:
        # ``A_face_free / L_dual_free`` above the 1 % floor) — so
        # ``eps_eff(omega) = sum_i f_i eps_i(omega)`` (and the mu
        # mirror) holds exactly, the identical mixing rule as the
        # static conformal averages.  Entities the classifier did not
        # process (NaN fractions — bulk, DD-053-promoted faces) keep
        # the staircase owning-cell rule; so do meshes without the
        # container (from_grid) — those paths stay bit-identical.
        frac_view = None
        geom_conf = geom
        usable_base = None
        if side == "E":
            em = getattr(mesh, "edge_material", None)
            if em is not None and em.material_fractions is not None:
                frac_view = em.fractions_by_mid
                from magnelio._operators.material_matrices import (  # noqa: PLC0415
                    _build_L_primal_E,
                )

                cat2 = em.category == 2
                if cat2.any():
                    geom_conf = geom.copy()
                    L_primal = _build_L_primal_E(grid)
                    geom_conf[cat2] = geom[cat2] * L_primal[cat2] / em.L_free[cat2]
                usable_base = np.ones(own.size, dtype=bool)
        else:
            # H side (WP-C3): mirror via FaceMaterialData fractions.
            # Conformal membership is restricted to faces whose OWN
            # M_mu / M_sigma_m entry is conformal too — cat 1, and
            # SAFE cat 2 (above the 1 % A_face_free floor, with the
            # same A_face_free / L_dual_free geometry).  Floored cat-2
            # faces book the bulk staircase in build_M_mu /
            # build_M_sigma_m, so they keep the staircase membership
            # here as well (a share-weighted W there could not match
            # its own booked sigma* — the mu-Drude-DC ≡ sigma* gate
            # enforces this face for face).  DD-053 pair-promoted
            # faces carry NaN fractions and fall back the same way.
            fm = getattr(mesh, "face_material", None)
            if fm is not None and fm.material_fractions is not None:
                frac_view = fm.fractions_by_mid
                from magnelio._operators.material_matrices import (  # noqa: PLC0415
                    _build_A_face_H,
                )

                A_face = _build_A_face_H(grid)
                cat2 = fm.category == 2
                safe = cat2 & (fm.A_face_free > 0.01 * A_face)
                if safe.any():
                    geom_conf = geom.copy()
                    geom_conf[safe] = fm.A_face_free[safe] / fm.L_dual_free[safe]
                usable_base = (fm.category == 1) | safe

        blocks = []
        for mid, mat in dispersive:
            frac = frac_view.get(mid) if frac_view is not None else None
            if frac is None:
                idx = np.nonzero((own == mid) & ~excluded)[0]
                if idx.size == 0:
                    continue
                g = const * geom[idx]
            else:
                usable = ~np.isnan(frac) & usable_base
                member = ~excluded & (((own == mid) & ~usable) | (usable & (frac > 0.0)))
                idx = np.nonzero(member)[0]
                if idx.size == 0:
                    continue
                share = np.where(
                    usable[idx],
                    np.nan_to_num(frac[idx]),
                    1.0,
                )
                # Staircase-fallback members (NaN fractions — bulk,
                # DD-053-promoted, floored cat 2) take the BULK
                # geometry factor: their own mass-matrix entry books
                # the staircase value there, never the conformal one.
                g = (
                    const
                    * np.where(
                        usable[idx],
                        geom_conf[idx],
                        geom[idx],
                    )
                    * share
                )
            poles = [_Pole(a, r, dt, idx.size) for a, r in getattr(mat, attr).poles]
            if not poles:  # pole-free model (e.g. tan_delta = 0)
                continue
            blocks.append(_Block(mid, idx, g, poles))
        if not blocks:
            return None
        return cls(blocks, n_states=own.size, side=side)

    # ------------------------------------------------------------------
    # Solver hooks
    # ------------------------------------------------------------------

    def bind(self, beta, xp) -> None:
        """Cache per-block ``beta`` slices; move state to the backend.

        The pole-current states, the geometry factor ``g`` and the field
        stash ``f_prev`` follow the field/coefficient precision (DD-094):
        ``beta`` was cast to the real field dtype by the solver, so its
        dtype drives the real state dtype and the matching complex dtype
        for conjugate-pair poles.  The recursion is a decaying IIR filter
        (``|k| < 1`` for every passive pole), so single storage carries no
        unbounded-accumulation hazard — the same argument that put the
        CPML psi state at the field dtype in WP1b.  The per-pole scalar
        coefficients (``k``/``c``) stay double: they are a handful of
        numbers and keeping them double makes each step a
        single-storage / double-op update.

        On the NumPy backend with Numba available this also stacks each
        block's pole states into per-dtype 2D arrays and precomputes the
        scalar coefficient tables the fused kernels consume.  ``p.J``
        becomes a row VIEW into the stack, so ``state_dict`` /
        ``load_state_dict`` (checkpointing) keep working on the same
        storage in either order relative to ``bind``.
        """
        real_dtype = beta.dtype
        complex_dtype = np.complex64 if real_dtype.itemsize == 4 else np.complex128
        for b in self.blocks:
            b.idx = xp.asarray(b.idx)
            b.g = xp.asarray(b.g.astype(real_dtype, copy=False))
            b.f_prev = xp.asarray(b.f_prev.astype(real_dtype, copy=False))
            b.beta = beta[b.idx]
            for p in b.poles:
                tgt = complex_dtype if np.iscomplexobj(p.J) else real_dtype
                p.J = xp.asarray(p.J.astype(tgt, copy=False))
        if HAS_NUMBA and xp is np:
            self._build_fused()

    def _build_fused(self) -> None:
        """Precompute the fused-kernel argument tables per block.

        Blocks below ``_PARALLEL_MIN_STATES`` are left on the NumPy
        branch (``b.fused = None``) — see the kernel comment for the
        measured cut-offs.
        """
        for b in self.blocks:
            if b.idx.size < _PARALLEL_MIN_STATES:
                continue
            nq = len(b.poles)
            n = b.idx.size
            n_c = sum(1 for p in b.poles if np.iscomplexobj(p.J))
            # State stacks follow the (already bound) field precision;
            # the scalar coefficient tables stay double (single-store /
            # double-op, DD-094).
            real_dtype = b.g.dtype
            complex_dtype = np.complex64 if real_dtype.itemsize == 4 else np.complex128
            kind = np.empty(nq, dtype=np.int64)
            row = np.empty(nq, dtype=np.int64)
            hw = np.empty(nq, dtype=np.float64)
            opk_r = np.zeros(nq, dtype=np.float64)
            k_r = np.zeros(nq, dtype=np.float64)
            c_r = np.zeros(nq, dtype=np.float64)
            opk_c = np.zeros(nq, dtype=np.complex128)
            k_c = np.zeros(nq, dtype=np.complex128)
            c_c = np.zeros(nq, dtype=np.complex128)
            Jr = np.zeros((nq - n_c, n), dtype=real_dtype)
            Jc = np.zeros((n_c, n), dtype=complex_dtype)
            ir = ic = 0
            for q, p in enumerate(b.poles):
                hw[q] = 0.5 * p.weight
                if np.iscomplexobj(p.J):
                    kind[q], row[q] = 1, ic
                    opk_c[q], k_c[q], c_c[q] = 1.0 + p.k, p.k, p.c
                    Jc[ic] = p.J
                    p.J = Jc[ic]
                    ic += 1
                else:
                    kind[q], row[q] = 0, ir
                    opk_r[q], k_r[q], c_r[q] = 1.0 + p.k, p.k, p.c
                    Jr[ir] = p.J
                    p.J = Jr[ir]
                    ir += 1
            b.fused = (kind, row, hw, opk_r, opk_c, k_r, c_r, k_c, c_c, Jr, Jc)

    def save_field(self, f) -> None:
        """Stash ``f^n`` on the subsets (top of the marching iteration).

        Taken at the loop top, not just before the kernel, so it is the
        FINAL field of the previous step — all BC/port/source
        corrections included.
        """
        for b in self.blocks:
            if b.fused is not None:
                _fused_save(f, b.idx, b.f_prev)
            else:
                b.f_prev[:] = f[b.idx]

    def update_field(self, f) -> None:
        """Complete the update with the pole-history current, advance poles.

        Runs right after this side's curl kernel: at that point
        ``f[idx]`` holds ``alpha f^n + beta R`` (``R = C^T h`` on the E
        side, ``-C e`` on the H side) with the ``W`` coefficient already
        folded into ``alpha``/``beta`` — subtracting ``beta * j_hist``
        completes the implicit solution exactly on every state no later
        stage rewrites.

        Two phases (WP-C2, DD-093), so blocks may share states — a
        conformal-boundary edge carrying two dispersive materials: (1)
        subtract EVERY block's history current (additive-exact — it
        depends only on the pre-update pole states), (2) advance EVERY
        block's pole states on the completed field.  On a shared state
        the implicit completion is the joint solve: both W terms were
        folded into ``alpha``/``beta`` and both histories are
        subtracted before any pole reads the field.  Disjoint meshes
        run the identical per-state arithmetic as the former single
        pass — bit-identical.

        The Numba path (``b.fused``, CPU backend only) runs the same
        arithmetic fused per state — bit-identical by construction, see
        the kernel comment — and exists purely because the NumPy branch
        allocates several full-subset temporaries per pole per step
        (measured 20-25x the per-state cost of the leapfrog kernels,
        PERFORMANCE_PROFILING_PLAN.md Workstream 2).  Both phases stay
        device-only array ops on the CuPy backend (WP-G3 graph
        segments — no host round trip).
        """
        for b in self.blocks:
            if b.fused is not None:
                (kind, row, hw, opk_r, opk_c, _k_r, _c_r, _k_c, _c_c, Jr, Jc) = b.fused
                _fused_subtract(f, b.idx, b.beta, kind, row, hw, opk_r, opk_c, Jr, Jc)
                continue
            j_hist = None
            for p in b.poles:
                term = (0.5 * p.weight) * ((1.0 + p.k) * p.J).real
                j_hist = term if j_hist is None else j_hist + term
            f[b.idx] = f[b.idx] - b.beta * j_hist
        for b in self.blocks:
            if b.fused is not None:
                (kind, row, _hw, _opk_r, _opk_c, k_r, c_r, k_c, c_c, Jr, Jc) = b.fused
                _fused_advance(f, b.idx, b.g, b.f_prev, kind, row, k_r, c_r, k_c, c_c, Jr, Jc)
                continue
            gdelta = b.g * (f[b.idx] - b.f_prev)
            for p in b.poles:
                p.J *= p.k
                p.J += p.c * gdelta

    # ------------------------------------------------------------------
    # Checkpoint (DD-070, WP-S6 pattern)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Pole-current states keyed by material id (``f_prev`` is not
        state: it is rewritten by ``save_field`` before every use)."""
        return {
            str(b.mat_id): {f"J{i}": p.J.copy() for i, p in enumerate(b.poles)} for b in self.blocks
        }

    def load_state_dict(self, state: dict) -> None:
        blocks_by_id = {str(b.mat_id): b for b in self.blocks}
        for key, pole_states in state.items():
            b = blocks_by_id[key]
            for i, p in enumerate(b.poles):
                copy_into(p.J, pole_states[f"J{i}"])
