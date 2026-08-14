"""PEC wall-surface enumeration for surface-loss models (DD-082).

Enumerates the tangential-H samples adjacent to PEC walls — both
material-PEC solids (staircase cell classification) and PEC domain-
boundary walls — with per-sample footprint areas, grouped by tag.

The perturbative power-loss method evaluates

    P_loss = 1/2 * R_s * sum_samples( weight * |H_tan|^2 )

where each tangential H sample carries the wall area it tiles
(``weight``).  Per wall face and tangential component the sample
weights sum to the full face area, so summing over BOTH tangential
components counts the face area twice — exactly the ``|H_tan|^2 =
|H_u|^2 + |H_v|^2`` bookkeeping.

Field states are FIT grid quantities (``h = H * l_dual``); the
``inv_l_dual`` factor converts a sampled state value to the physical
H in A/m.

Staircase walls are full cell faces from the cell classification.
When the mesh carries the conformal ``face_material.A_face_pec`` data
(DD-087) AND the geometry actually cuts faces, material-PEC solids
switch to the divergence-theorem cell path: per cell, the wall-area
vector ``w(C) = Σ_faces A_face_pec·n_out`` gives the local conformal
wall area ``||w||₂``, exact for a single plane cut.  A cell holding
two wall families at once — a grid-aligned lid AND the curved mantle —
would have them summed as vectors and booked short, so the flat family
is split off first via ``A_face_pec_jump`` (the wall lying in the
cell's own faces) and only the remainder goes through the norm.
Measured against the 4/π staircase over-count: cylinder side 0.03 % /
0.01 % and pillbox total (both families) 0.09 % / 0.04 % at 1 / 0.5 mm.

The tangential-H sampling uses all three components (H_normal = 0 on
the wall, for either family) — booked exclusively onto UNCUT,
Faraday-live faces found by a short walk along the cell's inward wall
normal.  Cut-face states are not clean grid integrals: their rim
edges live under the DD-051 sub-cell metric, so neither ``h/l_dual``
nor a flux/free-area reading has a well-defined meaning there
(measured: a resolution-INDEPENDENT ~18 % power over-read at generic
grid phase), and fully-masked faces are Faraday-dead outright
(``C e = 0`` ⇒ ``b ≡ 0``; their share grew 16 % → 35 % on
refinement — a diverging Q).  Sampling H_tan a small step inside the
volume along the surface normal is the standard perturbative-loss
practice; the walk direction comes from the wall vector itself
(air = −sign(w) per axis), so no geometry knowledge is needed.

What remains is the inward sample-position bias (|H| is larger a step
off the wall): pillbox TM010 Q reads −11 % at 10 cells/radius and
−7 % at 20, IDENTICAL at generic grid phase, and the J1
position-pullback experiment recovers ±1.9 % — pure position, O(h),
converging (internal record ``investigations/conformal_wall_area/``).
Purely staircase scenes (no cut faces) keep the historical face path
bit-identically; BC walls are always flat and stay on it.

SIBC update topology (WP-D3, ``enumerate_sibc_surfaces``): the TD-SIBC
needs UPDATE coefficients, not H samples — per driving face the
dimensionless ``G_f = A_f / l_dual_f^2`` of the restored wall-edge
voltage (internal dossier ``investigations/sibc/DERIVATION.md`` §3/§4).  Staircase
walls book one pair per frozen wall edge from the face circulation it
appears in (``G = l_edge / l_dual``, the circulation-exact choice; an
edge shared by two coplanar wall footprints contributes half per
footprint, so interior edges sum to the full voltage and a seam
between two metals splits between their impedances).  Conformal
scenes reuse the DD-087 cell booking above verbatim as coefficients.
Both paths land exclusively on uncut, Faraday-live faces — judged by
the solver's own freeze mask, so no coefficient ever sits on a state
the H update does not integrate.  The sampling enumeration is not
touched by any of this.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_BC_FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


@dataclass
class WallSurface:
    """Wall-adjacent tangential-H samples for one wall tag.

    Parameters
    ----------
    tag : int or str
        Material id (``int``) for a PEC solid's surface, or the domain
        face name (``"zmin"``, …) for a PEC boundary-condition wall.
    comp : np.ndarray
        Per-sample H component, ``uint8``: 0 = Hx, 1 = Hy, 2 = Hz.
    flat_idx : np.ndarray
        Per-sample flat index into the (C-order raveled) component array.
    weight : np.ndarray
        Per-sample wall footprint area [m²].
    inv_l_dual : np.ndarray
        Per-sample ``1 / l_dual`` [1/m] — converts the FIT state value
        (``H * l_dual``) to the physical H in A/m.
    """

    tag: object
    comp: np.ndarray
    flat_idx: np.ndarray
    weight: np.ndarray
    inv_l_dual: np.ndarray
    area_total: float

    @property
    def area(self) -> float:
        """Total wall area [m²].

        Explicit field (DD-087): the staircase path tiles the area
        once per tangential component (Σweight = 2·area), the
        conformal cell path spreads each cell's wall area over up to
        three components — the historical ``Σweight/2`` convention
        does not survive that, so the enumeration records the area.
        """
        return self.area_total

    def h_tan_sq_sum(self, Hx, Hy, Hz) -> float:
        """``sum( weight * |H_phys|^2 )`` over the samples.

        Accepts real state arrays (time domain / eigenmode) or complex
        DFT arrays of the same spatial shape.
        """
        total = 0.0
        for c, arr in enumerate((Hx, Hy, Hz)):
            sel = self.comp == c
            if not sel.any():
                continue
            vals = np.asarray(arr).reshape(-1)[self.flat_idx[sel]]
            h_phys = vals * self.inv_l_dual[sel]
            total += float(np.sum(self.weight[sel] * (h_phys.real**2 + h_phys.imag**2)))
        return total


def _state_dual_lengths(d: np.ndarray) -> np.ndarray:
    """Dual lengths in the SOLVER STATE convention (length N -> N+1).

    Must match ``material_matrices._build_avg_d``: the boundary entries
    are the FULL first/last cell (not the geometric half-cell) — that is
    the length ``build_M_mu`` divides by, hence the length the boundary
    H states carry (``h = H * l``).  Using the geometric half-cell here
    reads boundary H a factor 2 high (measured: PMC-wall Hx samples).
    """
    N = len(d)
    dual = np.empty(N + 1)
    dual[0] = d[0]
    if N > 1:
        dual[1:N] = 0.5 * (d[:-1] + d[1:])
    dual[N] = d[-1]
    return dual


class _SampleAccumulator:
    """Accumulates footprint weights per H sample for one tag."""

    def __init__(self, Nx: int, Ny: int, Nz: int) -> None:
        self._w = [
            np.zeros((Nx + 1, Ny, Nz)),  # Hx
            np.zeros((Nx, Ny + 1, Nz)),  # Hy
            np.zeros((Nx, Ny, Nz + 1)),  # Hz
        ]

    def add(self, comp: int, idx_tuple, w) -> None:
        np.add.at(self._w[comp], idx_tuple, w)

    def to_surface(
        self,
        tag,
        inv_dual: tuple,
        area_total: float | None = None,
    ) -> "WallSurface | None":
        comps, idxs, weights, invs = [], [], [], []
        for c in range(3):
            w = self._w[c]
            nz = np.nonzero(w.ravel())[0]
            if nz.size == 0:
                continue
            comps.append(np.full(nz.size, c, dtype=np.uint8))
            idxs.append(nz)
            weights.append(w.ravel()[nz])
            # inv l_dual varies only along the component's own axis
            multi = np.unravel_index(nz, w.shape)
            invs.append(inv_dual[c][multi[c]])
        if not comps:
            return None
        weight = np.concatenate(weights)
        if area_total is None:
            # staircase convention: each tangential component tiles the
            # wall once -> Σweight = 2·area
            area_total = float(weight.sum()) / 2.0
        return WallSurface(
            tag=tag,
            comp=np.concatenate(comps),
            flat_idx=np.concatenate(idxs),
            weight=weight,
            inv_l_dual=np.concatenate(invs),
            area_total=area_total,
        )


def _add_wall_faces(acc, axis: int, ii, jj, kk, cc, dx, dy, dz) -> None:
    """Register the 4 tangential H samples of wall faces with normal *axis*.

    ``ii, jj, kk`` are the face's cell-footprint indices, ``cc`` the
    air-side cell index along the normal axis.
    """
    if axis == 2:  # z-wall, footprint dx[i]*dy[j], tangential Hx & Hy
        wx = 0.5 * dx[ii] * dy[jj]  # per Hx sample (2 per face)
        wy = 0.5 * dx[ii] * dy[jj]  # per Hy sample (2 per face)
        acc.add(0, (ii, jj, cc), wx)
        acc.add(0, (ii + 1, jj, cc), wx)
        acc.add(1, (ii, jj, cc), wy)
        acc.add(1, (ii, jj + 1, cc), wy)
    elif axis == 1:  # y-wall, footprint dx[i]*dz[k], tangential Hx & Hz
        w = 0.5 * dx[ii] * dz[kk]
        acc.add(0, (ii, cc, kk), w)
        acc.add(0, (ii + 1, cc, kk), w)
        acc.add(2, (ii, cc, kk), w)
        acc.add(2, (ii, cc, kk + 1), w)
    else:  # x-wall, footprint dy[j]*dz[k], tangential Hy & Hz
        w = 0.5 * dy[jj] * dz[kk]
        acc.add(1, (cc, jj, kk), w)
        acc.add(1, (cc, jj + 1, kk), w)
        acc.add(2, (cc, jj, kk), w)
        acc.add(2, (cc, jj, kk + 1), w)


def _face_pec_views(mesh) -> tuple[list, list, list]:
    """(A_face_pec, A_face_full, A_face_pec_jump) as per-component views."""
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    shapes = [(Nx + 1, Ny, Nz), (Nx, Ny + 1, Nz), (Nx, Ny, Nz + 1)]
    sizes = [int(np.prod(s)) for s in shapes]
    cuts = np.cumsum(sizes)[:-1]
    a_pec = [p.reshape(s) for p, s in zip(np.split(mesh.face_material.A_face_pec, cuts), shapes)]
    jump_flat = mesh.face_material.A_face_pec_jump
    if jump_flat is None:  # store written before the jump channel
        jump_flat = np.zeros_like(mesh.face_material.A_face_pec)
    a_jump = [p.reshape(s) for p, s in zip(np.split(jump_flat, cuts), shapes)]
    a_full = [
        np.broadcast_to((dy[:, None] * dz[None, :])[None, :, :], shapes[0]),
        np.broadcast_to((dx[:, None] * dz[None, :])[:, None, :], shapes[1]),
        np.broadcast_to((dx[:, None] * dy[None, :])[:, :, None], shapes[2]),
    ]
    return a_pec, a_full, a_jump


def _masked_face_pec_views(mesh, masked_boundary_faces):
    """`_face_pec_views` with selected domain-boundary node planes
    rewritten to continuation semantics (on copies — the mesh arrays
    stay untouched).

    DD-099: a wall lying IN a domain-boundary plane exists only where
    the boundary is an effective PEC wall.  A face hosting a port (or
    a non-PEC BC) is the opposite — the structure CONTINUES through
    it — so the masked plane's coverage is replaced by the adjacent
    interior plane's coverage and its jump is zeroed: no wall family
    books there (the conductor cross-section where a line exits is
    not a wall), the cell wall vector ``w`` sees no coverage step
    (zeroing instead fabricates +43 mm² of phantom interior wall on
    the padded coax), and the SIBC target gate keeps excluding the
    faces inside the continuing conductor exactly as it did before
    the DD-099 seed registered the plane.
    """
    a_pec, a_full, a_jump = _face_pec_views(mesh)
    if not masked_boundary_faces:
        return a_pec, a_full, a_jump
    grid = mesh.grid
    n = (grid.Nx, grid.Ny, grid.Nz)
    a_pec = [a.copy() for a in a_pec]
    a_jump = [a.copy() for a in a_jump]
    for face in masked_boundary_faces:
        if face not in _BC_FACES:
            raise ValueError(f"unknown boundary face {face!r}; expected one of {_BC_FACES}")
        ax = "xyz".index(face[0])
        lo = face.endswith("min")
        sl = [slice(None)] * 3
        sl[ax] = 0 if lo else n[ax]
        sl_in = [slice(None)] * 3
        sl_in[ax] = 1 if lo else n[ax] - 1
        a_pec[ax][tuple(sl)] = a_pec[ax][tuple(sl_in)]
        a_jump[ax][tuple(sl)] = 0.0
    return a_pec, a_full, a_jump


def detect_unregistered_walls(mesh, threshold: float = 0.1):
    """Cells whose conductor wall cancels out of the registration.

    DD-099 (WP-B1.3): a conductor shell thinner than one cell carries
    near-equal PEC coverage on opposite cell faces, so the cell wall
    vector ``w = ΔA_face_pec`` cancels and the shell's loss surface
    books ~nothing — silently.  Flags NON-PEC cells with partial PEC
    face coverage whose combined wall magnitude
    ``Σ|w_flat| + ‖w − w_flat‖`` falls below ``threshold`` times the
    partial-coverage sum.

    The threshold sits inside a measured empty window
    (BOUNDARY_WALL_PLAN WP-B0/B1 census): ordinary cut cells never
    drop below ~0.49 across the suite fixtures, while an unresolved
    30 µm shell in 150 µm cells reads 0.0055 — the default 0.1 is
    ×18 above the must-fire signal and ×5 below the suite floor.

    Returns
    -------
    cells : np.ndarray, shape (n, 3)
        Flagged cell indices (empty when the mesh carries no
        conformal coverage data).
    ratios : np.ndarray, shape (n,)
        Their wall/coverage ratios (ascending order of severity is
        NOT applied; pair with ``cells`` row by row).
    """
    empty = (np.empty((0, 3), dtype=np.int64), np.empty(0))
    fm = getattr(mesh, "face_material", None)
    if fm is None or getattr(fm, "A_face_pec", None) is None:
        return empty
    grid = mesh.grid
    a_pec, a_full, a_jump = _face_pec_views(mesh)

    lib = mesh.material_library
    pec_table = np.zeros(max(lib.keys()) + 1, dtype=bool)
    for mid, mat in lib.items():
        pec_table[mid] = mat.is_pec
    pec = pec_table[mesh.material_id]

    tol = 1e-6
    w, w_flat = [], []
    cov = np.zeros(pec.shape)
    any_partial = np.zeros(pec.shape, dtype=bool)
    for ax in range(3):
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[ax] = slice(None, -1)
        sl_hi[ax] = slice(1, None)
        lo, hi = tuple(sl_lo), tuple(sl_hi)
        w.append(a_pec[ax][hi] - a_pec[ax][lo])
        j_lo, j_hi = a_jump[ax][lo], a_jump[ax][hi]
        w_flat.append(np.where(j_hi > 0.0, j_hi, 0.0) + np.where(j_lo < 0.0, j_lo, 0.0))
        af = np.asarray(a_full[ax])
        for sl in (lo, hi):
            frac = a_pec[ax][sl] / af[sl]
            partial = (frac > tol) & (frac < 1.0 - tol)
            any_partial |= partial
            cov += np.where(partial, a_pec[ax][sl], 0.0)

    a_wall = sum(np.abs(f) for f in w_flat) + np.sqrt(
        sum((wc - f) ** 2 for wc, f in zip(w, w_flat))
    )
    dxg, dyg, dzg = grid.dx, grid.dy, grid.dz
    scale = (
        dxg[:, None, None] * dyg[None, :, None]
        + dyg[None, :, None] * dzg[None, None, :]
        + dxg[:, None, None] * dzg[None, None, :]
    )
    cand = any_partial & ~pec & (cov > 1e-3 * scale)
    if not cand.any():
        return empty
    ratio = a_wall[cand] / cov[cand]
    flag = ratio < threshold
    if not flag.any():
        return empty
    return np.argwhere(cand)[flag], ratio[flag]


def _face_alive_from_edge_mask(grid, pec_mask) -> list:
    """Per-component boolean views: face has at least one free rim edge.

    A primal face whose four rim edges are ALL in the given PEC edge
    mask is Faraday-dead (``C e = 0`` ⇒ ``b ≡ 0`` for all time) — its
    H state carries no information.
    """
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    e_shapes = [(Nx, Ny + 1, Nz + 1), (Nx + 1, Ny, Nz + 1), (Nx + 1, Ny + 1, Nz)]
    ex, ey, ez = (
        np.asarray(pec_mask[c, : int(np.prod(s))]).reshape(s) for c, s in enumerate(e_shapes)
    )
    dead_hx = ey[:, :, :-1] & ey[:, :, 1:] & ez[:, :-1, :] & ez[:, 1:, :]
    dead_hy = ex[:, :, :-1] & ex[:, :, 1:] & ez[:-1, :, :] & ez[1:, :, :]
    dead_hz = ex[:, :-1, :] & ex[:, 1:, :] & ey[:-1, :, :] & ey[1:, :, :]
    return [~dead_hx, ~dead_hy, ~dead_hz]


def _face_alive_views(mesh) -> list | None:
    """Alive views from the CONFORMAL edge mask (DD-087 sampling path).

    ``None`` when the mesh has no conformal edge mask (then every face
    counts as alive).
    """
    em = getattr(mesh, "edge_material", None)
    if em is None or getattr(em, "pec_mask", None) is None:
        return None
    return _face_alive_from_edge_mask(mesh.grid, em.pec_mask)


def _conformal_solid_surfaces(
    mesh,
    pec: np.ndarray,
    inv_dual: tuple,
    curvature: bool = True,
    views: tuple | None = None,
) -> list[WallSurface]:
    """Divergence-theorem cell path for material-PEC solids (DD-087).

    Per cell the wall vector ``w = Σ_faces A_face_pec·n_out`` obeys
    ``||w||₂ = A_wall`` — but only for a SINGLE plane cut.  A corner
    cell holding two wall families at once (a flat lid AND the curved
    mantle) has them added as vectors, and ``||a+b|| < |a|+|b|`` books
    it short.  The grid-aligned family is separated out first: it is
    exactly the wall lying IN the cell's own faces, whose area is the
    signed jump ``A_face_pec_jump`` of the PEC coverage across that
    face (a face merely shadowed by a wall standing in front of it
    does not jump).  So

        w_flat = the jumps this cell owns, signed like w
        A_wall = Σ|w_flat| + ||w − w_flat||

    which leaves a pure single-normal remainder for the norm.  On a
    cell with only one family the split is inert (one term vanishes),
    and on a wall tangent to a grid plane both parts are near-parallel,
    so it stays a near-no-op there too.

    Sampling: per component the cell's lo/hi H-faces split the cell's
    wall area proportionally to their free fractions (H inside PEC is
    dead); ``|H_tan|² = Σ_c |H_c|²`` since H_normal vanishes on the
    wall — which holds for BOTH families of a corner cell, so the two
    share one sample layout.  Wall cells are tagged by their own
    material id (PEC-classified cut cells) or by flood-filling from
    PEC neighbours.
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    a_pec, a_full, a_jump = views if views is not None else _face_pec_views(mesh)

    lo = [None] * 3
    hi = [None] * 3
    w = []
    w_flat = []
    for ax in range(3):
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[ax] = slice(None, -1)
        sl_hi[ax] = slice(1, None)
        lo[ax], hi[ax] = tuple(sl_lo), tuple(sl_hi)
        w.append(a_pec[ax][hi[ax]] - a_pec[ax][lo[ax]])
        # A hi face jumping positive has PEC above, so this cell is the
        # non-PEC side that owns that wall and its outward normal is
        # +ax; a lo face jumping negative is the mirror.  Either way the
        # jump's own sign is already the cell-vector convention.
        j_lo, j_hi = a_jump[ax][lo[ax]], a_jump[ax][hi[ax]]
        w_flat.append(np.where(j_hi > 0.0, j_hi, 0.0) + np.where(j_lo < 0.0, j_lo, 0.0))
    a_wall = sum(np.abs(f) for f in w_flat) + np.sqrt(
        sum((wc - f) ** 2 for wc, f in zip(w, w_flat)),
    )

    # noise floor: exact zero differences dominate; keep genuinely cut
    # cells only
    dxg, dyg, dzg = grid.dx, grid.dy, grid.dz
    scale = (
        dxg[:, None, None] * dyg[None, :, None]
        + dyg[None, :, None] * dzg[None, None, :]
        + dxg[:, None, None] * dzg[None, None, :]
    )
    live = a_wall > 1e-9 * scale

    if not live.any():
        return []

    # DD-098: curvature pullback — every booked weight is scaled by
    # c_b² = max(1 + κ·d1, 0)² (exact for 1/r line fields, flat walls
    # bit-exact no-ops).  ``curvature=False`` is the escape hatch.
    curv = None
    if curvature:
        from ._curvature import CurvatureFactors

        curv = CurvatureFactors(grid, a_pec, a_full, a_jump, w, w_flat, live)

    # ── tag assignment ───────────────────────────────────────────────
    mat_id = mesh.material_id
    tag = np.where(live & pec, mat_id, -1)
    todo = live & (tag < 0)
    # flood from PEC neighbours first, then from already-tagged wall
    # cells (deterministic axis order)
    for _ in range(4):
        if not todo.any():
            break
        for src in (np.where(pec, mat_id, -1), tag):
            for ax in range(3):
                for sh in (1, -1):
                    cand = np.full_like(tag, -1)
                    if sh == 1:
                        idx_dst = lo[ax]
                        idx_src = hi[ax]
                    else:
                        idx_dst = hi[ax]
                        idx_src = lo[ax]
                    cand[idx_dst] = src[idx_src]
                    fill = todo & (cand >= 0)
                    tag[fill] = cand[fill]
                    todo &= tag < 0
        if not todo.any():
            break
    # leftovers (isolated cut air cells): unique PEC id if there is one
    if todo.any():
        pec_ids = sorted({int(m) for m in np.unique(mat_id[pec])})
        if len(pec_ids) == 1:
            tag[todo] = pec_ids[0]

    # ── booking gate: only UNCUT, Faraday-live faces carry samples ──
    # Cut-face states are not trustworthy grid integrals: their rim
    # edges live under the DD-051 sub-cell metric, so neither
    # ``h/l_dual`` nor the flux contour has a clean free-area meaning
    # (measured: a resolution-INDEPENDENT ~+18 % power over-read at
    # generic grid phase — only the symmetric centered fixture hides
    # it), and fully-masked faces are Faraday-dead outright
    # (``C e = 0`` ⇒ ``b ≡ 0``; their share grew 16 % → 35 % on
    # refinement).  Book weight exclusively onto uncut faces with at
    # least one free rim edge; the walk below carries a wall cell's
    # share to the nearest such face along each axis.
    f_free = [np.clip(1.0 - a_pec[ax] / a_full[ax], 0.0, 1.0) for ax in range(3)]
    uncut = [a_pec[ax] <= 1e-9 * a_full[ax] for ax in range(3)]
    face_alive = _face_alive_views(mesh)
    if face_alive is not None:
        uncut = [u & a for u, a in zip(uncut, face_alive)]
    f_free = [np.where(uncut[ax], f_free[ax], 0.0) for ax in range(3)]

    surfaces: list[WallSurface] = []
    for mid in sorted({int(t) for t in np.unique(tag[tag >= 0])}):
        sel = live & (tag == mid)
        if not sel.any():
            continue
        ii, jj, kk = np.nonzero(sel)
        aw = a_wall[sel]
        pos3 = [ii, jj, kk]
        acc = _SampleAccumulator(Nx, Ny, Nz)
        # Inward walk direction per cell from the wall vector itself:
        # w points towards the PEC, so −sign(w) steps into the air.
        # The walk displaces the CANDIDATE CELL diagonally along that
        # direction and books onto its lo/hi component faces — walking
        # only along the component's own axis cannot work in general
        # (a z-invariant mantle cuts every z-face in the column, so an
        # axial walk never finds an uncut one and the whole mantle's
        # z-weight would be dropped; measured 18-23 % unbooked).
        # Sampling H_tan a small step inside the volume along the
        # surface normal is the standard perturbative-loss practice;
        # the offset is O(h) and converges by measurement.
        d_in = [np.sign(-wc[sel]).astype(np.int64) for wc in w]
        n_cells = (Nx, Ny, Nz)
        for ax in range(3):
            fl_eff = np.zeros(aw.shape, dtype=float)
            fh_eff = np.zeros(aw.shape, dtype=float)
            # booked face indices: full triples of the DISPLACED cell
            tgt = [p.copy() for p in pos3]
            found = np.zeros(aw.shape, dtype=bool)
            for k in range(4):
                cell_k = [np.clip(pos3[a] + k * d_in[a], 0, n_cells[a] - 1) for a in range(3)]
                idx_lo_k = list(cell_k)
                idx_hi_k = list(cell_k)
                idx_hi_k[ax] = cell_k[ax] + 1
                v_l = f_free[ax][tuple(idx_lo_k)]
                v_h = f_free[ax][tuple(idx_hi_k)]
                hit = ~found & ((v_l > 0.0) | (v_h > 0.0))
                fl_eff[hit] = v_l[hit]
                fh_eff[hit] = v_h[hit]
                for a in range(3):
                    tgt[a][hit] = cell_k[a][hit]
                found |= hit
            s = fl_eff + fh_eff
            ok = s > 1e-12
            if not ok.any():
                continue
            wl = np.where(ok, aw * fl_eff / np.where(ok, s, 1.0), 0.0)
            wh = np.where(ok, aw * fh_eff / np.where(ok, s, 1.0), 0.0)
            idx_lo = list(tgt)
            idx_hi = list(tgt)
            idx_hi[ax] = tgt[ax] + 1
            if curv is not None:
                wl = wl * curv.scale(pos3, ax, idx_lo)
                wh = wh * curv.scale(pos3, ax, idx_hi)
            acc.add(ax, tuple(idx_lo), wl)
            acc.add(ax, tuple(idx_hi), wh)
        surf = acc.to_surface(mid, inv_dual, area_total=float(aw.sum()))
        if surf is not None:
            surfaces.append(surf)
    return surfaces


def enumerate_pec_surfaces(
    mesh,
    bc_pec_faces: tuple[str, ...] = (),
    curvature_correction: bool = True,
    masked_boundary_faces: tuple[str, ...] = (),
) -> list[WallSurface]:
    """Enumerate PEC wall surfaces with their tangential-H samples.

    Parameters
    ----------
    mesh : Mesh
        Consolidated mesh (cell classification drives the solid walls).
    bc_pec_faces : tuple[str, ...]
        Domain-boundary faces (``"xmin"`` … ``"zmax"``) to treat as PEC
        walls.  Faces hosting ports must not be passed.
    curvature_correction : bool
        Scale every conformal booking weight by the DD-098 curvature
        pullback ``c_b² = max(1 + κ·d1, 0)²`` (default).  Flat and
        staircase walls are bit-exact no-ops either way; ``False``
        restores the uncorrected booking.
    masked_boundary_faces : tuple[str, ...]
        Domain-boundary faces whose registered PEC coverage must NOT
        act as a wall (DD-099): faces hosting ports or a non-PEC BC.
        Their node-plane coverage is zeroed before any booking.

    Returns
    -------
    list[WallSurface]
        One entry per material id with exposed PEC surface, plus one per
        requested boundary face.  Solid faces on the domain boundary
        itself are skipped (no interior air cell to sample from).
    """
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    inv_dual = (
        1.0 / _state_dual_lengths(dx),
        1.0 / _state_dual_lengths(dy),
        1.0 / _state_dual_lengths(dz),
    )

    mat_id = mesh.material_id
    lib = mesh.material_library
    max_id = max(lib.keys())
    pec_table = np.zeros(max_id + 1, dtype=bool)
    for mid, mat in lib.items():
        pec_table[mid] = mat.is_pec
    pec = pec_table[mat_id]  # (Nx, Ny, Nz) bool

    surfaces: list[WallSurface] = []

    # ── DD-087: conformal cell path when the geometry cuts faces ────────
    fm = getattr(mesh, "face_material", None)
    use_conformal = False
    views = None
    if fm is not None and getattr(fm, "A_face_pec", None) is not None:
        views = _masked_face_pec_views(mesh, masked_boundary_faces)
        a_pec_v, a_full_v, _ = views
        tol = 1e-6
        for ax in range(3):
            frac = a_pec_v[ax] / a_full_v[ax]
            if np.any((frac > tol) & (frac < 1.0 - tol)):
                use_conformal = True
                break
    if use_conformal:
        surfaces.extend(
            _conformal_solid_surfaces(
                mesh, pec, inv_dual, curvature=curvature_correction, views=views
            )
        )

    # ── Material-PEC solid walls (interior faces only) ──────────────────
    pec_ids = [] if use_conformal else [mid for mid, mat in lib.items() if mat.is_pec]
    for mid in pec_ids:
        this = mat_id == mid
        acc = _SampleAccumulator(Nx, Ny, Nz)
        found = False
        for axis in range(3):
            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            sl_lo[axis] = slice(None, -1)
            sl_hi[axis] = slice(1, None)
            lo_this = this[tuple(sl_lo)] & ~pec[tuple(sl_hi)]  # air above
            hi_this = this[tuple(sl_hi)] & ~pec[tuple(sl_lo)]  # air below
            for mask, air_off in ((lo_this, 1), (hi_this, 0)):
                if not mask.any():
                    continue
                found = True
                ii, jj, kk = np.nonzero(mask)
                # air-side cell index along the normal axis
                base = (ii, jj, kk)[axis]
                cc = base + air_off
                idx = [ii, jj, kk]
                idx[axis] = cc
                _add_wall_faces(acc, axis, idx[0], idx[1], idx[2], cc, dx, dy, dz)
        if found:
            surf = acc.to_surface(mid, inv_dual)
            if surf is not None:
                surfaces.append(surf)

    # ── PEC domain-boundary walls ────────────────────────────────────────
    for face in bc_pec_faces:
        if face not in _BC_FACES:
            raise ValueError(f"unknown boundary face {face!r}; expected one of {_BC_FACES}")
        axis = "xyz".index(face[0])
        lo = face.endswith("min")
        n_ax = (Nx, Ny, Nz)[axis]
        cell = 0 if lo else n_ax - 1
        sl = [slice(None)] * 3
        sl[axis] = cell
        open_cells = ~pec[tuple(sl)]  # 2-D mask over the face
        acc = _SampleAccumulator(Nx, Ny, Nz)
        if open_cells.any():
            a, b = np.nonzero(open_cells)
            idx = [None, None, None]
            tang = [ax for ax in range(3) if ax != axis]
            idx[tang[0]], idx[tang[1]] = a, b
            idx[axis] = np.full(a.shape, cell)
            _add_wall_faces(acc, axis, idx[0], idx[1], idx[2], idx[axis], dx, dy, dz)
        surf = acc.to_surface(face, inv_dual)
        if surf is not None:
            surfaces.append(surf)

    return surfaces


# ══════════════════════════════════════════════════════════════════════
# SIBC update topology (WP-D3)
# ══════════════════════════════════════════════════════════════════════


class SIBCSurface(WallSurface):
    """Wall update topology for the TD-SIBC, one block per wall tag.

    Same array layout as :class:`WallSurface` (``comp`` / ``flat_idx``
    / ``weight`` / ``inv_l_dual``, one row per driving H face), but the
    weights are the wall areas the *update* books onto each face — the
    circulation-exact ``l_edge * l_dual`` tiles on staircase walls, the
    DD-087 cell booking verbatim on conformal walls.  The inherited
    :meth:`h_tan_sq_sum` therefore is exactly the SIBC's own loss
    accounting ``sum(w * |H_tan|^2)`` (SIBC_PLAN WP-D5 monitor).
    """

    @property
    def g(self) -> np.ndarray:
        """Dimensionless update coefficient per row.

        ``G_f = A_f / l_dual_f^2`` (DERIVATION.md §3): the wall term of
        face ``f`` is ``T_f = Z_s * G_f * h_f``, so ``G_f`` multiplies
        the fitted ``Z_s`` into the semi-implicit ``M_sigma_m``-style
        fold and the branch history (WP-D4).  Exact identity with the
        booked area: ``inv_l_dual`` is a per-face geometric constant.
        """
        return self.weight * self.inv_l_dual**2

    def state_indices(self, grid) -> np.ndarray:
        """Row indices into the concatenated H-state vector.

        The solver-side ordering ``[hx, hy, hz]`` (C-raveled per
        component) — the layout ``DispersionOperator.from_mesh`` uses
        on the H side, hence the one the ``SIBCOperator`` blocks index.
        """
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        offsets = np.array(
            [
                0,
                (Nx + 1) * Ny * Nz,
                (Nx + 1) * Ny * Nz + Nx * (Ny + 1) * Nz,
            ],
            dtype=np.int64,
        )
        return offsets[self.comp] + self.flat_idx


def _as_sibc(surf: WallSurface) -> SIBCSurface:
    return SIBCSurface(
        tag=surf.tag,
        comp=surf.comp,
        flat_idx=surf.flat_idx,
        weight=surf.weight,
        inv_l_dual=surf.inv_l_dual,
        area_total=surf.area_total,
    )


def _sibc_book_pairs(acc_of_tag, tags, n_ax, n_cell_off, d, dual, ok, n_cells) -> None:
    """Book the (wall-edge -> driving-face) pairs of one wall family.

    ``tags``: per n-NODE plane the wall's material id (−1 = no wall
    with this air side), cell-indexed on the tangential axes.  For each
    tangential edge direction ``t`` the wall edges at the u-nodes
    between footprints restore their Leontovich voltage in the ONE live
    circulation on the air side — the H_u face at ``n_cell = n_node +
    n_cell_off``.  An edge with wall footprints on both u-sides books
    half per footprint (interior edges of one wall sum to the full
    ``G = l_edge / l_dual``; a bimetal seam splits between the tags);
    a rim/arris edge with a single footprint restores its full voltage
    (DERIVATION.md §3 — the corner case receives one full term per
    orientation, each a self-damping term in its own circulation).
    """
    for t_ax in range(3):
        if t_ax == n_ax:
            continue
        u_ax = 3 - n_ax - t_ax
        pad = [(0, 0)] * 3
        pad[u_ax] = (1, 1)
        P = np.pad(tags, pad, constant_values=-1)
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[u_ax] = slice(0, n_cells[u_ax] + 1)  # footprint at u-cell j-1
        sl_b[u_ax] = slice(1, n_cells[u_ax] + 2)  # footprint at u-cell j
        A, B = P[tuple(sl_a)], P[tuple(sl_b)]
        both = (A >= 0) & (B >= 0)
        for F in (A, B):
            sel = F >= 0
            if not sel.any():
                continue
            ii = np.nonzero(sel)
            fidx = list(ii)
            fidx[n_ax] = ii[n_ax] + n_cell_off
            keep = ok[u_ax][tuple(fidx)]
            if not keep.any():
                continue
            fidx = tuple(fi[keep] for fi in fidx)
            factor = np.where(both[sel][keep], 0.5, 1.0)
            w = factor * d[t_ax][fidx[t_ax]] * dual[u_ax][fidx[u_ax]]
            tag_vals = F[sel][keep]
            for mid in np.unique(tag_vals):
                m = tag_vals == mid
                acc = acc_of_tag(mid)
                acc.add(u_ax, tuple(fi[m] for fi in fidx), w[m])


def _staircase_sibc_solids(mesh, pec, d, dual, inv_dual, ok) -> list:
    """Circulation-exact wall-edge pairs for material-PEC solids."""
    grid = mesh.grid
    n_cells = (grid.Nx, grid.Ny, grid.Nz)
    mat_id = mesh.material_id
    accs: dict[int, _SampleAccumulator] = {}

    def acc_of_tag(mid):
        return accs.setdefault(int(mid), _SampleAccumulator(*n_cells))

    for n_ax in range(3):
        shp = list(n_cells)
        shp[n_ax] += 1
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_int = [slice(None)] * 3
        sl_lo[n_ax] = slice(None, -1)
        sl_hi[n_ax] = slice(1, None)
        sl_int[n_ax] = slice(1, -1)  # interior n-nodes 1..N-1
        pec_lo = pec[tuple(sl_lo)]
        pec_hi = pec[tuple(sl_hi)]
        # (wall's PEC side, wall mask, driven n-cell = n-node + off):
        # solid faces ON the domain boundary have no interior air cell
        # and are skipped by construction (interior nodes only).
        for pec_side, wall, n_cell_off in (
            (tuple(sl_lo), pec_lo & ~pec_hi, 0),  # air on +n
            (tuple(sl_hi), pec_hi & ~pec_lo, -1),  # air on -n
        ):
            if not wall.any():
                continue
            tags = np.full(shp, -1, dtype=np.int64)
            tags[tuple(sl_int)][wall] = mat_id[pec_side][wall]
            _sibc_book_pairs(acc_of_tag, tags, n_ax, n_cell_off, d, dual, ok, n_cells)

    surfaces = []
    for mid in sorted(accs):
        surf = accs[mid].to_surface(mid, inv_dual)
        if surf is not None:
            surfaces.append(_as_sibc(surf))
    return surfaces


def _bc_sibc_wall(mesh, face, pec, d, dual, inv_dual, ok):
    """Wall-edge pairs of one PEC domain-boundary face."""
    grid = mesh.grid
    n_cells = (grid.Nx, grid.Ny, grid.Nz)
    n_ax = "xyz".index(face[0])
    layer = 0 if face.endswith("min") else n_cells[n_ax] - 1
    sl = [slice(None)] * 3
    sl[n_ax] = slice(layer, layer + 1)
    open3 = ~pec[tuple(sl)]  # keep n_ax as singleton
    if not open3.any():
        return None
    acc = _SampleAccumulator(*n_cells)

    def acc_of_tag(_):
        return acc

    # Reuse the solid pair machinery: a single-node tag plane (pseudo
    # tag 0) at the boundary layer, driven faces at the layer itself.
    shp = list(n_cells)
    shp[n_ax] += 1
    tags = np.full(shp, -1, dtype=np.int64)
    sl_node = [slice(None)] * 3
    sl_node[n_ax] = slice(
        layer if layer == 0 else layer + 1, (layer if layer == 0 else layer + 1) + 1
    )
    tags[tuple(sl_node)][open3] = 0
    n_cell_off = 0 if layer == 0 else -1
    _sibc_book_pairs(acc_of_tag, tags, n_ax, n_cell_off, d, dual, ok, n_cells)

    surf = acc.to_surface(face, inv_dual)
    return _as_sibc(surf) if surf is not None else None


def enumerate_sibc_surfaces(
    mesh,
    bc_pec_faces: tuple[str, ...] = (),
    curvature_correction: bool = True,
    masked_boundary_faces: tuple[str, ...] = (),
) -> list[SIBCSurface]:
    """Enumerate the SIBC update topology of all PEC walls (WP-D3).

    Where :func:`enumerate_pec_surfaces` books tangential-H *samples*
    for perturbative postprocessing, this returns per wall tag the
    *update coefficients* of the TD-SIBC: one row per driving H face
    with the booked wall area (``weight``), the state conversion
    (``inv_l_dual``) and hence the dimensionless damping coefficient
    ``g = weight * inv_l_dual**2`` (``T_f = Z_s * G_f * h_f``,
    DERIVATION.md §3).  Staircase walls book one pair per frozen wall
    edge (circulation-exact, ``G = l_edge / l_dual``); conformal scenes
    (DD-087 data present with real cuts) reuse the divergence-theorem
    cell booking verbatim.  Every coefficient lands on an uncut,
    Faraday-live face of the solver's own freeze mask — never on a cut
    face or a dead state (DERIVATION.md §4).

    Parameters
    ----------
    mesh : Mesh
        Consolidated mesh (cell classification drives the solid walls).
    bc_pec_faces : tuple[str, ...]
        Domain-boundary faces (``"xmin"`` … ``"zmax"``) to treat as PEC
        walls.  Faces hosting ports must not be passed.
    curvature_correction : bool
        Scale every conformal booking weight (and hence ``G_f``) by
        the DD-098 curvature pullback ``c_b² = max(1 + κ·d1, 0)²``
        (default).  A non-negative scalar per branch — the DD-091
        passivity identity is untouched.  Staircase/BC walls and flat
        conformal families are bit-exact no-ops either way.
    masked_boundary_faces : tuple[str, ...]
        Domain-boundary faces whose registered PEC coverage must NOT
        act as a wall (DD-099): faces hosting ports or a non-PEC BC.
        Their node-plane coverage is zeroed before booking AND before
        the target-face gate, restoring the pre-registration state
        there.

    Returns
    -------
    list[SIBCSurface]
        One entry per material id with exposed PEC surface (sorted),
        plus one per requested boundary face (caller order).  Tags
        resolve to surface impedances via
        :func:`resolve_wall_conductors` and
        ``materials.surface_impedance.fit_wall_impedances``.
    """
    grid = mesh.grid
    d = (grid.dx, grid.dy, grid.dz)
    dual = tuple(_state_dual_lengths(dc) for dc in d)
    inv_dual = tuple(1.0 / du for du in dual)

    mat_id = mesh.material_id
    lib = mesh.material_library
    pec_table = np.zeros(max(lib.keys()) + 1, dtype=bool)
    for mid, mat in lib.items():
        pec_table[mid] = mat.is_pec
    pec = pec_table[mat_id]

    # Target-face gate: Faraday-live under the solver freeze mask, and
    # uncut where conformal coverage data exists.
    ok = _face_alive_from_edge_mask(grid, mesh.pec_mask_edges)
    fm = getattr(mesh, "face_material", None)
    use_conformal = False
    views = None
    if fm is not None and getattr(fm, "A_face_pec", None) is not None:
        views = _masked_face_pec_views(mesh, masked_boundary_faces)
        a_pec_v, a_full_v, _ = views
        tol = 1e-6
        for ax in range(3):
            ok[ax] = ok[ax] & (a_pec_v[ax] <= 1e-9 * a_full_v[ax])
            frac = a_pec_v[ax] / a_full_v[ax]
            if np.any((frac > tol) & (frac < 1.0 - tol)):
                use_conformal = True

    surfaces: list[SIBCSurface] = []
    if use_conformal:
        surfaces.extend(
            _as_sibc(s)
            for s in _conformal_solid_surfaces(
                mesh, pec, inv_dual, curvature=curvature_correction, views=views
            )
        )
    else:
        surfaces.extend(_staircase_sibc_solids(mesh, pec, d, dual, inv_dual, ok))

    for face in bc_pec_faces:
        if face not in _BC_FACES:
            raise ValueError(f"unknown boundary face {face!r}; expected one of {_BC_FACES}")
        surf = _bc_sibc_wall(mesh, face, pec, d, dual, inv_dual, ok)
        if surf is not None:
            surfaces.append(surf)
    return surfaces


def resolve_wall_conductors(
    mesh,
    surfaces,
    sigma: float | None = None,
    mu: float = 1.0,
    roughness=None,
    overrides: dict | None = None,
) -> dict:
    """Per-tag conductor properties ``tag -> (sigma, mu, roughness)``.

    The single tag-resolution rule of both DD-082 channels, shared by
    the perturbative chain (``wall_loss_Q`` / ``MonitorWallLoss``) and
    the SIBC setup: lossy-metal solids (``Material.lossy_metal``,
    DD-081) carry their own conductivity, permeability and DD-088
    roughness model; plain-PEC solids and PEC boundary-condition walls
    have none of their own and take the caller-supplied fallback.

    Parameters
    ----------
    mesh : Mesh
    surfaces : list of WallSurface or SIBCSurface
        Enumerated walls whose tags need conductor properties.
    sigma : float, optional
        Fallback conductivity [S/m] for walls that are not lossy
        metals.  Lossy-metal solids always use their own values.
    mu : float, optional
        Relative permeability accompanying ``sigma`` (default 1).
    roughness : SurfaceRoughness, optional
        DD-088 roughness model for the same walls ``sigma`` applies to.
    overrides : dict, optional
        Per-tag conductor overrides ``tag -> (sigma, mu, roughness)``
        (DD-099: a ``PECBoundary`` carrying its own wall material).
        Consulted for walls WITHOUT own material values, ahead of the
        ``sigma`` fallback; lossy-metal solids keep their own values.

    Returns
    -------
    dict
        ``tag -> (sigma, mu, roughness)``.

    Raises
    ------
    ValueError
        When a wall has no conductivity source (not a lossy metal and
        no ``sigma`` fallback given).
    """
    resolved = {}
    for surf in surfaces:
        tag = surf.tag
        mat = mesh.material_library.get(tag) if isinstance(tag, int) else None
        if mat is not None and mat.is_lossy_metal:
            resolved[tag] = (
                float(mat.sigma[0]),
                float(mat.mu[0]),
                mat.roughness,
            )
        elif overrides is not None and tag in overrides:
            o_sigma, o_mu, o_rough = overrides[tag]
            resolved[tag] = (float(o_sigma), float(o_mu), o_rough)
        elif sigma is not None:
            resolved[tag] = (float(sigma), float(mu), roughness)
        else:
            kind = "material" if isinstance(tag, int) else "boundary face"
            raise ValueError(
                f"wall {tag!r} ({kind}) has no conductivity: it is not a "
                f"lossy metal — pass sigma= (and mu=) for plain-PEC walls"
            )
    return resolved
