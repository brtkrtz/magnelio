"""Static dead-tile analysis for the fused TD kernels (TILE_SKIP_PLAN).

Builds, from the solver-final coefficient arrays, the per-component
list of kernel tiles that contain at least one element whose update
is *not* a provable no-op.  The block-list kernels (WP-T3) launch
only those tiles; everything else is skipped, bit-identically:

* **E edge** — skippable iff ``alpha_E == 0 and beta_E == 0``
  (exactly the final PEC mask, port-plane flattening included).
  The kernel writes +0.0 there forever, and every runtime E writer
  either scales by ``beta_E`` (CPML, TF/SF, modal ports) or is
  followed by the per-step PEC re-enforcement.
* **H face** — skippable iff
  ``alpha_H == 1 and beta_H == 0`` (WP-R5 donated faces: the
  kernel computes ``h = h`` — a no-op regardless of the value), or
  the face is *curl-dead*: all four bounding E edges are skippable,
  so ``curl E == +0.0`` forever and ``h`` provably stays at its
  initial 0.0.  Curl-dead skipping is suppressed within
  ``boundary_shell_faces[face]`` cells of the named domain faces —
  needed only where a BC writes H unconditionally: PMC ``apply_H``
  mirrors the outermost face layer (shell 1); PEC is an E
  constraint and CPML corrections are recursions driven by the
  differentials of PEC edges, which vanish identically on
  curl-dead faces (shell 0 for both).  Periodic and unknown BC
  types self-disable the whole analysis instead.

Runtime-writer audit (session 132) backing the rules above: modal
and discrete port operators write E only (``h_flat`` unused);
CPML H corrections are recursions driven by the differentials of
PEC edges and vanish identically on curl-dead faces; SIBC and
mu-dispersive faces are never curl-dead (they border live E
edges); thin-wire edges are plain PEC edges at runtime (the wire
physics lives in the modified material matrices of the
*neighbouring* faces).  Two configurations self-disable instead of
being analysed: TF/SF field sources (``inject_H`` scales by
``beta_H``, which is nonzero inside PEC, so a TF/SF surface
through metal may drive curl-dead faces) and any boundary
condition outside the ``_pec_reenforce_after_bc`` safe list.

The one-time zeroing indices returned in the plan normalise the
provably-zero elements after a checkpoint resume, making the
skip invariant unconditional.  Donated no-op faces are *not*
zeroed — the dense kernel preserves their value, and so does the
skip.

All inputs are host (NumPy) arrays; run this before the setup
copies coefficients to the GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


def tile_skip_enabled() -> bool:
    """Whether dead-tile skipping is enabled (``MAGNELIO_TILE_SKIP``)."""
    return os.environ.get("MAGNELIO_TILE_SKIP", "1") != "0"


#: Default tile shape (i, j, k) — WP-T1 measurement (session 132):
#: dense-free on the RTX 4070 SUPER (−3.8 % … +0.1 % vs the flat
#: production block) with the best robust capture; cubic tiles fail
#: the dense gate (short k-runs), 1024-thread shapes the Ada
#: occupancy cap.
DEFAULT_TILE = (2, 4, 32)

#: Shell depth (cells) for BC types that write H unconditionally
#: on their face (PMC mirrors the outermost layer).
PMC_SHELL = 1

E_NAMES = ("Ex", "Ey", "Ez")
H_NAMES = ("Hx", "Hy", "Hz")


def component_shapes(Nx: int, Ny: int, Nz: int):
    """Kernel array shapes of the E-edge and H-face components."""
    e = {
        "Ex": (Nx, Ny + 1, Nz + 1),
        "Ey": (Nx + 1, Ny, Nz + 1),
        "Ez": (Nx + 1, Ny + 1, Nz),
    }
    h = {
        "Hx": (Nx + 1, Ny, Nz),
        "Hy": (Nx, Ny + 1, Nz),
        "Hz": (Nx, Ny, Nz + 1),
    }
    return e, h


@dataclass
class TileSkipPlan:
    """Result of the static liveness analysis.

    Attributes
    ----------
    tile : tuple of int
        Tile shape (ti, tj, tk) in array index order.
    block_grids : dict
        Component name -> (nbi, nbj, nbk) tile-grid dimensions.
    live_blocks : dict
        Component name -> sorted int32 array of linear tile ids
        (``bi * nbj * nbk + bj * nbk + bk``) that must be launched.
    dead_zero_idx_E, dead_zero_idx_H : ndarray
        Flat indices into ``e_flat`` / ``h_flat`` of the provably-
        zero elements — zeroed once before the march (resume
        safety).  Excludes donated no-op faces, whose values must
        be preserved.
    stats : dict
        Component name -> fraction of elements inside skipped
        tiles; key ``"total"`` is the size-weighted aggregate.
    """

    tile: tuple[int, int, int]
    block_grids: dict = field(default_factory=dict)
    live_blocks: dict = field(default_factory=dict)
    dead_zero_idx_E: np.ndarray = None
    dead_zero_idx_H: np.ndarray = None
    stats: dict = field(default_factory=dict)


def _split(flat, shapes, names):
    """Views of a flat concatenated component array, reshaped 3-D."""
    out, off = {}, 0
    for name in names:
        n = int(np.prod(shapes[name]))
        out[name] = np.asarray(flat[off : off + n]).reshape(shapes[name])
        off += n
    return out


def _curl_dead(dead_e):
    """H faces whose four bounding E edges are all skippable."""
    ex, ey, ez = dead_e["Ex"], dead_e["Ey"], dead_e["Ez"]
    return {
        "Hx": ey[:, :, :-1] & ey[:, :, 1:] & ez[:, :-1, :] & ez[:, 1:, :],
        "Hy": ez[:-1, :, :] & ez[1:, :, :] & ex[:, :, :-1] & ex[:, :, 1:],
        "Hz": ex[:, :-1, :] & ex[:, 1:, :] & ey[:-1, :, :] & ey[1:, :, :],
    }


_FACE_AXIS = {
    "xmin": (0, False),
    "xmax": (0, True),
    "ymin": (1, False),
    "ymax": (1, True),
    "zmin": (2, False),
    "zmax": (2, True),
}


def _clear_shell(mask, shell_faces):
    """Suppress skipping within per-face shells of the array bounds."""
    if not shell_faces:
        return mask
    out = mask.copy()
    for face, shell in shell_faces.items():
        if shell <= 0:
            continue
        ax, is_max = _FACE_AXIS[face]
        sl = [slice(None)] * 3
        sl[ax] = slice(-shell, None) if is_max else slice(0, shell)
        out[tuple(sl)] = False
    return out


def _live_block_ids(live, tile):
    """Linear ids of tiles containing >= 1 live real element."""
    ti, tj, tk = tile
    ni, nj, nk = live.shape
    pad = ((0, (-ni) % ti), (0, (-nj) % tj), (0, (-nk) % tk))
    padded = np.pad(live, pad, constant_values=False)
    nbi, nbj, nbk = (padded.shape[0] // ti, padded.shape[1] // tj, padded.shape[2] // tk)
    blocks = padded.reshape(nbi, ti, nbj, tj, nbk, tk)
    block_live = blocks.any(axis=(1, 3, 5))
    ids = np.flatnonzero(block_live).astype(np.int32)
    # Fraction of real elements inside skipped tiles (for stats):
    in_live = np.repeat(np.repeat(np.repeat(block_live, ti, axis=0), tj, axis=1), tk, axis=2)[
        :ni, :nj, :nk
    ]
    skipped_frac = 1.0 - in_live.sum() / live.size
    return (nbi, nbj, nbk), ids, skipped_frac


def build_tile_skip_plan(
    *,
    Nx: int,
    Ny: int,
    Nz: int,
    alpha_E: np.ndarray,
    beta_E: np.ndarray,
    alpha_H: np.ndarray,
    beta_H: np.ndarray,
    has_field_sources: bool = False,
    has_unsafe_bcs: bool = False,
    tile: tuple[int, int, int] = DEFAULT_TILE,
    boundary_shell_faces: dict | None = None,
) -> TileSkipPlan | None:
    """Build the live-block launch plan, or None to run dense.

    Returns None when a configuration outside the audited writer
    set is present (TF/SF field sources, unlisted BC types) — the
    caller then keeps the dense launch path.
    ``boundary_shell_faces`` maps face names to shell depths in
    cells; pass ``{face: PMC_SHELL}`` for each PMC face (PEC and
    CPML faces need no shell, see the module docstring).
    """
    if has_field_sources or has_unsafe_bcs:
        return None

    shapes_E, shapes_H = component_shapes(Nx, Ny, Nz)
    aE = _split(alpha_E, shapes_E, E_NAMES)
    bE = _split(beta_E, shapes_E, E_NAMES)
    aH = _split(alpha_H, shapes_H, H_NAMES)
    bH = _split(beta_H, shapes_H, H_NAMES)

    dead_E = {n: (aE[n] == 0.0) & (bE[n] == 0.0) for n in E_NAMES}
    curl_dead = _curl_dead(dead_E)

    plan = TileSkipPlan(tile=tuple(tile))
    zero_E, zero_H = [], []
    off_E = off_H = 0
    weighted = total = 0.0

    for name in E_NAMES:
        live = ~dead_E[name]
        grid_dims, ids, frac = _live_block_ids(live, tile)
        plan.block_grids[name] = grid_dims
        plan.live_blocks[name] = ids
        plan.stats[name] = frac
        zero_E.append(np.flatnonzero(dead_E[name].ravel()) + off_E)
        off_E += dead_E[name].size
        weighted += frac * dead_E[name].size
        total += dead_E[name].size

    for name in H_NAMES:
        noop = (aH[name] == 1.0) & (bH[name] == 0.0)
        cd = _clear_shell(curl_dead[name], boundary_shell_faces)
        live = ~(noop | cd)
        grid_dims, ids, frac = _live_block_ids(live, tile)
        plan.block_grids[name] = grid_dims
        plan.live_blocks[name] = ids
        plan.stats[name] = frac
        # Zero only the provably-zero faces; donated no-op faces
        # keep their (possibly nonzero) frozen value.
        zero_H.append(np.flatnonzero(cd.ravel()) + off_H)
        off_H += cd.size
        weighted += frac * cd.size
        total += cd.size

    plan.dead_zero_idx_E = np.concatenate(zero_E)
    plan.dead_zero_idx_H = np.concatenate(zero_H)
    plan.stats["total"] = weighted / total
    return plan
