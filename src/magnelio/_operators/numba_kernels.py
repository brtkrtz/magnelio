"""
Accelerated field-update kernels for FIT-TD leapfrog.

Three tiers of dispatch (fastest first):

1. **CUDA fused** (``update_E_fused_cuda`` / ``update_H_fused_cuda``):
   Custom CUDA kernels launched via CuPy ``RawModule``.  One thread per
   grid point, single-pass curl + material multiply.  GPU only.

2. **Numba CPU** (``update_E_fused`` / ``update_H_fused``):
   Fused curl + material multiply in a single pass per component.
   ``parallel=True`` for multi-threaded execution.

3. **Array stencil** (``update_E_stencil`` / ``update_H_stencil``):
   Uses only slice ops (``+=``, ``-=``, ``[:] =``) that work on both
   NumPy and CuPy arrays.  Requires pre-allocated curl buffers.
   CPU fallback when Numba is not installed, and GPU fallback when
   CUDA kernel compilation fails.

``HAS_NUMBA`` / ``HAS_CUPY`` flags indicate availability; the solver
uses these to pick the fastest path automatically.
"""

from __future__ import annotations

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def update_E_fused(Ex, Ey, Ez, Hx, Hy, Hz, aEx, bEx, aEy, bEy, aEz, bEz):
        """Fused dual-curl + material E-update: E = alpha*E + beta*curl_H.

        Boundary E-edges with fewer than four H-face neighbours are
        handled via conditional accumulation (same result as the
        zero+accumulate numpy stencil, but in one pass).
        """
        # --- Ex: shape (Nx, Ny+1, Nz+1) ---
        # curl_H component: dHz/dy - dHy/dz
        ni, nj, nk = Ex.shape
        for i in prange(ni):
            for j in range(nj):
                for k in range(nk):
                    curl = 0.0
                    if j < nj - 1:
                        curl += Hz[i, j, k]
                    if j > 0:
                        curl -= Hz[i, j - 1, k]
                    if k < nk - 1:
                        curl -= Hy[i, j, k]
                    if k > 0:
                        curl += Hy[i, j, k - 1]
                    Ex[i, j, k] = aEx[i, j, k] * Ex[i, j, k] + bEx[i, j, k] * curl

        # --- Ey: shape (Nx+1, Ny, Nz+1) ---
        # curl_H component: dHx/dz - dHz/dx
        ni, nj, nk = Ey.shape
        for i in prange(ni):
            for j in range(nj):
                for k in range(nk):
                    curl = 0.0
                    if k < nk - 1:
                        curl += Hx[i, j, k]
                    if k > 0:
                        curl -= Hx[i, j, k - 1]
                    if i < ni - 1:
                        curl -= Hz[i, j, k]
                    if i > 0:
                        curl += Hz[i - 1, j, k]
                    Ey[i, j, k] = aEy[i, j, k] * Ey[i, j, k] + bEy[i, j, k] * curl

        # --- Ez: shape (Nx+1, Ny+1, Nz) ---
        # curl_H component: dHy/dx - dHx/dy
        ni, nj, nk = Ez.shape
        for i in prange(ni):
            for j in range(nj):
                for k in range(nk):
                    curl = 0.0
                    if i < ni - 1:
                        curl += Hy[i, j, k]
                    if i > 0:
                        curl -= Hy[i - 1, j, k]
                    if j < nj - 1:
                        curl -= Hx[i, j, k]
                    if j > 0:
                        curl += Hx[i, j - 1, k]
                    Ez[i, j, k] = aEz[i, j, k] * Ez[i, j, k] + bEz[i, j, k] * curl

    @njit(parallel=True, cache=True)
    def update_H_fused(Ex, Ey, Ez, Hx, Hy, Hz, aHx, bHx, aHy, bHy, aHz, bHz):
        """Fused primal-curl + material H-update: H = alpha*H - beta*curl_E.

        The primal curl has no boundary issues — all face indices are
        guaranteed in-bounds, so no conditionals are needed.
        """
        # --- Hx: shape (Nx+1, Ny, Nz) ---
        # curl_E component: dEz/dy - dEy/dz
        ni, nj, nk = Hx.shape
        for i in prange(ni):
            for j in range(nj):
                for k in range(nk):
                    curl = Ez[i, j + 1, k] - Ez[i, j, k] - Ey[i, j, k + 1] + Ey[i, j, k]
                    Hx[i, j, k] = aHx[i, j, k] * Hx[i, j, k] - bHx[i, j, k] * curl

        # --- Hy: shape (Nx, Ny+1, Nz) ---
        # curl_E component: dEx/dz - dEz/dx
        ni, nj, nk = Hy.shape
        for i in prange(ni):
            for j in range(nj):
                for k in range(nk):
                    curl = Ex[i, j, k + 1] - Ex[i, j, k] - Ez[i + 1, j, k] + Ez[i, j, k]
                    Hy[i, j, k] = aHy[i, j, k] * Hy[i, j, k] - bHy[i, j, k] * curl

        # --- Hz: shape (Nx, Ny, Nz+1) ---
        # curl_E component: dEy/dx - dEx/dy
        ni, nj, nk = Hz.shape
        for i in prange(ni):
            for j in range(nj):
                for k in range(nk):
                    curl = Ey[i + 1, j, k] - Ey[i, j, k] - Ex[i, j + 1, k] + Ex[i, j, k]
                    Hz[i, j, k] = aHz[i, j, k] * Hz[i, j, k] - bHz[i, j, k] * curl

else:
    update_E_fused = None
    update_H_fused = None


# ── Array-stencil kernels (NumPy / CuPy compatible) ─────────────────────


def update_E_stencil(Ex, Ey, Ez, Hx, Hy, Hz, aEx, bEx, aEy, bEy, aEz, bEz, cEx, cEy, cEz):
    """E update via array stencil ops.  Works with NumPy and CuPy arrays.

    Combines dual curl (C^T @ h) and material multiply into one call.
    Requires pre-allocated curl buffers ``cEx, cEy, cEz``.
    """
    # --- Ex: (C^T h)_x = dHz/dy - dHy/dz ---
    cEx[:] = 0.0
    cEx[:, :-1, :] += Hz
    cEx[:, 1:, :] -= Hz
    cEx[:, :, :-1] -= Hy
    cEx[:, :, 1:] += Hy
    cEx *= bEx
    Ex *= aEx
    Ex += cEx

    # --- Ey: (C^T h)_y = dHx/dz - dHz/dx ---
    cEy[:] = 0.0
    cEy[:, :, :-1] += Hx
    cEy[:, :, 1:] -= Hx
    cEy[:-1, :, :] -= Hz
    cEy[1:, :, :] += Hz
    cEy *= bEy
    Ey *= aEy
    Ey += cEy

    # --- Ez: (C^T h)_z = dHy/dx - dHx/dy ---
    cEz[:] = 0.0
    cEz[:-1, :, :] += Hy
    cEz[1:, :, :] -= Hy
    cEz[:, :-1, :] -= Hx
    cEz[:, 1:, :] += Hx
    cEz *= bEz
    Ez *= aEz
    Ez += cEz


def update_H_stencil(Ex, Ey, Ez, Hx, Hy, Hz, aHx, bHx, aHy, bHy, aHz, bHz, cHx, cHy, cHz):
    """H update via array stencil ops.  Works with NumPy and CuPy arrays.

    Combines primal curl (C @ e) and material multiply into one call.
    Requires pre-allocated curl buffers ``cHx, cHy, cHz``.
    """
    # --- Hx: (curl E)_x = dEz/dy - dEy/dz ---
    cHx[:] = Ez[:, 1:, :]
    cHx -= Ez[:, :-1, :]
    cHx -= Ey[:, :, 1:]
    cHx += Ey[:, :, :-1]
    cHx *= bHx
    Hx *= aHx
    Hx -= cHx

    # --- Hy: (curl E)_y = dEx/dz - dEz/dx ---
    cHy[:] = Ex[:, :, 1:]
    cHy -= Ex[:, :, :-1]
    cHy -= Ez[1:, :, :]
    cHy += Ez[:-1, :, :]
    cHy *= bHy
    Hy *= aHy
    Hy -= cHy

    # --- Hz: (curl E)_z = dEy/dx - dEx/dy ---
    cHz[:] = Ey[1:, :, :]
    cHz -= Ey[:-1, :, :]
    cHz -= Ex[:, 1:, :]
    cHz += Ex[:, :-1, :]
    cHz *= bHz
    Hz *= aHz
    Hz -= cHz


# ── CUDA fused kernels (CuPy RawModule) ──────────────────────────────────
#
# One thread per grid point, single memory pass per component.
# Thread mapping: threadIdx.x → k (fastest, coalesced), .y → j, .z → i.
#
# Launches walk a per-component list of live tile ids (TILE_SKIP_PLAN):
# blockIdx.x indexes the list, DECODE_TILE recovers the (i, j, k) tile
# origin.  The dense path is the identity list — same code path, and the
# indirection plus the generalised i index measured free (WP-T1,
# internal dossier investigations/pec_fill/).  Skipped tiles contain
# only no-op updates (alpha_E = beta_E = 0 edges; alpha_H = 1,
# beta_H = 0 or forever-zero-curl faces), so omitting their writes is
# bit-identical; neighbouring live tiles read the frozen values as
# before.

try:
    import numpy as _np
    from cupy import RawModule as _RawModule
    from numpy import int32 as _i32

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

_CUDA_SOURCE = r"""
typedef SCALAR_T scalar_t;

/* Tile decode, two compile variants of the same kernels (WP-T3):
 *
 * LISTED: blockIdx.x walks the live-block list; each entry packs the
 * tile coordinates as (bi << 20) | (bj << 10) | bk — shift/mask
 * decode, no per-thread integer division.  10 bits per axis caps the
 * per-component tile grid at 1024 tiles per axis (asserted on the
 * Python side).
 *
 * Dense: plain 3-D grid, no list.  The dependent per-block list load
 * measured +12 % in the L2-resident float32 regime, so the dense path
 * (skip disabled, or a run without dead tiles) keeps the direct
 * launch; the listed path only ever runs where skipping repays it.
 *
 * blockDim == the tile shape (k, j, i order) in both variants. */
#if LISTED
#define DECODE_TILE                                             \
    const int b_ = blocks[blockIdx.x];                          \
    const int k = (b_ & 1023) * blockDim.x + threadIdx.x;       \
    const int j = ((b_ >> 10) & 1023) * blockDim.y + threadIdx.y; \
    const int i = (b_ >> 20) * blockDim.z + threadIdx.z;        \
    if (i >= ni || j >= nj || k >= nk) return
#else
#define DECODE_TILE                                             \
    const int k = blockIdx.x * blockDim.x + threadIdx.x;        \
    const int j = blockIdx.y * blockDim.y + threadIdx.y;        \
    const int i = blockIdx.z * blockDim.z + threadIdx.z;        \
    if (i >= ni || j >= nj || k >= nk) return
#endif

extern "C" {

/* ── E-field update kernels (dual curl, boundary-conditional) ──────────
 *
 * Arrays AND the local ``curl`` accumulator are ``scalar_t`` (float32 or
 * float64 — plan WP2).  The curl deliberately does NOT accumulate in double
 * for the float32 variant: WP0 measured that a ``double curl`` on the
 * FP64-crippled consumer target (RTX 4070 SUPER, 1:64) makes float32 *slower*
 * than float64 (0.63x) — the few FP64 register ops dominate this
 * bandwidth-bound kernel — whereas a pure float32 curl gives the expected
 * ~2.4x at 373k cells (see DD-094 / benchmarks/precision_kernel_ab.py).  The
 * 4-term curl is a sum of similar-magnitude neighbour differences, so float32
 * accumulation stays at the ~1e-7 field floor.  The CPU Numba kernels keep a
 * float64 ``curl = 0.0`` (free on a full-rate FP64 CPU); float64 GPU is
 * byte-identical to before (scalar_t == double there).
 */

__global__ void update_Ex(
    scalar_t* __restrict__ Ex,
    const scalar_t* __restrict__ Hy,
    const scalar_t* __restrict__ Hz,
    const scalar_t* __restrict__ aEx,
    const scalar_t* __restrict__ bEx,
    int ni, int nj, int nk,
    const int* __restrict__ blocks)
{
    /* Ex: (Nx, Ny+1, Nz+1)  Hz: (Nx, Ny, Nz+1)  Hy: (Nx, Ny+1, Nz) */
    DECODE_TILE;

    int idx = i * nj * nk + j * nk + k;

    int Hz_s1 = (nj - 1) * nk;
    int Hy_s2 = nk - 1;

    scalar_t curl = 0.0;
    if (j < nj - 1) curl += Hz[i * Hz_s1 + j * nk + k];
    if (j > 0)      curl -= Hz[i * Hz_s1 + (j - 1) * nk + k];
    if (k < nk - 1) curl -= Hy[i * nj * Hy_s2 + j * Hy_s2 + k];
    if (k > 0)      curl += Hy[i * nj * Hy_s2 + j * Hy_s2 + (k - 1)];

    Ex[idx] = aEx[idx] * Ex[idx] + bEx[idx] * curl;
}

__global__ void update_Ey(
    scalar_t* __restrict__ Ey,
    const scalar_t* __restrict__ Hx,
    const scalar_t* __restrict__ Hz,
    const scalar_t* __restrict__ aEy,
    const scalar_t* __restrict__ bEy,
    int ni, int nj, int nk,
    const int* __restrict__ blocks)
{
    /* Ey: (Nx+1, Ny, Nz+1)  Hx: (Nx+1, Ny, Nz)  Hz: (Nx, Ny, Nz+1) */
    DECODE_TILE;

    int idx = i * nj * nk + j * nk + k;

    int Hx_nk = nk - 1;
    int Hz_s1 = nj * nk;

    scalar_t curl = 0.0;
    if (k < nk - 1) curl += Hx[i * nj * Hx_nk + j * Hx_nk + k];
    if (k > 0)      curl -= Hx[i * nj * Hx_nk + j * Hx_nk + (k - 1)];
    if (i < ni - 1) curl -= Hz[i * Hz_s1 + j * nk + k];
    if (i > 0)      curl += Hz[(i - 1) * Hz_s1 + j * nk + k];

    Ey[idx] = aEy[idx] * Ey[idx] + bEy[idx] * curl;
}

__global__ void update_Ez(
    scalar_t* __restrict__ Ez,
    const scalar_t* __restrict__ Hx,
    const scalar_t* __restrict__ Hy,
    const scalar_t* __restrict__ aEz,
    const scalar_t* __restrict__ bEz,
    int ni, int nj, int nk,
    const int* __restrict__ blocks)
{
    /* Ez: (Nx+1, Ny+1, Nz)  Hy: (Nx, Ny+1, Nz)  Hx: (Nx+1, Ny, Nz) */
    DECODE_TILE;

    int idx = i * nj * nk + j * nk + k;

    int Hy_s1 = nj * nk;
    int Hx_nj = nj - 1;

    scalar_t curl = 0.0;
    if (i < ni - 1) curl += Hy[i * Hy_s1 + j * nk + k];
    if (i > 0)      curl -= Hy[(i - 1) * Hy_s1 + j * nk + k];
    if (j < nj - 1) curl -= Hx[i * Hx_nj * nk + j * nk + k];
    if (j > 0)      curl += Hx[i * Hx_nj * nk + (j - 1) * nk + k];

    Ez[idx] = aEz[idx] * Ez[idx] + bEz[idx] * curl;
}

/* ── H-field update kernels (primal curl, no boundary conditions) ────── */

__global__ void update_Hx(
    scalar_t* __restrict__ Hx,
    const scalar_t* __restrict__ Ey,
    const scalar_t* __restrict__ Ez,
    const scalar_t* __restrict__ aHx,
    const scalar_t* __restrict__ bHx,
    int ni, int nj, int nk,
    const int* __restrict__ blocks)
{
    /* Hx: (Nx+1, Ny, Nz)  Ez: (Nx+1, Ny+1, Nz)  Ey: (Nx+1, Ny, Nz+1) */
    DECODE_TILE;

    int idx = i * nj * nk + j * nk + k;

    int Ez_s1 = (nj + 1) * nk;
    int Ey_nk = nk + 1;

    scalar_t curl = Ez[i * Ez_s1 + (j + 1) * nk + k]
                - Ez[i * Ez_s1 + j * nk + k]
                - Ey[i * nj * Ey_nk + j * Ey_nk + (k + 1)]
                + Ey[i * nj * Ey_nk + j * Ey_nk + k];

    Hx[idx] = aHx[idx] * Hx[idx] - bHx[idx] * curl;
}

__global__ void update_Hy(
    scalar_t* __restrict__ Hy,
    const scalar_t* __restrict__ Ex,
    const scalar_t* __restrict__ Ez,
    const scalar_t* __restrict__ aHy,
    const scalar_t* __restrict__ bHy,
    int ni, int nj, int nk,
    const int* __restrict__ blocks)
{
    /* Hy: (Nx, Ny+1, Nz)  Ex: (Nx, Ny+1, Nz+1)  Ez: (Nx+1, Ny+1, Nz) */
    DECODE_TILE;

    int idx = i * nj * nk + j * nk + k;

    int Ex_nk = nk + 1;
    int Ez_s1 = nj * nk;

    scalar_t curl = Ex[i * nj * Ex_nk + j * Ex_nk + (k + 1)]
                - Ex[i * nj * Ex_nk + j * Ex_nk + k]
                - Ez[(i + 1) * Ez_s1 + j * nk + k]
                + Ez[i * Ez_s1 + j * nk + k];

    Hy[idx] = aHy[idx] * Hy[idx] - bHy[idx] * curl;
}

__global__ void update_Hz(
    scalar_t* __restrict__ Hz,
    const scalar_t* __restrict__ Ex,
    const scalar_t* __restrict__ Ey,
    const scalar_t* __restrict__ aHz,
    const scalar_t* __restrict__ bHz,
    int ni, int nj, int nk,
    const int* __restrict__ blocks)
{
    /* Hz: (Nx, Ny, Nz+1)  Ey: (Nx+1, Ny, Nz+1)  Ex: (Nx, Ny+1, Nz+1) */
    DECODE_TILE;

    int idx = i * nj * nk + j * nk + k;

    int Ey_s1 = nj * nk;
    int Ex_nj = nj + 1;

    scalar_t curl = Ey[(i + 1) * Ey_s1 + j * nk + k]
                - Ey[i * Ey_s1 + j * nk + k]
                - Ex[i * Ex_nj * nk + (j + 1) * nk + k]
                + Ex[i * Ex_nj * nk + j * nk + k];

    Hz[idx] = aHz[idx] * Hz[idx] - bHz[idx] * curl;
}

}  /* extern "C" */
"""

# dtype -> tuple of 6 compiled kernels (plan WP2: one specialisation per
# scalar precision; float32 and float64 modules coexist; dense and
# listed launch variants are separate modules).
_cuda_funcs: dict = {}
#: Thread-block shape in (k, j, i) order == tile (2, 4, 32) in array
#: order — the WP-T1 default (dense-free, best robust capture); must
#: match ``solver._tile_skip.DEFAULT_TILE``.
_BLK = (32, 4, 2)


def _ceildiv(a, b):
    return (a + b - 1) // b


def _block_grid(shape):
    """Tile-grid dimensions of a component array under ``_BLK``."""
    ni, nj, nk = shape
    return (_ceildiv(ni, _BLK[2]), _ceildiv(nj, _BLK[1]), _ceildiv(nk, _BLK[0]))


def pack_block_ids(linear_ids, dims):
    """Pack linear tile ids into the kernel's (bi<<20 | bj<<10 | bk) form.

    ``linear_ids`` index the tile grid ``dims = (nbi, nbj, nbk)`` in C
    order (the ``solver._tile_skip`` plan layout).  Host-side NumPy in,
    NumPy out — the caller moves the result to the device once.
    """
    nbi, nbj, nbk = dims
    if nbi > 1024 or nbj > 1024 or nbk > 1024:
        raise ValueError(
            f"tile grid {dims} exceeds the 10-bit packed-id range (1024 tiles per axis)"
        )
    ids = _np.asarray(linear_ids, dtype=_np.int64)
    bi, rem = _np.divmod(ids, nbj * nbk)
    bj, bk = _np.divmod(rem, nbk)
    return ((bi << 20) | (bj << 10) | bk).astype(_np.int32)


def _compile_cuda(dtype, listed=False):
    """Lazy-compile the CUDA kernels; cached per (dtype, launch mode)."""
    key = (_np.dtype(dtype), listed)
    cached = _cuda_funcs.get(key)
    if cached is not None:
        return cached
    if key[0] == _np.float32:
        ctype = "float"
    elif key[0] == _np.float64:
        ctype = "double"
    else:
        raise TypeError(f"CUDA fused kernels support float32/float64, got {key[0]}")
    src = _CUDA_SOURCE.replace("SCALAR_T", ctype)
    if not listed:
        # The dense variant has no list parameter at all.
        src = src.replace(",\n    const int* __restrict__ blocks)", ")")
    mod = _RawModule(code=src, options=(f"-DLISTED={int(listed)}",))
    funcs = tuple(
        mod.get_function(name)
        for name in ("update_Ex", "update_Ey", "update_Ez", "update_Hx", "update_Hy", "update_Hz")
    )
    _cuda_funcs[key] = funcs
    return funcs


if HAS_CUPY:

    def _launch_dense(fn, out, in1, in2, a, b):
        """Direct 3-D-grid launch covering every tile of ``out``."""
        ni, nj, nk = out.shape
        grid = (_ceildiv(nk, _BLK[0]), _ceildiv(nj, _BLK[1]), _ceildiv(ni, _BLK[2]))
        fn(grid, _BLK, (out, in1, in2, a, b, _i32(ni), _i32(nj), _i32(nk)))

    def _launch_listed(fn, out, in1, in2, a, b, ids):
        """Launch one component kernel over its live-block list."""
        n = int(ids.size)
        if n == 0:  # component fully skipped
            return
        ni, nj, nk = out.shape
        fn((n, 1, 1), _BLK, (out, in1, in2, a, b, _i32(ni), _i32(nj), _i32(nk), ids))

    def update_E_fused_cuda(Ex, Ey, Ez, Hx, Hy, Hz, aEx, bEx, aEy, bEy, aEz, bEz, blocks=None):
        """Fused dual-curl + material E-update on GPU (one CUDA thread per edge).

        ``blocks`` maps component names to device int32 arrays of
        packed live-tile ids (see :func:`pack_block_ids`; tile grid
        built with the ``_BLK`` tile shape); None runs the dense
        variant, which launches every tile without the list
        indirection.
        """
        listed = blocks is not None
        kEx, kEy, kEz, _, _, _ = _compile_cuda(Ex.dtype, listed)
        if listed:
            _launch_listed(kEx, Ex, Hy, Hz, aEx, bEx, blocks["Ex"])
            _launch_listed(kEy, Ey, Hx, Hz, aEy, bEy, blocks["Ey"])
            _launch_listed(kEz, Ez, Hx, Hy, aEz, bEz, blocks["Ez"])
        else:
            _launch_dense(kEx, Ex, Hy, Hz, aEx, bEx)
            _launch_dense(kEy, Ey, Hx, Hz, aEy, bEy)
            _launch_dense(kEz, Ez, Hx, Hy, aEz, bEz)

    def update_H_fused_cuda(Ex, Ey, Ez, Hx, Hy, Hz, aHx, bHx, aHy, bHy, aHz, bHz, blocks=None):
        """Fused primal-curl + material H-update on GPU (one CUDA thread per face).

        ``blocks`` as in :func:`update_E_fused_cuda`.
        """
        listed = blocks is not None
        _, _, _, kHx, kHy, kHz = _compile_cuda(Hx.dtype, listed)
        if listed:
            _launch_listed(kHx, Hx, Ey, Ez, aHx, bHx, blocks["Hx"])
            _launch_listed(kHy, Hy, Ex, Ez, aHy, bHy, blocks["Hy"])
            _launch_listed(kHz, Hz, Ex, Ey, aHz, bHz, blocks["Hz"])
        else:
            _launch_dense(kHx, Hx, Ey, Ez, aHx, bHx)
            _launch_dense(kHy, Hy, Ex, Ez, aHy, bHy)
            _launch_dense(kHz, Hz, Ex, Ey, aHz, bHz)
else:
    update_E_fused_cuda = None
    update_H_fused_cuda = None
