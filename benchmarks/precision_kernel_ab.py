"""WP0 measurement gate: raw single-vs-double throughput of the fused FIT-TD
E/H update kernels on the local CUDA device.

This is a *standalone* micro-benchmark — it does not import the solver.  It
templates the production CUDA kernel bodies (a verbatim copy of
``operators/numba_kernels.py`` ``_CUDA_SOURCE``) over the scalar pointer type
and times an E+H leapfrog sweep for ``float32`` and ``float64`` at the grid
sizes used by ``profile_solver.py`` (95k / 379k cells).  It also times a pure
CuPy elementwise leapfrog as a bandwidth upper-bound reference.

Purpose: decide, with real numbers on the target card (RTX 4070 SUPER, Ada,
FP64:FP32 = 1:64), whether the float32 CUDA-kernel work (plan WP2) is warranted
before it is invested.  The leapfrog is bandwidth-bound, so the *guaranteed*
win is ~2x from halved memory traffic; any excess is the crippled-FP64-ALU
bonus surfacing.

Run on the CUDA machine (needs a real GPU; this is NOT runnable on a CPU host):

    ~/.local/share/mamba/envs/mio/bin/python benchmarks/precision_kernel_ab.py

Note (per project memory): a fresh git worktree has a cold Numba cache — not
relevant here (pure CuPy), but the CUDA device is invisible to a
``python - <<EOF`` heredoc, so always run this as a real script file.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ImportError:
    raise SystemExit("cupy not importable — run this on the CUDA machine.")


# ── kernel source, templated over the array scalar type ──────────────────
# Verbatim from operators/numba_kernels.py, with the pointer type turned into
# `scalar_t` (typedef prelude) while the local `curl` accumulator stays
# `double` — the "single storage, double accumulation" policy of plan WP2.

_CUDA_TEMPLATE = r"""
typedef {scalar} scalar_t;
typedef {acc} acc_t;
extern "C" {{

__global__ void update_Ex(scalar_t* __restrict__ Ex,
    const scalar_t* __restrict__ Hy, const scalar_t* __restrict__ Hz,
    const scalar_t* __restrict__ aEx, const scalar_t* __restrict__ bEx,
    int ni, int nj, int nk)
{{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z;
    if (j >= nj || k >= nk) return;
    int idx = i*nj*nk + j*nk + k;
    int Hz_s1 = (nj-1)*nk;  int Hy_s2 = nk-1;
    acc_t curl = 0.0;
    if (j < nj-1) curl += Hz[i*Hz_s1 + j*nk + k];
    if (j > 0)    curl -= Hz[i*Hz_s1 + (j-1)*nk + k];
    if (k < nk-1) curl -= Hy[i*nj*Hy_s2 + j*Hy_s2 + k];
    if (k > 0)    curl += Hy[i*nj*Hy_s2 + j*Hy_s2 + (k-1)];
    Ex[idx] = aEx[idx]*Ex[idx] + bEx[idx]*curl;
}}

__global__ void update_Ey(scalar_t* __restrict__ Ey,
    const scalar_t* __restrict__ Hx, const scalar_t* __restrict__ Hz,
    const scalar_t* __restrict__ aEy, const scalar_t* __restrict__ bEy,
    int ni, int nj, int nk)
{{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z;
    if (j >= nj || k >= nk) return;
    int idx = i*nj*nk + j*nk + k;
    int Hx_nk = nk-1;  int Hz_s1 = nj*nk;
    acc_t curl = 0.0;
    if (k < nk-1) curl += Hx[i*nj*Hx_nk + j*Hx_nk + k];
    if (k > 0)    curl -= Hx[i*nj*Hx_nk + j*Hx_nk + (k-1)];
    if (i < ni-1) curl -= Hz[i*Hz_s1 + j*nk + k];
    if (i > 0)    curl += Hz[(i-1)*Hz_s1 + j*nk + k];
    Ey[idx] = aEy[idx]*Ey[idx] + bEy[idx]*curl;
}}

__global__ void update_Ez(scalar_t* __restrict__ Ez,
    const scalar_t* __restrict__ Hx, const scalar_t* __restrict__ Hy,
    const scalar_t* __restrict__ aEz, const scalar_t* __restrict__ bEz,
    int ni, int nj, int nk)
{{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z;
    if (j >= nj || k >= nk) return;
    int idx = i*nj*nk + j*nk + k;
    int Hy_s1 = nj*nk;  int Hx_nj = nj-1;
    acc_t curl = 0.0;
    if (i < ni-1) curl += Hy[i*Hy_s1 + j*nk + k];
    if (i > 0)    curl -= Hy[(i-1)*Hy_s1 + j*nk + k];
    if (j < nj-1) curl -= Hx[i*Hx_nj*nk + j*nk + k];
    if (j > 0)    curl += Hx[i*Hx_nj*nk + (j-1)*nk + k];
    Ez[idx] = aEz[idx]*Ez[idx] + bEz[idx]*curl;
}}

__global__ void update_Hx(scalar_t* __restrict__ Hx,
    const scalar_t* __restrict__ Ey, const scalar_t* __restrict__ Ez,
    const scalar_t* __restrict__ aHx, const scalar_t* __restrict__ bHx,
    int ni, int nj, int nk)
{{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z;
    if (j >= nj || k >= nk) return;
    int idx = i*nj*nk + j*nk + k;
    int Ez_s1 = (nj+1)*nk;  int Ey_nk = nk+1;
    acc_t curl = Ez[i*Ez_s1 + (j+1)*nk + k] - Ez[i*Ez_s1 + j*nk + k]
                - Ey[i*nj*Ey_nk + j*Ey_nk + (k+1)] + Ey[i*nj*Ey_nk + j*Ey_nk + k];
    Hx[idx] = aHx[idx]*Hx[idx] - bHx[idx]*curl;
}}

__global__ void update_Hy(scalar_t* __restrict__ Hy,
    const scalar_t* __restrict__ Ex, const scalar_t* __restrict__ Ez,
    const scalar_t* __restrict__ aHy, const scalar_t* __restrict__ bHy,
    int ni, int nj, int nk)
{{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z;
    if (j >= nj || k >= nk) return;
    int idx = i*nj*nk + j*nk + k;
    int Ex_nk = nk+1;  int Ez_s1 = nj*nk;
    acc_t curl = Ex[i*nj*Ex_nk + j*Ex_nk + (k+1)] - Ex[i*nj*Ex_nk + j*Ex_nk + k]
                - Ez[(i+1)*Ez_s1 + j*nk + k] + Ez[i*Ez_s1 + j*nk + k];
    Hy[idx] = aHy[idx]*Hy[idx] - bHy[idx]*curl;
}}

__global__ void update_Hz(scalar_t* __restrict__ Hz,
    const scalar_t* __restrict__ Ex, const scalar_t* __restrict__ Ey,
    const scalar_t* __restrict__ aHz, const scalar_t* __restrict__ bHz,
    int ni, int nj, int nk)
{{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z;
    if (j >= nj || k >= nk) return;
    int idx = i*nj*nk + j*nk + k;
    int Ey_s1 = nj*nk;  int Ex_nj = nj+1;
    acc_t curl = Ey[(i+1)*Ey_s1 + j*nk + k] - Ey[i*Ey_s1 + j*nk + k]
                - Ex[i*Ex_nj*nk + (j+1)*nk + k] + Ex[i*Ex_nj*nk + j*nk + k];
    Hz[idx] = aHz[idx]*Hz[idx] - bHz[idx]*curl;
}}

}}
"""

_BLK = (32, 8, 1)


def _ceildiv(a, b):
    return (a + b - 1) // b


def _grid3d(ni, nj, nk):
    return (_ceildiv(nk, _BLK[0]), _ceildiv(nj, _BLK[1]), ni)


def _yee_shapes(N):
    """Component shapes for an N x N x N cell grid (Yee staggering)."""
    return {
        "Ex": (N, N + 1, N + 1),
        "Ey": (N + 1, N, N + 1),
        "Ez": (N + 1, N + 1, N),
        "Hx": (N + 1, N, N),
        "Hy": (N, N + 1, N),
        "Hz": (N, N, N + 1),
    }


def _alloc(shapes, dtype):
    f = {}
    for name, shp in shapes.items():
        # small nonzero values so the update actually reads/writes
        f[name] = cp.full(shp, 1e-3, dtype=dtype)
        f["a" + name] = cp.full(shp, 0.999, dtype=dtype)
        f["b" + name] = cp.full(shp, 1e-3, dtype=dtype)
    return f


def _bench_raw_kernels(N, dtype, acc="double", n_steps=2000, warmup=50):
    shapes = _yee_shapes(N)
    f = _alloc(shapes, dtype)
    scalar = "float" if dtype == cp.float32 else "double"
    src = _CUDA_TEMPLATE.format(scalar=scalar, acc=acc)
    mod = cp.RawModule(code=src)
    K = {
        n: mod.get_function(n)
        for n in ("update_Ex", "update_Ey", "update_Ez", "update_Hx", "update_Hy", "update_Hz")
    }
    i32 = np.int32

    def step():
        for comp, (a, b) in (("Ex", ("Hy", "Hz")), ("Ey", ("Hx", "Hz")), ("Ez", ("Hx", "Hy"))):
            ni, nj, nk = shapes[comp]
            K["update_" + comp](
                _grid3d(ni, nj, nk),
                _BLK,
                (f[comp], f[a], f[b], f["a" + comp], f["b" + comp], i32(ni), i32(nj), i32(nk)),
            )
        for comp, (a, b) in (("Hx", ("Ey", "Ez")), ("Hy", ("Ex", "Ez")), ("Hz", ("Ex", "Ey"))):
            ni, nj, nk = shapes[comp]
            K["update_" + comp](
                _grid3d(ni, nj, nk),
                _BLK,
                (f[comp], f[a], f[b], f["a" + comp], f["b" + comp], i32(ni), i32(nj), i32(nk)),
            )

    for _ in range(warmup):
        step()
    cp.cuda.runtime.deviceSynchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    for _ in range(n_steps):
        step()
    end.record()
    end.synchronize()
    ms_total = cp.cuda.get_elapsed_time(start, end)
    return ms_total / n_steps


def main():
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print(f"device: {props['name'].decode()}")
    # Three variants: f64 baseline; f32 with a DOUBLE curl accumulator (the
    # plan's "single storage, double accumulation" policy); f32 with a FLOAT
    # curl accumulator (pure single).  On a 1:64-FP64 consumer card the double
    # curl may cost more than the halved memory traffic saves — measure it.
    hdr = (
        f"{'grid':>8} {'cells':>9} {'f64':>10} {'f32/dblcurl':>12} "
        f"{'f32/fltcurl':>12} {'sp(dbl)':>8} {'sp(flt)':>8}"
    )
    print(hdr)
    for N in (46, 72):  # ~97k and ~373k cells (profile_solver sizes)
        cells = N**3
        t64 = _bench_raw_kernels(N, cp.float64, acc="double")
        t32d = _bench_raw_kernels(N, cp.float32, acc="double")
        t32f = _bench_raw_kernels(N, cp.float32, acc="float")
        print(
            f"{N}^3   {cells:>9} {t64:>10.4f} {t32d:>12.4f} "
            f"{t32f:>12.4f} {t64 / t32d:>7.2f}x {t64 / t32f:>7.2f}x"
        )


if __name__ == "__main__":
    main()
