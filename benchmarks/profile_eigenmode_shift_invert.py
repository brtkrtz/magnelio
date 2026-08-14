"""DD-033 evaluation: GPU shift-invert for the 3D eigenmode solver.

Mirrors ``EigenmodeSolver3D.solve()`` up to the ARPACK call on a vacuum
PEC cavity, then times each stage separately:

  1. matrix assembly (M_eps/M_mu/curl, ``A_f = C^T diag(1/mu) C``)
  2. SuperLU factorisation of ``(A_f - sigma*B_f)`` — the DD-033
     bottleneck
  3. ARPACK ``eigsh`` iterations with an instrumented ``OPinv``:
       - CPU: scipy SuperLU triangular solves
       - GPU: ``cupyx...SuperLU`` wrapper around the *same host factors*
         (cuSPARSE spsm triangular solves on device)

Also measures raw per-solve latency CPU vs GPU on the identical
factors, outside ARPACK, separating transfer overhead from solve
throughput.

Verdict (2026-07-18, RTX 4070 SUPER, 16-core host — recorded in DD-033
and PERFORMANCE_PROFILING_PLAN.md): GPU shift-invert via CuPy is
rejected on two independent measured grounds.  (a) The factorisation
is 93-98 % of total time and runs on the CPU by cupyx design
(``splu``/``factorized`` call scipy on the host).  (b) The part that
does move — the triangular solves — is ~74x SLOWER on the GPU
(9.25 s vs 0.125 s per solve at N=30^3, device-resident, transfers
irrelevant): the 215x-fill LU factors have long sequential dependency
chains that defeat cuSPARSE spsm.  Results agree to ~5e-14.

Note: on hosts where the system CUDA headers (/usr/local/cuda) do not
match CuPy's bundled NVRTC, the CUB reduction kernel behind
``has_canonical_format`` fails to compile; run with
``CUPY_ACCELERATORS=""`` to fall back to the plain reduction path.

Usage:
  python benchmarks/profile_eigenmode_shift_invert.py --sizes 30,40 [--gpu]
"""

import argparse
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from magnelio._operators.curl import build_curl_matrix
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver._eigenmode_3d import (
    _build_pec_dof_mask,
    _estimate_sigma,
)

# Null-space (gradient) modes come out at ~0 Hz; only compare above this.
_F_PHYSICAL_MIN = 1e6


def build_case(N):
    a = 30e-3
    grid = GridLines(
        x=np.linspace(0, a, N + 1),
        y=np.linspace(0, a, N + 1),
        z=np.linspace(0, a, N + 1),
    )
    mesh = Mesh.from_grid(grid)
    g = mesh.grid

    t0 = time.perf_counter()
    M_eps = build_M_eps(mesh)
    M_mu = build_M_mu(mesh)
    M_mu_inv = np.where(M_mu > 0, 1.0 / np.where(M_mu > 0, M_mu, 1.0), 0.0)

    bcs = {f: "pec" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    pec_mask = _build_pec_dof_mask(g, bcs)
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    mat_pec = mesh.pec_mask_edges
    pec_mask[:n_Ex] |= mat_pec[0, :n_Ex]
    pec_mask[n_Ex : n_Ex + n_Ey] |= mat_pec[1, :n_Ey]
    pec_mask[n_Ex + n_Ey :] |= mat_pec[2, :n_Ez]
    free_idx = np.where(~pec_mask)[0]

    C = build_curl_matrix(g)
    A = C.T @ sp.diags(M_mu_inv, 0, format="csr") @ C
    M_eps_safe = np.where(M_eps > 0, M_eps, np.finfo(float).tiny)
    B = sp.diags(M_eps_safe, 0, format="csr")
    A_f = A[np.ix_(free_idx, free_idx)].tocsr()
    B_f = B[np.ix_(free_idx, free_idx)].tocsr()
    t_asm = time.perf_counter() - t0

    sigma = _estimate_sigma(g, bcs, 1.0)
    return A_f, B_f, sigma, t_asm


class CountingOp(LinearOperator):
    """OPinv wrapper counting calls and accumulated solve time."""

    def __init__(self, n, solve_fn):
        super().__init__(dtype=np.float64, shape=(n, n))
        self.solve_fn = solve_fn
        self.n_calls = 0
        self.t_total = 0.0

    def _matvec(self, b):
        t0 = time.perf_counter()
        x = self.solve_fn(b)
        self.t_total += time.perf_counter() - t0
        self.n_calls += 1
        return x


def _physical(freqs):
    return freqs[freqs >= _F_PHYSICAL_MIN]


def run_size(N, use_gpu):
    print(f"\n=== N={N}^3 cavity ===", flush=True)
    A_f, B_f, sigma, t_asm = build_case(N)
    n_free = A_f.shape[0]
    nnz = A_f.nnz
    print(f"n_free={n_free:,d}  nnz(A_f)={nnz:,d}  assembly {t_asm:.2f} s", flush=True)

    t0 = time.perf_counter()
    shifted = (A_f - sigma * B_f).tocsc()
    lu = splu(shifted)
    t_factor = time.perf_counter() - t0
    fill = (lu.L.nnz + lu.U.nnz) / nnz
    mem_gb = (lu.L.nnz + lu.U.nnz) * 12e-9
    print(
        f"SuperLU factor: {t_factor:.2f} s  fill {fill:.1f}x  ~{mem_gb:.2f} GB factors", flush=True
    )

    k = 9

    op_cpu = CountingOp(n_free, lu.solve)
    t0 = time.perf_counter()
    vals_cpu, _ = eigsh(A_f, M=B_f, k=k, which="LM", sigma=sigma, OPinv=op_cpu)
    t_arpack_cpu = time.perf_counter() - t0
    f_cpu = np.sqrt(np.maximum(np.sort(vals_cpu), 0)) / (2 * np.pi)
    print(
        f"ARPACK/CPU: {t_arpack_cpu:.2f} s total, "
        f"{op_cpu.n_calls} solves, {op_cpu.t_total:.2f} s in solves "
        f"({1e3 * op_cpu.t_total / op_cpu.n_calls:.1f} ms/solve)",
        flush=True,
    )
    print(f"  physical f = {_physical(f_cpu) / 1e9} GHz", flush=True)

    total_cpu = t_asm + t_factor + t_arpack_cpu
    print(
        f"CPU total: {total_cpu:.2f} s  "
        f"(assembly {100 * t_asm / total_cpu:.0f} % / "
        f"factor {100 * t_factor / total_cpu:.0f} % / "
        f"arpack {100 * t_arpack_cpu / total_cpu:.0f} %)",
        flush=True,
    )

    if not use_gpu:
        return

    import cupy
    import cupyx.scipy.sparse.linalg as csl

    t0 = time.perf_counter()
    lu_gpu = csl.SuperLU(lu)
    cupy.cuda.get_current_stream().synchronize()
    t_upload = time.perf_counter() - t0

    def gpu_solve(b):
        x = lu_gpu.solve(cupy.asarray(b))
        return cupy.asnumpy(x)

    gpu_solve(np.random.default_rng(0).standard_normal(n_free))  # warm-up

    b_host = np.random.default_rng(1).standard_normal(n_free)
    x_ref = lu.solve(b_host)
    x_gpu = gpu_solve(b_host)
    rel = np.linalg.norm(x_gpu - x_ref) / np.linalg.norm(x_ref)
    print(f"GPU factors upload: {t_upload:.2f} s; solve rel. diff vs CPU: {rel:.2e}", flush=True)

    n_rep = 20
    t0 = time.perf_counter()
    for _ in range(n_rep):
        lu.solve(b_host)
    t_cpu_solve = (time.perf_counter() - t0) / n_rep
    t0 = time.perf_counter()
    for _ in range(n_rep):
        gpu_solve(b_host)
    t_gpu_solve = (time.perf_counter() - t0) / n_rep
    b_dev = cupy.asarray(b_host)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        lu_gpu.solve(b_dev)
    cupy.cuda.get_current_stream().synchronize()
    t_gpu_dev = (time.perf_counter() - t0) / n_rep
    print(
        f"per-solve: CPU {1e3 * t_cpu_solve:.1f} ms | "
        f"GPU incl. transfers {1e3 * t_gpu_solve:.1f} ms | "
        f"GPU device-resident {1e3 * t_gpu_dev:.1f} ms",
        flush=True,
    )

    op_gpu = CountingOp(n_free, gpu_solve)
    t0 = time.perf_counter()
    vals_gpu, _ = eigsh(A_f, M=B_f, k=k, which="LM", sigma=sigma, OPinv=op_gpu)
    t_arpack_gpu = time.perf_counter() - t0
    f_gpu = np.sqrt(np.maximum(np.sort(vals_gpu), 0)) / (2 * np.pi)
    fp_cpu, fp_gpu = _physical(f_cpu), _physical(f_gpu)
    n_cmp = min(fp_cpu.size, fp_gpu.size)
    if n_cmp:
        df = np.max(np.abs(fp_gpu[:n_cmp] - fp_cpu[:n_cmp]) / fp_cpu[:n_cmp])
        agree = f"max rel. physical-freq diff vs CPU {df:.2e}"
    else:
        agree = "no physical modes among k requested (null-space only)"
    print(
        f"ARPACK/GPU-OPinv: {t_arpack_gpu:.2f} s total, "
        f"{op_gpu.n_calls} solves, {op_gpu.t_total:.2f} s in solves; "
        f"{agree}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="30,40")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    for N in [int(s) for s in args.sizes.split(",")]:
        run_size(N, args.gpu)


if __name__ == "__main__":
    main()
