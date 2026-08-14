"""GPU tests for the block-list fused kernels (TILE_SKIP_PLAN WP-T3).

The launch scheme is elementwise (one thread, one element, fixed
in-thread operation order), so every comparison here asserts BIT
identity, not closeness.
"""

import numpy as np
import pytest

from magnelio._backend.array_api import resolve_backend

try:
    resolve_backend("cupy")
    HAS_GPU = True
except Exception:
    HAS_GPU = False

gpu = pytest.mark.skipif(not HAS_GPU, reason="no usable CuPy/CUDA device")


def _component_shapes(Nx, Ny, Nz):
    e = {"Ex": (Nx, Ny + 1, Nz + 1), "Ey": (Nx + 1, Ny, Nz + 1), "Ez": (Nx + 1, Ny + 1, Nz)}
    h = {"Hx": (Nx + 1, Ny, Nz), "Hy": (Nx, Ny + 1, Nz), "Hz": (Nx, Ny, Nz + 1)}
    return e, h


def _random_state(Nx, Ny, Nz, dtype, seed=3):
    rng = np.random.default_rng(seed)
    shapes_E, shapes_H = _component_shapes(Nx, Ny, Nz)
    fields = {n: rng.standard_normal(s).astype(dtype) for n, s in {**shapes_E, **shapes_H}.items()}
    coefs = {
        n: (rng.standard_normal(s).astype(dtype), rng.standard_normal(s).astype(dtype))
        for n, s in {**shapes_E, **shapes_H}.items()
    }
    return fields, coefs


def _reference_sweep(f, c):
    """NumPy replica of both fused kernels, same in-thread op order."""
    aEx, bEx = c["Ex"]
    aEy, bEy = c["Ey"]
    aEz, bEz = c["Ez"]
    aHx, bHx = c["Hx"]
    aHy, bHy = c["Hy"]
    aHz, bHz = c["Hz"]
    Ex, Ey, Ez = f["Ex"], f["Ey"], f["Ez"]
    Hx, Hy, Hz = f["Hx"], f["Hy"], f["Hz"]

    curl = np.zeros_like(Ex)
    curl[:, :-1, :] += Hz
    curl[:, 1:, :] -= Hz
    curl[:, :, :-1] -= Hy
    curl[:, :, 1:] += Hy
    f["Ex"] = aEx * Ex + bEx * curl

    curl = np.zeros_like(Ey)
    curl[:, :, :-1] += Hx
    curl[:, :, 1:] -= Hx
    curl[:-1, :, :] -= Hz
    curl[1:, :, :] += Hz
    f["Ey"] = aEy * Ey + bEy * curl

    curl = np.zeros_like(Ez)
    curl[:-1, :, :] += Hy
    curl[1:, :, :] -= Hy
    curl[:, :-1, :] -= Hx
    curl[:, 1:, :] += Hx
    f["Ez"] = aEz * Ez + bEz * curl

    Ex, Ey, Ez = f["Ex"], f["Ey"], f["Ez"]
    curl = Ez[:, 1:, :] - Ez[:, :-1, :] - Ey[:, :, 1:] + Ey[:, :, :-1]
    f["Hx"] = aHx * Hx - bHx * curl
    curl = Ex[:, :, 1:] - Ex[:, :, :-1] - Ez[1:, :, :] + Ez[:-1, :, :]
    f["Hy"] = aHy * Hy - bHy * curl
    curl = Ey[1:, :, :] - Ey[:-1, :, :] - Ex[:, 1:, :] + Ex[:, :-1, :]
    f["Hz"] = aHz * Hz - bHz * curl


def _device_sweep(f, c, blocks_E=None, blocks_H=None):
    from magnelio._operators.numba_kernels import (
        update_E_fused_cuda,
        update_H_fused_cuda,
    )

    args = [f["Ex"], f["Ey"], f["Ez"], f["Hx"], f["Hy"], f["Hz"]]
    update_E_fused_cuda(*args, *c["Ex"], *c["Ey"], *c["Ez"], blocks=blocks_E)
    update_H_fused_cuda(*args, *c["Hx"], *c["Hy"], *c["Hz"], blocks=blocks_H)


@gpu
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_identity_blocklist_matches_reference(dtype):
    """Dense (identity-list) launch == elementwise NumPy replica.

    Tight allclose, not bitwise: the compiler may fuse
    ``a * f + b * curl`` into an FMA.  Bitwise identity is asserted
    where it is claimed — dense vs skipping, same kernel — below.
    """
    import cupy as cp

    Nx, Ny, Nz = 9, 7, 11  # deliberately non-multiples of the tile
    ref_f, coefs = _random_state(Nx, Ny, Nz, dtype)
    dev_f = {n: cp.asarray(v) for n, v in ref_f.items()}
    dev_c = {n: (cp.asarray(a), cp.asarray(b)) for n, (a, b) in coefs.items()}

    # One sweep: any decode/mapping defect shows immediately, while
    # chained sweeps with random O(1) coefficients would amplify the
    # FMA rounding difference exponentially.
    _reference_sweep(ref_f, coefs)
    _device_sweep(dev_f, dev_c)

    rtol = 1e-5 if dtype == np.float32 else 1e-13
    for name in ref_f:
        got = cp.asnumpy(dev_f[name])
        assert got.dtype == ref_f[name].dtype
        np.testing.assert_allclose(got, ref_f[name], rtol=rtol, atol=rtol, err_msg=name)


@gpu
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_skipping_dead_tiles_is_bit_identical(dtype):
    """Skipping no-op tiles leaves the marched state bit-identical."""
    import cupy as cp

    from magnelio.solver._tile_skip import build_tile_skip_plan

    Nx, Ny, Nz = 12, 9, 37
    fields, coefs = _random_state(Nx, Ny, Nz, dtype)
    shapes_E, shapes_H = _component_shapes(Nx, Ny, Nz)

    # PEC half-space i >= 6: alpha_E = beta_E = 0 and e = 0 there.
    # Every H face with all four bounding edges in the slab is
    # curl-dead; its h must start at the provable 0 (the invariant
    # the production plan enforces via its zeroing indices).
    for name in shapes_E:
        a, b = coefs[name]
        a[6:], b[6:] = 0.0, 0.0
        fields[name][6:] = 0.0
    for name in shapes_H:
        a, b = coefs[name]
        a[6:] = 1.0
        fields[name][6:] = 0.0

    plan = build_tile_skip_plan(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        alpha_E=np.concatenate([coefs[n][0].ravel() for n in shapes_E]),
        beta_E=np.concatenate([coefs[n][1].ravel() for n in shapes_E]),
        alpha_H=np.concatenate([coefs[n][0].ravel() for n in shapes_H]),
        beta_H=np.concatenate([coefs[n][1].ravel() for n in shapes_H]),
    )
    n_all = sum(int(np.prod(plan.block_grids[n])) for n in plan.block_grids)
    n_live = sum(v.size for v in plan.live_blocks.values())
    assert n_live < n_all  # something is actually skipped

    def to_dev(state):
        return (
            {n: cp.asarray(v) for n, v in state[0].items()},
            {n: (cp.asarray(a), cp.asarray(b)) for n, (a, b) in state[1].items()},
        )

    from magnelio._operators.numba_kernels import pack_block_ids

    dense_f, dense_c = to_dev((fields, coefs))
    skip_f, skip_c = to_dev((fields, coefs))
    blocks_E = {
        n: cp.asarray(pack_block_ids(plan.live_blocks[n], plan.block_grids[n])) for n in shapes_E
    }
    blocks_H = {
        n: cp.asarray(pack_block_ids(plan.live_blocks[n], plan.block_grids[n])) for n in shapes_H
    }

    for _ in range(5):
        _device_sweep(dense_f, dense_c)
        _device_sweep(skip_f, skip_c, blocks_E, blocks_H)

    for name in dense_f:
        np.testing.assert_array_equal(
            cp.asnumpy(skip_f[name]), cp.asnumpy(dense_f[name]), err_msg=name
        )


@gpu
def test_empty_component_list_is_noop():
    """A fully-skipped component must not launch (grid dim 0 guard)."""
    import cupy as cp

    Nx, Ny, Nz = 4, 4, 8
    fields, coefs = _random_state(Nx, Ny, Nz, np.float64)
    dev_f = {n: cp.asarray(v) for n, v in fields.items()}
    dev_c = {n: (cp.asarray(a), cp.asarray(b)) for n, (a, b) in coefs.items()}
    empty = cp.zeros(0, dtype=cp.int32)
    shapes_E, shapes_H = _component_shapes(Nx, Ny, Nz)
    blocks_E = {n: empty for n in shapes_E}
    blocks_H = {n: empty for n in shapes_H}
    before = {n: cp.asnumpy(v) for n, v in dev_f.items()}
    _device_sweep(dev_f, dev_c, blocks_E, blocks_H)
    for name, v in dev_f.items():
        np.testing.assert_array_equal(cp.asnumpy(v), before[name])
