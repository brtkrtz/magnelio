"""Numba CPU kernels: fused E/H updates against the NumPy stencil
reference on every shell configuration, the temporaries-free energy
reduction, and the PEC-face skip of the time loop."""

import numpy as np
import pytest

from magnelio._operators import numba_kernels as nk
from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

pytestmark = pytest.mark.skipif(not nk.HAS_NUMBA, reason="numba not installed")

# Degenerate extents exercise every shell path of the plane-by-plane
# sweep: single planes, single rows, single columns, and a box with
# a real interior.
SHAPES = [(1, 1, 1), (2, 1, 3), (1, 4, 2), (3, 3, 3), (5, 7, 6), (17, 9, 13)]
FACES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _state(rng, Nx, Ny, Nz, dtype):
    def r(*shape):
        return rng.random(shape).astype(dtype)

    fields = [
        r(Nx, Ny + 1, Nz + 1),
        r(Nx + 1, Ny, Nz + 1),
        r(Nx + 1, Ny + 1, Nz),
        r(Nx + 1, Ny, Nz),
        r(Nx, Ny + 1, Nz),
        r(Nx, Ny, Nz + 1),
    ]
    coef_E = [r(*fields[c // 2].shape) for c in range(6)]
    coef_H = [r(*fields[3 + c // 2].shape) for c in range(6)]
    return fields, coef_E, coef_H


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", SHAPES)
def test_fused_kernels_match_the_numpy_stencil(shape, dtype):
    rng = np.random.default_rng(7)
    fields, coef_E, coef_H = _state(rng, *shape, dtype)
    ref = [f.copy() for f in fields]
    got = [f.copy() for f in fields]

    bufs_E = [np.empty_like(f) for f in ref[:3]]
    bufs_H = [np.empty_like(f) for f in ref[3:]]
    nk.update_E_stencil(*ref, *coef_E, *bufs_E)
    nk.update_H_stencil(*ref, *coef_H, *bufs_H)

    nk.update_E_fused(*got, *coef_E)
    nk.update_H_fused(*got, *coef_H)

    # The fused E-update accumulates its curl in double; the stencil
    # works in the field dtype, so single precision differs at the ulp.
    tol = 1e-5 if dtype is np.float32 else 1e-12
    for a, b in zip(ref, got, strict=True):
        np.testing.assert_allclose(b, a, rtol=tol, atol=tol)


def test_weighted_dot_accumulates_in_double():
    rng = np.random.default_rng(3)
    w = rng.random(100_003).astype(np.float32)
    a = rng.standard_normal(100_003).astype(np.float32)
    b = rng.standard_normal(100_003).astype(np.float32)
    ref = float(np.sum(w.astype(np.float64) * a.astype(np.float64) * b.astype(np.float64)))
    assert nk.weighted_dot(w, a, b) == pytest.approx(ref, rel=1e-12)


def _solver(mesh, dt, seed):
    bcs = {f: PECBoundary(f) for f in FACES}
    s = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        sources=[],
        total_time_steps=24,
        dt=dt,
        verbose=False,
        backend="numpy",
        precision="single",
        energy_check_interval=4,
    )
    s.setup()
    rng = np.random.default_rng(seed)
    f = s._fields
    f.e_flat[:] = rng.standard_normal(f.e_flat.size).astype(np.float32)
    f.h_flat[:] = rng.standard_normal(f.h_flat.size).astype(np.float32)
    for face in FACES:
        PECBoundary(face).apply(f)
    return s


class TestPECFaceSkip:
    def _mesh(self, n=8):
        lines = np.linspace(0.0, 1e-2, n + 1)
        grid = GridLines(x=lines, y=lines, z=lines)
        return Mesh.from_grid(grid), 0.9 * courant_dt(grid)

    def test_consolidated_faces_are_frozen(self):
        mesh, dt = self._mesh()
        s = _solver(mesh, dt, 1)
        assert s._pec_faces_frozen == frozenset(FACES)

    def test_unmasked_face_is_not_frozen(self):
        mesh, dt = self._mesh()
        s = _solver(mesh, dt, 1)
        s._pec_mask_E[:] = False
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        assert not any(s._face_edges_all_pec(face, Nx, Ny, Nz) for face in FACES)

    def test_skipped_apply_leaves_the_march_bit_identical(self):
        mesh, dt = self._mesh()
        frozen = _solver(mesh, dt, 5)
        forced = _solver(mesh, dt, 5)
        forced._pec_faces_frozen = frozenset()
        a = frozen._run_loop()
        b = forced._run_loop()
        assert np.array_equal(a.e_flat, b.e_flat)
        assert np.array_equal(a.h_flat, b.h_flat)
        f = frozen._fields
        assert abs(f.Ey[0]).max() == 0.0 and abs(f.Ex[:, :, -1]).max() == 0.0
