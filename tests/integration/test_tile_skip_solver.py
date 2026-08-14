"""Solver-level gates for dead-tile skipping (TILE_SKIP_PLAN WP-T4).

Marches the same PEC-heavy cavity with skipping enabled and
disabled and asserts BIT identity of the final field state, with
CUDA graphs active (the launch lists must survive capture/replay).
"""

import numpy as np
import pytest

from magnelio._backend.array_api import resolve_backend
from magnelio.boundaries.cpml import CPMLBoundary
from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver._tile_skip import component_shapes
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

try:
    resolve_backend("cupy")
    HAS_GPU = True
except Exception:
    HAS_GPU = False

gpu = pytest.mark.skipif(not HAS_GPU, reason="no usable CuPy/CUDA device")


def _pec_slab_solver(precision, with_cpml=False, n_steps=40):
    """Cavity 20x16x24 with every edge at i >= 10 PEC-masked."""
    Nx, Ny, Nz = 20, 16, 24
    grid = GridLines(
        x=np.linspace(0, Nx * 1e-3, Nx + 1),
        y=np.linspace(0, Ny * 1e-3, Ny + 1),
        z=np.linspace(0, Nz * 1e-3, Nz + 1),
    )
    mesh = Mesh.from_grid(grid)
    shapes_E, _ = component_shapes(Nx, Ny, Nz)
    for axis, name in enumerate(("Ex", "Ey", "Ez")):
        n = int(np.prod(shapes_E[name]))
        mesh.pec_mask_edges[axis, :n].reshape(shapes_E[name])[10:] = True
    bcs = {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    if with_cpml:
        bcs["zmin"] = CPMLBoundary(face="zmin", grid=grid, thickness_cells=4)
        bcs["zmax"] = CPMLBoundary(face="zmax", grid=grid, thickness_cells=4)
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        backend="cupy",
        precision=precision,
        total_time_steps=n_steps,
        dt=courant_dt(grid, "draft"),
        verbose=False,
    )
    solver.setup()
    solver._fields.Ex[3, 5, 7] = 1.0  # impulse in the live half
    return solver


def _final_state(solver):
    import cupy as cp

    fields = solver.run()
    return {n: cp.asnumpy(getattr(fields, n)) for n in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}


@gpu
@pytest.mark.parametrize("precision", ["double", "single"])
@pytest.mark.parametrize("with_cpml", [False, True])
def test_skip_marches_bit_identical(monkeypatch, precision, with_cpml):
    skip = _pec_slab_solver(precision, with_cpml)
    assert skip._tile_skip_stats is not None
    assert skip._tile_skip_stats["total"] > 0.10
    state_skip = _final_state(skip)
    assert skip._gpu_graphs is not None and skip._gpu_graphs.ready

    monkeypatch.setenv("MAGNELIO_TILE_SKIP", "0")
    dense = _pec_slab_solver(precision, with_cpml)
    assert dense._tile_skip_stats is None
    state_dense = _final_state(dense)

    moved = max(np.abs(v).max() for v in state_skip.values())
    assert moved > 0  # the march actually marched
    for name in state_skip:
        np.testing.assert_array_equal(state_skip[name], state_dense[name], err_msg=name)


@gpu
def test_resume_normalisation_zeroes_dead_elements():
    """Garbage in provably-dead elements is cleared before the march."""
    import cupy as cp

    solver = _pec_slab_solver("double")
    # Pollute a deep-PEC Ex edge as a corrupted resume would.
    solver._fields.Ex[15, 8, 12] = 123.0
    fields = solver.run()
    assert float(cp.asnumpy(fields.Ex[15, 8, 12])) == 0.0


@gpu
def test_field_sources_fall_back_to_dense():
    class _FakeSource:
        def inject_E(self, fields, t):
            pass

    solver = _pec_slab_solver("double")
    assert solver._tile_skip_stats is not None
    src = FITTimeDomainSolver(
        mesh=solver.mesh,
        boundary_conditions=solver.boundary_conditions,
        backend="cupy",
        total_time_steps=5,
        dt=solver.dt,
        verbose=False,
        sources=(_FakeSource(),),
    )
    src.setup()
    assert src._tile_skip_stats is None
