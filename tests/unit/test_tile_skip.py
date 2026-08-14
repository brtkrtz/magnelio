"""Unit tests for the static dead-tile analysis (TILE_SKIP_PLAN WP-T2)."""

import numpy as np

from magnelio.boundaries.pec import PECBoundary
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.solver._tile_skip import (
    build_tile_skip_plan,
    component_shapes,
)
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import courant_dt

# DD-103: the closure these fixtures always assumed.  A face
# with no BC used to evolve under the free curl operator —
# which IS the natural magnetic wall, hence "PMC".
_BC_OPEN = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PMC",
    "ymax": "PMC",
    "zmin": "PMC",
    "zmax": "PMC",
}


def _cavity_solver(Nx=16, Ny=12, Nz=16, pec_slab_from=None):
    """Cavity solver; optionally mark all edges with i >= pec_slab_from PEC."""
    grid = GridLines(
        x=np.linspace(0, Nx * 1e-3, Nx + 1),
        y=np.linspace(0, Ny * 1e-3, Ny + 1),
        z=np.linspace(0, Nz * 1e-3, Nz + 1),
    )
    mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
    if pec_slab_from is not None:
        shapes_E, _ = component_shapes(Nx, Ny, Nz)
        for axis, name in enumerate(("Ex", "Ey", "Ez")):
            n = int(np.prod(shapes_E[name]))
            view = mesh.pec_mask_edges[axis, :n].reshape(shapes_E[name])
            view[pec_slab_from:] = True
    bcs = {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    solver = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions=bcs,
        total_time_steps=10,
        dt=courant_dt(grid, "draft"),
        verbose=False,
    )
    solver.setup()
    return solver


def _plan_from_solver(solver, **kw):
    mesh = solver.mesh
    return build_tile_skip_plan(
        Nx=mesh.Nx,
        Ny=mesh.Ny,
        Nz=mesh.Nz,
        alpha_E=np.asarray(solver._alpha_E),
        beta_E=np.asarray(solver._beta_E),
        alpha_H=np.asarray(solver._alpha_H),
        beta_H=np.asarray(solver._beta_H),
        **kw,
    )


def _element_launched(plan, name, shape, idx3):
    """Whether element ``idx3`` of component ``name`` is launched."""
    ti, tj, tk = plan.tile
    nbi, nbj, nbk = plan.block_grids[name]
    bi, bj, bk = idx3[0] // ti, idx3[1] // tj, idx3[2] // tk
    return (bi * nbj * nbk + bj * nbk + bk) in plan.live_blocks[name]


class TestSelfDisable:
    def test_field_sources_disable(self):
        solver = _cavity_solver()
        assert _plan_from_solver(solver, has_field_sources=True) is None

    def test_unsafe_bcs_disable(self):
        solver = _cavity_solver()
        assert _plan_from_solver(solver, has_unsafe_bcs=True) is None


class TestAirCavity:
    def test_everything_launched_nothing_zeroed(self):
        solver = _cavity_solver()
        plan = _plan_from_solver(solver, tile=(2, 2, 4))
        for name, dims in plan.block_grids.items():
            assert plan.live_blocks[name].size == np.prod(dims)
        assert plan.dead_zero_idx_E.size == 0
        assert plan.dead_zero_idx_H.size == 0
        assert plan.stats["total"] == 0.0


class TestPECSlab:
    """Half-space PEC slab (all edges with i >= 8 masked)."""

    def setup_method(self):
        self.solver = _cavity_solver(pec_slab_from=8)
        self.plan = _plan_from_solver(self.solver, tile=(2, 2, 4))
        self.shapes_E, self.shapes_H = component_shapes(16, 12, 16)

    def test_deep_pec_E_tile_skipped(self):
        assert not _element_launched(self.plan, "Ex", self.shapes_E["Ex"], (12, 6, 8))

    def test_live_E_tile_launched(self):
        assert _element_launched(self.plan, "Ex", self.shapes_E["Ex"], (2, 6, 8))

    def test_slab_boundary_E_tile_launched(self):
        # i = 7 (live) shares the ti=2 tile with i = 6 — launched.
        assert _element_launched(self.plan, "Ex", self.shapes_E["Ex"], (7, 6, 8))

    def test_interior_curl_dead_H_skipped(self):
        # Hx face deep in the slab, outside the boundary shell.
        assert not _element_launched(self.plan, "Hx", self.shapes_H["Hx"], (10, 6, 8))

    def test_pmc_shell_keeps_boundary_H_launched(self):
        # With a PMC-style shell on ymin the outermost curl-dead
        # layer must stay launched; without it, it is skipped.
        shelled = _plan_from_solver(
            self.solver,
            tile=(2, 2, 4),
            boundary_shell_faces={"ymin": 1},
        )
        assert _element_launched(shelled, "Hx", self.shapes_H["Hx"], (10, 0, 8))
        assert not _element_launched(self.plan, "Hx", self.shapes_H["Hx"], (10, 0, 8))
        # The shell must not resurrect deep-interior faces.
        assert not _element_launched(shelled, "Hx", self.shapes_H["Hx"], (10, 6, 8))

    def test_zero_idx_E_equals_solver_pec_idx(self):
        # dead E == final PEC mask == the solver's own index array.
        assert np.array_equal(
            np.sort(self.plan.dead_zero_idx_E),
            np.sort(np.asarray(self.solver._pec_idx_E)),
        )

    def test_zero_idx_H_are_curl_dead_faces(self):
        # Every zeroed H face lies fully inside the slab: check by
        # decoding a sample against the Hx shape.
        n_Hx = int(np.prod(self.shapes_H["Hx"]))
        hx_idx = self.plan.dead_zero_idx_H
        hx_idx = hx_idx[hx_idx < n_Hx]
        assert hx_idx.size > 0
        i = hx_idx // (12 * 16)
        assert (i >= 8).all()

    def test_stats_monotone(self):
        assert 0.0 < self.plan.stats["total"] < 1.0
        # Raw dead fraction bounds the tile capture from above.
        dead_frac = (
            np.asarray(self.solver._pec_mask_E).sum() / np.asarray(self.solver._pec_mask_E).size
        )
        assert self.plan.stats["Ex"] <= dead_frac + 0.25


class TestDonatedNoopFaces:
    def test_skipped_but_never_zeroed(self):
        Nx = Ny = Nz = 8
        shapes_E, shapes_H = component_shapes(Nx, Ny, Nz)
        n_E = sum(int(np.prod(s)) for s in shapes_E.values())
        n_H = sum(int(np.prod(s)) for s in shapes_H.values())
        alpha_E = np.full(n_E, 0.9)
        beta_E = np.full(n_E, 0.5)
        alpha_H = np.full(n_H, 0.9)
        beta_H = np.full(n_H, 0.5)
        # Mark ALL Hx faces as donated no-ops (alpha 1, beta 0).
        n_Hx = int(np.prod(shapes_H["Hx"]))
        alpha_H[:n_Hx] = 1.0
        beta_H[:n_Hx] = 0.0
        plan = build_tile_skip_plan(
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            alpha_E=alpha_E,
            beta_E=beta_E,
            alpha_H=alpha_H,
            beta_H=beta_H,
            tile=(2, 2, 4),
        )
        assert plan.live_blocks["Hx"].size == 0  # fully skipped
        assert plan.dead_zero_idx_H.size == 0  # value preserved
        assert plan.live_blocks["Hy"].size > 0


class TestBlockReduction:
    def test_against_bruteforce(self):
        rng = np.random.default_rng(7)
        from magnelio.solver._tile_skip import _live_block_ids

        for _ in range(5):
            live = rng.random((5, 6, 7)) < 0.3
            (nbi, nbj, nbk), ids, frac = _live_block_ids(live, (2, 3, 4))
            ref_ids = []
            n_in_live = 0
            for bi in range(nbi):
                for bj in range(nbj):
                    for bk in range(nbk):
                        blk = live[
                            bi * 2 : (bi + 1) * 2, bj * 3 : (bj + 1) * 3, bk * 4 : (bk + 1) * 4
                        ]
                        real = min(2, 5 - bi * 2) * min(3, 6 - bj * 3) * min(4, 7 - bk * 4)
                        if blk.any():
                            ref_ids.append(bi * nbj * nbk + bj * nbk + bk)
                            n_in_live += real
            assert np.array_equal(ids, np.array(ref_ids, dtype=np.int32))
            assert frac == 1.0 - n_in_live / live.size
