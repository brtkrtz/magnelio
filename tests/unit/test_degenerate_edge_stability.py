"""An edge that carries no electric energy must not run the solver (DD-147).

``build_M_eps`` gives a cat-2 edge with ``eps_avg = 0`` the value
``M_eps = 0``: the edge lies wholly inside a conductor, and the
classifier left it cat-2 and unmasked rather than cat-3.  Two places
used to take that at face value —

* :func:`compute_min_effective_eps` returned 0, and ``courant_dt``'s
  ``max(..., 1e-6)`` guard then pinned ``dt`` four decades below the
  geometric Courant limit;
* the E-side update coefficients divided straight through, so
  ``alpha_E`` and ``beta_E`` came out NaN and the NaN reached every
  field component on the first step.

Neither failed loudly.  Together they produced a run that advanced
0.003 ns in 44 200 steps and returned NaN power waves.  The H side had
handled its own version of this since DD-081 (``M_mu > 0``).
"""

from dataclasses import replace

import numpy as np
import pytest

from magnelio.geo import Brick, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.solver.stability import (
    _build_L_primal_E_local,
    compute_min_effective_eps,
    courant_dt,
)

occ = pytest.importorskip("OCC.Core.BRepPrimAPI")


def _meshed_box():
    """A plain air box — every edge healthy, nothing degenerate."""
    model = GeometryModel(background=Material.pec())
    model.add(Brick(origin=(0.0, 0.0, 0.0), size=(20e-3, 10e-3, 30e-3), material=Material.air()))
    return Mesh.from_geometry(model, MeshControl(), f_max=10e9)


def _solver_active_mask(mesh):
    """The edges the solver actually updates, flattened as stability.py does.

    ``edge_material.pec_mask`` is the classifier's own view and is not
    the mask the update uses — an edge free in one can be masked in the
    other, which makes it useless for picking a victim here.
    """
    g = mesh.grid
    n_ex = g.Nx * (g.Ny + 1) * (g.Nz + 1)
    n_ey = (g.Nx + 1) * g.Ny * (g.Nz + 1)
    n_ez = (g.Nx + 1) * (g.Ny + 1) * g.Nz
    pec = mesh.pec_mask_edges
    return ~np.concatenate([pec[0, :n_ex], pec[1, :n_ey], pec[2, :n_ez]])


def _victim_index(mesh):
    active = np.nonzero(_solver_active_mask(mesh))[0]
    assert active.size, "expected at least one active edge"
    return int(active[active.size // 2])


def _with_a_subcell_edge(mesh, *, eps_avg_value, f_A_value, l_free_fraction=1.0):
    """Plant one cat-2 edge with the given sub-cell numbers.

    ``eps_avg`` is area-weighted over the *whole* dual face, so it and
    ``f_A`` move together: on an air/PEC contour ``eps_avg = f_A``.
    """
    em = mesh.edge_material
    if em is None:
        pytest.skip("mesh carries no conformal edge data")

    victim = _victim_index(mesh)
    L_primal = _build_L_primal_E_local(mesh.grid)

    category = em.category.copy()
    eps_avg = em.eps_avg.copy()
    L_free = em.L_free.copy()
    f_A = em.f_A.copy()
    category[victim] = 2
    eps_avg[victim] = eps_avg_value
    f_A[victim] = f_A_value
    # cat-0 edges carry NaN in L_free; give it a real length.
    L_free[victim] = l_free_fraction * float(L_primal[victim])
    return replace(
        mesh,
        edge_material=replace(em, category=category, eps_avg=eps_avg, L_free=L_free, f_A=f_A),
    )


def _with_a_dead_edge(mesh):
    """Turn one active edge into the cat-2 / eps_avg = 0 degenerate case."""
    return _with_a_subcell_edge(mesh, eps_avg_value=0.0, f_A_value=0.0)


def _with_a_nearly_dead_edge(mesh):
    """The same edge, but with a rounding remainder instead of a clean zero.

    This is what a conformal coax feed actually produces: a dual face
    whose free area has collapsed to ~1e-15 of nominal.  An equality
    test against zero walks straight past it.
    """
    return _with_a_subcell_edge(
        mesh, eps_avg_value=3.2e-15, f_A_value=3.2e-15, l_free_fraction=0.44
    )


class TestDegenerateEdgeDoesNotSetTheTimeStep:
    def test_a_dead_edge_is_left_out_of_the_minimum(self):
        healthy = _meshed_box()
        eps_healthy = compute_min_effective_eps(healthy)

        eps_degenerate = compute_min_effective_eps(_with_a_dead_edge(healthy))

        assert eps_degenerate > 0.0
        assert eps_degenerate == pytest.approx(eps_healthy)

    def test_the_time_step_survives_it(self):
        """Without the fix dt drops by ~1/sqrt(1e-6) — three decades."""
        healthy = _meshed_box()
        degenerate = _with_a_dead_edge(healthy)

        dt_healthy = courant_dt(healthy.grid, min_effective_eps=compute_min_effective_eps(healthy))
        dt_degenerate = courant_dt(
            degenerate.grid, min_effective_eps=compute_min_effective_eps(degenerate)
        )

        assert dt_degenerate == pytest.approx(dt_healthy)


class TestDegenerateEdgeIsFrozenNotNaN:
    def test_update_coefficients_stay_finite(self):
        """alpha_E / beta_E must never carry NaN — it spreads on step one."""
        from magnelio.solver.fit_td import FITTimeDomainSolver

        degenerate = _with_a_dead_edge(_meshed_box())
        dt = courant_dt(degenerate.grid, min_effective_eps=compute_min_effective_eps(degenerate))

        solver = FITTimeDomainSolver(mesh=degenerate, dt=dt, total_time_steps=1)
        solver.run()  # the coefficients are built here, not in __init__

        assert np.isfinite(np.asarray(solver._alpha_E)).all()
        assert np.isfinite(np.asarray(solver._beta_E)).all()

    def test_the_dead_edge_is_frozen_like_a_masked_one(self):
        """beta_E = 0 there: no curl drives an edge with no energy."""
        from magnelio._operators.material_matrices import build_M_eps
        from magnelio.solver.fit_td import FITTimeDomainSolver

        degenerate = _with_a_dead_edge(_meshed_box())
        dt = courant_dt(degenerate.grid, min_effective_eps=compute_min_effective_eps(degenerate))
        solver = FITTimeDomainSolver(mesh=degenerate, dt=dt, total_time_steps=1)
        solver.run()

        dead = np.asarray(build_M_eps(degenerate)) == 0.0
        assert dead.any()
        assert np.all(np.asarray(solver._beta_E)[dead] == 0.0)
        assert np.all(np.asarray(solver._alpha_E)[dead] == 1.0)


class TestNearlyDeadEdgeIsFlooredToo:
    """A rounding remainder must not walk past the guard (DD-149).

    The DD-147 guard tested ``eps_avg == 0``.  A conformal coax feed on
    a 0.25 mm grid produced 10 edges at exactly zero *and* 3 more at
    ~3e-15 — physically the same edge, numerically past an equality
    test.  Those three alone held dt at 5.95e-17 s instead of 2.04e-14.
    """

    def test_the_time_step_survives_it(self):
        healthy = _meshed_box()
        nearly = _with_a_nearly_dead_edge(healthy)

        eps_healthy = compute_min_effective_eps(healthy)
        eps_nearly = compute_min_effective_eps(nearly)

        assert eps_nearly == pytest.approx(eps_healthy)
        assert courant_dt(nearly.grid, min_effective_eps=eps_nearly) == pytest.approx(
            courant_dt(healthy.grid, min_effective_eps=eps_healthy)
        )

    def test_the_floored_edge_is_frozen(self):
        """Frozen, not staircase: the edge lies inside a conductor.

        This is where the E side parts from the H side, whose floored
        faces keep their bulk value because they are Faraday-dead.
        """
        from magnelio._operators.material_matrices import build_M_eps

        nearly = _with_a_nearly_dead_edge(_meshed_box())

        assert float(np.asarray(build_M_eps(nearly))[_victim_index(nearly)]) == 0.0

    def test_a_sub_cell_edge_above_the_floor_keeps_its_reduction(self):
        """The floor must not swallow edges that still carry a field.

        2 % free area is thin but real — its ε_eff has to reach the
        CFL, or dt comes out too large and the run goes unstable.
        """
        healthy = _meshed_box()
        thin = _with_a_subcell_edge(healthy, eps_avg_value=0.02, f_A_value=0.02)

        assert compute_min_effective_eps(thin) == pytest.approx(0.02)

    def test_both_mass_matrices_use_the_same_floor(self):
        """One constant, mirrored in three places — pin them together."""
        from magnelio._operators import material_matrices
        from magnelio.solver import stability

        assert stability._FREE_AREA_FLOOR == material_matrices._FREE_AREA_FLOOR
