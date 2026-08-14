"""Unit tests for the H-face enlarged-cell donor mechanism (WP-R5).

The M_μ mirror of the DD-051 E-edge donor: cat-2 H-faces whose free
area collapsed below the 1 % floor donate their residual magnetic
inertia ``μ̄ · A_face_free`` to a neighbour face along the dual-edge
axis and are frozen (``M_μ = 0`` ⇒ exact ``1/M_μ = 0`` in the update
helpers).  Covers:

* donor assignment on a deep-PEC-inclusion geometry (miniature
  iris-loaded pillbox — the trigger geometry class of
  ``validation/iris_cavity_donor_trigger.py``),
* receiver validity (never floored, never a staircase interior-PEC
  face, always the flux-tube neighbour along the face normal),
* exact mass bookkeeping in ``build_M_mu`` against the donor-disabled
  reference state,
* the frozen-face contract in the FIT-TD update coefficients,
* the no-op guarantee on donor-free geometries.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import Material, Mesh, MeshControl
from magnelio._operators.material_matrices import (
    MU0,
    assign_h_face_donors,
    build_M_mu,
)
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.geo._subcell import _build_A_face_H, _build_L_dual_H
from magnelio.mesh.grid import GridLines

# Miniature iris-loaded pillbox (radii off every forced-grid multiple —
# tangent-plane / forced-node collisions produce sliver cells).
S_BBOX = 12.0e-3
R_CAV = 4.85e-3
L_HALF = 3.0e-3
T_IRIS = 1.0e-3
A_APER = 1.55e-3
X_PAD = 1.0e-3
L_TOT = 2 * L_HALF + T_IRIS
N_T = 13


def _build_iris_mesh() -> Mesh:
    pec = Material.pec()
    vacuum = Material.air()
    bbox = Brick(
        origin=(-X_PAD, -S_BBOX / 2, -S_BBOX / 2),
        size=(L_TOT + 2 * X_PAD, S_BBOX, S_BBOX),
        material=pec,
    )
    cavity = Cylinder(origin=(0, 0, 0), radius=R_CAV, height=L_TOT, axis="x", material=vacuum)
    slab = Cylinder(origin=(L_HALF, 0, 0), radius=R_CAV, height=T_IRIS, axis="x", material=pec)
    hole = Cylinder(origin=(L_HALF, 0, 0), radius=A_APER, height=T_IRIS, axis="x", material=vacuum)
    model = GeometryModel()
    model.add(Difference(bbox, cavity))
    model.add(Difference(cavity, slab))
    model.add(Difference(slab, hole))
    model.add(hole)

    h_t = S_BBOX / (N_T - 1)
    y_nodes = np.linspace(-S_BBOX / 2, S_BBOX / 2, N_T).tolist()
    x_breaks = [-X_PAD, 0.0, L_HALF, L_HALF + T_IRIS, L_TOT, L_TOT + X_PAD]
    x_nodes: list[float] = []
    for x0, x1 in zip(x_breaks[:-1], x_breaks[1:]):
        n = max(2, int(round((x1 - x0) / h_t)) + 1)
        seg = np.linspace(x0, x1, n).tolist()
        x_nodes.extend(seg[1:] if x_nodes else seg)
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=0,
        growth_factor=1.5,
        max_cell_size=4.0 * h_t,
        conformal=True,
        forced_planes={"x": x_nodes, "y": y_nodes, "z": y_nodes},
    )
    return Mesh.from_geometry(model, control, f_max=20.0e9)


@pytest.fixture(scope="module")
def iris_mesh() -> Mesh:
    # The donor pass is dormant in production (the trigger benchmark
    # measured it neutral — floored faces are Faraday-dead); it is
    # invoked explicitly here to validate the mechanism itself.
    mesh = _build_iris_mesh()
    assign_h_face_donors(mesh)
    return mesh


def _normal_stride(mesh: Mesh, flat: int) -> int:
    g = mesh.grid
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz
    if flat < n_Hx:
        return Ny * Nz
    if flat < n_Hx + n_Hy:
        return Nz
    return 1


class TestDonorAssignment:
    def test_floored_faces_exist_and_get_donors(self, iris_mesh):
        fm = iris_mesh.face_material
        A = _build_A_face_H(iris_mesh.grid)
        cat2 = fm.category == 2
        floored = cat2 & ~(fm.A_face_free > 0.01 * A)
        assert floored.sum() > 0, "trigger geometry must produce floored faces"
        donated = fm.enlarged_cell_donor >= 0
        assert donated.sum() > 0
        # Donors are assigned exclusively to floored cat-2 faces.
        assert not (donated & ~floored).any()

    def test_receivers_are_valid(self, iris_mesh):
        fm = iris_mesh.face_material
        A = _build_A_face_H(iris_mesh.grid)
        donated = np.nonzero(fm.enlarged_cell_donor >= 0)[0]
        receivers = fm.enlarged_cell_donor[donated]
        # Never floored / never themselves donated (no chains).
        assert (fm.enlarged_cell_donor[receivers] == -1).all()
        cat2_recv = fm.category[receivers] == 2
        ratio = fm.A_face_free[receivers] / A[receivers]
        assert (ratio[cat2_recv] > 0.01).all()
        # Never a dead receiver: every receiver stays a live DOF in the
        # assembled matrix (a staircase interior-PEC face would carry
        # the borrowed inertia into a frozen h — silently lost).
        M = build_M_mu(iris_mesh)
        assert (M[receivers] > 0.0).all()
        # Flux-tube neighbour: |receiver - flat| equals the stride of
        # the face-normal axis.
        for flat, recv in zip(donated, receivers):
            assert abs(int(recv) - int(flat)) == _normal_stride(
                iris_mesh,
                int(flat),
            )

    def test_borrowed_amount_is_mu_area(self, iris_mesh):
        fm = iris_mesh.face_material
        donated = np.nonzero(fm.enlarged_cell_donor >= 0)[0]
        mu = np.where(
            np.isfinite(fm.mu_avg[donated]) & (fm.mu_avg[donated] > 0),
            fm.mu_avg[donated],
            1.0,
        )
        a_free = np.where(
            np.isfinite(fm.A_face_free[donated]) & (fm.A_face_free[donated] > 0),
            fm.A_face_free[donated],
            0.0,
        )
        np.testing.assert_allclose(
            fm.enlarged_cell_area[donated],
            mu * a_free,
            rtol=1e-12,
        )


class TestMassBookkeeping:
    def test_build_M_mu_freezes_and_transfers(self, iris_mesh):
        fm = iris_mesh.face_material
        donor = fm.enlarged_cell_donor
        area = fm.enlarged_cell_area
        donated = np.nonzero(donor >= 0)[0]
        receivers = donor[donated]

        M_with = build_M_mu(iris_mesh)
        # Reference: donor-disabled state = historical staircase fallback.
        fm.enlarged_cell_donor = None
        try:
            M_without = build_M_mu(iris_mesh)
        finally:
            fm.enlarged_cell_donor = donor

        assert (M_with[donated] == 0.0).all()
        L_dual = _build_L_dual_H(iris_mesh.grid)
        expected = M_without.copy()
        expected[donated] = 0.0
        np.add.at(expected, receivers, MU0 * area[donated] / L_dual[receivers])
        np.testing.assert_allclose(M_with, expected, rtol=1e-13)
        # Everything else untouched.
        rest = np.ones(M_with.size, dtype=bool)
        rest[donated] = False
        rest[receivers] = False
        np.testing.assert_array_equal(M_with[rest], M_without[rest])

    def test_M_mu_finite_and_nonnegative(self, iris_mesh):
        M = build_M_mu(iris_mesh)
        assert np.isfinite(M).all()
        assert (M >= 0.0).all()


class TestFrozenFaceContract:
    def test_fit_td_beta_H_zero_on_donated(self, iris_mesh):
        from magnelio.boundaries.pec import PECBoundary
        from magnelio.solver.fit_td import FITTimeDomainSolver
        from magnelio.solver.stability import (
            compute_min_effective_eps,
            compute_min_effective_mu,
            courant_dt,
        )

        dt = courant_dt(
            iris_mesh.grid,
            "draft",
            min_effective_eps=compute_min_effective_eps(iris_mesh),
            min_effective_mu=compute_min_effective_mu(iris_mesh),
        )
        bcs = {f: PECBoundary(f) for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
        solver = FITTimeDomainSolver(
            mesh=iris_mesh,
            boundary_conditions=bcs,
            total_time_steps=1,
            dt=dt,
            verbose=False,
        )
        solver.setup()
        donated = iris_mesh.face_material.enlarged_cell_donor >= 0
        beta = np.asarray(solver._beta_H)
        assert (beta[donated] == 0.0).all()
        assert (beta[~donated] > 0.0).all()

    def test_eigensolver_spectrum_finite(self, iris_mesh):
        from magnelio import AnalysisEigenmode

        result = AnalysisEigenmode(mesh=iris_mesh, n_modes=3, verbose=False).run()
        assert np.isfinite(result.frequencies).all()
        phys = result.frequencies[result.frequencies > 1e6]
        assert len(phys) >= 1
        # Fundamental in the physically plausible band for this
        # geometry class (TM010-pair of the ~4.9 mm pillbox halves).
        assert 10e9 < phys[0] < 40e9


class TestNoOpOnSmoothGeometry:
    def test_uniform_air_cavity_has_no_donors(self):
        grid_mesh = Mesh.from_grid(
            GridLines(
                x=np.linspace(0, 6e-3, 7),
                y=np.linspace(0, 6e-3, 7),
                z=np.linspace(0, 6e-3, 7),
            ),
        )
        fm = grid_mesh.face_material
        if fm is not None and fm.enlarged_cell_donor is not None:
            assert (fm.enlarged_cell_donor == -1).all()
        M = build_M_mu(grid_mesh)
        assert (M > 0).all()
