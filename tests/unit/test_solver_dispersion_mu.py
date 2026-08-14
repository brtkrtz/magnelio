"""Unit gates for the mu(omega) ADE on the H side (DD-089).

The magnetic mirror of test_solver_dispersion.py.  The load-bearing one
is TestCoefficientReduction::test_drude_dc_pole_equals_sigma_m — the
mandatory exact-reduction gate the plan demands after DD-084's factor-2
lesson (a wrong W is nearly invisible in line physics but shows up here
immediately).
"""

from __future__ import annotations

import numpy as np

from magnelio import Material, Mesh
from magnelio._operators.material_matrices import (
    MU0,
    build_M_mu,
    build_M_sigma_m,
)
from magnelio.materials import DispersionModel
from magnelio.mesh.grid import GridLines
from magnelio.solver._dispersion import DispersionOperator
from magnelio.solver.fit_td import FITTimeDomainSolver
from magnelio.solver.stability import compute_min_effective_mu

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

DT = 1e-12


def _grid(n=6):
    lin = np.linspace(0.0, n * 1e-3, n + 1)
    return GridLines(x=lin, y=lin, z=lin)


def _debye_mu(name="disp_mu", mu_inf=2.0, delta_mu=1.0, tau=1e-11, sigma_m=0.0):
    return Material.dispersive_mu(
        name,
        DispersionModel.debye(mu_inf, delta_mu, tau),
        sigma_m=sigma_m,
    )


def _solver(mesh, steps=None):
    return FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={},
        dt=DT,
        total_time_steps=steps,
        verbose=False,
    )


class TestOperatorConstruction:
    def test_none_without_mu_dispersive_material(self):
        mesh = Mesh.from_grid(_grid(), background=Material.air(), boundary_conditions=_BC_OPEN)
        assert DispersionOperator.from_mesh(mesh, DT, side="H") is None

    def test_eps_dispersion_does_not_build_an_h_side_operator(self):
        """The two channels are independent: an eps-dispersive material
        must leave the H side empty (and vice versa)."""
        eps_mat = Material.dispersive(
            "eps_only",
            DispersionModel.debye(2.0, 1.0, 1e-11),
        )
        mesh = Mesh.from_grid(_grid(), background=eps_mat, boundary_conditions=_BC_OPEN)
        assert DispersionOperator.from_mesh(mesh, DT, side="H") is None
        assert DispersionOperator.from_mesh(mesh, DT, side="E") is not None

        mesh_mu = Mesh.from_grid(_grid(), background=_debye_mu(), boundary_conditions=_BC_OPEN)
        assert DispersionOperator.from_mesh(mesh_mu, DT, side="E") is None
        assert DispersionOperator.from_mesh(mesh_mu, DT, side="H") is not None

    def test_face_subset_single_cell(self):
        """One mu-dispersive cell in a 2x2x2 grid: the clamped one-sided
        lookup owns exactly one face per component — hand-checkable."""
        lin = np.linspace(0.0, 2e-3, 3)
        mesh = Mesh.from_grid(
            GridLines(x=lin, y=lin, z=lin),
            regions=[(_debye_mu(), (0.0, 0.0, 0.0, 1e-3, 1e-3, 1e-3))],
            boundary_conditions=_BC_OPEN,
        )
        op = DispersionOperator.from_mesh(mesh, DT, side="H")
        assert len(op.blocks) == 1
        idx = np.sort(op.blocks[0].idx)
        n_Hx = 3 * 2 * 2
        n_Hy = 2 * 3 * 2
        # Hx face (0,0,0); Hy face (0,0,0); Hz face (0,0,0) in flat order.
        expected = np.array([0, n_Hx + 0, n_Hx + n_Hy + 0])
        np.testing.assert_array_equal(idx, expected)

    def test_coupling_matches_M_mu_geometry(self):
        """On a homogeneous mu-dispersive fill, g must equal M_mu/mu_inf
        face for face (same geometry factor, same staircase sampling)."""
        mesh = Mesh.from_grid(
            _grid(), background=_debye_mu(mu_inf=2.0), boundary_conditions=_BC_OPEN
        )
        op = DispersionOperator.from_mesh(mesh, DT, side="H")
        (block,) = op.blocks
        M_mu = build_M_mu(mesh)
        np.testing.assert_allclose(block.g, M_mu[block.idx] / 2.0, rtol=1e-15)

    def test_w_positive_for_passive_model(self):
        mesh = Mesh.from_grid(_grid(), background=_debye_mu(), boundary_conditions=_BC_OPEN)
        op = DispersionOperator.from_mesh(mesh, DT, side="H")
        assert (op.W[op.blocks[0].idx] > 0).all()

    def test_frozen_faces_are_excluded(self):
        """WP-R5 donor faces (M_mu == 0) carry beta_H = 0 — a pole state
        there would never be integrated, so it must not be allocated."""
        mesh = Mesh.from_grid(_grid(), background=_debye_mu(), boundary_conditions=_BC_OPEN)
        frozen = np.zeros(build_M_mu(mesh).size, dtype=bool)
        frozen[[0, 5, 17]] = True
        op = DispersionOperator.from_mesh(mesh, DT, side="H", frozen=frozen)
        (block,) = op.blocks
        assert not np.isin(block.idx, np.array([0, 5, 17])).any()
        assert (op.W[frozen] == 0.0).all()


class TestCoefficientReduction:
    def test_no_mu_dispersion_bit_identical(self):
        """alpha_H/beta_H without a mu-dispersive material are the exact
        master expressions — no epsilon-perturbation from the new path."""
        mesh = Mesh.from_grid(
            _grid(),
            background=Material.from_isotropic("d", 2.0, mu=2.0, sigma_m=0.05),
            boundary_conditions=_BC_OPEN,
        )
        s = _solver(mesh, steps=1)
        s.setup()
        assert s._dispersion_mu is None
        M_mu = build_M_mu(mesh)
        M_sigma_m = np.where(M_mu > 0, build_M_sigma_m(mesh), 0.0)
        denom = np.where(M_mu > 0, M_mu + 0.5 * DT * M_sigma_m, 1.0)
        np.testing.assert_array_equal(
            s._alpha_H,
            np.where(M_mu > 0, (M_mu - 0.5 * DT * M_sigma_m) / denom, 1.0),
        )
        np.testing.assert_array_equal(
            s._beta_H,
            np.where(M_mu > 0, DT / denom, 0.0),
        )

    def test_drude_dc_pole_equals_sigma_m(self):
        """THE mandatory exact-reduction gate (DD-089).

        A pure DC pole (a=0, r=sigma_m/mu0) IS the semi-implicit magnetic
        conductor: with h^0 = 0 the trapezoidal state J^n = W h^n is exact
        by induction, so the marched field must match the sigma_m-material
        run to roundoff.  (Seeding E, not H — the mu-ADE integrates dH/dt,
        so a nonzero h^0 would be a deliberate initial-magnetisation
        offset, not a conductor.)

        This is the gate that would have caught DD-084's factor-2 W bug:
        the reduction is DYNAMIC (J accumulates into sigma_m*h over the
        run), so a wrong coefficient cannot hide the way it nearly did in
        the line physics.
        """
        sigma_m = 0.05

        def march(mat, n_steps=200):
            mesh = Mesh.from_grid(_grid(), background=mat, boundary_conditions=_BC_OPEN)
            s = _solver(mesh, steps=n_steps)
            s.setup()
            rng = np.random.default_rng(7)
            s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size)
            s.run()
            return s._fields.e_flat.copy(), s._fields.h_flat.copy()

        e_sig, h_sig = march(
            Material.from_isotropic("c", 2.0, mu=2.0, sigma_m=sigma_m),
        )
        # eps_inf here carries MU_INF (relative-units model); epsilon must
        # match the reference material's 2.0, or the two runs are not the
        # same problem.
        dc = DispersionModel(
            eps_inf=2.0,
            poles=((complex(0.0), complex(sigma_m / MU0)),),
            f_band=(1e8, 1e10),
        )
        e_ade, h_ade = march(Material.dispersive_mu("d", dc, epsilon=2.0))

        assert np.abs(h_ade - h_sig).max() < 1e-10 * np.abs(h_sig).max()
        assert np.abs(e_ade - e_sig).max() < 1e-10 * np.abs(e_sig).max()

    def test_drude_dc_W_equals_M_sigma_m(self):
        """The coefficient DD-084's factor-2 bug lived in, pinned
        directly: for a DC pole the W_m folded into alpha_H/beta_H must
        BE the M_sigma_m of the equivalent magnetic conductor, face for
        face.  The marching gate above proves the dynamics; this one
        localises a failure to the coefficient in one line."""
        sigma_m = 0.05
        dc = DispersionModel(
            eps_inf=2.0,
            poles=((complex(0.0), complex(sigma_m / MU0)),),
            f_band=(1e8, 1e10),
        )
        mesh_ade = Mesh.from_grid(
            _grid(),
            background=Material.dispersive_mu("d", dc, epsilon=2.0),
            boundary_conditions=_BC_OPEN,
        )
        op = DispersionOperator.from_mesh(mesh_ade, DT, side="H")
        mesh_sig = Mesh.from_grid(
            _grid(),
            background=Material.from_isotropic("c", 2.0, mu=2.0, sigma_m=sigma_m),
            boundary_conditions=_BC_OPEN,
        )
        np.testing.assert_allclose(
            op.W,
            build_M_sigma_m(mesh_sig),
            rtol=1e-12,
        )


class TestCFL:
    def test_cfl_uses_mu_inf(self):
        """mu_inf drives the stability limit (the poles are A-stable at
        any dt) — a mu-dispersive fill must give the same effective mu as
        a static mu_inf fill."""
        mu_inf = 2.5
        mesh_disp = Mesh.from_grid(
            _grid(), background=_debye_mu(mu_inf=mu_inf, delta_mu=8.0), boundary_conditions=_BC_OPEN
        )
        mesh_static = Mesh.from_grid(
            _grid(),
            background=Material.from_isotropic("s", 1.0, mu=mu_inf),
            boundary_conditions=_BC_OPEN,
        )
        assert compute_min_effective_mu(mesh_disp) == (compute_min_effective_mu(mesh_static))


class TestCheckpointRewind:
    def test_same_ops_rewind_bit_exact(self):
        """WP-S6 pattern with complex (Lorentz pair) and real (Debye)
        mu-pole states, plus an eps-dispersive block in the same run."""
        mu_model = DispersionModel.lorentz(
            2.0,
            0.5,
            2 * np.pi * 5e9,
            2 * np.pi * 5e8,
        )
        both = Material(
            name="both",
            epsilon=(3.0,) * 3,
            mu=(mu_model.eps_inf,) * 3,
            dispersion=DispersionModel.debye(3.0, 1.0, 1e-11),
            dispersion_mu=mu_model,
        )
        mesh = Mesh.from_grid(
            _grid(),
            regions=[
                (both, (0.0, 0.0, 0.0, 3e-3, 6e-3, 6e-3)),
                (_debye_mu("real_mu_poles"), (3e-3, 0.0, 0.0, 6e-3, 6e-3, 6e-3)),
            ],
            boundary_conditions=_BC_OPEN,
        )
        s = _solver(mesh, steps=40)
        s.setup()
        assert len(s._dispersion_mu.blocks) == 2
        assert len(s._dispersion.blocks) == 1
        rng = np.random.default_rng(3)
        s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size)
        s.run()

        sd = s.state_dict()
        assert "dispersion_mu" in sd and "dispersion" in sd
        s.total_time_steps = 80
        s.run()
        h_ref = s._fields.h_flat.copy()
        e_ref = s._fields.e_flat.copy()
        j_ref = {
            k: {kk: vv.copy() for kk, vv in v.items()}
            for k, v in s.state_dict()["dispersion_mu"].items()
        }

        s.load_state_dict(sd)
        s.run()
        np.testing.assert_array_equal(s._fields.h_flat, h_ref)
        np.testing.assert_array_equal(s._fields.e_flat, e_ref)
        j_back = s.state_dict()["dispersion_mu"]
        for key, poles in j_ref.items():
            for pk, val in poles.items():
                np.testing.assert_array_equal(j_back[key][pk], val)
