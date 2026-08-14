"""WP-C2 (DD-093): conformal E-side ADE blocks + two-phase overlap.

Boundary edges join every dispersive material's ADE block with the
WP-C1 per-material area share as the weight on the edge's own M_eps
geometry factor, so ``eps_eff(omega) = sum_i f_i eps_i(omega)`` holds
exactly — the arithmetic mixing rule of the static ``eps_avg``.
Blocks may share edges; ``update_field`` runs in two phases (subtract
every history, then advance every pole set on the completed field), so
the shared-state completion is the joint implicit solve.

Gates (CONFORMAL_DISPERSIVE_PLAN WP-C2):
* the mandatory exact reduction — a Drude-DC pole at a CONFORMAL
  boundary is the semi-implicit conductor: W equals the conformal
  ``build_M_sigma`` diagonal edge for edge (both channels share the
  same fractions), and the marched fields match to solver precision;
* shared-edge joint solve matched exactly against an independent
  scalar reference recursion;
* splitting one dispersive fill into two identical materials (blocks
  sharing every interface edge) reproduces the single-material run.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from magnelio import Material
from magnelio._backend.array_api import resolve_backend
from magnelio._operators.material_matrices import (
    EPS0,
    build_M_eps,
    build_M_sigma,
)
from magnelio.geo import Brick, GeometryModel
from magnelio.materials import DispersionModel
from magnelio.materials.material import Material as _Mat
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.solver._dispersion import DispersionOperator
from magnelio.solver.fit_td import FITTimeDomainSolver

DT = 1e-12
SIGMA = 0.05

try:
    resolve_backend("cupy")
    HAS_GPU = True
except Exception:
    HAS_GPU = False

gpu = pytest.mark.skipif(not HAS_GPU, reason="no usable CuPy/CUDA device")


def _ctrl():
    ax = np.linspace(0.0, 4e-3, 5)
    # min_cell_size pins the raster: the cylinder tangent planes leave
    # 0.7 mm boundary intervals, and the DD-107 domain-face buffer
    # would re-split them into 3 x 0.233 mm cells — below the CFL
    # margin of the fixed DT these gates march with.  The floor
    # invokes the DD-107 min_cell_size exception, keeping the grid
    # exactly as forced here.
    return MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=1.1e-3,
        min_cell_size=3e-4,
        forced_planes={"x": ax, "y": ax, "z": ax},
    )


def _half_filled(lower: Material, upper: Material | None = None) -> Mesh:
    m = GeometryModel()
    m.add(Brick(origin=(0, 0, 0), size=(4e-3, 2e-3, 4e-3), material=lower))
    m.add(
        Brick(
            origin=(0, 2e-3, 0),
            size=(4e-3, 2e-3, 4e-3),
            material=upper if upper is not None else Material.air(),
        )
    )
    mesh = Mesh.from_geometry(m, _ctrl(), f_max=5e9)
    # Bare material block: no walls anywhere, so the conformal boundary
    # edges the σ*/ADE gates compare stay live.  Declared after meshing
    # so the forced-plane grid this fixture relies on is untouched.
    return mesh.with_boundary_conditions(
        {f: "PMC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
    )


def _march(mesh, n_steps=400, seed=7):
    s = FITTimeDomainSolver(
        mesh=mesh,
        boundary_conditions={},
        dt=DT,
        total_time_steps=n_steps,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.setup()
        rng = np.random.default_rng(seed)
        s._fields.h_flat[:] = rng.standard_normal(s._fields.h_flat.size)
        s.run()
    return s._fields.e_flat.copy(), s._fields.h_flat.copy()


def _dc_material(eps_inf=2.0):
    dc = DispersionModel(
        eps_inf=eps_inf,
        poles=((complex(0.0), complex(SIGMA / EPS0)),),
        f_band=(1e8, 1e10),
    )
    return Material.dispersive("dc", dc)


class TestDrudeDCConformalReduction:
    def test_w_equals_conformal_sigma_diagonal(self):
        """The mandatory matrix-level gate: with a DC pole
        (a = 0 → c = r = σ/ε0, w_coeff = r), the ADE coefficient
        diagonal W must reproduce the conformal ``build_M_sigma`` of
        the same geometry with plain σ — both channels consume the
        SAME WP-C1 fractions, so a wrong weight cannot hide."""
        mesh_ade = _half_filled(_dc_material())
        mesh_sig = _half_filled(
            _Mat.from_isotropic("c", epsilon=2.0, sigma=SIGMA),
        )
        op = DispersionOperator.from_mesh(mesh_ade, DT)
        M_sigma = build_M_sigma(mesh_sig)
        # Compare where either side is nonzero (dispersive region +
        # its conformal boundary edges).
        sel = (op.W != 0.0) | (M_sigma != 0.0)
        assert sel.any()
        np.testing.assert_allclose(
            op.W[sel],
            M_sigma[sel],
            rtol=1e-12,
            atol=0.0,
        )

    def test_marched_fields_match_sigma_run(self):
        """TD leg of the mandatory gate: hundreds of steps through the
        production solver on the conformal-boundary mesh."""
        e_ade, h_ade = _march(_half_filled(_dc_material()))
        e_sig, h_sig = _march(
            _half_filled(
                _Mat.from_isotropic("c", epsilon=2.0, sigma=SIGMA),
            )
        )
        assert np.abs(e_ade - e_sig).max() < 1e-10 * np.abs(e_sig).max()
        assert np.abs(h_ade - h_sig).max() < 1e-10 * np.abs(h_sig).max()


class TestSharedEdgeJointSolve:
    def _two_material_mesh(self):
        """Lower/upper halves carry DIFFERENT dispersive materials —
        every interface-plane edge joins both blocks (shared states)."""
        a = Material.dispersive(
            "da",
            DispersionModel.debye(2.0, 1.0, 1e-11),
        )
        b = Material.dispersive(
            "db",
            DispersionModel.debye(3.0, 0.5, 3e-11),
        )
        return _half_filled(a, b), a, b

    def test_blocks_share_interface_edges(self):
        mesh, _, _ = self._two_material_mesh()
        op = DispersionOperator.from_mesh(mesh, DT)
        assert len(op.blocks) == 2
        shared = np.intersect1d(op.blocks[0].idx, op.blocks[1].idx)
        assert shared.size > 0
        em = mesh.edge_material
        fr = em.material_fractions
        # On shared edges both fractions are genuine partial shares.
        for row in range(2):
            vals = fr[row][shared]
            assert np.all(vals > 0.0)
            assert np.all(vals < 1.0)

    def test_joint_recursion_matches_scalar_reference(self):
        """Drive ONE shared edge with a synthetic RHS through the
        production operator (the solver's exact coefficient folding
        done by hand) and match an independently coded scalar joint
        recursion of the coupled implicit system — the DD-084/WP-D4
        exact-reference pattern."""
        mesh, mat_a, mat_b = self._two_material_mesh()
        op = DispersionOperator.from_mesh(mesh, DT)
        M_eps = build_M_eps(mesh)
        shared = np.intersect1d(op.blocks[0].idx, op.blocks[1].idx)
        edge = int(shared[0])

        # Solver coefficient folding on this edge (no static sigma).
        denom = M_eps[edge] + 0.5 * DT * op.W[edge]
        alpha = (M_eps[edge] - 0.5 * DT * op.W[edge]) / denom
        beta = DT / denom

        n_e = op.W.size
        op.bind(np.full(n_e, beta, dtype=np.float64), np)
        f = np.zeros(n_e, dtype=np.float64)
        rng = np.random.default_rng(3)
        rhs = rng.standard_normal(60)

        # Scalar reference: joint implicit system on this edge.
        blocks_ref = []
        for blk, mat in ((op.blocks[0], mat_a), (op.blocks[1], mat_b)):
            pos = int(np.nonzero(blk.idx == edge)[0][0])
            g = float(blk.g[pos])
            poles = []
            for a_p, r_p in mat.dispersion.poles:
                k = (1.0 + a_p * DT / 2.0) / (1.0 - a_p * DT / 2.0)
                c = r_p / (1.0 - a_p * DT / 2.0)
                poles.append([k, c, 0.0 + 0.0j])
            blocks_ref.append([g, poles])
        f_ref = 0.0

        for n in range(rhs.size):
            # Production path: save, curl surrogate, two-phase update.
            op.save_field(f)
            f[edge] = alpha * f[edge] + beta * rhs[n]
            op.update_field(f)

            # Reference: subtract both histories, then advance both.
            jh = 0.0
            for g, poles in blocks_ref:
                for k, c, J in poles:
                    w = 1.0 if k.imag == 0.0 and c.imag == 0.0 else 2.0
                    jh += 0.5 * w * ((1.0 + k) * J).real
            f_prev = f_ref
            f_ref = alpha * f_ref + beta * rhs[n] - beta * jh
            for g, poles in blocks_ref:
                gd = g * (f_ref - f_prev)
                for p in poles:
                    p[2] = p[2] * p[0] + p[1] * gd

            assert abs(f[edge] - f_ref) <= 1e-13 * max(1.0, abs(f_ref))

    def test_split_identical_material_matches_single_fill(self):
        """Two identical dispersive materials meeting at the interface
        (every interface edge shared f_a + f_b = 1) must reproduce the
        single-material full fill — the structural superposition check
        of the joint solve on the full production march."""
        model = DispersionModel.debye(2.0, 1.0, 1e-11)
        a = Material.dispersive("da", model)
        b = Material.dispersive("db", model)
        e_split, h_split = _march(_half_filled(a, b))

        single = Material.dispersive("ds", model)
        m = GeometryModel()
        m.add(Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=single))
        e_full, h_full = _march(
            Mesh.from_geometry(m, _ctrl(), f_max=5e9).with_boundary_conditions(
                {f: "PMC" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")}
            ),
        )

        assert np.abs(e_split - e_full).max() < 1e-11 * np.abs(e_full).max()
        assert np.abs(h_split - h_full).max() < 1e-11 * np.abs(h_full).max()


class TestConformalMuSide:
    """WP-C3 (DD-093): H-side mirror — conformal μ(ω) ADE membership
    via the FaceMaterialData fractions, with the mandatory
    μ-Drude-DC ≡ conformal-σ* exact reduction (WP-C4 supplies the
    comparison channel).  Fixture: a magnetic cylinder in air — a
    genuinely curved contour (planar axis-aligned interfaces are
    anchored onto grid planes and carry only 0/1 shares)."""

    SM = 0.05

    def _cyl_mesh(self, mat):
        from magnelio.geo import Difference
        from magnelio.geo.primitives import Cylinder

        cyl = Cylinder(origin=(2e-3, 2e-3, 0), radius=1.3e-3, height=4e-3, axis="z", material=mat)
        m = GeometryModel()
        m.add(
            Difference(
                Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=Material.air()),
                cyl,
            )
        )
        m.add(cyl)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # growth-factor advisory
            return Mesh.from_geometry(m, _ctrl(), f_max=5e9)

    def _mu_dc_material(self):
        from magnelio._operators.material_matrices import MU0

        dc = DispersionModel(
            eps_inf=2.0,  # carries MU_INF (relative-units model)
            poles=((complex(0.0), complex(self.SM / MU0)),),
            f_band=(1e8, 1e10),
        )
        return Material.dispersive_mu("mdc", dc, epsilon=2.0)

    def _sigma_m_material(self):
        return _Mat(name="ms", epsilon=(2.0,) * 3, mu=(2.0,) * 3, sigma_m=(self.SM,) * 3)

    def test_mu_dc_W_equals_conformal_sigma_m_matrix(self):
        """The mandatory matrix-level gate: the μ-DC pole's W_m must BE
        the conformal ``build_M_sigma_m`` of the same geometry with
        plain σ*, face for face — both consume the WP-C1 fractions on
        the identical cat-1 / safe-cat-2 geometry."""
        from magnelio._operators.material_matrices import build_M_sigma_m

        mesh_ade = self._cyl_mesh(self._mu_dc_material())
        mesh_sig = self._cyl_mesh(self._sigma_m_material())
        op = DispersionOperator.from_mesh(mesh_ade, DT, side="H")
        M_sm = build_M_sigma_m(mesh_sig)
        sel = (op.W != 0.0) | (M_sm != 0.0)
        assert sel.any()
        np.testing.assert_allclose(
            op.W[sel],
            M_sm[sel],
            rtol=1e-12,
            atol=0.0,
        )
        # The conformal membership genuinely engages: partial shares.
        fm = mesh_ade.face_material
        fr = fm.fractions_by_mid[int(fm.fraction_mids[0])]
        comp = ~np.isnan(fr)
        assert np.sum((fr[comp] > 0.1) & (fr[comp] < 0.9)) > 10

    def test_mu_dc_march_matches_sigma_m_run(self):
        """TD leg of the mandatory gate on the conformal contour."""

        def march(mat, n_steps=400):
            s = FITTimeDomainSolver(
                mesh=self._cyl_mesh(mat),
                boundary_conditions={},
                dt=DT,
                total_time_steps=n_steps,
                verbose=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s.setup()
                rng = np.random.default_rng(7)
                s._fields.e_flat[:] = rng.standard_normal(s._fields.e_flat.size)
                s.run()
            return s._fields.e_flat.copy(), s._fields.h_flat.copy()

        e_sig, h_sig = march(self._sigma_m_material())
        e_ade, h_ade = march(self._mu_dc_material())
        assert np.abs(h_ade - h_sig).max() < 1e-10 * np.abs(h_sig).max()
        assert np.abs(e_ade - e_sig).max() < 1e-10 * np.abs(e_sig).max()

    def test_two_mu_materials_share_contour_faces(self):
        a = Material.dispersive_mu(
            "ma",
            DispersionModel.debye(2.0, 1.0, 1e-11),
            epsilon=2.0,
        )
        mesh = self._cyl_mesh(a)
        # Make the surrounding air magnetic-dispersive too by swapping
        # material 1 (the Difference shell) — rebuild with both sides
        # dispersive instead.
        from magnelio.geo import Difference
        from magnelio.geo.primitives import Cylinder

        b = Material.dispersive_mu(
            "mb",
            DispersionModel.debye(3.0, 0.5, 3e-11),
            epsilon=2.0,
        )
        cyl = Cylinder(origin=(2e-3, 2e-3, 0), radius=1.3e-3, height=4e-3, axis="z", material=a)
        m = GeometryModel()
        m.add(
            Difference(
                Brick(origin=(0, 0, 0), size=(4e-3, 4e-3, 4e-3), material=b),
                cyl,
            )
        )
        m.add(cyl)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mesh = Mesh.from_geometry(m, _ctrl(), f_max=5e9)
        op = DispersionOperator.from_mesh(mesh, DT, side="H")
        assert len(op.blocks) == 2
        shared = np.intersect1d(op.blocks[0].idx, op.blocks[1].idx)
        assert shared.size > 0


@gpu
class TestConformalDispersionGPU:
    """WP-C2 GPU gate: the two-phase update stays inside the WP-G3
    graph segments (device-only array ops) — graph capture engages on
    a conformal shared-edge dispersive mesh and the march is
    bit-identical to the eager GPU path."""

    def _gpu_march(self, steps=150, seed=4):
        a = Material.dispersive(
            "da",
            DispersionModel.debye(2.0, 1.0, 1e-11),
        )
        b = Material.dispersive(
            "db",
            DispersionModel.debye(3.0, 0.5, 3e-11),
        )
        mesh = _half_filled(a, b)
        s = FITTimeDomainSolver(
            mesh=mesh,
            boundary_conditions={},
            dt=DT,
            total_time_steps=steps,
            verbose=False,
            backend="cupy",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.setup()
            rng = np.random.default_rng(seed)
            e0 = rng.standard_normal(s._fields.e_flat.size) * 1e-3
            s._fields.e_flat[:] = s._xp.asarray(e0)
            s.run()
        return s._fields.e_flat.get(), s._fields.h_flat.get(), s

    def test_graphs_engage_and_match_eager(self, monkeypatch):
        monkeypatch.delenv("MAGNELIO_GPU_GRAPHS", raising=False)
        e_g, h_g, s_g = self._gpu_march()
        assert s_g._gpu_graphs is not None
        assert s_g._gpu_graphs.ready
        assert not s_g._gpu_graphs.failed
        # Shared-edge blocks actually present on this mesh.
        assert len(s_g._dispersion.blocks) == 2

        monkeypatch.setenv("MAGNELIO_GPU_GRAPHS", "0")
        e_e, h_e, s_e = self._gpu_march()
        assert s_e._gpu_graphs is None
        np.testing.assert_array_equal(e_g, e_e)
        np.testing.assert_array_equal(h_g, h_e)
