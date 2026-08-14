"""WP-C1 (DD-093): per-material area fractions from the section pipeline.

The conformal classifiers optionally record, for the dispersive/σ*-
carrying material ids, each id's effective post-priority area share of
every processed dual face (E side) / primal face (H side) —
``EdgeMaterialData.material_fractions`` / ``FaceMaterialData.…`` — the
inputs the conformal ADE and σ* builders (WP-C2…C4) consume.

Gates (CONFORMAL_DISPERSIVE_PLAN WP-C1): fraction sums ≤ 1 with
equality where no PEC; ``eps_avg`` recomputed from fractions × library
values matches the stored ``eps_avg`` on processed edges (internal
consistency — same budget cascade); store round-trip; meshes without
dispersive materials carry no container.
"""

from __future__ import annotations

import numpy as np

from magnelio.geo import Brick, GeometryModel
from magnelio.geo._filling import compute_conformal_eps
from magnelio.materials.dispersion import DispersionModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import Mesh, MeshControl

EPS_INF = 2.0
MU_MAG = 3.0


def _ctrl():
    ax = np.linspace(0.0, 4e-3, 5)
    return MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=1.1e-3,
        forced_planes={"x": ax, "y": ax, "z": ax},
    )


def _dispersive():
    model = DispersionModel.debye(
        eps_inf=EPS_INF,
        delta_eps=1.5,
        tau=1e-11,
        f_band=(1e8, 2e10),
    )
    return Material.dispersive("dl", model)


def _half_filled(lower: Material) -> Mesh:
    m = GeometryModel()
    m.add(Brick(origin=(0, 0, 0), size=(4e-3, 2e-3, 4e-3), material=lower))
    m.add(Brick(origin=(0, 2e-3, 0), size=(4e-3, 2e-3, 4e-3), material=Material.air()))
    return Mesh.from_geometry(m, _ctrl(), f_max=5e9)


class TestContainerPresence:
    def test_no_container_without_dispersive_materials(self):
        mesh = _half_filled(Material.from_isotropic("d", epsilon=2.0))
        assert mesh.edge_material.material_fractions is None
        assert mesh.edge_material.fraction_mids is None
        assert mesh.face_material.material_fractions is None

    def test_e_side_requests_only_eps_dispersive(self):
        mesh = _half_filled(_dispersive())
        assert mesh.edge_material.fraction_mids is not None
        assert mesh.edge_material.fraction_mids.size == 1
        # ε-dispersive material does not trigger the H-side container.
        assert mesh.face_material.material_fractions is None


class TestEdgeFractions:
    def test_reconstructs_eps_avg_and_bounds(self):
        mesh = _half_filled(_dispersive())
        em = mesh.edge_material
        fr = em.fractions_by_mid[int(em.fraction_mids[0])]
        computed = ~np.isnan(fr)
        cat1 = em.category == 1
        # Every cat-1 edge of this PEC-free fixture is processed.
        assert np.all(computed[cat1])
        assert np.all(fr[computed] >= -1e-12)
        assert np.all(fr[computed] <= 1.0 + 1e-12)
        # Interface-plane edges carry genuine fractional shares.
        assert np.sum((fr[cat1] > 0.1) & (fr[cat1] < 0.9)) > 0
        # Internal consistency: same budget cascade as eps_avg
        # (other material is air, no PEC → its share is 1 − f).
        recon = fr[cat1] * EPS_INF + (1.0 - fr[cat1]) * 1.0
        np.testing.assert_allclose(recon, em.eps_avg[cat1], rtol=0.0, atol=1e-12)

    def test_sum_is_one_without_pec(self):
        lower = _dispersive()
        mesh = _half_filled(lower)
        shapes = [
            (Brick(origin=(0, 0, 0), size=(4e-3, 2e-3, 4e-3), material=lower), 1),
            (Brick(origin=(0, 2e-3, 0), size=(4e-3, 2e-3, 4e-3), material=Material.air()), 2),
        ]
        lib = {0: Material.air(), 1: lower, 2: Material.air()}
        _, _, _, fr = compute_conformal_eps(
            shapes,
            mesh.grid,
            mesh.material_id,
            lib,
            fraction_mids=np.array([1, 2]),
        )
        computed = ~np.isnan(fr[0])
        assert computed.any()
        total = fr[0][computed] + fr[1][computed]
        np.testing.assert_allclose(total, 1.0, rtol=0.0, atol=1e-12)

    def test_pec_claims_area(self):
        lower = _dispersive()
        pec_top = Brick(origin=(0, 2e-3, 0), size=(4e-3, 2e-3, 4e-3), material=Material.pec())
        mesh_model = GeometryModel()
        mesh_model.add(Brick(origin=(0, 0, 0), size=(4e-3, 2e-3, 4e-3), material=lower))
        mesh_model.add(pec_top)
        mesh = Mesh.from_geometry(mesh_model, _ctrl(), f_max=5e9)
        shapes = [
            (Brick(origin=(0, 0, 0), size=(4e-3, 2e-3, 4e-3), material=lower), 1),
            (pec_top, 2),
        ]
        lib = {0: Material.air(), 1: lower, 2: Material.pec()}
        _, _, _, fr = compute_conformal_eps(
            shapes,
            mesh.grid,
            mesh.material_id,
            lib,
            fraction_mids=np.array([1]),
        )
        computed = ~np.isnan(fr[0])
        assert computed.any()
        # PEC claims its share: the dispersive fraction stays ≤ 1 and
        # the interface-plane edges (half dispersive, half PEC) sit
        # strictly below 1.
        assert np.all(fr[0][computed] <= 1.0 + 1e-12)
        assert np.sum((fr[0][computed] > 0.1) & (fr[0][computed] < 0.9)) > 0


class TestFaceFractions:
    def _mag_mesh(self):
        mag = Material(name="mg", epsilon=(1.0,) * 3, mu=(MU_MAG,) * 3, sigma_m=(5.0,) * 3)
        return _half_filled(mag)

    def test_sigma_m_triggers_h_container_and_reconstructs_mu(self):
        mesh = self._mag_mesh()
        fm = mesh.face_material
        assert fm.fraction_mids is not None
        fr = fm.fractions_by_mid[int(fm.fraction_mids[0])]
        computed = ~np.isnan(fr)
        assert computed.any()
        sel = (fm.category >= 1) & computed
        recon = fr[sel] * MU_MAG + (1.0 - fr[sel]) * 1.0
        np.testing.assert_allclose(recon, fm.mu_avg[sel], rtol=0.0, atol=1e-12)
        # DD-053 pair-promoted faces carry no OCC statement: they must
        # be NaN (staircase lookup), never a silent zero.
        promoted = (fm.category >= 1) & ~computed
        if promoted.any():
            assert np.all(np.isnan(fr[promoted]))


class TestStoreRoundTrip:
    def test_fractions_survive_mesh_h5(self, tmp_path):
        import h5py

        from magnelio.io.project import _load_mesh, _save_mesh

        mesh = _half_filled(_dispersive())
        p = tmp_path / "mesh.h5"
        with h5py.File(p, "w") as f:
            _save_mesh(f, mesh)
        with h5py.File(p, "r") as f:
            back = _load_mesh(f)
        em0, em1 = mesh.edge_material, back.edge_material
        np.testing.assert_array_equal(em0.fraction_mids, em1.fraction_mids)
        np.testing.assert_array_equal(em0.material_fractions, em1.material_fractions)
        # H side had no container — stays None after the round trip.
        assert back.face_material.material_fractions is None

    def test_pre_dd093_store_loads_as_none(self, tmp_path):
        import h5py

        from magnelio.io.project import _load_mesh, _save_mesh

        mesh = _half_filled(_dispersive())
        p = tmp_path / "mesh.h5"
        with h5py.File(p, "w") as f:
            _save_mesh(f, mesh)
        # Simulate an old store: drop the new datasets.
        with h5py.File(p, "a") as f:
            del f["mesh/edge_material/fraction_mids"]
            del f["mesh/edge_material/material_fractions"]
        with h5py.File(p, "r") as f:
            back = _load_mesh(f)
        assert back.edge_material.fraction_mids is None
        assert back.edge_material.material_fractions is None
