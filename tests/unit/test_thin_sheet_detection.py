"""Unit tests for the WP-M2 thin-sheet pipeline.

``detect_thin_metallizations`` runs BEFORE grid-line generation against
the hard ``min_cell_size`` floor; a detected sheet gets ONE grid plane
at its substrate-side face (tangential-E mask) while the metal volume
stays in the DD-051 sub-cell classification of the adjacent cells.
"""

import numpy as np
import pytest

pytest.importorskip("OCC.Core.BRepPrimAPI")

from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh._conformal import detect_thin_metallizations
from magnelio.mesh.mesher import Mesh, MeshControl

H_SUB = 0.635e-3
T_MET = 35e-6
W_STRIP = 1.8e-3
W_DOM = 6.0e-3
H_AIR = 3.0e-3
FLOOR = 100e-6
L_Y = 4e-3


def _microstrip_shapes():
    sub = Material(name="sub", epsilon=(4.3, 4.3, 4.3))
    pec = Material.pec()
    air = Material.air()
    strip = Brick(
        origin=(-W_STRIP / 2, 0.0, H_SUB),
        size=(W_STRIP, L_Y, T_MET),
        material=pec,
    )
    air_brick = Brick(
        origin=(-W_DOM / 2, 0.0, H_SUB),
        size=(W_DOM, L_Y, H_AIR),
        material=air,
    )
    substrate = Brick(
        origin=(-W_DOM / 2, 0.0, 0.0),
        size=(W_DOM, L_Y, H_SUB),
        material=sub,
    )
    return substrate, air_brick, strip


def _microstrip_model():
    substrate, air_brick, strip = _microstrip_shapes()
    m = GeometryModel()
    m.add(substrate)
    m.add(Difference(air_brick, strip))
    m.add(strip)
    return m


class TestDetectThinMetallizations:
    def test_microstrip_strip_detected_substrate_side(self):
        substrate, air_brick, strip = _microstrip_shapes()
        shapes = [substrate, Difference(air_brick, strip), strip]
        specs = detect_thin_metallizations(shapes, FLOOR)
        assert len(specs) == 1
        spec = specs[0]
        assert spec.axis == "z"
        # Substrate (eps 4.3) below beats air above -> bottom face.
        assert spec.position == pytest.approx(H_SUB, abs=1e-12)
        assert spec.far_position == pytest.approx(H_SUB + T_MET, abs=1e-12)
        assert spec.shape is strip
        u0, v0, u1, v1 = spec.rect
        assert (u0, u1) == pytest.approx((-W_STRIP / 2, W_STRIP / 2))
        assert (v0, v1) == pytest.approx((0.0, L_Y))

    def test_flipped_substrate_picks_top_face(self):
        sub = Material(name="sub", epsilon=(4.3, 4.3, 4.3))
        pec = Material.pec()
        air = Material.air()
        # Substrate ABOVE the metallization: strip hangs below it.
        strip = Brick(
            origin=(-W_STRIP / 2, 0.0, H_SUB - T_MET),
            size=(W_STRIP, L_Y, T_MET),
            material=pec,
        )
        substrate = Brick(
            origin=(-W_DOM / 2, 0.0, H_SUB),
            size=(W_DOM, L_Y, 1e-3),
            material=sub,
        )
        air_below = Brick(
            origin=(-W_DOM / 2, 0.0, 0.0),
            size=(W_DOM, L_Y, H_SUB),
            material=air,
        )
        specs = detect_thin_metallizations(
            [Difference(air_below, strip), substrate, strip],
            FLOOR,
        )
        assert len(specs) == 1
        assert specs[0].position == pytest.approx(H_SUB, abs=1e-12)
        assert specs[0].far_position == pytest.approx(H_SUB - T_MET, abs=1e-12)

    def test_sheet_in_air_ties_to_lower_face(self):
        pec = Material.pec()
        sheet = Brick(origin=(0, 0, 1e-3), size=(2e-3, 2e-3, T_MET), material=pec)
        specs = detect_thin_metallizations([sheet], FLOOR)
        assert len(specs) == 1
        assert specs[0].position == pytest.approx(1e-3, abs=1e-12)

    def test_wire_not_detected(self):
        pec = Material.pec()
        wire = Brick(origin=(0, 0, 0), size=(30e-6, 2e-3, 30e-6), material=pec)
        assert detect_thin_metallizations([wire], FLOOR) == []

    def test_resolvable_layer_not_detected(self):
        pec = Material.pec()
        slab = Brick(origin=(0, 0, 0), size=(2e-3, 2e-3, 150e-6), material=pec)
        assert detect_thin_metallizations([slab], FLOOR) == []

    def test_thin_dielectric_not_detected(self):
        # Solder-mask class layers are deferred (PEC only).
        mask = Brick(
            origin=(0, 0, 0),
            size=(2e-3, 2e-3, T_MET),
            material=Material(name="mask", epsilon=(3.5,) * 3),
        )
        assert detect_thin_metallizations([mask], FLOOR) == []


class TestThinSheetPipeline:
    """End-to-end: the session-91 microstrip reproducer."""

    @pytest.fixture(scope="class")
    def mesh(self):
        ctrl = MeshControl(
            min_cells_per_feature=2,
            min_cell_size=FLOOR,
        )
        return Mesh.from_geometry(_microstrip_model(), ctrl, f_max=10e9)

    def test_one_plane_at_substrate_face(self, mesh):
        gz = np.asarray(mesh.grid.z)
        assert H_SUB in gz
        # The far-side face must NOT be a grid plane.
        assert not np.any(np.abs(gz - (H_SUB + T_MET)) < 1e-9)
        # No node inside the metal layer at all.
        assert not np.any((gz > H_SUB + 1e-12) & (gz < H_SUB + T_MET + 1e-9))

    def test_no_metal_thickness_cell(self, mesh):
        # The 35 um cell layer of the pre-WP-M2 pipeline is gone; the
        # cells adjacent to the sheet are floor-class or larger.
        k = int(np.argmin(np.abs(np.asarray(mesh.grid.z) - H_SUB)))
        dz_above = mesh.grid.z[k + 1] - mesh.grid.z[k]
        assert dz_above >= FLOOR

    def test_no_pec_cells_from_sheet(self, mesh):
        pec_ids = [mid for mid, mat in mesh.material_library.items() if mat.is_pec]
        for mid in pec_ids:
            assert not np.any(mesh.material_id == mid)

    def test_sheet_plane_tangential_edges_masked(self, mesh):
        grid = mesh.grid
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        k_h = int(np.argmin(np.abs(np.asarray(grid.z) - H_SUB)))
        x_c = 0.5 * (grid.x[:-1] + grid.x[1:])
        i_in = np.where((x_c >= -W_STRIP / 2) & (x_c <= W_STRIP / 2))[0]
        # Ex edges in the sheet plane over the strip footprint.
        ex = mesh.pec_mask_edges[0, : Nx * (Ny + 1) * (Nz + 1)].reshape(
            Nx,
            Ny + 1,
            Nz + 1,
        )
        assert np.all(ex[i_in, :, k_h])

    def test_normal_edges_above_strip_carry_thickness(self, mesh):
        # Ez edges from the sheet plane upward across the metal: cat 2
        # with L_free = dz_above - t (the DD-051 sub-cell thickness
        # effect).
        grid = mesh.grid
        Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
        k_h = int(np.argmin(np.abs(np.asarray(grid.z) - H_SUB)))
        dz_above = grid.z[k_h + 1] - grid.z[k_h]

        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        em = mesh.edge_material
        cat = em.category[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz)
        L_free = em.L_free[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz)

        # Strictly interior strip nodes (clear of the lateral edges).
        i_in = np.where((grid.x > -W_STRIP / 2 + 1e-9) & (grid.x < W_STRIP / 2 - 1e-9))[0]
        assert len(i_in) > 0
        sel_cat = cat[i_in, 1:-1, k_h]
        sel_L = L_free[i_in, 1:-1, k_h]
        assert np.all(sel_cat == 2)
        np.testing.assert_allclose(sel_L, dz_above - T_MET, rtol=1e-6)

    def test_mass_matrices_finite(self, mesh):
        from magnelio._operators.material_matrices import (
            build_M_eps,
            build_M_mu,
        )

        M_eps = np.asarray(build_M_eps(mesh))
        M_mu = np.asarray(build_M_mu(mesh))
        assert np.all(np.isfinite(M_eps))
        assert np.all(np.isfinite(M_mu))

    def test_no_floor_no_detection(self):
        # Opt-in via the hard floor: without min_cell_size the layer is
        # resolved with both faces (pre-WP-M2 behaviour).
        ctrl = MeshControl(min_cells_per_feature=2, min_cell_size=None)
        mesh = Mesh.from_geometry(_microstrip_model(), ctrl, f_max=10e9)
        gz = np.asarray(mesh.grid.z)
        assert np.any(np.abs(gz - (H_SUB + T_MET)) < 1e-9)
