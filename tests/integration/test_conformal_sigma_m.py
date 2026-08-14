"""WP-C4 (DD-093): conformal σ* through the face section pipeline.

``sigma_m_avg`` joins ``FaceMaterialData`` (same conventions and
NaN-marking as ``mu_avg``; PEC claims area with σ* = 0) and
``build_M_sigma_m`` applies cat-1/2 overrides mirroring
``build_M_mu``'s categorical form — the former "recorded non-goal"
retires.

Gates (CONFORMAL_DISPERSIVE_PLAN WP-C4): no-σ*-mesh bit-identity;
planar (from_grid) σ* slab unchanged; the conformal average equals the
WP-C1 fraction reconstruction exactly (same budget cascade); the
booked M_sigma_m follows the categorical form face for face.
"""

from __future__ import annotations

import numpy as np

from magnelio._operators.material_matrices import (
    _build_A_face_H,
    _build_geom_H,
    build_M_sigma_m,
)
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.geo.primitives import Cylinder
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh, MeshControl

SM = 5.0


def _ctrl():
    ax = np.linspace(0.0, 4e-3, 5)
    return MeshControl(
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        max_cell_size=1.1e-3,
        forced_planes={"x": ax, "y": ax, "z": ax},
    )


def _mag(sm=SM, mu=3.0):
    return Material(name="mg", epsilon=(1.0,) * 3, mu=(mu,) * 3, sigma_m=(sm,) * 3)


def _half_filled(lower):
    m = GeometryModel()
    m.add(Brick(origin=(0, 0, 0), size=(4e-3, 2e-3, 4e-3), material=lower))
    m.add(Brick(origin=(0, 2e-3, 0), size=(4e-3, 2e-3, 4e-3), material=Material.air()))
    return Mesh.from_geometry(m, _ctrl(), f_max=5e9)


class TestNoSigmaMPath:
    def test_no_sigma_m_material_keeps_container_free_staircase(self):
        mesh = _half_filled(
            Material(name="m", epsilon=(1.0,) * 3, mu=(3.0,) * 3),
        )
        assert mesh.face_material.sigma_m_avg is None
        # All-zero diagonal, and the conformal branch never engages.
        assert np.all(build_M_sigma_m(mesh) == 0.0)

    def test_from_grid_slab_stays_staircase(self):
        lin = np.linspace(0.0, 4e-3, 5)
        mesh = Mesh.from_grid(
            GridLines(x=lin, y=lin, z=lin),
            regions=[(_mag(), (0.0, 0.0, 0.0, 4e-3, 2e-3, 4e-3))],
        )
        assert mesh.face_material is None
        M = build_M_sigma_m(mesh)
        # Hand staircase check on one interior Hx face fully in the
        # magnetic slab: sigma_m * A_face/L_dual.
        geom_H = _build_geom_H(mesh.grid)
        own_sm = np.where(
            np.concatenate(
                [
                    mesh.material_id[np.clip(np.arange(5 - 1 + 1), 0, 3)].ravel(),
                    mesh.material_id[:, np.clip(np.arange(5), 0, 3)].ravel(),
                    mesh.material_id[:, :, np.clip(np.arange(5), 0, 3)].ravel(),
                ]
            )
            != 0,
            SM,
            0.0,
        )
        np.testing.assert_allclose(M, own_sm * geom_H, rtol=1e-15)


def _cylinder_mesh():
    """Magnetic cylinder in air — a genuinely CURVED σ* boundary.

    Planar axis-aligned interfaces are anchored onto grid planes by
    the mesher, so their H-faces carry only 0/1 shares; fractional
    conformal averages need an oblique contour.
    """
    import warnings

    cyl = Cylinder(origin=(2e-3, 2e-3, 0), radius=1.3e-3, height=4e-3, axis="z", material=_mag())
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


class TestConformalSigmaM:
    def test_average_matches_fraction_reconstruction(self):
        mesh = _cylinder_mesh()
        fm = mesh.face_material
        assert fm.sigma_m_avg is not None
        fr = fm.fractions_by_mid[int(fm.fraction_mids[0])]
        both = ~np.isnan(fm.sigma_m_avg) & ~np.isnan(fr)
        assert both.any()
        np.testing.assert_allclose(
            fm.sigma_m_avg[both],
            fr[both] * SM,
            rtol=0.0,
            atol=1e-12,
        )
        # The curved contour produces genuine partial averages.
        part = both & (fm.sigma_m_avg > 0.1 * SM) & (fm.sigma_m_avg < 0.9 * SM)
        assert part.sum() > 10

    def test_booked_matrix_follows_categorical_form(self):
        mesh = _half_filled(_mag())
        fm = mesh.face_material
        M = build_M_sigma_m(mesh)
        geom_H = _build_geom_H(mesh.grid)
        A_face = _build_A_face_H(mesh.grid)
        cat1 = (fm.category == 1) & ~np.isnan(fm.sigma_m_avg)
        assert cat1.any()
        np.testing.assert_allclose(
            M[cat1],
            fm.sigma_m_avg[cat1] * geom_H[cat1],
            rtol=1e-15,
        )
        cat2 = (fm.category == 2) & ~np.isnan(fm.sigma_m_avg) & (fm.A_face_free > 0.01 * A_face)
        if cat2.any():
            np.testing.assert_allclose(
                M[cat2],
                fm.sigma_m_avg[cat2] * fm.A_face_free[cat2] / fm.L_dual_free[cat2],
                rtol=1e-15,
            )

    def test_dd053_promoted_faces_keep_staircase(self):
        """Faces the DD-053 pair pass promoted to cat 2 without an OCC
        statement have NaN sigma_m_avg — their booked σ* must be the
        staircase value, not a silent zero."""
        mesh = _half_filled(_mag())
        fm = mesh.face_material
        promoted = (fm.category >= 1) & np.isnan(fm.sigma_m_avg)
        if not promoted.any():
            return
        M = build_M_sigma_m(mesh)
        # Staircase reference: rebuild without the conformal branch.
        sm_saved = fm.sigma_m_avg
        fm.sigma_m_avg = None
        M_stair = build_M_sigma_m(mesh)
        fm.sigma_m_avg = sm_saved
        np.testing.assert_array_equal(M[promoted], M_stair[promoted])
