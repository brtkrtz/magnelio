"""Unit tests for the SIBC update-topology enumeration (WP-D3).

Circulation-exact staircase pairs, tag handling (slots, bimetal
seams), BC walls, Faraday-dead filtering, and the tag -> Z_s
resolution chain (``resolve_wall_conductors`` +
``fit_wall_impedances``).  The conformal-reuse gate lives in
``tests/integration/test_conformal_wall_area.py`` (needs the real
mesher).  Reference: internal dossier
``investigations/sibc/DERIVATION.md`` §3/§4/§8.2.
"""

import numpy as np
import pytest

from magnelio.materials.material import Material
from magnelio.materials.surface_impedance import fit_wall_impedances
from magnelio.mesh._surfaces import (
    _face_alive_from_edge_mask,
    enumerate_pec_surfaces,
    enumerate_sibc_surfaces,
    resolve_wall_conductors,
)
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh
from magnelio.post.wall_loss import surface_resistance

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

D = 1e-3


def _grid(nx=8, ny=6, nz=5):
    return GridLines(x=np.arange(nx + 1) * D, y=np.arange(ny + 1) * D, z=np.arange(nz + 1) * D)


def _brick_mesh():
    pec = Material.lossy_metal("cu", sigma=5.8e7)
    return Mesh.from_grid(
        _grid(),
        regions=[(pec, (2 * D, 2 * D, 1 * D, 6 * D, 5 * D, 3 * D))],
        boundary_conditions=_BC_OPEN,
    )


def _assert_all_alive(mesh, surfaces):
    """Post-condition: every booked face is Faraday-live under the
    solver freeze mask (DERIVATION.md §4 'by construction')."""
    alive = _face_alive_from_edge_mask(mesh.grid, mesh.pec_mask_edges)
    for s in surfaces:
        for c in range(3):
            sel = s.comp == c
            if sel.any():
                assert alive[c].ravel()[s.flat_idx[sel]].all()


class TestStaircaseSolids:
    def test_flat_wall_g_is_one_uniform(self):
        """On a uniform grid every wall-edge pair has G = l_e/l_dual =
        1 exactly, and a free-standing brick has no face with two wall
        rim edges — all g == 1."""
        surfs = enumerate_sibc_surfaces(_brick_mesh())
        assert len(surfs) == 1
        np.testing.assert_allclose(surfs[0].g, 1.0, rtol=1e-12)

    def test_brick_pair_budget(self):
        """Hand-counted pair budget of the 4x3x2-cell brick.

        Per wall (a x b cells) the wall edges are a*(b+1) in one
        tangential direction and (a+1)*b in the other, each booking
        one D^2 tile: z-walls 16+15, y-walls 12+10, x-walls 9+8 —
        sum(weight) = 2*(31+22+17)*D^2 = 140 D^2, all faces distinct."""
        surf = enumerate_sibc_surfaces(_brick_mesh())[0]
        assert surf.weight.sum() == pytest.approx(140 * D * D)
        assert surf.comp.size == 140  # one row per pair
        assert surf.area_total == pytest.approx(70 * D * D)
        assert surf.tag == 1

    def test_graded_grid_circulation_exact(self):
        """G = dx[i]/l_dual_y[j] and weight = dx[i]*l_dual_y[j] per
        pair, checked on a graded grid against hand values."""
        x = np.array([0.0, 1.0, 2.5, 4.5, 7.0, 10.0]) * 1e-3
        y = np.array([0.0, 1.0, 2.0, 3.5, 5.5, 8.0]) * 1e-3
        grid = GridLines(x=x, y=y, z=np.arange(6) * D)
        pec = Material.lossy_metal("cu", sigma=5.8e7)
        # brick spans x-cells 1..3, y-cells 1..3, z-cells 1..2
        mesh = Mesh.from_grid(
            grid,
            regions=[(pec, (1e-3, 1e-3, 1 * D, 7e-3, 5.5e-3, 3 * D))],
            boundary_conditions=_BC_OPEN,
        )
        surf = enumerate_sibc_surfaces(mesh)[0]
        dx, dy = np.diff(x), np.diff(y)
        dual_y = np.array([dy[0], *(0.5 * (dy[:-1] + dy[1:])), dy[-1]])
        # top-wall pair: Ex edge at (x-cell 2, y-node 2, z-node 3)
        # drives Hy(2, 2, 3); interior wall edge -> full G.
        hy = surf.comp == 1
        idx = np.unravel_index(surf.flat_idx[hy], (grid.Nx, grid.Ny + 1, grid.Nz))
        m = (idx[0] == 2) & (idx[1] == 2) & (idx[2] == 3)
        assert m.sum() == 1
        assert surf.weight[hy][m][0] == pytest.approx(dx[2] * dual_y[2])
        assert surf.g[hy][m][0] == pytest.approx(dx[2] / dual_y[2])

    def test_inside_corner_two_terms(self):
        """A one-cell slot between two walls of one metal: the slot
        faces carry two wall rim edges -> g == 2 (DERIVATION.md §3)."""
        pec = Material.lossy_metal("cu", sigma=5.8e7)
        mesh = Mesh.from_grid(
            GridLines(x=np.arange(9) * D, y=np.arange(7) * D, z=np.arange(8) * D),
            regions=[
                (pec, (0.0, 0.0, 1 * D, 8 * D, 6 * D, 3 * D)),
                (pec, (0.0, 0.0, 4 * D, 8 * D, 6 * D, 6 * D)),
            ],
            boundary_conditions=_BC_OPEN,
        )
        surfs = enumerate_sibc_surfaces(mesh)
        assert len(surfs) == 1
        surf = surfs[0]
        # slot layer z-cell 3: all tangential faces get both walls
        for c, shape in ((0, (9, 6, 7)), (1, (8, 7, 7))):
            sel = surf.comp == c
            idx = np.unravel_index(surf.flat_idx[sel], shape)
            slot = idx[2] == 3
            assert slot.any()
            np.testing.assert_allclose(surf.g[sel][slot], 2.0, rtol=1e-12)

    def test_two_metals_slot_separate_tags(self):
        """Same slot with two different metals: two surfaces, each
        contributing g == 1 on the shared slot faces."""
        m1 = Material.lossy_metal("cu", sigma=5.8e7)
        m2 = Material.lossy_metal("steel", sigma=1.4e6, mu=100.0)
        mesh = Mesh.from_grid(
            GridLines(x=np.arange(9) * D, y=np.arange(7) * D, z=np.arange(8) * D),
            regions=[
                (m1, (0.0, 0.0, 1 * D, 8 * D, 6 * D, 3 * D)),
                (m2, (0.0, 0.0, 4 * D, 8 * D, 6 * D, 6 * D)),
            ],
            boundary_conditions=_BC_OPEN,
        )
        surfs = {s.tag: s for s in enumerate_sibc_surfaces(mesh)}
        assert set(surfs) == {1, 2}
        for s in surfs.values():
            sel = s.comp == 0
            idx = np.unravel_index(s.flat_idx[sel], (9, 6, 7))
            slot = idx[2] == 3
            assert slot.any()
            np.testing.assert_allclose(s.g[sel][slot], 1.0, rtol=1e-12)

    def test_bimetal_seam_halved(self):
        """One flat wall from two metals side by side: the seam edge
        splits half per footprint, interior edges stay full."""
        m1 = Material.lossy_metal("cu", sigma=5.8e7)
        m2 = Material.lossy_metal("steel", sigma=1.4e6, mu=100.0)
        mesh = Mesh.from_grid(
            _grid(),
            regions=[
                (m1, (0.0, 0.0, 0.0, 4 * D, 6 * D, 2 * D)),
                (m2, (4 * D, 0.0, 0.0, 8 * D, 6 * D, 2 * D)),
            ],
            boundary_conditions=_BC_OPEN,
        )
        surfs = {s.tag: s for s in enumerate_sibc_surfaces(mesh)}
        assert set(surfs) == {1, 2}
        for tag, s in surfs.items():
            hx = s.comp == 0
            idx = np.unravel_index(s.flat_idx[hx], (9, 6, 5))
            top = idx[2] == 2  # top-wall Hx faces
            seam = top & (idx[0] == 4)
            interior = top & (idx[0] != 4) & (idx[0] != 0) & (idx[0] != 8)
            assert seam.any() and interior.any()
            np.testing.assert_allclose(s.g[hx][seam], 0.5, rtol=1e-12)
            np.testing.assert_allclose(s.g[hx][interior], 1.0, rtol=1e-12)

    def test_booked_faces_alive(self):
        mesh = _brick_mesh()
        _assert_all_alive(mesh, enumerate_sibc_surfaces(mesh))


class TestBoundaryWalls:
    def test_bc_wall_uniform(self):
        """Empty domain, zmin: g == 1 on every first-layer tangential
        face; hand-counted budget (state convention: full end cells,
        so the edge rows at the domain rim carry full tiles)."""
        mesh = Mesh.from_grid(_grid(), boundary_conditions=_BC_OPEN)
        surfs = enumerate_sibc_surfaces(mesh, bc_pec_faces=("zmin",))
        assert len(surfs) == 1
        surf = surfs[0]
        assert surf.tag == "zmin"
        np.testing.assert_allclose(surf.g, 1.0, rtol=1e-12)
        # Ex edges: 8*7 pairs; Ey edges: 9*6 pairs -> 110 D^2 booked
        assert surf.comp.size == 56 + 54
        assert surf.weight.sum() == pytest.approx(110 * D * D)
        assert (np.unravel_index(surf.flat_idx[surf.comp == 1], (8, 7, 5))[2] == 0).all()

    def test_solid_on_bc_wall(self):
        """A PEC brick on zmin: no pairs under the brick footprint, and
        the side-wall-plane faces at the rim are Faraday-dead ->
        dropped; every remaining booking is live."""
        pec = Material.pec()
        mesh = Mesh.from_grid(
            _grid(),
            regions=[(pec, (2 * D, 2 * D, 0.0, 6 * D, 5 * D, 2 * D))],
            boundary_conditions=_BC_OPEN,
        )
        surfs = enumerate_sibc_surfaces(mesh, bc_pec_faces=("zmin",))
        zmin = [s for s in surfs if s.tag == "zmin"][0]
        hy = zmin.comp == 1
        idx = np.unravel_index(zmin.flat_idx[hy], (8, 7, 5))
        under = (idx[0] >= 2) & (idx[0] < 6) & (idx[1] >= 2) & (idx[1] <= 5)
        assert not under.any()
        _assert_all_alive(mesh, surfs)

    def test_unknown_face_raises(self):
        mesh = Mesh.from_grid(_grid(), boundary_conditions=_BC_OPEN)
        with pytest.raises(ValueError, match="unknown boundary face"):
            enumerate_sibc_surfaces(mesh, bc_pec_faces=("bottom",))

    def test_sampling_path_untouched(self):
        """The DD-082 sampling enumeration is independent of the SIBC
        one (guard: same mesh, both run, sampling convention holds)."""
        mesh = _brick_mesh()
        samp = enumerate_pec_surfaces(mesh)[0]
        assert samp.weight.sum() == pytest.approx(2 * samp.area)
        assert samp.area == pytest.approx(52 * D * D)


class TestResolution:
    def test_lossy_metal_brings_own_values(self):
        mesh = _brick_mesh()
        surfs = enumerate_sibc_surfaces(mesh, bc_pec_faces=("zmin",))
        rough = None
        resolved = resolve_wall_conductors(mesh, surfs, sigma=1e6, mu=2.0, roughness=rough)
        assert resolved[1] == (5.8e7, 1.0, None)  # material's own
        assert resolved["zmin"] == (1e6, 2.0, rough)  # caller override

    def test_no_conductivity_raises(self):
        pec = Material.pec()
        mesh = Mesh.from_grid(
            _grid(),
            regions=[(pec, (2 * D, 2 * D, 1 * D, 6 * D, 5 * D, 3 * D))],
            boundary_conditions=_BC_OPEN,
        )
        surfs = enumerate_sibc_surfaces(mesh)
        with pytest.raises(ValueError, match="has no conductivity"):
            resolve_wall_conductors(mesh, surfs)

    def test_fit_wall_impedances_shared_and_accurate(self):
        """Identical conductor triples share ONE fit object; the fit's
        real part reproduces the DD-082 surface resistance."""
        resolved = {
            1: (5.8e7, 1.0, None),
            "zmin": (5.8e7, 1.0, None),
            2: (1.4e6, 100.0, None),
        }
        fits = fit_wall_impedances(resolved, 1e9, 1e10, tol=1e-3)
        assert fits[1] is fits["zmin"]
        assert fits[2] is not fits[1]
        f = np.logspace(9, 10, 31)
        for tag, (sig, mur, _) in resolved.items():
            r_ref = surface_resistance(f, sig, mur)
            err = np.abs(fits[tag].impedance(f).real - r_ref) / r_ref
            assert err.max() < 1.1e-3
