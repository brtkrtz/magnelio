"""Tests for the auto-derive path of ``PortSpecMultiConductor`` (Phase-2 cleanup 2).

The auto-derive path lets the user omit the declarative ``conductors``
list entirely; the factory walks the mesh's PEC mask on the port plane,
groups PEC-edge connected components, and uses the resulting
``conductor_node_groups`` directly.

Cleanup-2 also includes :meth:`Mesh.with_pec_boundaries`, which
consolidates ``BoundaryConditions(<face>="PEC")`` into the mesh's
``pec_mask_edges`` so that a PEC bbox face described via the BC dict
and a PEC material brick on that face produce indistinguishable
mesh state for every downstream consumer.

Validates:

1. The standalone helper :func:`extract_conductor_groups_from_mesh`
   returns K=2 groups on a rect-coax geometry whose mesh PEC mask
   carries both the outer-conductor body and the inner conductor.
2. The first group (ground) is larger than the second (signal),
   matching the geometry's outer-vs-inner area split.
3. A hollow-waveguide-shaped geometry (single PEC outer body, no
   inner) raises ``ValueError("at least 2")``.
4. ``PortSpecMultiConductor(conductors=None)`` is accepted at
   construction; ``n_modes > K-1`` raises at *factory* time after
   the runtime auto-detection.
5. ``build_modal_port`` on the auto-derive path yields a positive
   physically-sensible ``z_line`` and a single discrete TEM mode.
6. ``Mesh.with_pec_boundaries`` rejects unknown face names.
7. ``Mesh.with_pec_boundaries`` is in-plane-PEC-equivalent to a
   PEC-shell geometry built via ``Mesh.from_grid`` regions overlap.
8. Parallel-plate cross-section (2× PEC + 2× implicit-PMC) yields
   K=2 conductors and a ``z_line`` close to the analytical
   ``η₀ · d / W``.
9. WP2.3 cell-material fallback: when the PEC-edge graph yields < 2
   components (under-resolved conductor on the conformal path), the
   detection falls back to ``material_id`` + ``is_pec`` grouping with
   a ``UserWarning`` instead of a hard error; a mesh with no PEC at
   all still raises.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh, MeshControl
from magnelio.ports._modal import (
    BoxFace,
    PortPlane,
    PortSpecMultiConductor,
    build_modal_port,
    extract_conductor_groups_from_mesh,
)
from magnelio.solver.stability import courant_dt

# DD-103: the closure these fixtures always assumed.  A face
# with no BC used to evolve under the free curl operator —
# which IS the natural magnetic wall, hence "PMC".
_BC_CLOSED = {
    "xmin": "PEC",
    "xmax": "PEC",
    "ymin": "PEC",
    "ymax": "PEC",
    "zmin": "PEC",
    "zmax": "PEC",
}

_BC_OPEN = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PMC",
    "ymax": "PMC",
    "zmin": "PMC",
    "zmax": "PMC",
}

_BC_PEC_YMIN_YMAX_ZMIN_ZMAX = {
    "ymin": "PEC",
    "ymax": "PEC",
    "zmin": "PEC",
    "zmax": "PEC",
    "xmin": "PMC",
    "xmax": "PMC",
}

_BC_PEC_Z = {
    "zmin": "PEC",
    "zmax": "PEC",
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PMC",
    "ymax": "PMC",
}


def _rect_coax_mesh(
    *, B: float = 8e-3, b_air: float = 6e-3, a: float = 2e-3, L_x: float = 20e-3, f_max: float = 8e9
):
    """Rect coax with PEC outer body in the mesh + inner PEC brick.

    Geometry: ``bbox_pec`` (B×B×L_x, PEC) ∖ ``air`` (b_air×b_air×L_x)
    ∖ ``inner`` (a×a×L_x, PEC) — so the mesh PEC mask carries both
    the outer-conductor body (the shell between b_air and B) and the
    inner conductor.  Two independent PEC connected components on
    every cross-section.
    """
    pec = Material.pec()
    air = Material.air()
    y0_air = z0_air = (B - b_air) / 2
    y0_inner = z0_inner = (B - a) / 2
    bbox = Brick(origin=(0, 0, 0), size=(L_x, B, B), material=pec)
    air_region = Brick(origin=(0, y0_air, z0_air), size=(L_x, b_air, b_air), material=air)
    inner = Brick(origin=(0, y0_inner, z0_inner), size=(L_x, a, a), material=pec)
    model = GeometryModel()
    model.add(Difference(bbox, air_region))
    model.add(Difference(air_region, inner))
    model.add(inner)
    mesh = Mesh.from_geometry(
        model,
        MeshControl(
            min_nodes_per_wavelength=4,
            max_cell_size=0.4e-3,
            min_cells_per_feature=4,
        ),
        f_max=f_max,
    )
    return mesh, y0_inner, z0_inner, a


def _hollow_wg_mesh(
    *, B: float = 8e-3, b_air: float = 6e-3, L_x: float = 20e-3, f_max: float = 8e9
):
    """Hollow waveguide — single PEC outer-body component, no signal."""
    pec = Material.pec()
    air = Material.air()
    y0_air = z0_air = (B - b_air) / 2
    bbox = Brick(origin=(0, 0, 0), size=(L_x, B, B), material=pec)
    air_region = Brick(origin=(0, y0_air, z0_air), size=(L_x, b_air, b_air), material=air)
    model = GeometryModel()
    model.add(Difference(bbox, air_region))
    model.add(air_region)
    return Mesh.from_geometry(
        model,
        MeshControl(
            min_nodes_per_wavelength=4,
            max_cell_size=0.4e-3,
            min_cells_per_feature=4,
        ),
        f_max=f_max,
    )


def _detection_mesh(mesh, face=BoxFace.X_MIN):
    """The mesh the extractor sees in production.

    ``build_modal_port`` and ``resolve_declarative_port`` both flatten
    the port-plane PEC slab in line with the first interior slab before
    extracting conductors — without it the port face's own PEC closure
    (all-PEC is the default) covers the whole cross-section and the
    conductors merge into one component.
    """
    import dataclasses

    from magnelio._operators.material_matrices import (
        flatten_port_plane_pec_mask,
    )

    return dataclasses.replace(
        mesh,
        pec_mask_edges=flatten_port_plane_pec_mask(
            mesh.pec_mask_edges,
            mesh,
            face,
        ),
    )


class TestExtractor:
    def test_rect_coax_yields_two_groups(self):
        mesh, *_ = _rect_coax_mesh()
        mesh = _detection_mesh(mesh)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        groups = extract_conductor_groups_from_mesh(plane, mesh)
        assert len(groups) == 2
        assert all(g.size > 0 for g in groups)
        assert groups[0].size > groups[1].size

    def test_groups_disjoint_and_in_range(self):
        mesh, *_ = _rect_coax_mesh()
        mesh = _detection_mesh(mesh)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        groups = extract_conductor_groups_from_mesh(plane, mesh)
        all_nodes = np.concatenate(groups)
        assert all_nodes.size == np.unique(all_nodes).size
        Nu = mesh.Ny + 1
        Nv = mesh.Nz + 1
        assert all_nodes.min() >= 0
        assert all_nodes.max() < Nu * Nv

    def test_hollow_waveguide_rejected(self):
        mesh = _hollow_wg_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        with pytest.raises(ValueError, match="at least 2"):
            extract_conductor_groups_from_mesh(plane, mesh)


class TestSpecAcceptsAuto:
    def test_conductors_none_accepted(self):
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        assert spec.conductors is None

    def test_n_modes_zero_still_rejected_at_construction(self):
        with pytest.raises(ValueError, match="n_modes must be positive"):
            PortSpecMultiConductor(
                name="p",
                plane=BoxFace.X_MIN,
                conductors=None,
                n_modes=0,
            )


class TestFactoryAuto:
    def test_factory_auto_yields_one_tem_mode(self):
        mesh, *_ = _rect_coax_mesh()
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(mesh.grid, accuracy="normal")

        spec = PortSpecMultiConductor(
            name="auto",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=8e9)
        assert op.n_modes == 1
        assert op.discrete_modes[0].mode.z_line > 0.0

    def test_factory_auto_n_modes_beyond_k_minus_1(self):
        """WP-U2: on a homogeneous filling, ``n_modes > K-1`` no
        longer raises — the port is extended by the lowest TE/TM
        curl-curl channels, merged by ascending cut-off (TEM first).
        The QTEM path (``epsilon_r=None``) keeps the cap and points
        at WP-U6."""
        mesh, *_ = _rect_coax_mesh()
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(mesh.grid, accuracy="normal")

        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=5,
        )
        with pytest.warns(UserWarning, match="degenerate"):
            op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=8e9)
        assert op.n_modes == 5
        cutoffs = [dm.mode.omega_c for dm in op.discrete_modes]
        assert cutoffs[0] == 0.0
        assert all(c > 0.0 for c in cutoffs[1:])
        assert cutoffs == sorted(cutoffs)

        # QTEM path (epsilon_r=None): n_modes > K-1 goes through the
        # WP-U6 zeta-pencil channel source; at f_calc = 8 GHz no
        # higher mode propagates on this cross-section, so the
        # factory raises with guidance.
        qtem_spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=None,
            n_modes=5,
        )
        with pytest.raises(ValueError, match="propagate at f_calc"):
            build_modal_port(qtem_spec, mesh, m_eps, m_mu, dt=dt, f_calc=8e9)

    def test_factory_auto_background_pec_difference_hole(self):
        """WP2.1 regression (finding F1, stage 2): the full auto-derive
        chain works when the inner conductor exists only as a
        ``Difference`` hole exposing the PEC background — no explicit
        inner-PEC shape.  Requires the mesher to resolve the
        background-region feature (grid nodes inside the hole) so the
        PEC mask carries a second connected component.
        """
        pytest.importorskip("OCC.Core.BRepPrimAPI")
        from magnelio.geo import Cylinder

        D_i, D_a, L = 0.41e-3, 5.0e-3, 10.0e-3
        eps_r = 9.0
        pec = Material.pec()
        diel = Material.from_isotropic("alumina", epsilon=eps_r)

        out_cyl = Cylinder(origin=(0, 0, 0), radius=D_a / 2, height=L, axis="z", material=diel)
        in_cyl = Cylinder(origin=(0, 0, 0), radius=D_i / 2, height=L, axis="z", material=pec)
        model = GeometryModel(background=pec)
        model.add(Difference(out_cyl, in_cyl))  # no explicit inner shape

        mesh = Mesh.from_geometry(model, MeshControl(), f_max=10e9)
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(mesh.grid, accuracy="normal")

        spec = PortSpecMultiConductor(
            name="auto",
            plane=BoxFace.Z_MIN,
            conductors=None,
            epsilon_r=eps_r,
            n_modes=1,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
        assert op.n_modes == 1
        # Analytical: Z = eta0/(2*pi*sqrt(eps_r)) * ln(D_a/D_i) = 50.0 ohm.
        # Coarse transversal staircase leaves O(10 %) discretisation error.
        z_line = op.discrete_modes[0].mode.z_line
        assert 40.0 < z_line < 60.0, f"z_line = {z_line:.2f} ohm"


class TestWithPecBoundaries:
    def test_unknown_face_rejected(self):
        grid = GridLines(
            x=np.linspace(0, 1e-3, 4),
            y=np.linspace(0, 1e-3, 4),
            z=np.linspace(0, 1e-3, 4),
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
        with pytest.raises(ValueError, match="unknown bbox face"):
            mesh.with_pec_boundaries(["xmin", "frobnicate"])

    def test_in_plane_equivalent_to_pec_shell_geometry(self):
        """Vacuum + with_pec_boundaries vs PEC-shell geometry — same mesh PEC."""
        pec = Material.pec()
        air = Material.air()
        nx, ny, nz = 12, 8, 4
        L, W, H = 12e-3, 8e-3, 4e-3
        grid = GridLines(
            x=np.linspace(0, L, nx + 1),
            y=np.linspace(0, W, ny + 1),
            z=np.linspace(0, H, nz + 1),
        )
        dx, dy, dz = L / nx, W / ny, H / nz
        # Path A: PEC-shell geometry via overlap order in Mesh.from_grid.
        mesh_a = Mesh.from_grid(
            grid,
            regions=[
                (pec, (0.0, 0.0, 0.0, L, W, H)),
                (air, (dx, dy, dz, L - dx, W - dy, H - dz)),
            ],
        )
        # Path B: Vacuum mesh + with_pec_boundaries on all six faces.
        mesh_b = Mesh.from_grid(grid, boundary_conditions=_BC_CLOSED)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh_a)
        n_Ex = mesh_a.Nx * (mesh_a.Ny + 1) * (mesh_a.Nz + 1)
        n_Ey = (mesh_a.Nx + 1) * mesh_a.Ny * (mesh_a.Nz + 1)
        n_Ez = (mesh_a.Nx + 1) * (mesh_a.Ny + 1) * mesh_a.Nz
        flat_a = np.concatenate(
            [
                mesh_a.pec_mask_edges[0, :n_Ex],
                mesh_a.pec_mask_edges[1, :n_Ey],
                mesh_a.pec_mask_edges[2, :n_Ez],
            ]
        )
        flat_b = np.concatenate(
            [
                mesh_b.pec_mask_edges[0, :n_Ex],
                mesh_b.pec_mask_edges[1, :n_Ey],
                mesh_b.pec_mask_edges[2, :n_Ez],
            ]
        )
        in_plane = np.concatenate([plane.e_u_indices, plane.e_v_indices])
        assert np.array_equal(flat_a[in_plane], flat_b[in_plane])

    def test_does_not_mutate_input(self):
        grid = GridLines(
            x=np.linspace(0, 1e-3, 4),
            y=np.linspace(0, 1e-3, 4),
            z=np.linspace(0, 1e-3, 4),
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
        before = mesh.pec_mask_edges.copy()
        _ = mesh.with_pec_boundaries(["xmin"])
        assert np.array_equal(mesh.pec_mask_edges, before)


def _rect_coax_from_grid(*, with_inner: bool = True):
    """OCC-free rect coax on a uniform 0.5 mm grid, port on X_MIN.

    Outer PEC shell 1 mm thick (B = 8 mm, air window 6 mm), inner PEC
    brick 2 mm — all region boxes grid-aligned.
    """
    pec = Material.pec()
    air = Material.air()
    L, B = 10e-3, 8e-3
    grid = GridLines(
        x=np.linspace(0, L, 21),
        y=np.linspace(0, B, 17),
        z=np.linspace(0, B, 17),
    )
    regions = [
        (pec, (0.0, 0.0, 0.0, L, B, B)),
        (air, (0.0, 1e-3, 1e-3, L, 7e-3, 7e-3)),
    ]
    if with_inner:
        regions.append((pec, (0.0, 3e-3, 3e-3, L, 5e-3, 5e-3)))
    return Mesh.from_grid(grid, regions=regions, boundary_conditions=_BC_OPEN)


class TestCellMaterialFallback:
    """WP2.3: fallback to cell-material grouping when the PEC-edge
    graph yields < 2 components (locked decision: warn + fallback,
    no hard error)."""

    def _under_resolved_mesh(self):
        """Material carries both conductors, edge mask only the outer.

        Simulates the conformal-path under-resolution (finding F2): the
        sub-cell classifier admitted the inner conductor's edges, but
        the cell centres still classify as PEC.
        """
        mesh_full = _rect_coax_from_grid(with_inner=True)
        mesh_outer_only = _rect_coax_from_grid(with_inner=False)
        return Mesh(
            grid=mesh_full.grid,
            material_id=mesh_full.material_id,
            material_library=mesh_full.material_library,
            pec_mask_edges=mesh_outer_only.pec_mask_edges,
        ), mesh_full

    def test_fallback_recovers_under_resolved_inner_conductor(self):
        mesh, mesh_ref = self._under_resolved_mesh()
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        with pytest.warns(UserWarning, match="cell-material"):
            groups = extract_conductor_groups_from_mesh(plane, mesh)
        assert len(groups) == 2
        # On the staircase path the edge-derived node sets equal the
        # PEC-cell corner sets, so the fallback must reproduce the
        # reference grouping of the fully-masked mesh exactly.
        groups_ref = extract_conductor_groups_from_mesh(plane, mesh_ref)
        for g, g_ref in zip(groups, groups_ref):
            assert np.array_equal(np.sort(g), np.sort(g_ref))

    def test_mask_only_ground_plus_cell_only_signal(self):
        """Mixed sources merge: ground exists only as BC-PEC mask,
        signal only as PEC cell material."""
        pec = Material.pec()
        L, B = 10e-3, 8e-3
        grid = GridLines(
            x=np.linspace(0, L, 21),
            y=np.linspace(0, B, 17),
            z=np.linspace(0, B, 17),
        )
        mesh_mat = Mesh.from_grid(
            grid,
            regions=[(pec, (0.0, 3e-3, 3e-3, L, 5e-3, 5e-3))],
        )
        mesh_mask = Mesh.from_grid(grid, boundary_conditions=_BC_PEC_YMIN_YMAX_ZMIN_ZMAX)
        mesh = Mesh(
            grid=grid,
            material_id=mesh_mat.material_id,
            material_library=mesh_mat.material_library,
            pec_mask_edges=mesh_mask.pec_mask_edges,
        )
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        with pytest.warns(UserWarning, match="cell-material"):
            groups = extract_conductor_groups_from_mesh(plane, mesh)
        assert len(groups) == 2
        # Ground = the bbox wall ring, signal = the 4x4-cell inner
        # brick's 5x5 corner nodes.
        Nu, Nv = mesh.Ny + 1, mesh.Nz + 1
        assert groups[0].size == 2 * Nu + 2 * Nv - 4
        assert groups[1].size == 25

    def test_no_pec_anywhere_still_raises(self):
        grid = GridLines(
            x=np.linspace(0, 1e-3, 4),
            y=np.linspace(0, 1e-3, 4),
            z=np.linspace(0, 1e-3, 4),
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_OPEN)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        with pytest.raises(ValueError, match="no PEC edges"):
            extract_conductor_groups_from_mesh(plane, mesh)

    def test_healthy_path_emits_no_warning(self):
        mesh = _rect_coax_from_grid(with_inner=True)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            groups = extract_conductor_groups_from_mesh(plane, mesh)
        assert len(groups) == 2

    def test_single_conductor_after_fallback_still_raises(self):
        """Hollow-waveguide-like: one PEC component in both edge mask
        and cell material — fallback runs but must not warn, and the
        ValueError is preserved."""
        mesh_full = _rect_coax_from_grid(with_inner=False)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh_full)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError, match="at least 2"):
                extract_conductor_groups_from_mesh(plane, mesh_full)


class TestParallelPlate:
    def test_two_pec_two_pmc_yields_two_conductors(self):
        """Parallel-plate (z=PEC plates, y=PMC walls) → K=2 by auto-detect."""
        W, d, L = 8e-3, 2e-3, 10e-3
        grid = GridLines(
            x=np.linspace(0, L, 11),
            y=np.linspace(0, W, 17),
            z=np.linspace(0, d, 7),
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_PEC_Z)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, mesh)
        groups = extract_conductor_groups_from_mesh(plane, mesh)
        assert len(groups) == 2
        assert groups[0].size == groups[1].size

    def test_parallel_plate_z_line_close_to_analytical(self):
        """Z_line numerical agrees with η₀·d/W within 10 %."""
        eta0 = 376.730313668
        W, d, L = 8e-3, 2e-3, 10e-3
        grid = GridLines(
            x=np.linspace(0, L, 11),
            y=np.linspace(0, W, 17),
            z=np.linspace(0, d, 7),
        )
        mesh = Mesh.from_grid(grid, boundary_conditions=_BC_PEC_Z)
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)
        dt = courant_dt(grid, accuracy="normal")
        spec = PortSpecMultiConductor(
            name="pp",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        op = build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=10e9)
        z_num = op.discrete_modes[0].mode.z_line
        z_anal = eta0 * d / W
        rel_err = abs(z_num - z_anal) / z_anal
        assert rel_err < 0.10, (
            f"|Z_num - Z_anal| / Z_anal = {rel_err:.3f} (target < 0.10);"
            f" Z_num={z_num:.3f} Ω, Z_anal={z_anal:.3f} Ω"
        )


class TestSurfaceFragmentAbsorption:
    """Isolated PEC surface fragments must not become phantom conductors.

    On a curved conductor the staircase edge graph can leave a handful
    of flagged edges near the apex disconnected from the conductor
    body (their connecting edges fall below the classifier threshold).
    Grouped as its own "conductor", such a fragment forms a
    near-zero-gap TEM channel with an enormous C' that sorts first in
    the capacitance-ordered channel basis and shadows the real mode
    (measured z_line 0.95 Ω instead of ~50 Ω on a curved-electrode
    stripline).  The PEC-cell corner links restore the physical
    connectivity, so the fragment joins the conductor it sits on.
    """

    @staticmethod
    def _fragment_mesh():
        """Rect-coax detection mesh with one isolated surface u-edge.

        Cuts the flagged edges around one u-edge on the inner
        conductor's top surface, leaving that edge as an isolated
        two-node fragment in the PEC *edge* graph while the PEC cells
        underneath still carry the conductor body.
        """
        import dataclasses

        from magnelio._operators.curl import build_gradient_matrix
        from magnelio.ports._modal.curl_curl_2d import build_2d_gradient

        mesh, y0, z0, a = _rect_coax_mesh()
        det = _detection_mesh(mesh)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, det)

        g_3d = build_gradient_matrix(det.grid)
        g_2d, _, primal_2d = build_2d_gradient(plane, det.grid, g_3d)
        g = g_2d.tocsr()
        pair_to_row = {}
        for r in range(g.shape[0]):
            nz = g.indices[g.indptr[r] : g.indptr[r + 1]]
            if nz.size == 2:
                pair_to_row[frozenset((int(nz[0]), int(nz[1])))] = r

        # Local raster node = i_u * Nv_node + i_v; on X_MIN u = y, v = z.
        Nv_node = plane.n_nodes_v
        iu_mid = int(np.argmin(np.abs(det.grid.y - (y0 + a / 2))))
        iv_top = int(np.argmin(np.abs(det.grid.z - (z0 + a))))
        node_a, node_b, node_c, node_d = ((iu_mid + k) * Nv_node + iv_top for k in (-1, 0, 1, 2))
        assert frozenset((node_b, node_c)) in pair_to_row

        cut = [frozenset((node_a, node_b)), frozenset((node_c, node_d))]
        for n in (node_b, node_c):
            cut.append(frozenset((n, n - 1)))
            cut.append(frozenset((n, n + 1)))

        n_ex = det.Nx * (det.Ny + 1) * (det.Nz + 1)
        n_ey = (det.Nx + 1) * det.Ny * (det.Nz + 1)
        mask = det.pec_mask_edges.copy()
        for pair in cut:
            r = pair_to_row.get(pair)
            if r is None:
                continue
            gidx = int(primal_2d[r])
            if gidx < n_ex:
                mask[0, gidx] = False
            elif gidx < n_ex + n_ey:
                mask[1, gidx - n_ex] = False
            else:
                mask[2, gidx - n_ex - n_ey] = False

        det = dataclasses.replace(det, pec_mask_edges=mask)
        plane = PortPlane.from_mesh(BoxFace.X_MIN, det)
        return det, plane, (node_b, node_c)

    def test_fixture_is_a_phantom_on_the_edge_graph_alone(self):
        """Sanity: without cell links the fragment is its own group."""
        from unittest import mock

        from magnelio.ports._modal import auto_conductors

        det, plane, fragment = self._fragment_mesh()
        empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        with mock.patch.object(auto_conductors, "_pec_cell_corner_links", return_value=empty):
            groups = extract_conductor_groups_from_mesh(plane, det)
        assert len(groups) == 3
        assert sorted(groups[-1].tolist()) == sorted(fragment)

    def test_fragment_joins_its_conductor(self):
        det, plane, fragment = self._fragment_mesh()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            groups = extract_conductor_groups_from_mesh(plane, det)
        assert len(groups) == 2
        signal = set(groups[1].tolist())
        assert set(fragment) <= signal
