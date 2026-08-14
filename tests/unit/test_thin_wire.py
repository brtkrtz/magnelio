"""DD-080 unit gates: the Holland/Simpson thin-wire sub-cell model.

Machine-precision checks of the paired (m, 1/m) correction: the factor
formula on uniform and graded grids, the ring stencil for all three
segment axes, the pair-product preservation (the DD-053 identity the
paired encoding exists for), the conservative min composition rule at
corners, precedence against conformal-solid data, radius validation,
and the CFL minima.  The OCC-marked tests cover the geometry object,
the mesher end-to-end wiring and the project-store round-trip.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.circuit.rasterize import EdgePath
from magnelio.mesh._thin_wire import (
    KAPPA0,
    _collect_requests,
    _ring_stencil,
    apply_thin_wire_path,
)
from magnelio.mesh.grid import GridLines
from magnelio.mesh.indexing import (
    edge_index_Ex,
    edge_index_Ey,
    edge_index_Ez,
)
from magnelio.mesh.mesher import Mesh


def _uniform_grid(n: int = 8, L: float = 8e-3) -> GridLines:
    ax = np.linspace(0.0, L, n + 1)
    return GridLines(x=ax.copy(), y=ax.copy(), z=ax.copy())


def _offsets(grid: GridLines) -> tuple[int, int]:
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    return (
        Nx * (Ny + 1) * (Nz + 1),
        (Nx + 1) * Ny * (Nz + 1),
    )


def _z_path(grid: GridLines, i: int, j: int, ks) -> EdgePath:
    n_Ex, n_Ey = _offsets(grid)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    return EdgePath(
        axes=["z"] * len(ks),
        ijk=[(i, j, k) for k in ks],
        signs=[1] * len(ks),
        dls=[float(grid.dz[k]) for k in ks],
        flat_indices=[n_Ex + n_Ey + edge_index_Ez(i, j, k, Nx, Ny, Nz) for k in ks],
    )


def _mask_path(mesh, path) -> None:
    grid = mesh.grid
    n_Ex, n_Ey = _offsets(grid)
    for axis, flat in zip(path.axes, path.flat_indices):
        comp = {"x": 0, "y": 1, "z": 2}[axis]
        local = flat - (0, n_Ex, n_Ex + n_Ey)[comp]
        mesh.pec_mask_edges[comp, local] = True


# ---------------------------------------------------------------------------
# Factor formula + stencil
# ---------------------------------------------------------------------------


def test_factor_uniform_grid():
    """m = ln(d/a)/ln(1/KAPPA0) on a uniform grid, identical on all faces."""
    grid = _uniform_grid()
    d = float(grid.dx[0])
    a = 0.05 * d
    path = _z_path(grid, 4, 4, [3, 4])
    face_m, edge_m = _collect_requests(path, a, grid)
    m_expect = math.log(d / a) / math.log(1.0 / KAPPA0)
    assert len(face_m) == 8  # 2 segments x 4 faces
    assert len(edge_m) == 12  # 2 x 8 partners, 4 shared at the joint
    for m in list(face_m.values()) + list(edge_m.values()):
        assert m == pytest.approx(m_expect, rel=1e-14)


def test_factor_graded_grid():
    """Per-face delta and the geometric-mean r0 on a graded transverse axis."""
    z = np.linspace(0.0, 8e-3, 9)
    x = np.concatenate([[0.0], np.cumsum(1e-3 * 1.3 ** np.arange(8))])
    grid = GridLines(x=x, y=np.linspace(0.0, 8e-3, 9).copy(), z=z)
    i, j, k = 4, 4, 3
    a = 0.03e-3
    path = _z_path(grid, i, j, [k])
    # Single-axis grading is deliberately anisotropic here (formula
    # check only) — the accuracy warning is expected.
    with pytest.warns(UserWarning, match="anisotropic"):
        face_m, _ = _collect_requests(path, a, grid)
    ds = [float(grid.dy[j]), float(grid.dy[j - 1]), float(grid.dx[i]), float(grid.dx[i - 1])]
    r0 = KAPPA0 * math.exp(sum(math.log(d) for d in ds) / 4.0)
    ring = _ring_stencil("z", (i, j, k), grid)
    assert len(ring) == 4
    for face, delta_t, _ in ring:
        m_expect = math.log(delta_t / a) / math.log(delta_t / r0)
        assert face_m[face] == pytest.approx(m_expect, rel=1e-14)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_ring_stencil_faces_encircle_edge(axis):
    """The 4 stencil faces are exactly the faces whose curl contains the edge.

    Cross-checked against the discrete curl: an impulse on the wire edge
    must produce nonzero circulation exactly on the 4 stencil faces.
    """
    from magnelio._operators.curl import build_curl_matrix

    grid = _uniform_grid(6)
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    i, j, k = 3, 3, 3
    if axis == "x":
        flat = edge_index_Ex(i, j, k, Nx, Ny, Nz)
    elif axis == "y":
        flat = n_Ex + edge_index_Ey(i, j, k, Nx, Ny, Nz)
    else:
        flat = n_Ex + n_Ey + edge_index_Ez(i, j, k, Nx, Ny, Nz)

    C = build_curl_matrix(grid)
    e = np.zeros(C.shape[1])
    e[flat] = 1.0
    circ_faces = set(np.nonzero(C @ e)[0])

    ring = _ring_stencil(axis, (i, j, k), grid)
    assert len(ring) == 4
    assert {face for face, _, _ in ring} == circ_faces


def test_ring_stencil_clips_at_domain_boundary():
    grid = _uniform_grid(6)
    ring_corner = _ring_stencil("z", (0, 0, 3), grid)
    assert len(ring_corner) == 2  # i-1 and j-1 sides are outside
    ring_face = _ring_stencil("z", (0, 3, 3), grid)
    assert len(ring_face) == 3


# ---------------------------------------------------------------------------
# Pair-product preservation + mechanism
# ---------------------------------------------------------------------------


def test_pair_product_preserved_machine_precision():
    """M_eps x M_mu on every (face, partner edge) pair is untouched."""
    grid = _uniform_grid()
    mesh = Mesh.from_grid(grid)
    path = _z_path(grid, 4, 4, [2, 3, 4, 5])
    _mask_path(mesh, path)
    m_eps0, m_mu0 = build_M_eps(mesh), build_M_mu(mesh)

    apply_thin_wire_path(mesh, path, 0.05 * float(grid.dx[0]))
    m_eps1, m_mu1 = build_M_eps(mesh), build_M_mu(mesh)

    n_pairs = 0
    for axis, ijk in zip(path.axes, path.ijk):
        for face, _, edges in _ring_stencil(axis, ijk, grid):
            for e in edges:
                p0 = m_eps0[e] * m_mu0[face]
                p1 = m_eps1[e] * m_mu1[face]
                assert abs(p1 - p0) <= 1e-14 * p0
                n_pairs += 1
    assert n_pairs == 32


def test_correction_scales_mu_and_eps_reciprocally():
    grid = _uniform_grid()
    mesh = Mesh.from_grid(grid)
    d = float(grid.dx[0])
    a = 0.05 * d
    path = _z_path(grid, 4, 4, [3, 4])
    _mask_path(mesh, path)
    m_eps0, m_mu0 = build_M_eps(mesh), build_M_mu(mesh)
    apply_thin_wire_path(mesh, path, a)
    m_eps1, m_mu1 = build_M_eps(mesh), build_M_mu(mesh)

    m = math.log(d / a) / math.log(1.0 / KAPPA0)
    assert m > 1.0  # thin wire: added inductance
    fm, em = mesh.face_material, mesh.edge_material
    faces = np.nonzero(fm.category == 2)[0]
    edges = np.nonzero(em.category == 1)[0]
    assert faces.size == 8 and edges.size == 12
    np.testing.assert_allclose(m_mu1[faces] / m_mu0[faces], m, rtol=1e-14)
    np.testing.assert_allclose(m_eps1[edges] / m_eps0[edges], 1.0 / m, rtol=1e-14)


def test_fat_wire_reduces_mu():
    """a > r0: the bare grid over-estimates the inductance, m < 1."""
    grid = _uniform_grid()
    mesh = Mesh.from_grid(grid)
    d = float(grid.dx[0])
    a = 0.25 * d  # r0 ~ 0.1985 d < a < 0.30 d
    path = _z_path(grid, 4, 4, [3, 4])
    _mask_path(mesh, path)
    with pytest.warns(UserWarning, match="fatter than the bare-grid"):
        apply_thin_wire_path(mesh, path, a)
    fm = mesh.face_material
    faces = np.nonzero(fm.category == 2)[0]
    m = math.log(d / a) / math.log(1.0 / KAPPA0)
    assert m < 1.0
    A_face = d * d
    np.testing.assert_allclose(fm.A_face_free[faces] / A_face, m, rtol=1e-14)
    # The legal radius range never reaches the build_M_mu 1% floor.
    assert fm.A_face_free[faces].min() / A_face > 0.74


# ---------------------------------------------------------------------------
# Composition + precedence
# ---------------------------------------------------------------------------


def test_corner_min_rule():
    """At an x->z bend the shared face takes the smaller m exactly once."""
    grid = GridLines(
        x=np.linspace(0.0, 8e-3, 9),
        y=np.linspace(0.0, 8e-3, 9),
        z=np.concatenate([[0.0], np.cumsum(0.5e-3 * 1.25 ** np.arange(8))]),
    )
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    i, j, k = 4, 4, 3
    a = 0.02e-3
    # x-segment into the corner node, then a z-segment out of it.
    path = EdgePath(
        axes=["x", "z"],
        ijk=[(i - 1, j, k), (i, j, k)],
        signs=[1, 1],
        dls=[float(grid.dx[i - 1]), float(grid.dz[k])],
        flat_indices=[
            edge_index_Ex(i - 1, j, k, Nx, Ny, Nz),
            n_Ex + n_Ey + edge_index_Ez(i, j, k, Nx, Ny, Nz),
        ],
    )
    face_m, _ = _collect_requests(path, a, grid)
    ring_x = dict((f, (d, e)) for f, d, e in _ring_stencil("x", (i - 1, j, k), grid))
    ring_z = dict((f, (d, e)) for f, d, e in _ring_stencil("z", (i, j, k), grid))
    shared = set(ring_x) & set(ring_z)
    assert shared, "fixture must produce a shared corner face"
    for f in shared:
        m_x = _collect_requests(
            EdgePath(
                axes=["x"],
                ijk=[(i - 1, j, k)],
                signs=[1],
                dls=[float(grid.dx[i - 1])],
                flat_indices=[edge_index_Ex(i - 1, j, k, Nx, Ny, Nz)],
            ),
            a,
            grid,
        )[0][f]
        m_z = _collect_requests(
            EdgePath(
                axes=["z"],
                ijk=[(i, j, k)],
                signs=[1],
                dls=[float(grid.dz[k])],
                flat_indices=[n_Ex + n_Ey + edge_index_Ez(i, j, k, Nx, Ny, Nz)],
            ),
            a,
            grid,
        )[0][f]
        assert face_m[f] == min(m_x, m_z)


def test_conformal_solid_precedence():
    """A face already claimed by a conformal solid (cat 2) is skipped."""
    grid = _uniform_grid()
    mesh = Mesh.from_grid(grid)
    path = _z_path(grid, 4, 4, [3, 4])
    _mask_path(mesh, path)

    from magnelio.mesh._thin_wire import _ensure_face_material

    fm = _ensure_face_material(mesh)
    ring = _ring_stencil("z", (4, 4, 3), grid)
    claimed_face = ring[0][0]
    d = float(grid.dx[0])
    fm.category[claimed_face] = 2
    fm.mu_avg[claimed_face] = 1.0
    fm.A_face_free[claimed_face] = 0.5 * d * d
    fm.L_dual_free[claimed_face] = d

    with pytest.warns(UserWarning, match="claimed by a conformal solid"):
        apply_thin_wire_path(mesh, path, 0.05 * d)
    assert fm.A_face_free[claimed_face] == 0.5 * d * d  # untouched


def test_dielectric_cat1_composes_multiplicatively():
    """On a dielectric-boundary edge the wire divides the existing eps_avg."""
    from magnelio.materials.material import Material

    grid = _uniform_grid()
    fr4 = Material(name="FR4", epsilon=(4.0, 4.0, 4.0))
    mesh = Mesh.from_grid(
        grid,
        regions=[(fr4, (0.0, 0.0, 0.0, 8e-3, 4e-3, 8e-3))],
    )
    d = float(grid.dx[0])
    a = 0.05 * d
    # Wire along the dielectric interface plane y = 4 mm (j = 4).
    path = _z_path(grid, 4, 4, [3, 4])
    _mask_path(mesh, path)
    m_eps0 = build_M_eps(mesh)
    apply_thin_wire_path(mesh, path, a)
    m_eps1 = build_M_eps(mesh)
    em = mesh.edge_material
    edges = np.nonzero(em.category == 1)[0]
    m = math.log(d / a) / math.log(1.0 / KAPPA0)
    # Every corrected edge scales by exactly 1/m regardless of its
    # staircase eps (2.5 on the interface plane, 1 or 4 off it).
    np.testing.assert_allclose(m_eps1[edges] / m_eps0[edges], 1.0 / m, rtol=1e-14)
    assert not np.allclose(m_eps0[edges] / m_eps0[edges][0], 1.0), (
        "fixture should mix eps values across the corrected edges"
    )


def test_masked_edges_silently_skipped():
    """Radial edges inside a PEC solid (monopole base) skip WITHOUT warning."""
    grid = _uniform_grid()
    mesh = Mesh.from_grid(grid)
    path = _z_path(grid, 4, 4, [3, 4])
    _mask_path(mesh, path)
    # Mask the two radial partner edges of the first ring face (as if
    # the wire landed on a PEC plate there).
    ring = _ring_stencil("z", (4, 4, 3), grid)
    n_Ex, n_Ey = _offsets(grid)
    for e in ring[0][2]:
        comp = 0 if e < n_Ex else (1 if e < n_Ex + n_Ey else 2)
        local = e - (0, n_Ex, n_Ex + n_Ey)[comp]
        mesh.pec_mask_edges[comp, local] = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_thin_wire_path(mesh, path, 0.05 * float(grid.dx[0]))
    em = mesh.edge_material
    for e in ring[0][2]:
        assert em.category[e] == 0  # untouched


# ---------------------------------------------------------------------------
# Radius validation + CFL
# ---------------------------------------------------------------------------


def test_anisotropic_transverse_cells_warn():
    """Single-axis grading at the wire triggers the anisotropy warning."""
    x = np.concatenate([[0.0], np.cumsum(0.2e-3 * 1.45 ** np.arange(8))])
    grid = GridLines(x=x, y=np.linspace(0.0, 8e-3, 9).copy(), z=np.linspace(0.0, 8e-3, 9).copy())
    path = _z_path(grid, 4, 4, [3, 4])
    with pytest.warns(UserWarning, match="anisotropic"):
        _collect_requests(path, 0.02e-3, grid)


def test_radius_validation_errors():
    from magnelio.geo.wire import ThinWire  # noqa: F401 (ctor check below)

    grid = _uniform_grid()
    mesh = Mesh.from_grid(grid)
    d = float(grid.dx[0])
    path = _z_path(grid, 4, 4, [3, 4])
    _mask_path(mesh, path)
    with pytest.raises(ValueError, match="refine the mesh or reduce the radius"):
        apply_thin_wire_path(mesh, path, 0.35 * d)


def test_thinwire_constructor_errors():
    pytest.importorskip("OCC.Core.BRepBuilderAPI")
    from magnelio.geo import Curve, ThinWire

    c = Curve.polyline([(0, 0, 0), (0, 0, 5e-3)])
    with pytest.raises(ValueError, match="radius must be positive"):
        ThinWire(c, radius=0.0)
    with pytest.raises(TypeError, match="must be a Curve"):
        ThinWire("not a curve", radius=1e-4)
    w = ThinWire(c, radius=1e-4, name="w")
    assert w.material.is_pec


def test_cfl_minima():
    """Thin wire: min eps_eff = 1/m (mu side inert); fat wire: min mu_eff = m."""
    from magnelio.solver.stability import (
        compute_min_effective_eps,
        compute_min_effective_mu,
    )

    grid = _uniform_grid()
    d = float(grid.dx[0])

    mesh = Mesh.from_grid(grid)
    path = _z_path(grid, 4, 4, [3, 4])
    _mask_path(mesh, path)
    a_thin = 0.05 * d
    apply_thin_wire_path(mesh, path, a_thin)
    m = math.log(d / a_thin) / math.log(1.0 / KAPPA0)
    assert compute_min_effective_eps(mesh) == pytest.approx(1.0 / m, rel=1e-12)
    assert compute_min_effective_mu(mesh) == 1.0

    mesh2 = Mesh.from_grid(_uniform_grid())
    path2 = _z_path(mesh2.grid, 4, 4, [3, 4])
    _mask_path(mesh2, path2)
    a_fat = 0.25 * d
    with pytest.warns(UserWarning):
        apply_thin_wire_path(mesh2, path2, a_fat)
    m_fat = math.log(d / a_fat) / math.log(1.0 / KAPPA0)
    assert compute_min_effective_mu(mesh2) == pytest.approx(m_fat, rel=1e-12)
    assert compute_min_effective_eps(mesh2) == 1.0


# ---------------------------------------------------------------------------
# OCC end-to-end (geometry object -> mesher -> store)
# ---------------------------------------------------------------------------


@pytest.fixture
def _occ_model():
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    import magnelio as em
    from magnelio.geo import Brick, Curve, GeometryModel
    from magnelio.geo import ThinWire as _ThinWire

    model = GeometryModel()
    model.add(Brick(origin=(0, 0, 0), size=(20e-3, 20e-3, 2e-3), material=em.Material.pec()))
    model.add(
        _ThinWire(
            Curve.polyline([(10e-3, 10e-3, 2e-3), (10e-3, 10e-3, 14e-3)]),
            radius=0.1e-3,
            name="stub",
        )
    )
    return model


def test_wire_endpoint_in_pec_solid_validates(_occ_model):
    _occ_model.validate()  # must not raise despite the contact


def test_from_geometry_end_to_end(_occ_model):
    from magnelio.mesh.mesher import Mesh as M
    from magnelio.mesh.mesher import MeshControl

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mesh = M.from_geometry(
            _occ_model,
            MeshControl(min_nodes_per_wavelength=15),
            f_max=10e9,
        )
    # Wire vertex planes anchored the grid on all axes.
    for arr, v in (
        (mesh.grid.x, 10e-3),
        (mesh.grid.y, 10e-3),
        (mesh.grid.z, 2e-3),
        (mesh.grid.z, 14e-3),
    ):
        assert np.any(np.isclose(arr, v))
    # The wire's Ez chain is masked and its stencil corrected.
    assert (mesh.face_material.category == 2).any()
    assert (mesh.edge_material.category == 1).any()
    i = int(np.argmin(np.abs(mesh.grid.x - 10e-3)))
    j = int(np.argmin(np.abs(mesh.grid.y - 10e-3)))
    k0 = int(np.argmin(np.abs(mesh.grid.z - 2e-3)))
    Nx, Ny, Nz = mesh.grid.Nx, mesh.grid.Ny, mesh.grid.Nz
    assert mesh.pec_mask_edges[2, edge_index_Ez(i, j, k0, Nx, Ny, Nz)]


def test_project_store_roundtrip(_occ_model, tmp_path):
    import json

    import magnelio as em
    from magnelio.io.project import ProjectStore
    from magnelio.mesh.mesher import Mesh as M
    from magnelio.mesh.mesher import MeshControl

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = M.from_geometry(
            _occ_model,
            MeshControl(min_nodes_per_wavelength=15),
            f_max=10e9,
        )
    ProjectStore.create(tmp_path / "p", geometry=_occ_model, mesh=mesh)
    gj = json.loads((tmp_path / "p" / "geometry.json").read_text())
    assert gj["kinds"] == ["solid", "wire"]
    assert gj["radii"] == [None, 0.1e-3]

    proj = em.open_project(tmp_path / "p")
    m2 = proj.mesh
    assert np.array_equal(m2.pec_mask_edges, mesh.pec_mask_edges)
    assert np.array_equal(m2.face_material.category, mesh.face_material.category)
    assert np.allclose(m2.face_material.A_face_free, mesh.face_material.A_face_free, equal_nan=True)
    assert np.allclose(m2.edge_material.eps_avg, mesh.edge_material.eps_avg, equal_nan=True)
    with pytest.warns(UserWarning, match="thin wires"):
        g = proj.geometry
    assert len(g.shapes) == 1  # the wire is metadata-only in v1
