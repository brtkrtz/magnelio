"""Pilot 4: force-bbox-wall variant — does forcing the whole bbox wall
to ground fix Z_line?

Pilot 1 showed that the clean ``D_a × D_a + background=pec`` setup
produces a fragmented PEC mask along the bbox wall: edges in the
non-tangent region (i.e. in the corners of the dielectric square)
are PEC, but edges close to the cylinder tangent point are not.
``extract_conductor_groups_from_mesh`` then groups the bbox-wall
PEC-edge-connected nodes plus the inner cylinder, but leaves the
"non-PEC" bbox-wall nodes (along the cylinder tangent) free to
hold non-zero potential.

In a continuous coax with a square outer conductor at radius D_a/2,
the entire bbox wall is at φ=0.  Does forcing the entire bbox-wall
node row into the ground group fix Z_line?  If yes, the bug is in
the auto-detection's PEC-edge connectivity step (it doesn't see the
"PEC by background-material" walls correctly).
"""

from __future__ import annotations

import numpy as np

from magnelio import Material, Mesh, MeshControl
from magnelio._operators.curl import build_curl_matrix, build_gradient_matrix
from magnelio._operators.material_matrices import (
    build_M_eps,
    build_M_mu,
    flatten_port_plane_mass,
    flatten_port_plane_pec_mask,
)
from magnelio.geo import Cylinder, Difference, GeometryModel
from magnelio.ports._modal import BoxFace
from magnelio.ports._modal.auto_conductors import extract_conductor_groups_from_mesh
from magnelio.ports._modal.curl_curl_2d import build_2d_curl_curl, build_2d_gradient
from magnelio.ports._modal.port_plane import PortPlane
from magnelio.ports._modal.tem_laplace import solve_tem_laplace

D_i, D_a, EPS_R, L, F_MAX = 0.41e-3, 5.0e-3, 9.0, 10e-3, 10e9


def _model():
    pec = Material.pec()
    diel = Material(name="dielectric", epsilon=(EPS_R,) * 3)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=D_a / 2, height=L, axis="z", material=diel)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=D_i / 2, height=L, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    return model


def _examine(label: str, force_bbox_wall: bool, conformal: bool) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    mesh = Mesh.from_geometry(
        _model(),
        MeshControl(
            min_nodes_per_wavelength=8,
            min_cells_per_feature=3,
            growth_factor=1.4,
            conformal=conformal,
            max_cell_size=0.4e-3,
            min_cell_size=50e-6,
            min_feature_gap=20e-6,
        ),
        f_max=F_MAX,
    )
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    print(f"  Mesh: {Nx} x {Ny} x {Nz}")

    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    m_eps = flatten_port_plane_mass(m_eps, mesh, BoxFace.Z_MIN)
    object.__setattr__(
        mesh,
        "pec_mask_edges",
        flatten_port_plane_pec_mask(mesh.pec_mask_edges, mesh, BoxFace.Z_MIN),
    )

    plane = PortPlane.from_mesh(BoxFace.Z_MIN, mesh)
    c_3d = build_curl_matrix(mesh.grid)
    g_3d = build_gradient_matrix(mesh.grid)
    _, M_2d, _ = build_2d_curl_curl(plane, mesh.grid, m_eps, m_mu, c_3d)
    g_2d, _, _ = build_2d_gradient(plane, mesh.grid, g_3d)

    groups = extract_conductor_groups_from_mesh(plane, mesh)
    print(f"  auto-detect: {len(groups)} groups, sizes {[len(g) for g in groups]}")

    if force_bbox_wall:
        # Force every bbox-wall node into ground group.
        u_axis = plane.face.u_axis
        v_axis = plane.face.v_axis
        Nu_node = (Nx, Ny, Nz)[u_axis] + 1
        Nv_node = (Nx, Ny, Nz)[v_axis] + 1
        bbox_wall_nodes = []
        for iu in range(Nu_node):
            for iv in range(Nv_node):
                if iu in (0, Nu_node - 1) or iv in (0, Nv_node - 1):
                    bbox_wall_nodes.append(iu * Nv_node + iv)
        bbox_wall_set = set(bbox_wall_nodes)
        # Merge with existing ground group; remove from signal groups.
        ground = set(groups[0].tolist()) | bbox_wall_set
        signal = [
            np.asarray([n for n in g.tolist() if n not in bbox_wall_set], dtype=np.int64)
            for g in groups[1:]
        ]
        groups = [np.asarray(sorted(ground), dtype=np.int64)] + signal
        print(f"  forced:      {len(groups)} groups, sizes {[len(g) for g in groups]}")

    modes = solve_tem_laplace(plane, g_2d, M_2d, groups, EPS_R, grid=mesh.grid, m_mu_flat=m_mu)
    print(f"  Z_line_num = {modes[0].z_line:.3f} Ohm")


if __name__ == "__main__":
    _examine(
        "A — clean, conformal=True, auto-detect (status quo)", force_bbox_wall=False, conformal=True
    )
    _examine(
        "B — clean, conformal=True, ground = auto + bbox-wall", force_bbox_wall=True, conformal=True
    )
    _examine("C — clean, conformal=False, auto-detect", force_bbox_wall=False, conformal=False)
    _examine(
        "D — clean, conformal=False, ground = auto + bbox-wall",
        force_bbox_wall=True,
        conformal=False,
    )
