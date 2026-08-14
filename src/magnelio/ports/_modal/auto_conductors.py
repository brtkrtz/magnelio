"""Auto-derive conductor node groups from the FIT mesh PEC mask.

Phase-2 cleanup item 2 (`reference_architecture_phase2_mode_solver.md`
§2.6 — "All material information comes from the 3D mesh").

Given a :class:`PortPlane` and the underlying :class:`Mesh`, this module
walks the in-plane primal-2D edge graph, collects the edges flagged as
PEC in ``mesh.pec_mask_edges``, computes connected components on the
incident-node graph, and returns the resulting groups in
``conductor_node_groups`` form (i.e. lists of local-2D-node indices in
the ``i_u * Nv_node + i_v`` basis used by :func:`build_2d_gradient` and
consumed by :func:`solve_tem_laplace` / :func:`solve_qtem_laplace`).

Group ordering follows the "ground-first wins" convention: the
component touching at least one bbox-lateral-wall node is placed first
(= φ = 0 reference), the remaining components are placed in descending
node count.  When no component touches the bbox walls (rare — typically
"floating" geometries), the largest component is used as ground.

Two material sources feed the grouping: the PEC *edge* mask carries
the conductor node sets, and the corner links of PEC *cells* in the
port-adjacent slab (``material_id`` + ``Material.is_pec``) decide
which edge components belong to the same conductor — isolated
staircase fragments on a curved conductor's surface would otherwise
form phantom conductors.  The cell links fuse component labels only;
they never add nodes to a conductor (that would widen staircased
conductors and shift line impedances).  The exception is the
under-resolved rescue: when the edge graph alone yields fewer than
two components (a thin conductor whose edges pass the sub-cell
classifier's η threshold while only its cell centres classify PEC),
the merged node sets are used as the only material source, with a
:class:`UserWarning` (refine the transversal mesh).

This closes the gap between the high-level ``PortSpecMultiConductor``
and a true OCC-cross-section discretisation: the user's ``Mesh`` already
carries the staircase/Dey-Mittra PEC mask of the actual conductors, so
the modal solver does not need an additional declarative description.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from magnelio._operators.curl import build_gradient_matrix
from magnelio.mesh.mesher import Mesh
from magnelio.ports._modal.curl_curl_2d import build_2d_gradient
from magnelio.ports._modal.port_plane import PortPlane


def extract_conductor_groups_from_mesh(
    plane: PortPlane,
    mesh: Mesh,
    extra_pec_edge_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Auto-derive conductor node groups from the mesh PEC mask.

    Parameters
    ----------
    plane : PortPlane
        Port-plane geometry built via :meth:`PortPlane.from_mesh`.
    extra_pec_edge_mask : np.ndarray of bool, optional
        Additional PEC edges in the plane's ``[e_u | e_v]`` basis,
        OR-ed onto the mesh-mask edges before the node graph is built.
        Used by sub-face ports to feed the window-boundary Dirichlet
        ring from the edge-BC rule
        (:func:`magnelio.ports._modal.port_plane.build_port_edge_pec_mask`)
        into the grouping, so an embedded port's PEC frame can act as
        its ground conductor.
    mesh : Mesh
        FIT mesh whose ``pec_mask_edges`` carries the (staircased /
        Dey-Mittra-refined) PEC information of the OCC geometry.

    Returns
    -------
    list of np.ndarray
        ``[ground, signal_1, signal_2, ...]`` lists of *local* 2D
        primal-node indices.  Indices are in the
        ``i_u * Nv_node + i_v`` raster ordering used by
        :func:`build_2d_gradient` and consumed by
        :func:`solve_tem_laplace` / :func:`solve_qtem_laplace`.

        - ``ground`` (groups[0]) is the connected component that
          contains at least one bbox-lateral-wall node, or — if no
          component touches the walls — the largest component.
        - Remaining components are ordered by descending node count.

    Warns
    -----
    UserWarning
        If the PEC-edge graph yields fewer than two connected components
        but the cell-material fallback (see module docstring) recovers a
        valid multi-conductor grouping — typically an under-resolved
        conductor on the port plane; consider refining the transversal
        mesh there.

    Raises
    ------
    ValueError
        If the port plane has no PEC edges and no PEC cells, or if fewer
        than two connected components are found even after the
        cell-material fallback (no signal conductor — not a
        multi-conductor port).
    """
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    pec_E_flat = np.concatenate(
        [
            mesh.pec_mask_edges[0, :n_Ex],
            mesh.pec_mask_edges[1, :n_Ey],
            mesh.pec_mask_edges[2, :n_Ez],
        ]
    )

    g_3d = build_gradient_matrix(mesh.grid)
    g_2d, _, primal_2d_edges = build_2d_gradient(plane, mesh.grid, g_3d)
    n_2d_nodes = g_2d.shape[1]

    pec_2d_edge_mask = pec_E_flat[primal_2d_edges]
    if extra_pec_edge_mask is not None:
        if extra_pec_edge_mask.shape != pec_2d_edge_mask.shape:
            raise ValueError(
                f"extra_pec_edge_mask shape {extra_pec_edge_mask.shape} "
                f"does not match the plane's [e_u | e_v] basis "
                f"{pec_2d_edge_mask.shape}."
            )
        pec_2d_edge_mask = pec_2d_edge_mask | extra_pec_edge_mask

    pec_rows = g_2d[pec_2d_edge_mask].tocsr()

    indices = pec_rows.indices
    indptr = pec_rows.indptr
    n_pec_edges = pec_rows.shape[0]
    edges_a = np.empty(n_pec_edges, dtype=np.int64)
    edges_b = np.empty(n_pec_edges, dtype=np.int64)
    n_kept = 0
    for k in range(n_pec_edges):
        nz = indices[indptr[k] : indptr[k + 1]]
        if nz.size == 2:
            edges_a[n_kept] = nz[0]
            edges_b[n_kept] = nz[1]
            n_kept += 1
    edges_a = edges_a[:n_kept]
    edges_b = edges_b[:n_kept]

    comp_edges = _connected_node_groups(edges_a, edges_b, n_2d_nodes)
    comp_to_nodes = comp_edges

    # PEC continuity through cell bodies: the edge graph alone can
    # leave sub-cell fragments of a curved conductor's surface as
    # separate components (a handful of edges above the apex whose
    # connecting edges fall below the classifier threshold).  Such a
    # fragment forms a phantom conductor with a near-zero gap — its
    # phantom TEM channel has an enormous C', sorts FIRST in the
    # capacitance-ordered channel basis, and shadows the real mode at
    # small n_modes (measured z_line 0.95 Ω instead of ~50 Ω on a
    # curved-electrode stripline).  The PEC-cell corner links decide
    # ONLY which edge components belong to the same conductor (label
    # fusion) — the conductor node sets stay those of the edge graph,
    # so the Dirichlet sets of the mode solvers are unchanged wherever
    # the edge graph was already whole (a cell-corner node without a
    # PEC edge widens a staircased conductor and was measured to move
    # a coax z_line by −10 %).  Distinct conductors stay distinct
    # because the mesher's feature-gap floor keeps them at least one
    # non-PEC cell apart.  Only the under-resolved rescue (< 2 edge
    # components) books the full merged node sets — there the cell
    # graph is the only material source — and warns.
    cells_a, cells_b = _pec_cell_corner_links(plane, mesh)
    if cells_a.size:
        merged = _connected_node_groups(
            np.concatenate([edges_a, cells_a]),
            np.concatenate([edges_b, cells_b]),
            n_2d_nodes,
        )
        if len(comp_edges) >= 2:
            merged_label: dict[int, int] = {}
            for lbl, nodes in merged.items():
                for n in nodes:
                    merged_label[n] = lbl
            fused: dict[int, list[int]] = {}
            for nodes in comp_edges.values():
                fused.setdefault(merged_label[nodes[0]], []).extend(nodes)
            comp_to_nodes = {lbl: sorted(nodes) for lbl, nodes in fused.items()}
        else:
            comp_to_nodes = merged
            if len(merged) >= 2:
                warnings.warn(
                    f"port plane on {plane.face.value!r}: the PEC "
                    f"staircase-edge graph yields "
                    f"{len(comp_edges)} connected component(s); "
                    f"fell back to cell-material (is_pec) grouping "
                    f"({len(merged)} components).  This typically "
                    f"means a conductor is under-resolved by the grid "
                    f"on the port plane — consider refining the "
                    f"transversal mesh there.",
                    UserWarning,
                    stacklevel=2,
                )

    if len(comp_to_nodes) == 0:
        raise ValueError(
            f"port plane on {plane.face.value!r} has no PEC edges and "
            f"no PEC cells; cannot auto-derive conductor groups for a "
            f"multi-conductor port."
        )
    if len(comp_to_nodes) < 2:
        raise ValueError(
            f"port plane has {len(comp_to_nodes)} PEC-connected "
            f"component (including the cell-material fallback); need "
            f"at least 2 for a multi-conductor port (ground + ≥1 "
            f"signal)."
        )

    Nu_node = plane.n_nodes_u
    Nv_node = plane.n_nodes_v

    def _touches_bbox_wall(comp_nodes: list[int]) -> bool:
        for n in comp_nodes:
            iu, iv = divmod(n, Nv_node)
            if iu == 0 or iu == Nu_node - 1 or iv == 0 or iv == Nv_node - 1:
                return True
        return False

    bbox_touching = [c for c, nodes in comp_to_nodes.items() if _touches_bbox_wall(nodes)]
    if bbox_touching:
        ground_label = max(
            bbox_touching,
            key=lambda c: len(comp_to_nodes[c]),
        )
    else:
        ground_label = max(
            comp_to_nodes.keys(),
            key=lambda c: len(comp_to_nodes[c]),
        )

    other_labels = sorted(
        (c for c in comp_to_nodes if c != ground_label),
        key=lambda c: -len(comp_to_nodes[c]),
    )

    return [
        np.asarray(comp_to_nodes[ground_label], dtype=np.int64),
    ] + [np.asarray(comp_to_nodes[c], dtype=np.int64) for c in other_labels]


def _connected_node_groups(
    links_a: np.ndarray,
    links_b: np.ndarray,
    n_2d_nodes: int,
) -> dict[int, list[int]]:
    """Group linked 2D nodes into connected components.

    ``links_a[i]``–``links_b[i]`` are undirected node–node links (from
    PEC edges and/or PEC-cell corners).  Returns a dict mapping the
    component label to the sorted list of member node indices; nodes
    not touched by any link do not appear.
    """
    if links_a.size == 0:
        return {}

    rows = np.concatenate([links_a, links_b])
    cols = np.concatenate([links_b, links_a])
    data = np.ones(rows.size, dtype=np.int8)
    adj = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_2d_nodes, n_2d_nodes),
    )

    _, labels = connected_components(adj, directed=False)

    linked_nodes = np.unique(rows)
    comp_to_nodes: dict[int, list[int]] = {}
    for node in linked_nodes:
        c = int(labels[node])
        comp_to_nodes.setdefault(c, []).append(int(node))
    return comp_to_nodes


def _pec_cell_corner_links(
    plane: PortPlane,
    mesh: Mesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Node–node links from PEC *cells* in the port-adjacent cell slab.

    Second material source alongside the PEC edge mask (finding F2):
    on the conformal path the sub-cell classifier's η threshold can
    admit the edges of a thin conductor, while the cell centres still
    classify as PEC in ``material_id``.  Each PEC cell links its four
    corner nodes in the local ``i_u * Nv_node + i_v`` raster (three
    star links from the (u, v) corner reach all four).

    Returns
    -------
    (links_a, links_b) : tuple of np.ndarray
        Undirected link endpoint arrays; both empty when the slab has
        no PEC cells.
    """
    face = plane.face
    n_axis = face.normal_axis
    n_cells = (mesh.Nx, mesh.Ny, mesh.Nz)
    slab_idx = n_cells[n_axis] - 1 if face.is_max else 0
    slab = np.take(mesh.material_id, slab_idx, axis=n_axis)
    # np.take keeps the remaining axes in ascending global-axis order;
    # transpose when the face's local axes are ordered the other way.
    if face.u_axis > face.v_axis:
        slab = slab.T

    # Clip the slab to the plane's cell windows (whole face: no-op) so
    # sub-face planes only link corner nodes inside their own raster.
    u_lo, u_hi = plane.u_node_window
    v_lo, v_hi = plane.v_node_window
    slab = slab[u_lo:u_hi, v_lo:v_hi]

    pec_ids = [mid for mid, mat in mesh.material_library.items() if mat.is_pec]
    if not pec_ids:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    pec_cells = np.isin(slab, pec_ids)

    cu, cv = np.nonzero(pec_cells)
    if cu.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    Nv_node = plane.n_nodes_v
    n00 = cu * Nv_node + cv
    n10 = (cu + 1) * Nv_node + cv
    n01 = cu * Nv_node + (cv + 1)
    n11 = (cu + 1) * Nv_node + (cv + 1)
    links_a = np.concatenate([n00, n00, n00]).astype(np.int64)
    links_b = np.concatenate([n10, n01, n11]).astype(np.int64)
    return links_a, links_b
