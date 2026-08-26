"""Continue the sub-cell classification into the PML extension slabs.

The mesher grows an absorbing (CPML) face outward by ``n_pml`` uniform
cells and continues the cell materials into them (``from_geometry``
step 3b), so a body touching that face — a waveguide neck ending on the
wall, a ground plane running into the absorber — keeps its cross-section
through the absorbing layer.  The conformal sub-cell classifier,
however, works against the B-rep solids, and those end at the nominal
bounding box: every edge and face inside the extension is "entirely
outside" all solids, the classifier reports free space there, and the
PEC mask of a conductor's surface falls away in exactly the slabs the
absorber occupies (DD-198, KB-028).

This module mirrors step 3b for the sub-cell data: the extension slabs
receive a copy of the first interior slab that the solids fully
describe — the same translation-invariant continuation the material
ids already have.  Quantities sampled on nodes along the face normal
(edges and faces lying *in* the wall plane) copy the plane one cell
inside the interface, because the interface plane's own dual cell
straddles the extension; quantities sampled on cells copy the first
interior cell.  Enlarged-cell donor links are re-pointed within the
copied slab and dropped where they crossed slabs.
"""

from __future__ import annotations

import numpy as np

_AXIS = {"x": 0, "y": 1, "z": 2}


def _edge_shapes(Nx: int, Ny: int, Nz: int) -> tuple[tuple[int, int, int], ...]:
    return ((Nx, Ny + 1, Nz + 1), (Nx + 1, Ny, Nz + 1), (Nx + 1, Ny + 1, Nz))


def _face_shapes(Nx: int, Ny: int, Nz: int) -> tuple[tuple[int, int, int], ...]:
    return ((Nx + 1, Ny, Nz), (Nx, Ny + 1, Nz), (Nx, Ny, Nz + 1))


def _slab_pairs(n_along: int, n_pml: int, side: str, node_sampled: bool):
    """(source index, [destination indices]) along the face normal."""
    if side == "min":
        src = n_pml + 1 if node_sampled else n_pml
        dsts = list(range(n_pml + 1 if node_sampled else n_pml))
    else:
        n_cells = n_along - 1 if node_sampled else n_along
        src = n_cells - n_pml - 1
        dsts = list(range(n_cells - n_pml, n_cells + 1 if node_sampled else n_cells))
    return src, dsts


def _copy_slabs(block: np.ndarray, ax: int, src: int, dsts: list[int]) -> None:
    index = [slice(None)] * 3
    index[ax] = src
    source = block[tuple(index)].copy()
    for dst in dsts:
        index[ax] = dst
        block[tuple(index)] = source


def _decode(flat: np.ndarray, shapes, offsets):
    """Flat component-concatenated index → (component, i, j, k) arrays."""
    comp = np.full(flat.shape, -1, dtype=np.int64)
    ijk = np.zeros(flat.shape + (3,), dtype=np.int64)
    for c, (shape, off) in enumerate(zip(shapes, offsets)):
        size = shape[0] * shape[1] * shape[2]
        sel = (flat >= off) & (flat < off + size)
        if sel.any():
            comp[sel] = c
            ijk[sel] = np.stack(np.unravel_index(flat[sel] - off, shape), axis=-1)
    return comp, ijk


def _encode(comp: np.ndarray, ijk: np.ndarray, shapes, offsets) -> np.ndarray:
    out = np.full(comp.shape, -1, dtype=np.int64)
    for c, (shape, off) in enumerate(zip(shapes, offsets)):
        sel = comp == c
        if sel.any():
            out[sel] = off + np.ravel_multi_index((ijk[sel, 0], ijk[sel, 1], ijk[sel, 2]), shape)
    return out


def _extend_component_arrays(arrays, shapes, offsets, ax, n_pml, side, node_sampled_of):
    """Copy the source slab into the extension slabs for every array.

    *arrays* are flat component-concatenated 1D arrays (or 2D with the
    flat axis last); *node_sampled_of(c)* says whether component *c* is
    node-sampled along axis *ax*.
    """
    for arr in arrays:
        if arr is None:
            continue
        for c, (shape, off) in enumerate(zip(shapes, offsets)):
            size = shape[0] * shape[1] * shape[2]
            src, dsts = _slab_pairs(shape[ax], n_pml, side, node_sampled_of(c))
            if arr.ndim == 1:
                block = arr[off : off + size].reshape(shape)
                _copy_slabs(block, ax, src, dsts)
            else:
                for row in range(arr.shape[0]):
                    block = arr[row, off : off + size].reshape(shape)
                    _copy_slabs(block, ax, src, dsts)


def _remap_donors(donor, area, shapes, offsets, ax, n_pml, side, node_sampled_of):
    """Re-point copied donor links into their own slab, drop cross-slab ones."""
    if donor is None:
        return
    comp, ijk = _decode(donor, shapes, offsets)
    for c, shape in enumerate(shapes):
        src, dsts = _slab_pairs(shape[ax], n_pml, side, node_sampled_of(c))
        size = shape[0] * shape[1] * shape[2]
        off = offsets[c]
        owner_block = np.arange(off, off + size).reshape(shape)
        index = [slice(None)] * 3
        for dst in dsts:
            index[ax] = dst
            owners = owner_block[tuple(index)].ravel()
            d = donor[owners]
            linked = d >= 0
            if not linked.any():
                continue
            dc, dijk = comp[owners][linked], ijk[owners][linked]
            # A donor in the source slab of ITS component moves to the
            # matching destination; anything else crossed slabs.
            src_of_donor = np.array(
                [_slab_pairs(shapes[k][ax], n_pml, side, node_sampled_of(k))[0] for k in dc]
            )
            same = dijk[:, ax] == src_of_donor
            dijk = dijk.copy()
            dijk[same, ax] = dst
            new = _encode(dc, dijk, shapes, offsets)
            new[~same] = -1
            owners_linked = owners[linked]
            donor[owners_linked] = new
            if area is not None:
                area[owners_linked[~same]] = 0.0


def extend_subcell_data_into_pml(pec_mask, edge_material, face_material, grid, pml_cells):
    """Mirror step 3b for the conformal sub-cell data (in place).

    Parameters
    ----------
    pec_mask : np.ndarray, shape (3, n_max)
        The E-edge PEC mask in the ``Mesh.pec_mask_edges`` layout.
    edge_material : EdgeMaterialData or None
    face_material : FaceMaterialData or None
    grid : GridLines
    pml_cells : dict
        ``{face: n_pml}`` as recorded by the mesher.
    """
    if not pml_cells:
        return
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    e_shapes = _edge_shapes(Nx, Ny, Nz)
    h_shapes = _face_shapes(Nx, Ny, Nz)
    e_sizes = [s[0] * s[1] * s[2] for s in e_shapes]
    h_sizes = [s[0] * s[1] * s[2] for s in h_shapes]
    e_offsets = [0, e_sizes[0], e_sizes[0] + e_sizes[1]]
    h_offsets = [0, h_sizes[0], h_sizes[0] + h_sizes[1]]

    for face, n_pml in pml_cells.items():
        ax, side = _AXIS[face[0]], face[1:]
        if n_pml <= 0:
            continue

        def edge_node_sampled(c, ax=ax):
            return c != ax  # an edge is cell-sampled along its own axis only

        def face_node_sampled(c, ax=ax):
            return c == ax  # a face is node-sampled along its own normal only

        # PEC mask: one row per component, padded to n_max.
        for c, (shape, size) in enumerate(zip(e_shapes, e_sizes)):
            block = pec_mask[c, :size].reshape(shape)
            src, dsts = _slab_pairs(shape[ax], n_pml, side, edge_node_sampled(c))
            _copy_slabs(block, ax, src, dsts)

        if edge_material is not None:
            fields = [
                edge_material.category,
                edge_material.eps_avg,
                edge_material.sigma_avg,
                edge_material.A_free,
                edge_material.L_free,
                edge_material.f_A,
                edge_material.enlarged_cell_donor,
                edge_material.enlarged_cell_area,
                edge_material.material_fractions,
            ]
            _extend_component_arrays(
                fields, e_shapes, e_offsets, ax, n_pml, side, edge_node_sampled
            )
            _remap_donors(
                edge_material.enlarged_cell_donor,
                edge_material.enlarged_cell_area,
                e_shapes,
                e_offsets,
                ax,
                n_pml,
                side,
                edge_node_sampled,
            )

        if face_material is not None:
            fields = [
                face_material.category,
                face_material.mu_avg,
                face_material.A_face_free,
                face_material.L_dual_free,
                face_material.enlarged_cell_donor,
                face_material.enlarged_cell_area,
                face_material.A_face_pec,
                face_material.A_face_pec_jump,
                face_material.material_fractions,
                face_material.sigma_m_avg,
            ]
            _extend_component_arrays(
                fields, h_shapes, h_offsets, ax, n_pml, side, face_node_sampled
            )
            _remap_donors(
                face_material.enlarged_cell_donor,
                face_material.enlarged_cell_area,
                h_shapes,
                h_offsets,
                ax,
                n_pml,
                side,
                face_node_sampled,
            )
