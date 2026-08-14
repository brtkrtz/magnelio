"""Holland/Simpson thin-wire sub-cell model (DD-080).

A :class:`~magnelio.geo.wire.ThinWire` keeps a PEC constraint on its
rasterised E-edge chain and corrects the material matrices of the
encircling cells so the wire presents the physical per-length inductance
``L' = (mu/2pi)·ln(delta/a)`` instead of the bare-grid value: a masked
edge alone behaves like a conductor of equivalent radius
``r0 = KAPPA0·delta`` (square-lattice Green's function,
``KAPPA0 = e^(-gamma)/2^(3/2) ~ 0.1985``).

The correction is the paired Noda–Yokoyama (2002) encoding of Holland &
Simpson's (1981) ``L'·C' = eps·mu`` closure: per axis-aligned segment the
4 encircling H-faces scale ``M_mu`` by ``m = ln(delta_f/a)/ln(delta_f/r0)``
and the co-located radial E-edges scale ``M_eps`` by ``1/m``.  The pair
product ``M_eps·M_mu`` is untouched, so the DD-053 pair identity — and
with it the exact discrete travelling wave — survives on the wire line.
A mu-only correction would leave the in-cell capacitance at the bare-grid
value and the wire wave speed at ``c·sqrt(ln(delta/r0)/ln(delta/a))``
(~0.73c at ``a = 0.05·delta``).

Everything happens at mesh-build time through the existing sub-cell
channels (``FaceMaterialData`` category 2 / ``EdgeMaterialData``
category 1, the ``couple_face_material_pairs`` conventions); the solver
is untouched.  Composition is conservative: a face or edge requested by
several segments (staircase corners, parallel wires closer than two
cells) takes the *smallest* m — erring toward the bare-grid baseline,
never double-booking inductance.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from magnelio.mesh.indexing import (
    edge_index_Ex,
    edge_index_Ey,
    edge_index_Ez,
    face_index_Hx,
    face_index_Hy,
    face_index_Hz,
)

# Equivalent radius of a bare masked grid edge, r0 = KAPPA0 * delta:
# the square-lattice Laplace Green's function gives
# KAPPA0 = e^(-gamma) / 2^(3/2) (gamma = Euler-Mascheroni).  Yee-grid
# measurements in the literature scatter around this value (0.135-0.23);
# the T3 gate measures magnelio's own r0 against a closed form.
KAPPA0 = math.exp(-0.5772156649015329) / (2.0 * math.sqrt(2.0))

# Radius validity bounds relative to the smallest transverse cell along
# the path.  Above 0.30 the log model degrades quickly (and m drops
# toward the build_M_mu A_face_free floor); between 0.20 and 0.30 the
# wire is fatter than the bare-grid equivalent radius (m < 1) and the
# CFL dt shrinks by up to sqrt(0.744) ~ 0.86.
_RADIUS_MAX_FRAC = 0.30
_RADIUS_WARN_FRAC = 0.20

# A face whose own transverse extent is not clearly above r0 cannot
# carry a meaningful log correction (pathological grading); it keeps the
# bare-grid value with a warning.
_MIN_LOG_DEN = 0.1

# The log correction is derived for locally isotropic transverse cells.
# Measured on a square duct: at a 2.5:1 transverse aspect ratio at the
# wire the corrected Z0 is ~11% low (the four-face lift under-shoots) —
# warn above this ratio.  Both-axes grading (locally isotropic) is fine.
_ANISO_WARN_RATIO = 1.5


def _ring_stencil(axis, ijk, grid):
    """The <=4 encircling H-faces of one wire segment, with partners.

    Returns a list of ``(face_flat, delta_t, (edge_flat, edge_flat))``:
    the flat H index (``Hx|Hy|Hz`` layout), the face's transverse extent
    measured from the wire, and the two co-located radial E-edges (flat
    ``Ex|Ey|Ez`` layout) — exactly the face's DD-053 ladder partners
    along the wire axis.  Faces outside the domain are omitted.
    """
    i, j, k = ijk
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    dx, dy, dz = grid.dx, grid.dy, grid.dz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Hx = (Nx + 1) * Ny * Nz
    n_Hy = Nx * (Ny + 1) * Nz

    out = []
    if axis == "z":
        # Radial +/-y: Hx faces; partners are the radial Ey edges at k, k+1.
        for jf in (j, j - 1):
            if 0 <= jf <= Ny - 1:
                out.append(
                    (
                        face_index_Hx(i, jf, k, Nx, Ny, Nz),
                        float(dy[jf]),
                        (
                            n_Ex + edge_index_Ey(i, jf, k, Nx, Ny, Nz),
                            n_Ex + edge_index_Ey(i, jf, k + 1, Nx, Ny, Nz),
                        ),
                    )
                )
        # Radial +/-x: Hy faces; partners are the radial Ex edges at k, k+1.
        for i_f in (i, i - 1):
            if 0 <= i_f <= Nx - 1:
                out.append(
                    (
                        n_Hx + face_index_Hy(i_f, j, k, Nx, Ny, Nz),
                        float(dx[i_f]),
                        (
                            edge_index_Ex(i_f, j, k, Nx, Ny, Nz),
                            edge_index_Ex(i_f, j, k + 1, Nx, Ny, Nz),
                        ),
                    )
                )
    elif axis == "x":
        # Radial +/-y: Hz faces; partners are the radial Ey edges at i, i+1.
        for jf in (j, j - 1):
            if 0 <= jf <= Ny - 1:
                out.append(
                    (
                        n_Hx + n_Hy + face_index_Hz(i, jf, k, Nx, Ny, Nz),
                        float(dy[jf]),
                        (
                            n_Ex + edge_index_Ey(i, jf, k, Nx, Ny, Nz),
                            n_Ex + edge_index_Ey(i + 1, jf, k, Nx, Ny, Nz),
                        ),
                    )
                )
        # Radial +/-z: Hy faces; partners are the radial Ez edges at i, i+1.
        for kf in (k, k - 1):
            if 0 <= kf <= Nz - 1:
                out.append(
                    (
                        n_Hx + face_index_Hy(i, j, kf, Nx, Ny, Nz),
                        float(dz[kf]),
                        (
                            n_Ex + n_Ey + edge_index_Ez(i, j, kf, Nx, Ny, Nz),
                            n_Ex + n_Ey + edge_index_Ez(i + 1, j, kf, Nx, Ny, Nz),
                        ),
                    )
                )
    elif axis == "y":
        # Radial +/-x: Hz faces; partners are the radial Ex edges at j, j+1.
        for i_f in (i, i - 1):
            if 0 <= i_f <= Nx - 1:
                out.append(
                    (
                        n_Hx + n_Hy + face_index_Hz(i_f, j, k, Nx, Ny, Nz),
                        float(dx[i_f]),
                        (
                            edge_index_Ex(i_f, j, k, Nx, Ny, Nz),
                            edge_index_Ex(i_f, j + 1, k, Nx, Ny, Nz),
                        ),
                    )
                )
        # Radial +/-z: Hx faces; partners are the radial Ez edges at j, j+1.
        for kf in (k, k - 1):
            if 0 <= kf <= Nz - 1:
                out.append(
                    (
                        face_index_Hx(i, j, kf, Nx, Ny, Nz),
                        float(dz[kf]),
                        (
                            n_Ex + n_Ey + edge_index_Ez(i, j, kf, Nx, Ny, Nz),
                            n_Ex + n_Ey + edge_index_Ez(i, j + 1, kf, Nx, Ny, Nz),
                        ),
                    )
                )
    else:
        raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
    return out


def _min_transverse_extent(path, grid) -> float:
    """Smallest transverse cell extent over all segments of the path."""
    d_min = math.inf
    for axis, ijk in zip(path.axes, path.ijk):
        for _, delta_t, _ in _ring_stencil(axis, ijk, grid):
            d_min = min(d_min, delta_t)
    return d_min


def _validate_radius(radius, d_min, name) -> None:
    label = f"ThinWire {name!r}" if name else "ThinWire"
    if radius >= _RADIUS_MAX_FRAC * d_min:
        raise ValueError(
            f"{label}: radius {radius:.3e} m >= {_RADIUS_MAX_FRAC}x the "
            f"smallest transverse cell along the path ({d_min:.3e} m). "
            f"The thin-wire model needs a << cell size — refine the mesh "
            f"or reduce the radius (a resolved cylinder is the alternative)."
        )
    if radius > _RADIUS_WARN_FRAC * d_min:
        warnings.warn(
            f"{label}: radius {radius:.3e} m is fatter than the bare-grid "
            f"equivalent radius (~{KAPPA0:.3f}x cell); the correction factor "
            f"drops below 1 and the CFL time step shrinks by up to "
            f"~{math.sqrt(math.log(1 / _RADIUS_MAX_FRAC) / math.log(1 / KAPPA0)):.2f}x.",
            UserWarning,
            stacklevel=3,
        )


def _collect_requests(path, radius, grid, name=None):
    """Per-face / per-edge correction factors requested by one wire path.

    Returns ``(face_m, edge_m)`` dicts mapping flat H-face / E-edge
    indices to the requested factor m.  Multiple requests onto the same
    object (corners, revisited cells) already collapse to the minimum
    here; cross-wire collapsing happens in the caller.
    """
    label = f"ThinWire {name!r}" if name else "ThinWire"
    face_m: dict[int, float] = {}
    edge_m: dict[int, float] = {}
    n_weak = 0
    max_aniso = 1.0
    for axis, ijk in zip(path.axes, path.ijk):
        ring = _ring_stencil(axis, ijk, grid)
        if not ring:
            continue
        ds = [d for _, d, _ in ring]
        max_aniso = max(max_aniso, max(ds) / min(ds))
        # r0 from the geometric mean of the available transverse extents.
        d_bar = math.exp(sum(math.log(d) for _, d, _ in ring) / len(ring))
        r0 = KAPPA0 * d_bar
        for face, delta_t, edges in ring:
            den = math.log(delta_t / r0)
            if den <= _MIN_LOG_DEN:
                # Pathological grading: this face's own extent is not
                # clearly above the bare-grid equivalent radius — no
                # meaningful log correction exists; keep the bare value.
                n_weak += 1
                continue
            m = math.log(delta_t / radius) / den
            if face not in face_m or m < face_m[face]:
                face_m[face] = m
            for e in edges:
                if e not in edge_m or m < edge_m[e]:
                    edge_m[e] = m
    if n_weak:
        warnings.warn(
            f"{label}: {n_weak} encircling face(s) have a transverse "
            f"extent within ~{_MIN_LOG_DEN} log units of the bare-grid "
            f"equivalent radius (extreme grading) and keep the "
            f"uncorrected value.",
            UserWarning,
            stacklevel=3,
        )
    if max_aniso > _ANISO_WARN_RATIO:
        warnings.warn(
            f"{label}: the transverse cells at the wire are anisotropic "
            f"(aspect ratio up to {max_aniso:.2f}).  The thin-wire "
            f"correction is derived for locally isotropic transverse "
            f"cells; the wire impedance degrades with anisotropy "
            f"(~10% low at 2.5:1).  Grade both transverse axes alike "
            f"around the wire.",
            UserWarning,
            stacklevel=3,
        )
    return face_m, edge_m


def _ensure_face_material(mesh):
    from magnelio.geo._subcell import FaceMaterialData  # noqa: PLC0415

    if mesh.face_material is None:
        grid = mesh.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        n_H = (Nx + 1) * Ny * Nz + Nx * (Ny + 1) * Nz + Nx * Ny * (Nz + 1)
        mesh.face_material = FaceMaterialData(
            category=np.zeros(n_H, dtype=np.int8),
            mu_avg=np.full(n_H, np.nan),
            A_face_free=np.full(n_H, np.nan),
            L_dual_free=np.full(n_H, np.nan),
        )
    return mesh.face_material


def _ensure_edge_material(mesh):
    from magnelio.geo._subcell import EdgeMaterialData  # noqa: PLC0415

    if mesh.edge_material is None:
        grid = mesh.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        n_E = Nx * (Ny + 1) * (Nz + 1) + (Nx + 1) * Ny * (Nz + 1) + (Nx + 1) * (Ny + 1) * Nz
        mesh.edge_material = EdgeMaterialData(
            category=np.zeros(n_E, dtype=np.int8),
            eps_avg=np.full(n_E, np.nan),
            sigma_avg=np.full(n_E, np.nan),
            A_free=np.full(n_E, np.nan),
            L_free=np.full(n_E, np.nan),
            f_A=np.full(n_E, np.nan),
            pec_mask=mesh.pec_mask_edges,
            enlarged_cell_donor=np.full(n_E, -1, dtype=np.int64),
            enlarged_cell_area=np.zeros(n_E),
        )
    return mesh.edge_material


def _write_corrections(mesh, face_m, edge_m, name=None) -> None:
    """Encode collapsed (face -> m) / (edge -> m) into the mesh channels.

    Faces become category 2 with the equivalent ``A_face_free`` carrying
    ``m x`` the face's *current* M_mu (so a dielectric cat-1 value
    composes multiplicatively); edges become category 1 with
    ``eps_avg = eps_eff/m`` of their current effective permittivity.
    Objects already claimed by a conformal solid (cat-2 faces,
    cat-2/cat-3/PEC-masked edges) are skipped — the solid wins.
    """
    from magnelio._operators.material_matrices import (  # noqa: PLC0415
        EPS0,
        MU0,
        _build_geom_E,
        _build_L_dual_H,
        _build_L_primal_E,
        _staircase_mu_faces,
        build_M_eps,
        build_M_mu,
    )

    label = f"ThinWire {name!r}" if name else "ThinWire"
    grid = mesh.grid
    fm = _ensure_face_material(mesh)
    em = _ensure_edge_material(mesh)

    m_mu = build_M_mu(mesh)
    m_eps = build_M_eps(mesh)
    L_dual = _build_L_dual_H(grid)
    geom_E = _build_geom_E(grid)
    L_primal = _build_L_primal_E(grid)

    mu_face = _staircase_mu_faces(mesh)
    has_mu = ~np.isnan(fm.mu_avg)
    mu_face[has_mu] = fm.mu_avg[has_mu]

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    n_Ez = (Nx + 1) * (Ny + 1) * Nz
    pec = mesh.pec_mask_edges
    pec_flat = np.concatenate(
        [
            pec[0, :n_Ex],
            pec[1, :n_Ey],
            pec[2, :n_Ez],
        ]
    )

    n_face_skip = 0
    for face, m in face_m.items():
        if fm.category[face] == 2:
            n_face_skip += 1  # conformal solid / DD-053 pair value wins
            continue
        fm.category[face] = 2
        fm.mu_avg[face] = mu_face[face]
        fm.A_face_free[face] = m * m_mu[face] * L_dual[face] / (MU0 * mu_face[face])
        fm.L_dual_free[face] = L_dual[face]

    n_edge_skip = 0
    for edge, m in edge_m.items():
        if pec_flat[edge] or em.category[edge] == 3:
            # Masked edge (e.g. the radial edges where the wire lands
            # on a PEC solid — the monopole connection): its M_eps is
            # never read, silently skip.
            continue
        if em.category[edge] == 2:
            n_edge_skip += 1  # conformal-solid edge — solid wins
            continue
        eps_eff = m_eps[edge] / (EPS0 * geom_E[edge])
        if em.category[edge] == 0:
            em.category[edge] = 1
            em.A_free[edge] = geom_E[edge] * L_primal[edge]  # = A_dual
            em.L_free[edge] = L_primal[edge]
            em.f_A[edge] = 1.0
            # sigma_avg stays NaN: build_M_sigma keeps the staircase value.
        em.eps_avg[edge] = eps_eff / m

    if n_face_skip or n_edge_skip:
        warnings.warn(
            f"{label}: {n_face_skip} face(s) / {n_edge_skip} edge(s) of the "
            f"correction stencil are already claimed by a conformal solid "
            f"(or PEC-masked) and keep the solid's value.",
            UserWarning,
            stacklevel=3,
        )


def apply_thin_wire_path(mesh, path, radius, *, name=None) -> None:
    """Apply the DD-080 correction for one already-rasterised wire path.

    The OCC-free core (unit tests build the ``EdgePath`` by hand): the
    path's edges must already be PEC-masked (:func:`mask_thin_wires`).
    For several wires use :func:`correct_thin_wire_materials`, which
    collapses shared faces across wires before writing.
    """
    d_min = _min_transverse_extent(path, grid=mesh.grid)
    _validate_radius(radius, d_min, name)
    face_m, edge_m = _collect_requests(path, radius, mesh.grid, name=name)
    _write_corrections(mesh, face_m, edge_m, name=name)


def mask_thin_wires(mesh, wires, *, samples_per_cell: int = 4, scale: float = 1.0):
    """Rasterise *wires* and mask their E-edge chains PEC (in place).

    Runs before ``couple_face_material_pairs`` so the DD-053 pass never
    certifies a ladder through a wire edge.  Returns the rasterised
    ``EdgePath`` per wire (consumed by
    :func:`correct_thin_wire_materials` after that pass).
    """
    from magnelio.circuit.rasterize import rasterize_curve  # noqa: PLC0415
    from magnelio.geo._occ_backend import sample_wire  # noqa: PLC0415

    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    min_cell = min(grid.dx_min, grid.dy_min, grid.dz_min)

    paths = []
    for wire in wires:
        label = f"ThinWire {wire.name!r}" if wire.name else "ThinWire"
        path = rasterize_curve(wire.curve, grid, samples_per_cell=samples_per_cell, scale=scale)
        _validate_radius(
            radius=wire.radius, d_min=_min_transverse_extent(path, grid), name=wire.name
        )

        # Snap-displacement check on the curve's ENDPOINTS: those are
        # the electrically meaningful spots (PEC / gap / PMC-wall
        # connections).  Interior samples of a deliberately oblique
        # curve sit between nodes by construction (the staircase
        # carries them), so they are not checked.
        pts = sample_wire(wire.curve._occ_shape(scale), min_cell / samples_per_cell, scale=scale)
        off = 0.0
        for p in (pts[0], pts[-1]):
            off = max(
                off,
                float(np.min(np.abs(grid.x - p[0]))),
                float(np.min(np.abs(grid.y - p[1]))),
                float(np.min(np.abs(grid.z - p[2]))),
            )
        if off > 0.3 * min_cell:
            warnings.warn(
                f"{label}: an endpoint lies {off:.3e} m off its snapped "
                f"grid node (> 0.3x the smallest cell) — the wire does "
                f"not end where the curve does.  Align the endpoint with "
                f"grid lines (polyline vertices become anchor planes "
                f"automatically) or refine the mesh.",
                UserWarning,
                stacklevel=2,
            )

        # OR the path's edges into the PEC mask (per-component layout).
        for axis, flat in zip(path.axes, path.flat_indices):
            if axis == "x":
                mesh.pec_mask_edges[0, flat] = True
            elif axis == "y":
                mesh.pec_mask_edges[1, flat - n_Ex] = True
            else:
                mesh.pec_mask_edges[2, flat - n_Ex - n_Ey] = True
        em = mesh.edge_material
        if em is not None and em.pec_mask is not mesh.pec_mask_edges:
            em.pec_mask |= mesh.pec_mask_edges
        paths.append(path)
    return paths


def correct_thin_wire_materials(mesh, wires, paths) -> None:
    """Apply the DD-080 (m, 1/m) correction for all *wires* at once.

    Requests from all wires are collapsed first (minimum m per face /
    edge — the conservative composition rule), then written once, so
    parallel wires sharing an encircling face never double-book
    inductance.
    """
    face_m: dict[int, float] = {}
    edge_m: dict[int, float] = {}
    face_owner: dict[int, int] = {}
    n_shared = 0
    for idx, (wire, path) in enumerate(zip(wires, paths)):
        f_m, e_m = _collect_requests(path, wire.radius, mesh.grid, name=wire.name)
        for face, m in f_m.items():
            if face in face_m:
                if face_owner[face] != idx:
                    n_shared += 1
                face_m[face] = min(face_m[face], m)
            else:
                face_m[face] = m
                face_owner[face] = idx
        for edge, m in e_m.items():
            edge_m[edge] = min(m, edge_m.get(edge, math.inf))
    if n_shared:
        warnings.warn(
            f"{n_shared} encircling face(s) are shared between different "
            f"thin wires (closer than two cells); the correction degrades "
            f"to the conservative minimum there.",
            UserWarning,
            stacklevel=2,
        )
    name = None
    if len(wires) == 1:
        name = wires[0].name
    _write_corrections(mesh, face_m, edge_m, name=name)
