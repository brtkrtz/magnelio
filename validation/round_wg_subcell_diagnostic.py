"""Diagnose: where does the Round-WG conformal-branch O(1) defect sit?

Variante B (DD-051 partial, session 62) refactored the data flow but
kept the two original OCC pipelines (cross-section / thin-box for ε̄
and line-solid for f_L).  The Round-WG TE11 cut-off still converges
non-monotonically (rel.err 8.59 % → 3.83 % → 4.36 % → 2.84 % across
n_t ∈ {17, 25, 33, 49}), with p_obs ≪ 1.8 between adjacent levels.

Three candidate root causes:

(a) **Tessellation drift** between the two OCC algorithms — line-solid
    (BRepIntCurveSurface_Inter, no explicit deflection) vs cross-
    section (BRepAlgoAPI_Section, deflection ~ h*1e-2 = 1e-5 .. 1e-7 m
    on the round-WG mesh).
(b) **Tessellation universally too coarse** — even at the script's
    own deflection=1e-4 m default, a 10 mm radius cylinder is
    tessellated with O(100) chords; the resulting per-edge tangential
    polygon has tangential straightness error of ~50 µm, comparable
    to the cell width and to f_L itself.
(c) **Enlarged-cell threshold discontinuity** — cat-2/cat-3 split at
    f_L = eta = 0.4 is mesh-dependent and refinement-discrete; an
    edge that flips from cat 3 to cat 2 between n_t=25 and n_t=33
    contributes a step change in the operator spectrum.

Method.  Build the round-WG mesh at n_t=17 (the worst-case in the
convergence study).  For each cat-2 edge near the cylinder mantle:

1. Read the per-edge data already produced by Variante B
   (eps_avg, L_free, A_dual, f_L = L_free / L_primal).
2. Re-compute "ground-truth" reference values with a 1000× finer
   tessellation (deflection=1e-9 m) by calling the same OCC functions
   in isolation.
3. Aggregate Δf_L, Δeps_avg, Δ(A_PEC/A_dual) and report the
   distribution.

Reading.  If max |Δf_L| ≫ 1 % or max |Δeps_avg| ≫ 1 %, candidate (a)
or (b) is confirmed and the fix path is to consolidate the OCC
tessellation.  If the deltas are all below ~0.1 % yet the
convergence still fails, candidate (c) is the dominant defect and
the fix path is a continuous (no enlarged-cell threshold) cat-2
formulation.

Usage::

    conda run --no-capture-output -n occ python validation/round_wg_subcell_diagnostic.py
"""

from __future__ import annotations

import numpy as np

from magnelio import Material, Mesh, MeshControl
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.geo._filling import compute_conformal_eps
from magnelio.geo._occ_backend import (
    build_effective_pec_solid,
    compute_edge_pec_fractions,
)
from magnelio.mesh.grid import GridLines

# ---------------------------------------------------------------------------
# Round-WG geometry — identical to round_wg_conformal_convergence.py n_t=17
# ---------------------------------------------------------------------------

R = 10.0e-3
D = 2.0 * R
S_BBOX = 1.2 * D
L_X = 60.0e-3
F_MAX = 14.0e9
N_T = 17


def _geometry() -> GeometryModel:
    pec = Material.pec()
    vacuum = Material.air()
    bbox = Brick(
        origin=(0.0, -S_BBOX / 2, -S_BBOX / 2),
        size=(L_X, S_BBOX, S_BBOX),
        material=pec,
    )
    inner = Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=R,
        height=L_X,
        axis="x",
        material=vacuum,
    )
    model = GeometryModel()
    model.add(Difference(bbox, inner))
    model.add(inner)
    return model


def _shapes_with_material():
    """Shapes list mirroring the GeometryModel for direct OCC re-calls."""
    pec = Material.pec()
    vacuum = Material.air()
    bbox = Brick(
        origin=(0.0, -S_BBOX / 2, -S_BBOX / 2),
        size=(L_X, S_BBOX, S_BBOX),
        material=pec,
    )
    inner = Cylinder(
        origin=(0.0, 0.0, 0.0),
        radius=R,
        height=L_X,
        axis="x",
        material=vacuum,
    )
    pec_jacket = Difference(bbox, inner)
    return [(pec_jacket, 1), (inner, 0)], {0: vacuum, 1: pec}


def build_mesh(n_t: int = N_T) -> Mesh:
    """Build the round-WG mesh at the given transversal resolution."""
    model = _geometry()
    y_nodes = np.linspace(-S_BBOX / 2, S_BBOX / 2, n_t).tolist()
    z_nodes = np.linspace(-S_BBOX / 2, S_BBOX / 2, n_t).tolist()
    x_nodes = np.linspace(0.0, L_X, 31).tolist()
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=0,
        growth_factor=1.5,
        max_cell_size=4.0 * S_BBOX / max(n_t - 1, 1),
        conformal=True,
        forced_planes={"x": x_nodes, "y": y_nodes, "z": z_nodes},
    )
    return Mesh.from_geometry(model, control, f_max=F_MAX)


# ---------------------------------------------------------------------------
# Edge index → (component, i, j, k) decoding
# ---------------------------------------------------------------------------


def decode_edge(flat: int, Nx: int, Ny: int, Nz: int) -> tuple[str, int, int, int]:
    n_Ex = Nx * (Ny + 1) * (Nz + 1)
    n_Ey = (Nx + 1) * Ny * (Nz + 1)
    if flat < n_Ex:
        rem = flat
        sj = Nz + 1
        si = (Ny + 1) * sj
        i = rem // si
        rem %= si
        j = rem // sj
        k = rem % sj
        return ("x", i, j, k)
    elif flat < n_Ex + n_Ey:
        local = flat - n_Ex
        sk = Nz + 1
        sj = Ny * sk
        i = local // sj
        rem = local % sj
        j = rem // sk
        k = rem % sk
        return ("y", i, j, k)
    else:
        local = flat - n_Ex - n_Ey
        sk = Nz
        sj = (Ny + 1) * sk
        i = local // sj
        rem = local % sj
        j = rem // sk
        k = rem % sk
        return ("z", i, j, k)


def edge_endpoints(component: str, i: int, j: int, k: int, grid: GridLines):
    if component == "x":
        return (grid.x[i], grid.y[j], grid.z[k]), (grid.x[i + 1], grid.y[j], grid.z[k])
    if component == "y":
        return (grid.x[i], grid.y[j], grid.z[k]), (grid.x[i], grid.y[j + 1], grid.z[k])
    return (grid.x[i], grid.y[j], grid.z[k]), (grid.x[i], grid.y[j], grid.z[k + 1])


def primal_length(component: str, i: int, j: int, k: int, grid: GridLines) -> float:
    if component == "x":
        return grid.dx[i]
    if component == "y":
        return grid.dy[j]
    return grid.dz[k]


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


def main():
    print("=" * 76)
    print(f"Round-WG sub-cell diagnostic, n_t = {N_T}")
    print("=" * 76)

    mesh = build_mesh()
    em = mesh.edge_material
    grid = mesh.grid
    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz

    cat2_idx = np.nonzero(em.category == 2)[0]
    print(f"\n  total cat-2 edges (curved-PEC sub-cell): {len(cat2_idx)}")
    print(f"  total cat-3 edges (interior PEC):        {(em.category == 3).sum()}")
    print(f"  total cat-1 edges (dielectric):          {(em.category == 1).sum()}")
    print(f"  total cat-0 edges (bulk):                {(em.category == 0).sum()}")

    if len(cat2_idx) == 0:
        print("\n  ERROR: no cat-2 edges found.  Bad geometry?")
        return

    # ----- Build a "ground-truth" fresh pec_solid with super-fine tessellation
    shapes_with_material, material_library = _shapes_with_material()
    pec_solid_fine = build_effective_pec_solid(shapes_with_material, material_library)

    # Force a fine OCC tessellation on the pec_solid before any subsequent calls.
    fine_deflection = 1e-7
    try:
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh  # noqa: PLC0415

        BRepMesh_IncrementalMesh(pec_solid_fine, fine_deflection, False, 0.1, True)
        print(
            f"\n  pec_solid_fine tessellated with deflection = {fine_deflection:g} m (ground truth)"
        )
    except ImportError:
        print("\n  WARNING: BRepMesh not available; using default tessellation")

    # ----- Sample some cat-2 edges (limit to 80 for output volume)
    n_sample = min(80, len(cat2_idx))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(cat2_idx, size=n_sample, replace=False)
    sample_idx.sort()

    # Build edges array for compute_edge_pec_fractions
    sample_edges = np.empty((n_sample, 2, 3), dtype=np.float64)
    sample_components: list[str] = []
    sample_ijk: list[tuple[int, int, int]] = []
    for n, flat in enumerate(sample_idx):
        comp, i, j, k = decode_edge(int(flat), Nx, Ny, Nz)
        p0, p1 = edge_endpoints(comp, i, j, k, grid)
        sample_edges[n, 0] = p0
        sample_edges[n, 1] = p1
        sample_components.append(comp)
        sample_ijk.append((i, j, k))

    # ----- Reference f_L from ground-truth pec_solid
    f_L_ref = compute_edge_pec_fractions([pec_solid_fine], sample_edges)

    # Current f_L from Variante B
    L_primal = np.array(
        [primal_length(sample_components[n], *sample_ijk[n], grid) for n in range(n_sample)]
    )
    f_L_cur = em.L_free[sample_idx] / L_primal

    # ----- Reference eps_avg from compute_conformal_eps re-call with super-fine
    eps_ref, sigma_ref = compute_conformal_eps(
        shapes_with_material,
        grid,
        mesh.material_id,
        material_library,
        section_cache=None,
    )
    eps_cur = em.eps_avg[sample_idx]
    eps_ref_sample = eps_ref[sample_idx]

    # ----- Stats
    df_L = f_L_ref - f_L_cur
    deps = eps_ref_sample - eps_cur

    print()
    print("─" * 76)
    print(" Per-edge consistency (sample of 80 cat-2 edges)")
    print("─" * 76)
    print(
        f"  f_L:      cur in [{f_L_cur.min():.4f}, {f_L_cur.max():.4f}],  "
        f"ref in [{f_L_ref.min():.4f}, {f_L_ref.max():.4f}]"
    )
    print(
        f"  Δf_L:     mean={np.abs(df_L).mean():.2e},  "
        f"max={np.abs(df_L).max():.2e},  "
        f"max-rel = {(np.abs(df_L) / np.maximum(f_L_cur, 1e-12)).max() * 100:.3f} %"
    )
    print(
        f"  eps_avg:  cur in [{eps_cur.min():.4f}, {eps_cur.max():.4f}],  "
        f"ref in [{eps_ref_sample.min():.4f}, {eps_ref_sample.max():.4f}]"
    )
    print(f"  Δeps:     mean={np.abs(deps).mean():.2e},  max={np.abs(deps).max():.2e}")

    # ----- Top-10 worst Δf_L edges
    worst = np.argsort(np.abs(df_L))[::-1][:10]
    print()
    print("─" * 76)
    print(" Top-10 worst Δf_L edges")
    print("─" * 76)
    print(f"  {'idx':>8} {'comp':>4} {'(i,j,k)':>15} {'f_L cur':>10} {'f_L ref':>10} {'Δf_L':>10}")
    for w in worst:
        flat = int(sample_idx[w])
        comp, i, j, k = decode_edge(flat, Nx, Ny, Nz)
        print(
            f"  {flat:>8} {comp:>4} {f'({i},{j},{k})':>15} "
            f"{f_L_cur[w]:>10.5f} {f_L_ref[w]:>10.5f} {df_L[w]:>+10.5f}"
        )

    # ----- Now: how does Δf_L map onto ΔM_eps?
    # cat-2 formula: M_eps ∝ eps_avg / f_L (with eps_avg over A_dual normalisation)
    # Relative ΔM_eps caused by Δf_L on cat-2 edges:
    #   ΔM_eps / M_eps ≈ -Δf_L / f_L_cur
    rel_dM = -df_L / np.maximum(f_L_cur, 1e-12)
    print()
    print("─" * 76)
    print(" Implied ΔM_eps from Δf_L (M_eps ∝ 1 / f_L)")
    print("─" * 76)
    print(
        f"  rel ΔM_eps:  mean = {np.abs(rel_dM).mean() * 100:.3f} %, "
        f"max = {np.abs(rel_dM).max() * 100:.3f} %"
    )

    # ----- And: how many cat-2 edges have f_L close to the eta threshold (=0.4)?
    near_eta = np.sum((f_L_cur < 0.45) & (f_L_cur > 0.35))
    print()
    print("─" * 76)
    print(" Threshold proximity (eta = 0.4 — enlarged-cell boundary)")
    print("─" * 76)
    print(f"  cat-2 edges with f_L in [0.35, 0.45]:  {near_eta} / {n_sample}")
    print(f"  cat-2 edges with f_L < 0.5:            {(f_L_cur < 0.5).sum()} / {n_sample}")

    print("\nDone.")


if __name__ == "__main__":
    main()
