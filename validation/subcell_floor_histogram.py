"""Sub-cell classification + Krietenstein-floor histogram on a mesh.

Diagnostic for the DD-051 sub-cell pipeline: counts how many
E-edges land in each ``EdgeMaterialData.category`` and how many
H-faces land in each ``FaceMaterialData.category``, then bins the
cat-2 H-face ``A_face_free / A_face`` ratio so the user can see
how often the 1 % stability floor in ``build_M_mu`` actually fires.

Why this matters.  The cat-2 floor falls back to the bulk-staircase
value for faces almost fully in PEC.  This is the FIT-TD analogue
of the "cell completely filled with PEC" fallback of conformal
hexahedral solvers — both keep
``1/M_μ`` finite and the local wave speed bounded.  On a hollow
round waveguide the floor catches ~48 % of cat-2 H-faces, but
those are exactly the faces that carry no mode energy, so the
sub-percent TE11 cut-off accuracy is preserved.  This script makes
that statement quantitatively visible on any conformal-meshed
geometry.

Usage::

    conda run --no-capture-output -n occ python validation/subcell_floor_histogram.py

The default geometry is the round waveguide
(``round_wg_subcell_diagnostic.build_mesh``) at four transversal
resolutions ``n_t ∈ {17, 25, 33, 49}``.  Replace ``build_mesh`` /
``RESOLUTIONS`` with your own geometry to run the diagnostic on
arbitrary conformal meshes.
"""

from __future__ import annotations

import warnings

import numpy as np

# Re-use the round-WG mesh builder.
from round_wg_subcell_diagnostic import build_mesh  # noqa: PLC0415 — local example

from magnelio.geo._subcell import _build_A_face_H

RESOLUTIONS = (17, 25, 33, 49)
RATIO_BINS = (0.0, 0.01, 0.10, 0.25, 0.50, 0.99, 1.0001)
RATIO_LABELS = (
    "<1 % (floor)",
    "1–10 %",
    "10–25 %",
    "25–50 %",
    "50–99 %",
    "99–100 %",
)


def _print_histogram(label: str, counts: list[int], total: int) -> None:
    print(f"  {label}:")
    for ratio_label, count in zip(RATIO_LABELS, counts):
        pct = 100 * count / total if total else 0.0
        print(f"    {ratio_label:>15s}: {count:>6d}  ({pct:5.1f} %)")


_FLOOR_TRIGGER_PCT = 70.0  # cf. STATUS.md construction site 2 / DD-058


def _diagnose(n_t: int) -> float | None:
    """Run the per-resolution diagnostic.

    Returns the cat-2 floor share in percent (or ``None`` if there
    are no cat-2 H-faces) so the caller can issue a roll-up trigger
    warning when the share crosses the H-face donor threshold.
    """
    mesh = build_mesh(n_t)
    grid = mesh.grid
    em = mesh.edge_material
    fm = mesh.face_material

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    n_E = em.category.size
    n_H = fm.category.size

    print(f"=== n_t = {n_t}  (Nx = {Nx}, Ny = {Ny}, Nz = {Nz}) ===")

    # E-edge categories
    print(f"E-edges total: {n_E}")
    e_labels = [
        (0, "bulk"),
        (1, "dielectric boundary"),
        (2, "curved-PEC sub-cell"),
        (3, "interior PEC (masked)"),
    ]
    for c, name in e_labels:
        n = int((em.category == c).sum())
        print(f"  cat-{c} {name:<26s}: {n:>6d}  ({100 * n / n_E:5.1f} %)")
    n_short = int((em.enlarged_cell_donor >= 0).sum())
    print(f"  enlarged-cell-borrowed (f_L < eta)    : {n_short:>6d}")

    # H-face categories
    print(f"H-faces total: {n_H}")
    h_labels = [
        (0, "bulk"),
        (1, "dielectric boundary"),
        (2, "curved-PEC sub-cell"),
    ]
    for c, name in h_labels:
        n = int((fm.category == c).sum())
        print(f"  cat-{c} {name:<26s}: {n:>6d}  ({100 * n / n_H:5.1f} %)")

    # Cat-2 H-face A_face_free / A_face histogram
    A_face = _build_A_face_H(grid)
    cat2 = fm.category == 2
    n_cat2 = int(cat2.sum())
    if n_cat2 == 0:
        print("  (no cat-2 H-faces — geometry has no curved PEC contour)")
        print()
        return None

    ratio = fm.A_face_free[cat2] / A_face[cat2]
    hist, _ = np.histogram(ratio, bins=RATIO_BINS)
    _print_histogram(
        "Cat-2 H-face A_face_free / A_face distribution",
        hist.tolist(),
        n_cat2,
    )
    floor_count = int(hist[0])
    floor_pct = 100 * floor_count / n_cat2
    print(
        f"  → {floor_count} faces ({floor_pct:.1f} %) land in the 1 % floor "
        f"and fall back to staircase M_μ.  This is the FIT-TD analogue of "
        f"the ‘cell completely filled with PEC’ fallback of conformal "
        f"hexahedral solvers."
    )
    print()
    return floor_pct


def main() -> None:
    warnings.filterwarnings("ignore")  # mute aspect-ratio mesher warnings
    floor_shares: list[float] = []
    for n_t in RESOLUTIONS:
        share = _diagnose(n_t)
        if share is not None:
            floor_shares.append(share)

    print("=" * 72)
    print("Reading guide")
    print("=" * 72)
    print(
        "  The 1 % A_face_free floor is the FIT-TD analogue of the\n"
        "  'cell completely filled with PEC' fallback of conformal\n"
        "  hexahedral solvers.  Faces below the\n"
        "  threshold receive the bulk-staircase value of M_μ rather than the\n"
        "  Krietenstein-reduced one.  On smooth-walled curved-PEC geometries\n"
        "  (round/rect waveguides, hollow cavities, coax) this is harmless:\n"
        "  the Round-WG TE11 cut-off lands sub-percent at floor shares of\n"
        "  ~48 % because the floor catches faces that carry no mode energy.\n"
    )
    if floor_shares:
        peak = max(floor_shares)
        if peak > _FLOOR_TRIGGER_PCT:
            print(
                f"  ⚠  PEAK FLOOR SHARE = {peak:.1f} %  "
                f"(> {_FLOOR_TRIGGER_PCT:.0f} % trigger threshold)\n"
                "  This geometry has many fast-PEC H-faces.  Measured on the\n"
                "  WP-R5 trigger geometry (iris-loaded pillbox, 70-72 %\n"
                "  floor share): the floored faces are Faraday-dead — their\n"
                "  circulation edges sit inside the PEC mask — so the\n"
                "  staircase fallback stayed exactly neutral and the\n"
                "  H-face donor pass (assign_h_face_donors) is dormant.\n"
                "  Re-check with validation/\n"
                "  iris_cavity_donor_trigger.py adapted to this geometry\n"
                "  before wiring the donor — see DD-058."
            )
        else:
            print(
                f"  Peak floor share across resolutions = {peak:.1f} %\n"
                f"  (below the {_FLOOR_TRIGGER_PCT:.0f} % H-face donor trigger threshold).\n"
                "  Staircase fallback is fine for this geometry."
            )


if __name__ == "__main__":
    main()
