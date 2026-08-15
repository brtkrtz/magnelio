"""Certificate: a symmetry plane cutting matter costs no accuracy.

A half model closed with a magnetic symmetry wall must reproduce its
full model exactly -- the discrete half update is the restriction of the
full one, so for a symmetric mode the two eigenfrequencies agree to
machine precision.  Anything else is a defect of the boundary layer, and
this certificate is built so that the boundary layer is the *only* thing
that can differ.

Grid.  The magnetic wall sits half a boundary cell outside the outermost
primal line, and the mesher realises that by moving the clipped line at
the plane to ``h/3`` (the cell shrinks to ``2h/3``, its outer half
reaches back to the plane).  A half grid is therefore never the
restriction of a *uniform* full grid.  Both models are meshed on a
forced ladder that carries that same ``h/3`` line, so the two grids
agree on ``x >= 0`` line for line -- asserted, not assumed.

Fixtures, in order of what they put into the plane:

* ``empty``    -- nothing.  Floor: the wall convention alone.
* ``offset``   -- a mirrored pair of dielectric bricks two cells clear
  of the plane.  Floor with material contrast present, but nothing for
  the boundary layer itself to classify.
* ``brick``    -- a dielectric brick through the plane.  Its trace there
  is a rectangle on grid lines: the staircase reproduces the *contour*
  exactly, yet the dual faces of the edges in the plane still straddle
  two materials and need their average.
* ``cylinder`` -- a dielectric cylinder along x through the plane.  Its
  trace is a circle, curved inside the plane, so no grid line can snap
  to it.

The floor fixtures separate a grid or wall-convention error from a
classification error: they must stay at machine precision under any
change to the conformal pass, while the two loaded fixtures move.

Measured (relative eigenfrequency difference, full vs. half):

    fixture     boundary layer blind      boundary layer classified
    empty                0                          0
    offset            2.2e-15                    1.8e-15
    brick            +2.0e-04                    4.2e-15
    cylinder         -2.3e-03                    4.7e-15

Run from ``magnelio/``:

    CUPY_ACCELERATORS= python validation/symmetry_boundary_face_certificate.py
"""

from __future__ import annotations

import sys

import numpy as np

from magnelio import AnalysisEigenmode, Material, Mesh, MeshControl
from magnelio.geo import Brick, Cylinder, GeometryModel
from magnelio.ports import PortWaveguide

C0 = 299792458.0

H = 1.0e-3  # ladder cell size
K_X = 10  # ladder runs to +-K_X * H
NY, NZ = 10, 20
A, B, D = 2 * K_X * H, NY * H, NZ * H

R_LOAD = 3.0e-3
L_LOAD = 6.0e-3
EPS_R = 4.0

N_MODES = 4
TOL = 1e-12  # machine precision, four orders below the tightest floor

F_EMPTY = 0.5 * C0 * np.hypot(1.0 / A, 1.0 / D)

LOADS = ("empty", "offset", "brick", "cylinder")


def _ladder(half: bool) -> list[float]:
    """The x plane ladder that makes both grids agree on ``x >= 0``."""
    pos = np.concatenate(([H / 3.0], np.arange(1, K_X + 1) * H))
    if half:
        return np.concatenate(([0.0], pos[1:])).tolist()
    return np.concatenate((-pos[::-1], pos)).tolist()


def cavity(half: bool, load: str) -> Mesh:
    diel = Material.from_isotropic(name="LOAD", epsilon=EPS_R)
    model = GeometryModel(
        background=Material.air(),
        boundary_conditions={"xmin": "SymmetryPMC"} if half else None,
        allow_overlaps=True,
    )
    # The air cavity is the whole domain; its faces are the PEC walls.
    model.add(Brick(origin=(-A / 2, 0.0, 0.0), size=(A, B, D), material=Material.air()))

    if load == "brick":
        model.add(
            Brick(
                origin=(-L_LOAD / 2, B / 2 - R_LOAD, D / 2 - R_LOAD),
                size=(L_LOAD, 2 * R_LOAD, 2 * R_LOAD),
                material=diel,
            )
        )
    elif load == "cylinder":
        model.add(
            Cylinder(
                origin=(-L_LOAD / 2, B / 2, D / 2),
                radius=R_LOAD,
                height=L_LOAD,
                axis="x",
                material=diel,
            )
        )
    elif load == "offset":
        for x0 in (-6 * H, 2 * H):
            model.add(
                Brick(
                    origin=(x0, B / 2 - R_LOAD, D / 2 - R_LOAD),
                    size=(4 * H, 2 * R_LOAD, 2 * R_LOAD),
                    material=diel,
                )
            )
    elif load != "empty":
        raise ValueError(load)

    # Mesher hint only.  Without a declared port the equidistant-cell
    # buffer applies to every domain face; at the symmetry face it then
    # splits the half-cell first interval and the growth rule refines
    # the whole axis.  The eigenmode analysis is handed the mesh alone,
    # so this leaves the physics untouched -- the closure stays PEC on
    # all six faces, PMC on the symmetry face.
    model.add_port(PortWaveguide(name="mesher_hint", plane="zmax"))

    ctrl = MeshControl(
        # The grid is dictated by the ladder, not by accuracy: this
        # certificate compares two discretisations of one problem, so
        # the wavelength rule must not add lines of its own.
        min_nodes_per_wavelength=4,
        min_cells_per_feature=0,
        growth_factor=1.5,
        max_cell_size=1.5 * H,
        conformal=True,
        forced_planes={
            "x": _ladder(half),
            "y": np.linspace(0.0, B, NY + 1).tolist(),
            "z": np.linspace(0.0, D, NZ + 1).tolist(),
        },
    )
    return Mesh.from_geometry(model, ctrl, f_max=2.0 * F_EMPTY)


def conformal_on_face(mesh: Mesh, axis: int) -> int:
    """Category-1/2 E-edges lying in the low bbox face of ``axis``."""
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    shapes = ((Nx, Ny + 1, Nz + 1), (Nx + 1, Ny, Nz + 1), (Nx + 1, Ny + 1, Nz))
    offs = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    cat = mesh.edge_material.category
    fam = [cat[offs[k] : offs[k + 1]].reshape(shapes[k]) for k in range(3)]
    cats = np.concatenate([np.take(fam[k], 0, axis=axis).ravel() for k in range(3) if k != axis])
    return int(np.count_nonzero((cats == 1) | (cats == 2)))


def measure(load: str) -> dict:
    mesh_full = cavity(False, load)
    x = mesh_full.grid.x
    i0 = int(np.searchsorted(x, 0.0))
    f_full = np.atleast_1d(
        AnalysisEigenmode(mesh=mesh_full, n_modes=N_MODES, verbose=False).run().frequencies
    )

    mesh_half = cavity(True, load)
    grid_match = bool(
        x[i0:].shape == mesh_half.grid.x.shape
        and np.allclose(x[i0:], mesh_half.grid.x, rtol=0, atol=1e-15)
    )
    f_half = np.atleast_1d(
        AnalysisEigenmode(mesh=mesh_half, n_modes=N_MODES, verbose=False).run().frequencies
    )

    # The half model carries only the symmetric modes; match its
    # fundamental against the nearest full-model mode.
    fh = float(f_half[0])
    ff = float(f_full[int(np.argmin(np.abs(f_full - fh)))])
    return {
        "load": load,
        "grid_match": grid_match,
        "face": conformal_on_face(mesh_half, 0),
        "f_full": ff,
        "f_half": fh,
        "rel": fh / ff - 1.0,
    }


def main() -> None:
    print(f"empty-cavity TE101 = {F_EMPTY / 1e9:.6f} GHz   ({A * 1e3} x {B * 1e3} x {D * 1e3} mm)")
    print(
        f"{'fixture':10s} {'grid':>5s} {'face':>5s} "
        f"{'f_full [GHz]':>15s} {'f_half [GHz]':>15s} {'rel':>11s}"
    )
    rows = []
    for load in LOADS:
        r = measure(load)
        rows.append(r)
        print(
            f"{r['load']:10s} {str(r['grid_match']):>5s} {r['face']:5d} "
            f"{r['f_full'] / 1e9:15.9f} {r['f_half'] / 1e9:15.9f} {r['rel']:+11.3e}"
        )
    print("\ngrid = half grid equals the full grid on x >= 0, line for line")
    print("face = conformal E-edges in the half model's symmetry face")

    for r in rows:
        assert r["grid_match"], f"{r['load']}: the two grids differ on x >= 0"
        assert abs(r["rel"]) < TOL, (
            f"{r['load']}: full vs half {r['rel']:+.3e} exceeds {TOL:.0e} "
            f"(boundary layer blind: brick +2.0e-04, cylinder -2.3e-03)"
        )
    loaded = [r for r in rows if r["load"] in ("brick", "cylinder")]
    assert all(r["face"] > 0 for r in loaded), (
        "the loaded fixtures no longer classify their symmetry face — "
        "the certificate would pass without testing anything"
    )
    print("CERTIFICATE PASSED")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
