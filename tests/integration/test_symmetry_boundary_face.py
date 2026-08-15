"""Integration: a symmetry plane cutting matter is exact (DD-164).

A half model closed with a magnetic symmetry wall is the discrete
restriction of its full model, so for a symmetric mode the two
eigenfrequencies must agree to machine precision.  The conformal
classifier used to skip every bbox face, which left the boundary layer
of the half model staircased where the full model's interior cut was
conformal: the dielectric cylinder below read -2.3e-03 and a
grid-aligned dielectric brick still read +2.0e-04.

The magnetic wall sits half a boundary cell outside the outermost primal
line, realised by moving the clipped line to h/3, so a half grid is
never the restriction of a *uniform* full grid.  The forced ladder gives
the full model that same h/3 line; that the two grids then agree on
x >= 0 is asserted, not assumed.

``offset`` is the control: same material contrast, two cells clear of
the plane.  It separates a grid or wall-convention error (which moves
both fixtures) from a classification error (which moves only
``cylinder``).
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisEigenmode, Material, Mesh, MeshControl
from magnelio.geo import Brick, Cylinder, GeometryModel
from magnelio.ports import PortWaveguide

C0 = 299792458.0

H = 1.0e-3
K_X = 10
NY, NZ = 10, 20
A, B, D = 2 * K_X * H, NY * H, NZ * H
R_LOAD, L_LOAD, EPS_R = 3.0e-3, 6.0e-3, 4.0
F_EMPTY = 0.5 * C0 * np.hypot(1.0 / A, 1.0 / D)


def _ladder(half: bool) -> list[float]:
    pos = np.concatenate(([H / 3.0], np.arange(1, K_X + 1) * H))
    if half:
        return np.concatenate(([0.0], pos[1:])).tolist()
    return np.concatenate((-pos[::-1], pos)).tolist()


def _cavity(half: bool, load: str) -> Mesh:
    diel = Material.from_isotropic(name="LOAD", epsilon=EPS_R)
    model = GeometryModel(
        background=Material.air(),
        boundary_conditions={"xmin": "SymmetryPMC"} if half else None,
        allow_overlaps=True,
    )
    model.add(Brick(origin=(-A / 2, 0.0, 0.0), size=(A, B, D), material=Material.air()))
    if load == "cylinder":
        model.add(
            Cylinder(
                origin=(-L_LOAD / 2, B / 2, D / 2),
                radius=R_LOAD,
                height=L_LOAD,
                axis="x",
                material=diel,
            )
        )
    else:
        for x0 in (-6 * H, 2 * H):
            model.add(
                Brick(
                    origin=(x0, B / 2 - R_LOAD, D / 2 - R_LOAD),
                    size=(4 * H, 2 * R_LOAD, 2 * R_LOAD),
                    material=diel,
                )
            )
    # Mesher hint: confines the equidistant-cell buffer to that face, so
    # the forced ladder survives at the symmetry face.  The eigenmode
    # analysis is handed the mesh alone — the closure is unaffected.
    model.add_port(PortWaveguide(name="mesher_hint", plane="zmax"))
    ctrl = MeshControl(
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


def _conformal_on_face(mesh: Mesh, axis: int) -> int:
    Nx, Ny, Nz = mesh.Nx, mesh.Ny, mesh.Nz
    shapes = ((Nx, Ny + 1, Nz + 1), (Nx + 1, Ny, Nz + 1), (Nx + 1, Ny + 1, Nz))
    offs = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    cat = mesh.edge_material.category
    fam = [cat[offs[k] : offs[k + 1]].reshape(shapes[k]) for k in range(3)]
    cats = np.concatenate([np.take(fam[k], 0, axis=axis).ravel() for k in range(3) if k != axis])
    return int(np.count_nonzero((cats == 1) | (cats == 2)))


@pytest.fixture(scope="module")
def measured():
    out = {}
    for load in ("offset", "cylinder"):
        full = _cavity(False, load)
        half = _cavity(True, load)
        i0 = int(np.searchsorted(full.grid.x, 0.0))
        f_full = np.atleast_1d(
            AnalysisEigenmode(mesh=full, n_modes=4, verbose=False).run().frequencies
        )
        f_half = np.atleast_1d(
            AnalysisEigenmode(mesh=half, n_modes=4, verbose=False).run().frequencies
        )
        fh = float(f_half[0])
        out[load] = {
            "x_full_pos": full.grid.x[i0:],
            "x_half": half.grid.x,
            "face": _conformal_on_face(half, 0),
            "f_full": float(f_full[int(np.argmin(np.abs(f_full - fh)))]),
            "f_half": fh,
        }
    return out


class TestSymmetryBoundaryFace:
    @pytest.mark.parametrize("load", ["offset", "cylinder"])
    def test_grids_agree_on_the_shared_half(self, measured, load):
        """Without this the frequency comparison measures the grid."""
        r = measured[load]
        assert r["x_full_pos"].shape == r["x_half"].shape
        assert np.allclose(r["x_full_pos"], r["x_half"], rtol=0, atol=1e-15)

    def test_the_cut_fixture_classifies_its_symmetry_face(self, measured):
        """Guards the guard: no conformal edges there, nothing is tested."""
        assert measured["cylinder"]["face"] > 0
        assert measured["offset"]["face"] == 0

    @pytest.mark.parametrize("load", ["offset", "cylinder"])
    def test_half_model_reproduces_the_full_model(self, measured, load):
        r = measured[load]
        rel = r["f_half"] / r["f_full"] - 1.0
        assert abs(rel) < 1e-12, (
            f"{load}: full vs half {rel:+.3e} — the half model is no longer "
            f"the restriction of the full model (boundary layer blind "
            f"before DD-164: cylinder -2.3e-03, control 2.2e-15)"
        )
