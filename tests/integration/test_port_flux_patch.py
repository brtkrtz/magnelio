"""Integration: the DD-095 conformality patch sees enlarged-cell donations.

A short curved-PEC edge is masked and hands its dual-face mass to a
neighbour (the enlarged-cell technique).  When that neighbour is a
category-0/1 edge, its M_eps exceeds the staircase value while the
frozen DD-095 spec set chi = 1 — the power patch then transports the
neighbour's share without accounting for it.  DD-163 folds the
donation into chi; before it, every conformal coax port warned about
the residual bias instead of correcting it.

The round coax below is the DD-053 port-floor fixture, small enough
for a unit-speed run and dense enough to produce category-0/1
receivers on the port plane — the trigger fixture the DD-095 dossier
asked for (internal dossier investigations/port_power/DERIVATION.md
section 5c).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio._operators.material_matrices import build_M_eps
from magnelio.geo import Cylinder, Difference, GeometryModel
from magnelio.mesh import BoxFace
from magnelio.ports import PortSpecMultiConductor
from magnelio.ports._modal import PortPlane
from magnelio.ports._modal.operator import conformal_flux_patch_scale

D_I = 0.41e-3
D_A = 5.0e-3
EPS_R = 9.0
LENGTH = 10.0e-3
F_MAX = 10e9


def _mesh() -> Mesh:
    pec = Material.pec()
    diel = Material.from_isotropic(name="dielectric", epsilon=EPS_R)
    out_cyl = Cylinder(origin=(0, 0, 0), radius=D_A / 2, height=LENGTH, axis="z", material=diel)
    in_cyl = Cylinder(origin=(0, 0, 0), radius=D_I / 2, height=LENGTH, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(out_cyl, in_cyl))
    model.add(in_cyl)
    control = MeshControl(
        min_nodes_per_wavelength=8,
        min_cells_per_feature=3,
        growth_factor=1.4,
        conformal=True,
        max_cell_size=0.4e-3,
        min_cell_size=50e-6,
        min_feature_gap=20e-6,
    )
    return Mesh.from_geometry(model, control, f_max=F_MAX)


@pytest.fixture(scope="module")
def coax_mesh() -> Mesh:
    return _mesh()


def _patch_and_receivers(mesh: Mesh):
    """chi arrays plus the interior-slab bookkeeping they are built from."""
    plane = PortPlane.from_mesh(BoxFace.Z_MIN, mesh)
    m_eps = build_M_eps(mesh)
    chi_u, chi_v = conformal_flux_patch_scale(plane, mesh, m_eps)

    em = mesh.edge_material
    grid = mesh.grid
    n_sizes = (grid.Nx, grid.Ny, grid.Nz)

    def stride(comp_axis: int, along_axis: int) -> int:
        shape = [n_sizes[ax] + (0 if ax == comp_axis else 1) for ax in range(3)]
        s = 1
        for ax in range(along_axis + 1, 3):
            s *= shape[ax]
        return s

    face = plane.face
    n_ax = face.normal_axis
    idx_u = plane.e_u_indices + face.inward_sign * stride(face.u_axis, n_ax)
    idx_v = plane.e_v_indices + face.inward_sign * stride(face.v_axis, n_ax)

    donor = em.enlarged_cell_donor
    shorts = np.nonzero(donor >= 0)[0]
    out = []
    for idx, chi in ((idx_u, chi_u), (idx_v, chi_v)):
        cat = em.category[idx]
        borrowed = np.zeros(idx.size, dtype=float)
        pos = {int(e): i for i, e in enumerate(idx)}
        for s in shorts:
            p = pos.get(int(donor[s]))
            if p is None or cat[p] not in (0, 1):
                continue
            borrowed[p] += em.enlarged_cell_area[s]
        out.append((chi, cat, borrowed))
    return out


def test_fixture_triggers_category_01_receivers(coax_mesh):
    """Guard the guard: without a receiver the test below proves nothing."""
    n_recv = sum(int(np.count_nonzero(b > 0.0)) for _, _, b in _patch_and_receivers(coax_mesh))
    assert n_recv > 0, (
        "fixture no longer produces enlarged-cell donations parked on "
        "category-0/1 port-plane edges — pick a mesh that does, or the "
        "patch test below is vacuous"
    )


def test_donation_receivers_carry_a_patch(coax_mesh):
    """chi > 1 exactly on the receivers, exactly 1 on every other cat-0/1 edge."""
    for chi, cat, borrowed in _patch_and_receivers(coax_mesh):
        stair = np.isin(cat, (0, 1))
        recv = stair & (borrowed > 0.0)
        plain = stair & (borrowed <= 0.0)
        assert np.all(chi[recv] > 1.0), (
            "an enlarged-cell receiver kept the staircase patch weight "
            f"(min chi = {chi[recv].min() if recv.any() else float('nan')})"
        )
        assert np.all(chi[plain] == 1.0), (
            "the patch moved a category-0/1 edge that received nothing — "
            "DD-095 stays conformality-only"
        )


def test_solve_ports_is_quiet_on_a_conformal_coax(coax_mesh):
    """The user-visible half: no port-power warning on a plain coax."""
    specs = [
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, epsilon_r=EPS_R, n_modes=1),
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, epsilon_r=EPS_R, n_modes=1),
    ]
    analysis = AnalysisScatteringTD(mesh=coax_mesh, ports=specs, f_max=F_MAX, verbose=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analysis.solve_ports()
    power_warnings = [str(w.message) for w in caught if "power" in str(w.message)]
    assert not power_warnings, power_warnings
