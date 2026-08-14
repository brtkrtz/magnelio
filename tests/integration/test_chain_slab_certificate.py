"""DD-067 regression: feed-chain slab certificate (stage 2) + mu-flatten.

The transversal pair-product chain certificate is structurally blind
to the *normal-face* M_mu (it enters the TE transversal curl-curl
operator but forms no co-located pair).  Measured on the conformal
RG-58 coax (DD-066 open item): the boundary-slab Hz-M_mu deviated
36 % from the interior while the chain certified "uniform" — TE11 CW
floor -42 dB instead of the -158 dB the same channel reaches on
staircase / grid-aligned meshes.

Two-part closure (DD-067):

* **Fix** — ``flatten_port_plane_mu``: the normal H-faces ON the port
  plane get the first-interior slab values, exactly like the existing
  M_eps / PEC-mask flatten and with the same rationale; applied by
  the factory (mode solve) and ``FITTimeDomainSolver.setup`` (volume
  update).
* **Guard** — ``_port_chain_slab_defect``: the factory measures the
  max relative slab deviation of every mass entry across the first
  feed cells (plane vs. slab 1 vs. slab 2); above 1e-8 the operator
  withholds the exact DTBC on every channel (loud Mur fallback).

Pinned here: the guard trips on a feed that is not translation-
invariant behind the port (dielectric step in the second cell), stays
silent on invariant feeds, and the conformal-coax defect is at the
roundoff floor after the mu-flatten.  The end-to-end CW gate (coax
TE11 < -100 dB) lives in ``test_unified_multimode_port.py`` /
``merged_port_cw_floors.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import Material, Mesh
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecMultiConductor
from magnelio.ports._modal import BoxFace, build_modal_port
from magnelio.ports._modal.factory import _port_chain_slab_defect
from magnelio.ports._modal.port_plane import BoxFace as _BF
from magnelio.solver.stability import courant_dt


def _rect_coax_grid(nz=21):
    return GridLines(
        x=np.linspace(-5e-3, 5e-3, 21),
        y=np.linspace(-5e-3, 5e-3, 21),
        z=np.linspace(0.0, 1e-3 * (nz - 1) * 0.5, nz),
    )


def _rect_coax_mesh(dielectric_step: bool):
    grid = _rect_coax_grid()
    pec = Material.pec()
    regions = [
        (pec, (-1e-3, -1e-3, grid.z[0], 1e-3, 1e-3, grid.z[-1])),
    ]
    if dielectric_step:
        # eps step starting in the second cell behind the z_min port:
        # the feed is NOT translation-invariant along the port normal.
        diel = Material.from_isotropic(name="er4", epsilon=4.0)
        regions.insert(
            0,
            (
                diel,
                (-5e-3, -5e-3, grid.z[1], 5e-3, 5e-3, grid.z[-1]),
            ),
        )
    mesh = Mesh.from_grid(grid, regions=regions)
    return mesh.with_boundary_conditions(
        {
            "xmin": "PEC",
            "xmax": "PEC",
            "ymin": "PEC",
            "ymax": "PEC",
            "zmin": "PMC",
            "zmax": "PMC",
        }
    )


def _build(mesh, epsilon_r=1.0):
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(mesh.grid, "normal")
    spec = PortSpecMultiConductor(
        name="p",
        plane=BoxFace.Z_MIN,
        epsilon_r=epsilon_r,
        n_modes=1,
    )
    return build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=8.0e9)


class TestChainSlabCertificate:
    def test_invariant_feed_keeps_dtbc(self):
        op = _build(_rect_coax_mesh(dielectric_step=False))
        assert op.termination_kinds == ["dtbc"]

    def test_non_invariant_feed_falls_back_to_mur_loudly(self):
        with pytest.warns(UserWarning, match="certificate stage 2"):
            op = _build(_rect_coax_mesh(dielectric_step=True))
        assert op.termination_kinds == ["mur"]

    def test_conformal_coax_defect_at_roundoff_after_mu_flatten(self):
        """The DD-066 case: 0.36 boundary-slab Hz-M_mu deviation
        before the mu-flatten; at the roundoff floor after."""
        from magnelio import MeshControl
        from magnelio._operators.material_matrices import (
            flatten_port_plane_mass,
            flatten_port_plane_mu,
        )
        from magnelio.geo import Cylinder, Difference, GeometryModel

        pec = Material.pec()
        diel = Material.from_isotropic(name="pe", epsilon=2.25)
        dx = 0.12e-3
        outer = Cylinder(
            origin=(0.0, 0.0, 0.0), radius=1.475e-3, height=12 * dx, axis="z", material=diel
        )
        inner = Cylinder(
            origin=(0.0, 0.0, 0.0), radius=0.405e-3, height=12 * dx, axis="z", material=pec
        )
        model = GeometryModel(background=pec)
        model.add(Difference(outer, inner))
        model.add(inner)
        mesh = Mesh.from_geometry(
            model,
            MeshControl(min_nodes_per_wavelength=15, max_cell_size=dx),
            f_max=50.0e9,
        )
        m_eps = build_M_eps(mesh)
        m_mu = build_M_mu(mesh)

        raw = _port_chain_slab_defect(
            mesh,
            flatten_port_plane_mass(m_eps, mesh, _BF.Z_MIN),
            m_mu,
            _BF.Z_MIN,
        )
        assert raw > 0.1, (
            f"expected the conformal boundary-slab M_mu deviation "
            f"(measured 0.36) without the mu-flatten; got {raw:.2e}"
        )
        flat = _port_chain_slab_defect(
            mesh,
            flatten_port_plane_mass(m_eps, mesh, _BF.Z_MIN),
            flatten_port_plane_mu(m_mu, mesh, _BF.Z_MIN),
            _BF.Z_MIN,
        )
        assert flat < 1e-12, f"defect after mu-flatten: {flat:.2e}"
