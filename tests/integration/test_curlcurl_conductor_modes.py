"""WP-U1 regression: curl-curl TE/TM modes on conductor cross-sections.

The TE/TM curl-curl path was documented for hollow cross-sections
only; WP-U1 certifies it on multiply-connected domains (the WP-U2
TEM (+) TE/TM merge builds on this).  Full measurement:
``validation/curlcurl_conductor_certification.py``.

Pinned here (coarse, fast meshes):

* coax TE11 pair / TM01 cut-offs within the conformal-mesh error
  class of the analytic Bessel cross-product roots (the benchmark
  measures the convergence: |err| 3-5 % at dx = 0.24 mm falling to
  < 0.3 % at 0.03 mm),
* PEC/PMC parallel-plate ladders (PMC walls make Hz Dirichlet: TE
  needs m >= 1; the n*c/2b family is TM),
* TEM x TE/TM cross-orthogonality through the production projections
  at solver tolerance — the derivation gate for the WP-U2 merge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from magnelio import Material, Mesh, MeshControl
from magnelio._operators.material_matrices import build_M_eps, build_M_mu
from magnelio.geo import Brick, Cylinder, Difference, GeometryModel
from magnelio.ports import PortSpecNumerical
from magnelio.ports._modal import (
    BoxFace,
    ModeType,
    PortSpecMultiConductor,
    build_modal_port,
)
from magnelio.solver.stability import courant_dt

C0 = 299_792_458.0
R_I, R_A, EPS_R = 0.405e-3, 1.475e-3, 2.25


def _coax_mesh(dx=0.12e-3):
    pec = Material.pec()
    diel = Material.from_isotropic(name="pe", epsilon=EPS_R)
    length = 12.0 * dx
    outer = Cylinder(origin=(0.0, 0.0, 0.0), radius=R_A, height=length, axis="z", material=diel)
    inner = Cylinder(origin=(0.0, 0.0, 0.0), radius=R_I, height=length, axis="z", material=pec)
    model = GeometryModel(background=pec)
    model.add(Difference(outer, inner))
    model.add(inner)
    return Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=15, max_cell_size=dx),
        f_max=50.0e9,
    )


def _build(mesh, spec, f_calc):
    m_eps = build_M_eps(mesh)
    m_mu = build_M_mu(mesh)
    dt = courant_dt(mesh.grid, "normal")
    return build_modal_port(spec, mesh, m_eps, m_mu, dt=dt, f_calc=f_calc)


def _cutoffs(op):
    return [dm.mode.omega_c / (2.0 * math.pi) for dm in op.discrete_modes]


class TestCoaxHigherModes:
    """Bessel cross-product references: TE11 34.803 GHz,
    TM01 91.603 GHz (RG-58 class, eps_r = 2.25)."""

    def test_te11_pair(self):
        mesh = _coax_mesh()
        with pytest.warns(UserWarning, match="degenerate"):
            op = _build(
                mesh,
                PortSpecNumerical(name="p", plane=BoxFace.Z_MIN, n_modes=2, mode_type=ModeType.TE),
                f_calc=40.0e9,
            )
        f_te11 = 34.803e9
        pair = _cutoffs(op)
        assert abs(np.mean(pair) - f_te11) / f_te11 < 0.03
        assert abs(pair[0] - pair[1]) / np.mean(pair) < 1e-3

    def test_tm01(self):
        mesh = _coax_mesh()
        op = _build(
            mesh,
            PortSpecNumerical(name="p", plane=BoxFace.Z_MIN, n_modes=1, mode_type=ModeType.TM),
            f_calc=60.0e9,
        )
        f_tm01 = 91.603e9
        assert abs(_cutoffs(op)[0] - f_tm01) / f_tm01 < 0.03


class TestParallelPlateLadder:
    """PMC x-walls (wall ON the face, declared on the model):
    Hz Dirichlet,
    TE ladder m >= 1; PEC plates: TM ladder n >= 1."""

    A, B = 10.0e-3, 5.0e-3

    @classmethod
    def _mesh(cls):
        model = GeometryModel(
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            }
        )
        model.add(
            Brick(
                origin=(-cls.A / 2, -cls.B / 2, 0.0),
                size=(cls.A, cls.B, 10.0e-3),
                material=Material.from_isotropic(name="air", epsilon=1.0),
            )
        )
        return Mesh.from_geometry(
            model,
            MeshControl(min_nodes_per_wavelength=15, max_cell_size=0.4e-3),
            f_max=40.0e9,
        )

    def _f_mn(self, m, n):
        return math.hypot(m * C0 / (2 * self.A), n * C0 / (2 * self.B))

    def test_te_ladder(self):
        mesh = self._mesh()
        op = _build(
            mesh,
            PortSpecNumerical(name="p", plane=BoxFace.Z_MIN, n_modes=3, mode_type=ModeType.TE),
            f_calc=25.0e9,
        )
        ladder = sorted([self._f_mn(1, 0), self._f_mn(2, 0), self._f_mn(1, 1)])
        for got, ref in zip(sorted(_cutoffs(op)), ladder):
            assert abs(got - ref) / ref < 0.01

    def test_tm_ladder(self):
        mesh = self._mesh()
        op = _build(
            mesh,
            PortSpecNumerical(name="p", plane=BoxFace.Z_MIN, n_modes=2, mode_type=ModeType.TM),
            f_calc=40.0e9,
        )
        ladder = sorted([self._f_mn(0, 1), self._f_mn(1, 1)])
        for got, ref in zip(sorted(_cutoffs(op)), ladder):
            assert abs(got - ref) / ref < 0.01


class TestCrossOrthogonality:
    """TEM x TE/TM crosstalk through the production projections must
    sit at solver tolerance (measured 1e-14..1e-16) — the analytic-
    derivation gate for the WP-U2 merged port."""

    def test_coax_tem_vs_te_tm(self):
        mesh = _coax_mesh()
        n_e = build_M_eps(mesh).size
        n_h = build_M_mu(mesh).size
        op_tem = _build(
            mesh,
            PortSpecMultiConductor(name="p", plane=BoxFace.Z_MIN, epsilon_r=EPS_R, n_modes=1),
            f_calc=10.0e9,
        )
        with pytest.warns(UserWarning, match="degenerate"):
            op_te = _build(
                mesh,
                PortSpecNumerical(name="p", plane=BoxFace.Z_MIN, n_modes=2, mode_type=ModeType.TE),
                f_calc=40.0e9,
            )

        for op_x, op_y in ((op_tem, op_te), (op_te, op_tem)):
            pl = op_y.plane
            for j, dm in enumerate(op_y.discrete_modes):
                e = np.zeros(n_e)
                e[pl.e_u_indices] = dm.e_u_profile
                e[pl.e_v_indices] = dm.e_v_profile
                h = np.zeros(n_h)
                h[pl.h_u_indices] = dm.h_u_profile
                h[pl.h_v_indices] = dm.h_v_profile
                v_rel = np.max(np.abs(op_x.project_V(e))) / abs(op_y.project_V(e)[j])
                i_rel = np.max(np.abs(op_x.project_I(h))) / abs(op_y.project_I(h)[j])
                assert v_rel < 1e-12, f"V crosstalk {v_rel:.2e}"
                assert i_rel < 1e-12, f"I crosstalk {i_rel:.2e}"
