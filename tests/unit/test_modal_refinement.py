"""Tests for ``solve_modes_refined`` (Phase-2 cleanup item 3).

Validates the refinement-based mode-parameter pipeline:

1. **Convergence on rectangular coax** (PEC outer body + inner brick):
   ``z_line`` should converge to the IPC-2141A closed-form within the
   IPC's intrinsic 5–10 % accuracy after 2–3 refinement levels.
2. **Stopping behaviour**: ``target_rel_err`` is honoured (returned
   ``converged=True`` once the threshold is reached), ``max_levels``
   is honoured (``converged=False`` if the target is too tight).
3. **Richardson extrapolation** is populated when ``extrapolate=True``
   and ≥ 2 levels were run.
4. **Convergence order** is computed when ≥ 3 levels were run.
5. **Argument validation**: unknown ``target``, non-positive
   ``max_levels`` and ``target_rel_err`` rejected.

Tests use small geometries and modest base mesh to keep per-test
runtime under ~30 s.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from magnelio.geo import Brick, Difference, GeometryModel
from magnelio.materials.material import Material
from magnelio.mesh.mesher import MeshControl
from magnelio.ports._modal import (
    BoxFace,
    PortSpecMultiConductor,
    solve_modes_refined,
)


def _rect_coax_geometry(
    *, B: float = 8e-3, b_air: float = 6e-3, a: float = 2e-3, L_x: float = 6e-3
):
    """Rect coax with PEC outer body + inner brick — known IPC Z_line."""
    pec = Material.pec()
    air = Material.air()
    y0_air = z0_air = (B - b_air) / 2
    y0_inner = z0_inner = (B - a) / 2
    bbox = Brick(origin=(0, 0, 0), size=(L_x, B, B), material=pec)
    air_region = Brick(origin=(0, y0_air, z0_air), size=(L_x, b_air, b_air), material=air)
    inner = Brick(origin=(0, y0_inner, z0_inner), size=(L_x, a, a), material=pec)
    model = GeometryModel()
    model.add(Difference(bbox, air_region))
    model.add(Difference(air_region, inner))
    model.add(inner)
    return model, B, a, b_air


def _ipc_2141a_z_line(B: float, a: float, eps_r: float = 1.0) -> float:
    """IPC-2141A square coax: Z_0 ≈ (60/√εr) · ln(1.0787 · B/a)."""
    eta_factor = 60.0 / math.sqrt(eps_r)
    return eta_factor * math.log(1.0787 * B / a)


class TestConvergenceRectCoax:
    def test_z_line_converges_toward_ipc_reference(self):
        """Refined Z_line lands within ±20 % of the IPC-2141A reference.

        The IPC formula uses the *outer-conductor inner edge* as ``B``
        (= ``b_air`` in this fixture), not the bbox outer edge.  20 %
        band accommodates: IPC's intrinsic 5–10 % accuracy + the
        Cartesian-staircase residual at the inner brick edges (no
        Dey-Mittra in this fixture's MeshControl).
        """
        model, _, a, b_air = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            growth_factor=1.4,
            conformal=False,
            max_cell_size=0.4e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        report = solve_modes_refined(
            spec,
            model,
            base,
            f_max=8e9,
            target_rel_err=5e-3,
            max_levels=3,
        )
        z_ipc = _ipc_2141a_z_line(b_air, a)
        rel_err = abs(report.converged_value - z_ipc) / z_ipc
        assert rel_err < 0.20, (
            f"refined z_line {report.converged_value:.3f} Ω vs "
            f"IPC {z_ipc:.3f} Ω (rel err {rel_err:.3f})"
        )

    def test_z_line_history_moves_toward_reference(self):
        """Successive levels reduce the absolute error vs IPC reference."""
        model, _, a, b_air = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            growth_factor=1.4,
            conformal=False,
            max_cell_size=0.4e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        report = solve_modes_refined(
            spec,
            model,
            base,
            f_max=8e9,
            target_rel_err=1e-9,
            max_levels=3,
        )
        z_ipc = _ipc_2141a_z_line(b_air, a)
        errs = [abs(lr.value - z_ipc) for lr in report.history]
        # Last level no further from reference than the first.
        assert errs[-1] <= errs[0], f"refinement did not move toward reference: errors {errs}"

    def test_history_monotonic_in_n_cells(self):
        """Cell count strictly increases level-by-level."""
        model, *_ = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            growth_factor=1.4,
            conformal=False,
            max_cell_size=0.6e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        report = solve_modes_refined(
            spec,
            model,
            base,
            f_max=8e9,
            target_rel_err=1e-9,
            max_levels=3,
        )
        n3d = [lr.n_cells_3d for lr in report.history]
        npp = [lr.n_cells_port_plane for lr in report.history]
        assert all(n3d[i + 1] > n3d[i] for i in range(len(n3d) - 1))
        assert all(npp[i + 1] > npp[i] for i in range(len(npp) - 1))


class TestStoppingBehaviour:
    def test_unreachable_target_hits_max_levels(self):
        """target_rel_err = 1e-9 won't be reached → converged=False at cap."""
        model, *_ = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            growth_factor=1.4,
            conformal=False,
            max_cell_size=0.6e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        report = solve_modes_refined(
            spec,
            model,
            base,
            f_max=8e9,
            target_rel_err=1e-9,
            max_levels=2,
        )
        assert report.converged is False
        assert report.n_levels == 2

    def test_loose_target_converges_early(self):
        """target_rel_err = 1.0 (loose) → converged after 2 levels."""
        model, *_ = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            growth_factor=1.4,
            conformal=False,
            max_cell_size=0.6e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        report = solve_modes_refined(
            spec,
            model,
            base,
            f_max=8e9,
            target_rel_err=1.0,
            max_levels=4,
        )
        assert report.converged is True
        assert report.n_levels == 2


class TestExtrapolation:
    def test_extrapolation_value_present_when_two_levels(self):
        model, *_ = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            growth_factor=1.4,
            conformal=False,
            max_cell_size=0.6e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        report = solve_modes_refined(
            spec,
            model,
            base,
            f_max=8e9,
            target_rel_err=1e-9,
            max_levels=2,
            extrapolate=True,
        )
        assert report.extrapolated_value is not None
        assert math.isfinite(report.extrapolated_value)
        assert report.converged_value == report.extrapolated_value

    def test_extrapolation_disabled(self):
        model, *_ = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            growth_factor=1.4,
            conformal=False,
            max_cell_size=0.6e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        report = solve_modes_refined(
            spec,
            model,
            base,
            f_max=8e9,
            target_rel_err=1e-9,
            max_levels=2,
            extrapolate=False,
        )
        assert report.extrapolated_value is None
        assert report.converged_value == report.history[-1].value


class TestConvergenceOrder:
    def test_order_populated_after_three_levels(self):
        model, *_ = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            growth_factor=1.4,
            conformal=False,
            max_cell_size=0.6e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        report = solve_modes_refined(
            spec,
            model,
            base,
            f_max=8e9,
            target_rel_err=1e-9,
            max_levels=3,
        )
        assert report.convergence_order is not None
        assert math.isfinite(report.convergence_order)


class TestValidation:
    def test_unknown_target_rejected(self):
        model, *_ = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            max_cell_size=1e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        with pytest.raises(ValueError, match="unknown refinement target"):
            solve_modes_refined(
                spec,
                model,
                base,
                f_max=8e9,
                target="frobnicate",
            )

    def test_zero_max_levels_rejected(self):
        model, *_ = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            max_cell_size=1e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        with pytest.raises(ValueError, match="max_levels"):
            solve_modes_refined(
                spec,
                model,
                base,
                f_max=8e9,
                max_levels=0,
            )

    def test_zero_target_rel_err_rejected(self):
        model, *_ = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            max_cell_size=1e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        with pytest.raises(ValueError, match="target_rel_err"):
            solve_modes_refined(
                spec,
                model,
                base,
                f_max=8e9,
                target_rel_err=0.0,
            )


class TestReportSemantics:
    def test_report_is_frozen(self):
        model, *_ = _rect_coax_geometry()
        base = MeshControl(
            min_nodes_per_wavelength=4,
            min_cells_per_feature=3,
            max_cell_size=1e-3,
        )
        spec = PortSpecMultiConductor(
            name="p",
            plane=BoxFace.X_MIN,
            conductors=None,
            epsilon_r=1.0,
            n_modes=1,
        )
        report = solve_modes_refined(
            spec,
            model,
            base,
            f_max=8e9,
            max_levels=1,
        )
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            report.target = "epsilon_r"  # type: ignore
