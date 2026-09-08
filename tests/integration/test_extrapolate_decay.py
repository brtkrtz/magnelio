"""Extrapolating a truncated run against an exact reference (DD-272).

An iris-coupled cavity fed through a below-cut-off slot stores energy
for far longer than any affordable march.  It is lossless, so a
one-port has to return every watt: ``|S11| = 1`` at every frequency,
exactly.  What a run reports instead of one is the energy still sitting
in the cavity when the march stopped — the truncation itself, measured
against a right answer that needs no reference run.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, MeshControl
from magnelio.geo import Brick, GeometryModel
from magnelio.mesh.mesher import Mesh
from magnelio.ports import PortWaveguide

A, B = 22.86e-3, 10.16e-3  # WR-90
L_FEED, L_IRIS, L_CAV, W_IRIS = 8e-3, 2e-3, 24e-3, 8e-3
F_MAX = 12.0e9
F_AXIS = np.linspace(8.5e9, 9.3e9, 161)
STEPS = 6000


@pytest.fixture(scope="module")
def truncated_run():
    air = Material.air()
    model = GeometryModel(
        background=Material.pec(),
        boundary_conditions={"zmin": "PEC", "zmax": "PEC"},
    )
    model.add(Brick(origin=(-A / 2, -B / 2, 0.0), size=(A, B, L_FEED), material=air))
    model.add(
        Brick(
            origin=(-W_IRIS / 2, -B / 2, L_FEED),
            size=(W_IRIS, B, L_IRIS),
            material=air,
        )
    )
    model.add(
        Brick(
            origin=(-A / 2, -B / 2, L_FEED + L_IRIS),
            size=(A, B, L_CAV),
            material=air,
        )
    )
    mesh = Mesh.from_geometry(model, MeshControl(min_nodes_per_wavelength=12), f_max=F_MAX)
    analysis = AnalysisScatteringTD(
        mesh=mesh,
        ports=[PortWaveguide(name="p1", plane="zmin", n_modes=1)],
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
        geometry=model,
    )
    return analysis.run(
        f_axis=F_AXIS,
        energy_stop_db=None,
        total_time_steps=STEPS,
        port_signal_stop_db=None,
        max_time_steps=None,
    )


def _unitarity_defect(result) -> float:
    return float(np.max(np.abs(np.abs(result.S("p1", "p1")) - 1.0)))


def test_the_truncated_run_loses_a_third_of_the_returned_power(truncated_run):
    """Without this the rest of the file would be measuring nothing."""
    assert _unitarity_defect(truncated_run) > 0.1


def test_the_continuation_restores_unitarity(truncated_run):
    raw = _unitarity_defect(truncated_run)
    extended = truncated_run.extrapolate()
    assert _unitarity_defect(extended) < 0.02
    assert _unitarity_defect(extended) < raw / 10.0


def test_the_report_names_the_trapped_resonance(truncated_run):
    extended = truncated_run.extrapolate()
    report = extended.extrapolation[("p1", 0)]
    assert report.residual < 0.05, "the record is a free decay; the fit must predict it"
    assert not report.capped
    # The cavity mode inside the band, and its decay time far beyond
    # the march that recorded it.
    in_band = (report.frequencies > F_AXIS[0]) & (report.frequencies < F_AXIS[-1])
    assert in_band.any()
    slow = report.decay_times[in_band].max()
    # 29.7 ns against a 9.1 ns march: the mode the run could not wait out.
    assert slow > 3.0 * truncated_run.dt * STEPS


def test_the_original_result_is_untouched(truncated_run):
    before = _unitarity_defect(truncated_run)
    n_before = truncated_run.n_actual_steps
    extended = truncated_run.extrapolate()
    assert extended is not truncated_run
    assert truncated_run.extrapolation is None
    assert truncated_run.n_actual_steps == n_before
    assert extended.n_actual_steps > n_before
    assert _unitarity_defect(truncated_run) == before


def test_the_records_grew_and_still_start_with_what_was_recorded(truncated_run):
    extended = truncated_run.extrapolate()
    for excited, channels in truncated_run.signals.items():
        for key, (v_old, i_old) in channels.items():
            v_new, i_new = extended.signals[excited][key]
            assert v_new.values.size > v_old.values.size
            np.testing.assert_array_equal(v_new.values[: v_old.values.size], v_old.values)
            np.testing.assert_array_equal(i_new.values[: i_old.values.size], i_old.values)
            np.testing.assert_allclose(np.diff(v_new.t), truncated_run.dt, rtol=1e-9)
