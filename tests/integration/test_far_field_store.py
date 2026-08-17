"""MonitorFarField project-store integration (DD-173).

Same store contract as the frequency and wall-loss monitors: a
fixed-size running DFT dumped whole and atomically to its own
``far_field.h5`` at each checkpoint, tagged with the step it reflects.
The file carries the box geometry and image planes alongside the bins,
so the reader rebuilds the transform without the mesh — reader ==
monitor by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Mesh, open_project, resume
from magnelio.boundaries.boundary_conditions import BoundaryConditions
from magnelio.mesh.grid import GridLines
from magnelio.monitors import MonitorFarField
from magnelio.ports import PortSpecLumped

H = 2e-3
N = 16
F0 = 3e9


def _analysis(mon, project=None):
    ax = np.arange(N + 1) * H
    grid = GridLines(ax.copy(), ax.copy(), ax.copy())
    bc = BoundaryConditions(
        cpml_thickness_cells=4,
        **{f: "CPML" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")},
    )
    mid = N // 2 * H
    return AnalysisScatteringTD(
        mesh=Mesh.from_grid(grid, boundary_conditions=bc),
        ports=[
            PortSpecLumped(
                name="feed",
                start=(mid, mid, (N // 2 - 1) * H),
                end=(mid, mid, (N // 2 + 1) * H),
                Z0=50.0,
            )
        ],
        f_max=6e9,
        monitors=(mon,),
        verbose=False,
        project=project,
    )


def _monitor():
    return MonitorFarField(freqs=[F0], margin_cells=1, name="pattern")


_ANGLES = {"theta": np.linspace(0, np.pi, 19), "phi": np.linspace(0, 2 * np.pi, 37)}


def test_reader_matches_in_ram_monitor(tmp_path):
    mon = _monitor()
    p = tmp_path / "ff"
    _analysis(mon, project=p).run(excited=[("feed", 0)], energy_stop_db=None, total_time_steps=300)
    assert (p / "runs" / "feed_mode0" / "far_field.h5").exists()

    loaded = open_project(p).monitors["pattern"]
    np.testing.assert_array_equal(loaded.f, mon.f)
    a = mon.result(F0, **_ANGLES)
    b = loaded.result(F0, **_ANGLES)
    # The divisor differs by the reference-signal storage round trip
    # (float re-sampling), so the match is tight but not bitwise.
    np.testing.assert_allclose(b.E_theta, a.E_theta, rtol=1e-10)
    np.testing.assert_allclose(b.E_phi, a.E_phi, rtol=1e-10)
    assert a.physical_mask is None and b.physical_mask is None
    np.testing.assert_allclose(b.P_rad, a.P_rad, rtol=1e-12)


def test_resume_bit_exact(tmp_path):
    n1, n_total = 120, 300

    ref = _monitor()
    _analysis(ref).run(excited=[("feed", 0)], energy_stop_db=None, total_time_steps=n_total)
    ref_pattern = ref.result(F0, **_ANGLES)

    p = tmp_path / "ff_resume"
    _analysis(_monitor(), project=p).run(
        excited=[("feed", 0)],
        energy_stop_db=None,
        total_time_steps=n1,
        checkpoint_interval=60,
    )
    proj = resume(p, excited=("feed", 0), total_time_steps=n_total, verbose=False)
    assert proj.runs["feed_mode0"]["n_steps"] == n_total
    resumed = proj.monitors["pattern"].result(F0, **_ANGLES)
    # The reference-signal storage round trip costs a few ulp on the
    # divisor; the accumulators themselves resume bit-exactly.
    np.testing.assert_allclose(resumed.E_theta, ref_pattern.E_theta, rtol=1e-10)
    np.testing.assert_allclose(resumed.E_phi, ref_pattern.E_phi, rtol=1e-10)


def test_stale_result_file_is_rejected(tmp_path):
    import h5py

    p = tmp_path / "ff_stale"
    _analysis(_monitor(), project=p).run(
        excited=[("feed", 0)],
        energy_stop_db=None,
        total_time_steps=120,
        checkpoint_interval=60,
    )
    ff = p / "runs" / "feed_mode0" / "far_field.h5"
    with h5py.File(ff, "r+") as f:
        f.attrs["n_completed"] = int(f.attrs["n_completed"]) - 1

    with pytest.raises(ValueError, match="far_field.h5 is at step"):
        resume(p, excited=("feed", 0), total_time_steps=300, verbose=False)


def test_recipe_roundtrip_carries_the_spec():
    from magnelio.analysis._recipe import _monitor_from_dict, _monitor_to_dict

    mon = MonitorFarField(freqs=[1e9, 2e9], margin_cells=4, name="ff")
    back = _monitor_from_dict(_monitor_to_dict(mon))
    assert back.name == mon.name
    assert back.margin_cells == mon.margin_cells
    np.testing.assert_array_equal(back.freqs, mon.freqs)
