"""MonitorWallLoss project-store integration (DD-082 addendum).

The monitor is the same KIND as a MonitorFieldFrequency — a fixed-size
running DFT, not an append stream — so it follows the session-99 Freq
pattern: its own result file, written whole and atomically at each
checkpoint, tagged with the n_completed that ties it to checkpoint.h5.

It differs in one way, which these gates pin: the RESULT is a reduction
(P_loss/P_flow per tag), not the accumulators.  wall_loss.h5 therefore
carries both — the reduction a reader serves, and the raw accumulators a
resume reloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Mesh, open_project, resume
from magnelio.materials import Hammerstad
from magnelio.mesh.grid import GridLines
from magnelio.monitors.wall_loss import MonitorWallLoss
from magnelio.ports import PortWaveguide

SIGMA_CU = 5.8e7
W_A, GAP_B, LENGTH = 10e-3, 5e-3, 30e-3
FREQS = np.linspace(2e9, 10e9, 5)

_BCS = {
    "xmin": "PMC",
    "xmax": "PMC",
    "ymin": "PEC",
    "ymax": "PEC",
    "zmin": "PEC",
    "zmax": "PEC",
}


def _monitor(**kw):
    return MonitorWallLoss(
        freqs=FREQS,
        reference_plane=("z", 2e-3),
        sigma=SIGMA_CU,
        bc_faces=("ymin", "ymax"),
        name="walls",
        **kw,
    )


def _analysis(mon, project=None):
    grid = GridLines(
        x=np.linspace(0, W_A, 11),
        y=np.linspace(0, GAP_B, 6),
        z=np.linspace(0, LENGTH, 61),
    )
    return AnalysisScatteringTD(
        mesh=Mesh.from_grid(grid, boundary_conditions=_BCS),
        ports=[
            PortWaveguide(name="port1", plane="zmin", n_modes=1),
            PortWaveguide(name="port2", plane="zmax", n_modes=1),
        ],
        f_max=10e9,
        monitors=(mon,),
        verbose=False,
        project=project,
    )


def _assert_same_fraction(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for tag in a:
        np.testing.assert_array_equal(np.asarray(a[tag]), np.asarray(b[tag]))


def test_reader_matches_in_ram_monitor(tmp_path):
    """Project.monitors[name] serves the same dissipated_fraction /
    power_loss the in-RAM monitor holds — bit for bit, since the file
    carries what the monitor reduced rather than a re-derivation."""
    mon = _monitor()
    p = tmp_path / "wl"
    _analysis(mon, project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=200,
    )

    loaded = open_project(p).monitors["walls"]
    _assert_same_fraction(loaded.dissipated_fraction, mon.dissipated_fraction)
    np.testing.assert_array_equal(loaded.f, mon.f)
    _assert_same_fraction(loaded.power_loss(2.5), mon.power_loss(2.5))
    # The tag types survive: BC walls stay face-name strings.
    assert "ymin" in loaded.dissipated_fraction
    assert "total" in loaded.dissipated_fraction


def test_resume_bit_exact(tmp_path):
    """A run resumed across a checkpoint seam gives the same fractions as
    an uninterrupted one — the raw accumulators reload from wall_loss.h5,
    and the monitor spec rebuilds from the recipe."""
    n1, n_total = 80, 200

    ref_mon = _monitor()
    _analysis(ref_mon).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n_total,
    )
    ref_frac = ref_mon.dissipated_fraction

    p = tmp_path / "wl_resume"
    _analysis(_monitor(), project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=n1,
        checkpoint_interval=40,
    )
    assert open_project(p).runs["port1_mode0"]["n_steps"] == n1

    proj = resume(p, excited=("port1", 0), total_time_steps=n_total, verbose=False)
    assert proj.runs["port1_mode0"]["n_steps"] == n_total
    _assert_same_fraction(proj.monitors["walls"].dissipated_fraction, ref_frac)


def test_recipe_roundtrip_carries_the_spec():
    """The monitor spec survives the recipe — including the DD-088
    roughness, which rides the store's material serialiser."""
    from magnelio.analysis._recipe import _monitor_from_dict, _monitor_to_dict

    mon = _monitor(roughness=Hammerstad(1e-6), mu=1.0)
    back = _monitor_from_dict(_monitor_to_dict(mon))
    assert back.name == mon.name
    assert back.reference_plane == mon.reference_plane
    assert back.sigma == mon.sigma
    assert back.mu == mon.mu
    assert back.bc_faces == mon.bc_faces
    assert back.roughness == mon.roughness
    np.testing.assert_array_equal(back.freqs, mon.freqs)

    smooth = _monitor()
    assert _monitor_from_dict(_monitor_to_dict(smooth)).roughness is None


def test_roughness_survives_the_store(tmp_path):
    """A rough monitor's persisted fractions are the rough ones — the
    K(f) factor must ride the recipe, not silently drop to smooth."""
    rough = _monitor(roughness=Hammerstad(1e-6))
    p = tmp_path / "wl_rough"
    _analysis(rough, project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=120,
    )
    loaded = open_project(p).monitors["walls"]
    _assert_same_fraction(loaded.dissipated_fraction, rough.dissipated_fraction)

    smooth = _monitor()
    _analysis(smooth, project=tmp_path / "wl_smooth").run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=120,
    )
    k = Hammerstad(1e-6).factor(FREQS, SIGMA_CU)
    np.testing.assert_allclose(
        loaded.dissipated_fraction["total"] / smooth.dissipated_fraction["total"],
        k,
        rtol=1e-12,
    )


def test_legacy_project_without_the_group_loads(tmp_path):
    """A run with no wall-loss monitor writes no file, and the reader
    simply has no such name — projects written before this change load
    unchanged."""
    from magnelio.monitors.flux import MonitorFluxTime

    p = tmp_path / "no_wl"
    _analysis(
        MonitorFluxTime(plane=("z", 0.015), name="flux"),
        project=p,
    ).run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=60)

    assert not (p / "runs" / "port1_mode0" / "wall_loss.h5").exists()
    mons = open_project(p).monitors
    assert "flux" in mons and "walls" not in mons


def test_stale_result_file_is_rejected(tmp_path):
    """A wall_loss.h5 whose n_completed disagrees with the checkpoint (a
    hard crash between the two writes) must raise, not integrate on from
    a wrong partial DFT."""
    import h5py

    p = tmp_path / "wl_stale"
    _analysis(_monitor(), project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=80,
        checkpoint_interval=40,
    )
    wl = p / "runs" / "port1_mode0" / "wall_loss.h5"
    with h5py.File(wl, "r+") as f:
        f.attrs["n_completed"] = int(f.attrs["n_completed"]) - 1

    with pytest.raises(ValueError, match="wall_loss.h5 is at step"):
        resume(p, excited=("port1", 0), total_time_steps=200, verbose=False)
