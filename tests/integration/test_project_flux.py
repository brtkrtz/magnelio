"""Flux-monitor write-through — DD-070 follow-up.

A :class:`MonitorFluxTime` on a project-backed run streams its scalar
Poynting-flux time series (time + power) into the run's ``results.h5``
(``flux/<name>/``), append-only exactly like the V/I channels, and
``Project.monitors`` resolves it by name into a ``_LoadedFluxMonitor``.
Same three properties the field-monitor gate pins:

* **Parity** — the reader reproduces an in-RAM monitor's ``power`` / ``t``
  / ``total_energy`` bit-for-bit.
* **Memory-bounded** — after the run the monitor holds no samples (the sink
  drained every one to disk).
* **Resume continuity** — a flux series recorded across a checkpoint/resume
  seam is bit-identical to one from an uninterrupted run.
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, MeshControl, open_project, resume
from magnelio.geo import Brick
from magnelio.mesh.mesher import Mesh
from magnelio.monitors import MonitorFluxTime
from magnelio.ports import PortWaveguide

A, B, LZ = 10.0e-3, 5.0e-3, 20.0e-3
F_MAX = 12.0e9
N_TOTAL = 300


def _tem_analysis(project=None):
    """A parallel-plate TEM two-port with a z-normal Poynting-flux monitor."""
    from magnelio.geo import GeometryModel  # noqa: PLC0415

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
            origin=(-A / 2, -B / 2, -LZ / 2),
            size=(A, B, LZ),
            material=Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=8),
        f_max=F_MAX,
    )
    analysis = AnalysisScatteringTD(
        mesh=mesh,
        ports=[
            PortWaveguide(name="port1", plane="zmin", n_modes=1),
            PortWaveguide(name="port2", plane="zmax", n_modes=1),
        ],
        f_max=F_MAX,
        verbose=False,
        project=project,
        geometry=model,
    )
    analysis.monitors = (
        MonitorFluxTime(
            normal="z",
            position=0.0,
            name="flux_z",
        ),
    )
    return analysis


def _ref_flux():
    """Run in-RAM and return the monitor's full (t, power)."""
    an = _tem_analysis()
    an.run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL)
    mon = an.monitors[0]
    return mon.t.copy(), mon.power.copy()


# ═════════════════════════════════════════════════════════════════════
# Parity + memory-bounded
# ═════════════════════════════════════════════════════════════════════


def test_streamed_flux_matches_in_ram_and_is_memory_bounded(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    ref_t, ref_p = _ref_flux()
    assert ref_t.size == N_TOTAL  # flux records every step

    p = tmp_path / "pp"
    an = _tem_analysis(project=p)
    proj = an.run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL)

    # Memory-bounded: the sink drained every sample during the run, so the
    # live monitor holds none — only its running count.
    live = an.monitors[0]
    assert len(live._power) == 0
    assert live._next_idx == N_TOTAL

    # Parity: the reader reproduces the in-RAM flux bit-for-bit.
    rmon = proj.monitors["flux_z"]
    assert isinstance(rmon, type(open_project(p).monitors["flux_z"]))
    assert rmon.t.shape == ref_t.shape
    assert np.array_equal(rmon.t, ref_t)
    assert np.array_equal(rmon.power, ref_p)
    assert rmon.total_energy == pytest.approx(float(np.trapezoid(ref_p, ref_t)), rel=0, abs=0)


# ═════════════════════════════════════════════════════════════════════
# Resume continuity
# ═════════════════════════════════════════════════════════════════════


def test_flux_bit_exact_across_resume(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    ref_t, ref_p = _ref_flux()

    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=120,
        checkpoint_interval=40,
    )
    # A partial flux series is already readable at the cut (streamed live).
    partial = open_project(p).monitors["flux_z"]
    assert 0 < partial.t.size < ref_t.size

    proj = resume(p, excited=("port1", 0), total_time_steps=N_TOTAL, verbose=False)
    rmon = proj.monitors["flux_z"]
    assert rmon.t.shape == ref_t.shape
    assert np.array_equal(rmon.t, ref_t), (
        f"flux t not bit-exact across resume, max|Δ|={float(np.max(np.abs(rmon.t - ref_t))):.3e}"
    )
    assert np.array_equal(rmon.power, ref_p), (
        "flux power not bit-exact across resume, max|Δ|="
        f"{float(np.max(np.abs(rmon.power - ref_p))):.3e}"
    )
