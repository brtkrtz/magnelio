"""Field-monitor write-through — the WP-S9 gate (DD-070).

A ``MonitorFieldTime`` on a project-backed run streams its snapshots into
the run's ``results.h5`` instead of accumulating them in RAM, and a
separate reader (``Project.monitors``) loads them lazily.  These tests
pin the three properties that matter:

* **Parity** — the reader reproduces an in-RAM monitor's ``data`` /
  ``component`` bit-for-bit (the store round-trips losslessly through the
  row-major ParaView layout).
* **Memory-bounded** — after the run the monitor holds no snapshots (the
  sink drained every one to disk).
* **Resume continuity** — a monitor recorded across a checkpoint/resume
  seam is bit-identical to one from an uninterrupted run.

Plus: ``fields.xdmf`` is written for ParaView, and multi-run projects
disambiguate the monitor name by excitation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, MeshControl, open_project, resume
from magnelio.geo import Brick
from magnelio.mesh.mesher import Mesh
from magnelio.monitors import MonitorFieldTime
from magnelio.ports import PortWaveguide
from magnelio.solver.stability import (
    compute_min_effective_eps,
    compute_min_effective_mu,
    courant_dt,
)

A, B, LZ = 10.0e-3, 5.0e-3, 20.0e-3
F_MAX = 12.0e9
N_TOTAL = 300


def _tem_analysis(project=None):
    """A parallel-plate TEM two-port with an xy-plane field-time monitor."""
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
    dt = courant_dt(
        mesh.grid,
        "normal",
        min_effective_eps=compute_min_effective_eps(mesh),
        min_effective_mu=compute_min_effective_mu(mesh),
    )
    # Target times inside the run (some before, some after the N1 resume cut).
    times = np.linspace(0.15, 0.9, 5) * N_TOTAL * dt
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
        MonitorFieldTime(
            corners=((None, None, 0.0), (None, None, 0.0)),
            times=times,
            fields=["E"],
            name="Eplane",
        ),
    )
    return analysis


def _ref_monitor_data():
    """Run in-RAM and return the monitor's full recorded data."""
    an = _tem_analysis()
    an.run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL)
    mon = an.monitors[0]
    return {c: v.copy() for c, v in mon.data.items()}, mon.t.copy()


def _assert_monitor_matches(reader_mon, ref_data, tag):
    assert set(reader_mon.components) == set(ref_data), f"{tag}: comps differ"
    for comp, ref in ref_data.items():
        got = reader_mon.component(comp)
        assert got.shape == ref.shape, f"{tag} {comp}: shape {got.shape} != {ref.shape}"
        assert np.array_equal(got, ref), (
            f"{tag} {comp}: not bit-exact, max|Δ|={float(np.max(np.abs(got - ref))):.3e}"
        )


# ═════════════════════════════════════════════════════════════════════
# Parity + memory-bounded
# ═════════════════════════════════════════════════════════════════════


def test_streamed_monitor_matches_in_ram_and_is_memory_bounded(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    ref_data, ref_t = _ref_monitor_data()
    assert ref_t.size > 0 and ref_data["Ez"].shape[0] == ref_t.size

    p = tmp_path / "pp"
    an = _tem_analysis(project=p)
    proj = an.run(excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL)

    # Memory-bounded: the sink drained every snapshot to disk during the
    # run, so the live monitor holds none — only its target-time cursor.
    live = an.monitors[0]
    assert len(live._snapshots) == 0
    assert live._next_idx == ref_t.size

    # Parity: the reader reproduces the in-RAM monitor bit-for-bit.
    rmon = proj.monitors["Eplane"]
    assert rmon.t.shape == ref_t.shape
    _assert_monitor_matches(rmon, ref_data, "streamed")


# ═════════════════════════════════════════════════════════════════════
# Resume continuity
# ═════════════════════════════════════════════════════════════════════


def test_monitor_bit_exact_across_resume(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    ref_data, ref_t = _ref_monitor_data()

    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=120,
        checkpoint_interval=40,
    )
    # Some snapshots recorded before the cut, the rest after — the seam is
    # inside the monitor's target-time range.
    partial = open_project(p).monitors["Eplane"]
    assert 0 < partial.t.size < ref_t.size

    proj = resume(p, excited=("port1", 0), total_time_steps=N_TOTAL, verbose=False)
    rmon = proj.monitors["Eplane"]
    assert rmon.t.shape == ref_t.shape
    _assert_monitor_matches(rmon, ref_data, "resume")


# ═════════════════════════════════════════════════════════════════════
# ParaView descriptor
# ═════════════════════════════════════════════════════════════════════


def test_fields_xdmf_written_and_valid(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)],
        energy_stop_db=None,
        total_time_steps=N_TOTAL,
    )
    xdmf = p / "runs" / "port1_mode0" / "fields.xdmf"
    assert xdmf.exists()

    root = ET.parse(xdmf).getroot()
    # One temporal collection for the monitor, one uniform grid per time.
    colls = root.findall(".//{*}Grid[@GridType='Collection']")
    assert len(colls) == 1
    times = root.findall(".//{*}Time")
    assert len(times) > 0
    # Every data reference points into this run's results.h5.
    refs = [
        d.text.strip() for d in root.findall(".//{*}DataItem") if d.text and "results.h5" in d.text
    ]
    assert refs and all(r.startswith("results.h5:/monitors/Eplane/") for r in refs)
    # Complete component triples are also exposed as a JOINed vector
    # attribute so ParaView offers Glyph / streamline filters directly.
    vecs = root.findall(".//{*}Attribute[@AttributeType='Vector']")
    assert vecs
    for vec in vecs:
        join = vec.find("{*}DataItem[@ItemType='Function']")
        assert join is not None and len(join.findall("{*}DataItem")) == 3


# ═════════════════════════════════════════════════════════════════════
# Multi-run disambiguation
# ═════════════════════════════════════════════════════════════════════


def test_multi_run_monitor_selection(tmp_path):
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    p = tmp_path / "pp"
    _tem_analysis(project=p).run(
        excited=[("port1", 0)], energy_stop_db=None, total_time_steps=N_TOTAL
    )
    _tem_analysis(project=p).run(
        excited=[("port2", 0)], energy_stop_db=None, total_time_steps=N_TOTAL
    )

    proj = open_project(p)
    # Ambiguous by name across two runs → guides to monitors_for.
    with pytest.raises(ValueError, match="per-run"):
        _ = proj.monitors
    m1 = proj.monitors_for(("port1", 0))["Eplane"]
    m2 = proj.monitors_for(("port2", 0))["Eplane"]
    assert m1.t.size > 0 and m2.t.size > 0
