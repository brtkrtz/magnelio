"""Single-vs-double S-parameter agreement (DD-094, plan WP4).

End-to-end evidence that the single-precision default is faithful for the
physically meaningful quantities while the ultra-deep reflection floor is a
double-only feature:

- ``|S21|`` (insertion loss) is identical to ~1e-6 — single changes nothing
  a user reads off the transmission.
- The ``|S11|`` reflection floor rises from the double value (~-138 dB here)
  toward the float32 field floor (~-113 dB), staying well below the -100 dB
  reflection-free acceptance line.  This is the high-dynamic-range regime
  where a user opts into ``precision="double"``.

The discretisation error in ``|S21|`` (~1e-2 dB on this coarse mesh) dwarfs
the single-vs-double difference by four orders of magnitude — the whole
point of the single default.
"""

import numpy as np

from magnelio import (
    AnalysisScatteringTD,
    Material,
    Mesh,
)
from magnelio.materials import DispersionModel
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecMultiConductor

WIDTH_A = 10e-3
GAP_B = 5e-3
LENGTH = 20e-3
F_MAX = 10e9


def _build(precision):
    grid = GridLines(
        x=np.linspace(-WIDTH_A / 2, WIDTH_A / 2, 11),
        y=np.linspace(-GAP_B / 2, GAP_B / 2, 6),
        z=np.linspace(-LENGTH / 2, LENGTH / 2, 41),
    )
    mesh = Mesh.from_grid(grid)
    specs = [
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, n_modes=1),
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, n_modes=1),
    ]
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            }
        ),
        ports=specs,
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
        precision=precision,
    )


def test_single_matches_double_on_transmission():
    """|S21| and the linear S-matrix agree to ~1e-5 between single and double."""
    f_axis = np.linspace(F_MAX / 40, F_MAX, 81)
    Sd = _build("double").run(f_axis=f_axis, excited=["port1"])
    Ss = _build("single").run(f_axis=f_axis, excited=["port1"])

    S21d, S21s = Sd.S("port2", "port1"), Ss.S("port2", "port1")
    S11d, S11s = Sd.S("port1", "port1"), Ss.S("port1", "port1")

    # Physical transmission unaffected by the field precision.
    assert np.max(np.abs(np.abs(S21s) - np.abs(S21d))) < 1e-4, (
        "single |S21| departs from double beyond 1e-4 — the physical "
        "transmission must be precision-independent"
    )
    # Whole-matrix linear agreement well under any discretisation error.
    assert np.max(np.abs(S21s - S21d)) < 1e-4
    assert np.max(np.abs(S11s - S11d)) < 1e-4


def test_single_reflection_floor_bounded():
    """Single |S11| floor rises off the double value but stays below -100 dB."""
    f_axis = np.linspace(F_MAX / 40, F_MAX, 81)
    Ss = _build("single").run(f_axis=f_axis, excited=["port1"])
    s11_db = 20 * np.log10(np.abs(Ss.S("port1", "port1")) + 1e-30)

    assert np.all(np.isfinite(s11_db))
    # Float32 field floor lands well below the -100 dB reflection-free line;
    # bound with margin (measured ~-113 dB max on this fixture).
    assert s11_db.max() < -100.0, (
        f"single-precision |S11| floor {s11_db.max():.1f} dB breached the "
        f"-100 dB reflection-free acceptance line"
    )


def _build_dispersive(precision):
    """Debye-filled two-port line — exercises the ADE aux-state path
    (pole current + g + f_prev) at the selected precision."""
    grid = GridLines(
        x=np.linspace(-WIDTH_A / 2, WIDTH_A / 2, 11),
        y=np.linspace(-GAP_B / 2, GAP_B / 2, 6),
        z=np.linspace(0.0, LENGTH, 41),
    )
    model = DispersionModel.debye(2.0, 1.5, tau=1.0 / (2 * np.pi * 5e9))
    mesh = Mesh.from_grid(grid, background=Material.dispersive("debye", model))
    specs = [
        PortSpecMultiConductor(name="port1", plane=BoxFace.Z_MIN, n_modes=1),
        PortSpecMultiConductor(name="port2", plane=BoxFace.Z_MAX, n_modes=1),
    ]
    return AnalysisScatteringTD(
        mesh=mesh.with_boundary_conditions(
            {
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PEC",
                "zmax": "PEC",
            }
        ),
        ports=specs,
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
        precision=precision,
    )


def test_single_matches_double_on_dispersive_line():
    """The ADE pole-current path in single (float32 g/f_prev, complex64
    pole state) tracks double to the float32 floor — dispersive loss and
    phase are precision-independent (DD-094 aux-state follow-up)."""
    f_axis = np.linspace(2e9, F_MAX, 41)
    Sd = _build_dispersive("double").run(f_axis=f_axis, excited=["port1"])
    Ss = _build_dispersive("single").run(f_axis=f_axis, excited=["port1"])

    S21d, S21s = Sd.S("port2", "port1"), Ss.S("port2", "port1")
    # Insertion loss + phase through a lossy dispersive fill are physical
    # quantities: single must not move them beyond the float32 floor.
    assert np.max(np.abs(np.abs(S21s) - np.abs(S21d))) < 1e-4
    assert np.max(np.abs(S21s - S21d)) < 1e-4
    assert np.all(np.isfinite(S21s))
