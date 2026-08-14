"""3b part 2 gates: unified LumpedElementOperator (DD-077 companion wiring).

Four gates pin the RLC-element port:

1. **Passive impedance** — a passive ``PortSpecLumped(element=…)`` on a
   TEM plate must present exactly the trapezoidal element impedance:
   ``DFT(V)/DFT(I) = −Z_trap(ω)`` with the bilinear frequency map
   ``jω̃ = j·(2/dt)·tan(ω·dt/2)``.  This is a *per-step identity* of the
   companion update (KVL: ``V_gap = −V_elem``), so it holds to DFT
   accuracy regardless of the line side — no transverse-coupling
   confounder.
2. **Recipe round-trip** — the element survives the DD-070 store recipe.
3. **state_dict round-trip** — companion state is checkpointed; a
   pre-unification checkpoint (no ``element`` group) still loads.
4. **Resume bit-exact** — an RLC-backed source port resumed across a
   checkpoint seam is bit-identical to one uninterrupted run (the L/C
   history crosses the seam losslessly).
"""

from __future__ import annotations

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Mesh
from magnelio.circuit import ParallelRLC, SeriesRLC
from magnelio.mesh import BoxFace
from magnelio.mesh.grid import GridLines
from magnelio.ports import PortSpecLumped, PortSpecMultiConductor

GAP, WIDTH, LENGTH = 5e-3, 16e-3, 60e-3

_BC = {"xmin": "PMC", "xmax": "PMC", "ymin": "PEC", "ymax": "PEC", "zmin": "PEC", "zmax": "PMC"}


def _plate_grid() -> GridLines:
    return GridLines(
        x=np.linspace(-WIDTH / 2, WIDTH / 2, 9),
        y=np.linspace(-GAP / 2, GAP / 2, 6),
        z=np.linspace(-LENGTH / 2, LENGTH / 2, 121),
    )


def _z_trap(element, omega: np.ndarray, dt: float) -> np.ndarray:
    """Exact trapezoidal element impedance at DFT frequencies."""
    jwt = 1j * (2.0 / dt) * np.tan(omega * dt / 2.0)
    if isinstance(element, SeriesRLC):
        z = np.zeros_like(jwt)
        if element.R is not None:
            z = z + element.R
        if element.L is not None:
            z = z + jwt * element.L
        if element.C is not None:
            z = z + 1.0 / (jwt * element.C)
        return z
    y = np.zeros_like(jwt)
    if element.R is not None:
        y = y + 1.0 / element.R
    if element.L is not None:
        y = y + 1.0 / (jwt * element.L)
    if element.C is not None:
        y = y + jwt * element.C
    return 1.0 / y


def _measure_load_impedance(element, f_axis: np.ndarray):
    grid = _plate_grid()
    ana = AnalysisScatteringTD(
        mesh=Mesh.from_grid(grid, boundary_conditions=dict(_BC)),
        ports=[
            PortSpecMultiConductor(name="m1", plane=BoxFace.Z_MIN, n_modes=1),
            PortSpecLumped(
                name="load",
                start=(0.0, -GAP / 2, grid.z[-3]),
                end=(0.0, GAP / 2, grid.z[-3]),
                Z0=50.0,
                element=element,
            ),
        ],
        f_max=6e9,
        verbose=False,
    )
    res = ana.run(f_axis=f_axis, excited=["m1"], total_time_steps=20000, energy_stop_db=None)
    V, I = res.signals[("m1", 0)][("load", 0)]
    Vf = V.at_frequencies(f_axis)
    If = I.at_frequencies(f_axis)
    return -Vf / If, V.dt


@pytest.mark.parametrize(
    "element",
    [
        SeriesRLC(R=75.0, L=5e-9, C=2e-12),
        SeriesRLC(L=3e-9),
        ParallelRLC(R=120.0, C=1.5e-12),
    ],
)
def test_passive_rlc_presents_trapezoidal_impedance(element):
    f_axis = np.array([0.8e9, 1.7e9, 3.1e9])
    z_meas, dt = _measure_load_impedance(element, f_axis)
    z_ref = _z_trap(element, 2.0 * np.pi * f_axis, dt)
    rel = np.abs(z_meas - z_ref) / np.abs(z_ref)
    assert np.all(rel < 1e-6), (
        f"{element!r}: measured Z = {z_meas} vs trapezoidal {z_ref} (rel {rel})"
    )


def test_rlc_element_recipe_roundtrip():
    from magnelio.analysis._recipe import _spec_from_dict, _spec_to_dict

    for element in (SeriesRLC(R=75.0, L=5e-9, C=2e-12), ParallelRLC(L=3e-9), None):
        spec = PortSpecLumped(
            name="p",
            start=(0.0, -GAP / 2, 0.0),
            end=(0.0, GAP / 2, 0.0),
            Z0=42.0,
            element=element,
        )
        back = _spec_from_dict(_spec_to_dict(spec))
        assert back.Z0 == spec.Z0
        if element is None:
            assert back.element is None
        else:
            assert type(back.element) is type(element)
            assert (back.element.R, back.element.L, back.element.C) == (
                element.R,
                element.L,
                element.C,
            )


def test_state_dict_roundtrip_and_legacy_checkpoint():
    from magnelio.ports._lumped.operator import PortOperatorLumped

    op = PortOperatorLumped(
        name="p",
        Z0=50.0,
        direction="y",
        flat_edge_indices=[0, 1],
        ijk_list=[(0, 0, 0), (0, 1, 0)],
        dl_list=[1e-3, 1e-3],
        beta_E=np.array([0.0, 0.0]),
        element=SeriesRLC(R=50.0, L=1e-9),
    )
    op.element._i, op.element._vL = 0.25, -3.5
    op._last_V, op._last_I = 1.5, 0.03
    sd = op.state_dict()
    assert sd["element"] == {"i": 0.25, "vL": -3.5, "vC": 0.0}

    op2 = PortOperatorLumped(
        name="p",
        Z0=50.0,
        direction="y",
        flat_edge_indices=[0, 1],
        ijk_list=[(0, 0, 0), (0, 1, 0)],
        dl_list=[1e-3, 1e-3],
        beta_E=np.array([0.0, 0.0]),
        element=SeriesRLC(R=50.0, L=1e-9),
    )
    op2.load_state_dict(sd)
    assert (op2.element._i, op2.element._vL) == (0.25, -3.5)
    assert (op2._last_V, op2._last_I) == (1.5, 0.03)

    # Pre-unification checkpoint: no "element" group — must still load.
    op2.load_state_dict({"last_V": 0.5, "last_I": 0.01})
    assert (op2._last_V, op2._last_I) == (0.5, 0.01)


def test_rlc_source_resume_bit_exact(tmp_path):
    """WP hard gate: RLC companion state crosses the checkpoint seam."""
    pytest.importorskip("OCC.Core.BRepPrimAPI")
    from magnelio import Material, MeshControl, open_project, resume
    from magnelio.geo import Brick, GeometryModel
    from magnelio.ports import PortWaveguide

    A, B, LZ = 10.0e-3, 5.0e-3, 20.0e-3
    f_max = 12.0e9

    def _analysis(project=None):
        model = GeometryModel(
            boundary_conditions={
                "xmin": "PMC",
                "xmax": "PMC",
                "ymin": "PEC",
                "ymax": "PEC",
                "zmin": "PMC",
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
            f_max=f_max,
        )
        z0 = mesh.grid.z[2]
        return AnalysisScatteringTD(
            mesh=mesh,
            ports=[
                PortSpecLumped(
                    name="feed",
                    start=(0.0, -B / 2, z0),
                    end=(0.0, B / 2, z0),
                    Z0=100.0,
                    element=SeriesRLC(R=100.0, L=2e-9, C=3e-12),
                ),
                PortWaveguide(name="port2", plane="zmax", n_modes=1),
            ],
            f_max=f_max,
            verbose=False,
            project=project,
            geometry=model,
        )

    n1, n_total = 120, 300
    ref = _analysis().run(
        excited=[("feed", 0)],
        energy_stop_db=None,
        total_time_steps=n_total,
    )
    ref_vi = {
        k: (v[0].values.copy(), v[1].values.copy()) for k, v in ref.signals[("feed", 0)].items()
    }

    p = tmp_path / "rlc"
    _analysis(project=p).run(
        excited=[("feed", 0)],
        energy_stop_db=None,
        total_time_steps=n1,
        checkpoint_interval=40,
    )
    assert open_project(p).runs["feed_mode0"]["n_steps"] == n1

    proj = resume(p, excited=("feed", 0), total_time_steps=n_total, verbose=False)
    assert proj.runs["feed_mode0"]["n_steps"] == n_total
    for chan, (rv, ri) in ref_vi.items():
        gv, gi = proj.signals[("feed", 0)][chan]
        assert np.array_equal(rv, gv.values), (
            f"{chan}: V not bit-exact across the RLC resume seam, "
            f"max|Δ|={float(np.max(np.abs(rv - gv.values))):.3e}"
        )
        assert np.array_equal(ri, gi.values), (
            f"{chan}: I not bit-exact across the RLC resume seam, "
            f"max|Δ|={float(np.max(np.abs(ri - gi.values))):.3e}"
        )
