"""Result contract: RAM result and store-backed Project agree exactly.

The same tiny two-port parallel-plate run is executed twice — once
without a project (in-RAM ScatteringTDResult) and once streamed through
the project store (Project reader) — and the full result contract
(magnelio.analysis.result_interface.ScatteringResult) is asserted over
both with identical checks.
"""

import numpy as np
import pytest

from magnelio import AnalysisScatteringTD, Material, Mesh, MeshControl
from magnelio.analysis.result_interface import ScatteringResult
from magnelio.geo import Brick, GeometryModel
from magnelio.ports import PortWaveguide

pytest.importorskip("OCC.Core.BRepPrimAPI")

F_MAX = 20e9


def _analysis(project=None):
    a, b, L = 10e-3, 2e-3, 20e-3
    model = GeometryModel()
    model.add(
        Brick(
            origin=(-a / 2, -b / 2, -L / 2),
            size=(a, b, L),
            material=Material.from_isotropic(name="air", epsilon=1.0),
        )
    )
    model.add_port(PortWaveguide(name="p1", plane="zmin"))
    model.add_port(PortWaveguide(name="p2", plane="zmax"))
    mesh = Mesh.from_geometry(
        model,
        MeshControl(min_nodes_per_wavelength=8),
        f_max=F_MAX,
    )
    return AnalysisScatteringTD(
        mesh=mesh,
        f_max=F_MAX,
        verbose=False,
        backend="numpy",
        project=project,
        params={"a_mm": 10.0, "note": "contract"},
    )


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    ram = _analysis().run(excited=["p1"], energy_stop_db=40.0)
    proj_dir = tmp_path_factory.mktemp("contract") / "proj"
    store = _analysis(project=str(proj_dir)).run(
        excited=["p1"],
        energy_stop_db=40.0,
    )
    return {"ram": ram, "store": store}


@pytest.fixture(params=["ram", "store"])
def result(request, results):
    return results[request.param]


class TestContractShape:
    def test_satisfies_protocol(self, result):
        assert isinstance(result, ScatteringResult)

    def test_axes_and_channels(self, result):
        assert result.f_axis.ndim == 1
        assert ("p1", 0) in result.channels
        assert ("p2", 0) in result.channels
        assert result.excitations == (("p1", 0),)

    def test_settings_populated(self, result):
        s = result.settings
        assert s.f_max == pytest.approx(F_MAX)
        assert s.n_freq == 201
        assert s.dt is not None and s.dt > 0
        assert s.n_actual_steps is not None and s.n_actual_steps > 0
        assert s.port_model_used == "modal"

    def test_timing_populated(self, result):
        assert result.elapsed > 0.0
        assert result.started <= result.finished

    def test_phase_matches_s(self, result):
        s = result.S("p2", "p1")
        ph = result.phase("p2", "p1", deg=False, unwrap=False)
        np.testing.assert_allclose(ph, np.angle(s), rtol=0, atol=1e-12)
        ph_deg = result.phase("p2", "p1", deg=True, unwrap=False)
        np.testing.assert_allclose(ph_deg, np.degrees(np.angle(s)))

    def test_db_matches_s(self, result):
        s = result.S("p2", "p1")
        db = result.db("p2", "p1")
        np.testing.assert_allclose(
            db,
            20 * np.log10(np.maximum(np.abs(s), 1e-10)),
            atol=1e-9,
        )

    def test_custom_f_axis_recompute(self, result):
        f_axis = np.linspace(14e9, 18e9, 31)
        s = result.S("p2", "p1", f_axis=f_axis)
        assert s.shape == (31,)
        # The recompute agrees with the default axis where they overlap.
        s_default = result.S("p2", "p1")
        k_default = int(np.argmin(np.abs(result.f_axis - 16e9)))
        k_custom = int(np.argmin(np.abs(f_axis - result.f_axis[k_default])))
        if abs(f_axis[k_custom] - result.f_axis[k_default]) < 1e6:
            np.testing.assert_allclose(
                s[k_custom],
                s_default[k_default],
                rtol=1e-9,
            )

    def test_power_waves(self, result):
        # The TE fundamental cuts off near 14.8 GHz -> in-band f_ref.
        a = result.a("p1", excited=("p1", 0), f_ref=17e9)
        b = result.b("p2", excited=("p1", 0), f_ref=17e9)
        assert len(a.values) == len(b.values) > 0

    def test_power_wave_evanescent_f_ref_raises(self, result):
        with pytest.raises(ValueError, match="not real"):
            result.a("p1", excited=("p1", 0), f_ref=5e9)


class TestCrossImplementation:
    """RAM and store answers agree.

    The two results come from two *runs* (the in-RAM and the streamed
    execution paths are not bit-identical — measured max |dV| ~ 7e-15),
    so the comparison allows eps-level run divergence; the accessor
    *semantics* are pinned exactly by TestContractShape on each
    implementation.
    """

    def test_s_agrees(self, results):
        for pair in (("p2", "p1"), ("p1", "p1")):
            s_ram = results["ram"].S(*pair)
            s_store = results["store"].S(*pair)
            scale = float(np.max(np.abs(s_ram)))
            np.testing.assert_allclose(
                s_ram,
                s_store,
                rtol=1e-6,
                atol=1e-9 * scale,
            )

    def test_phase_agrees_in_band(self, results):
        f = results["ram"].f_axis
        band = f >= 16e9  # propagating, strong signal
        np.testing.assert_allclose(
            results["ram"].phase("p2", "p1")[band],
            results["store"].phase("p2", "p1")[band],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_f_axis_identical(self, results):
        np.testing.assert_array_equal(
            results["ram"].f_axis,
            results["store"].f_axis,
        )

    def test_deembed_agrees(self, results):
        # Exercises the stored line records (dt, line params, normal
        # dx, modes) against the in-RAM ones on a Klein-Gordon (q > 0)
        # channel; the shift factors must match exactly, so the only
        # tolerance is the run divergence already allowed on S itself.
        d = {"p1": 4e-3, "p2": 2e-3}
        de_ram = results["ram"].deembed(d)
        de_store = results["store"].deembed(d)
        f = np.asarray(de_ram.f_axis)
        band = f >= 16e9  # propagating, bounded shift factors
        scale = float(np.max(np.abs(de_ram.matrix[band])))
        np.testing.assert_allclose(
            de_ram.matrix[band],
            de_store.matrix[band],
            rtol=1e-6,
            atol=1e-9 * scale,
        )

    def test_store_records_run_settings(self, results):
        s = results["store"].settings
        assert s.energy_stop_db == pytest.approx(40.0)

    def test_store_params_roundtrip(self, results):
        assert results["store"].params == {"a_mm": 10.0, "note": "contract"}

    def test_store_mesh_carries_ports(self, results):
        labels = [p.name for p in results["store"].mesh.ports]
        assert labels == ["p1", "p2"]

    def test_touchstone_extension_must_match_export(self, results, tmp_path):
        """Only p1 was excited, so a .s2p is a mis-declared file."""
        with pytest.raises(ValueError, match="declares a 2-port network"):
            results["ram"].to_touchstone(tmp_path / "x.s2p")

    def test_touchstone_one_port_agrees_across_implementations(self, results, tmp_path):
        """Both readers reduce to the same measured reflection."""
        written = {}
        for key in ("ram", "store"):
            out = tmp_path / f"{key}.s1p"
            results[key].to_touchstone(out)
            data = np.loadtxt(out, comments=("!", "#"))
            assert data.shape[1] == 3
            written[key] = data[:, 1] + 1j * data[:, 2]
        scale = float(np.max(np.abs(written["ram"])))
        np.testing.assert_allclose(
            written["ram"],
            written["store"],
            rtol=1e-6,
            atol=1e-9 * scale,
        )

    def test_recompute_agrees(self, results):
        f_axis = np.linspace(16e9, 19e9, 31)
        s_ram = results["ram"].S("p2", "p1", f_axis=f_axis)
        s_store = results["store"].S("p2", "p1", f_axis=f_axis)
        scale = float(np.max(np.abs(s_ram)))
        np.testing.assert_allclose(
            s_ram,
            s_store,
            rtol=1e-6,
            atol=1e-9 * scale,
        )
