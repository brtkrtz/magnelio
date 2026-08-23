"""Store schema v1.0 hard validation + Touchstone/scikit-rf export."""

import json

import numpy as np
import pytest

from magnelio import open_project
from magnelio.io._schema import SCHEMA_VERSION, ProjectSchemaError
from magnelio.post import SParameterResult


class TestSchemaValidation:
    def test_current_version_is_one_dot_zero(self):
        assert SCHEMA_VERSION == "1.0"

    def test_old_project_json_raises(self, tmp_path):
        (tmp_path / "project.json").write_text(
            json.dumps(
                {
                    "schema_version": "2.0",
                    "setup": {},
                    "runs": {},
                }
            )
        )
        with pytest.raises(ProjectSchemaError, match="re-run"):
            open_project(tmp_path).meta

    def test_unversioned_project_json_raises(self, tmp_path):
        (tmp_path / "project.json").write_text(json.dumps({"runs": {}}))
        with pytest.raises(ProjectSchemaError, match="not supported"):
            open_project(tmp_path).meta


def _complete_result(n_freq=5):
    f = np.linspace(1e9, 5e9, n_freq)
    channels = (("p1", 0), ("p2", 0))
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(n_freq, 2, 2)) + 1j * rng.normal(size=(n_freq, 2, 2))
    return SParameterResult(
        f_axis=f,
        channels=channels,
        excitations=channels,
        matrix=matrix,
    )


def _partial_result(n_freq=5):
    f = np.linspace(1e9, 5e9, n_freq)
    rng = np.random.default_rng(7)
    return SParameterResult(
        f_axis=f,
        channels=(("p1", 0), ("p2", 0)),
        excitations=(("p1", 0),),
        matrix=rng.normal(size=(n_freq, 2, 1)) * (1 + 0j),
    )


def _multimode_result(n_freq=5):
    """Two 3-mode ports, only mode 0 excited on each (issue #3)."""
    f = np.linspace(1e9, 5e9, n_freq)
    channels = tuple((p, m) for p in ("p1", "p2") for m in range(3))
    excitations = (("p1", 0), ("p2", 0))
    matrix = (np.arange(n_freq * 6 * 2).reshape(n_freq, 6, 2) + 1) * (1 + 0j)
    return SParameterResult(
        f_axis=f,
        channels=channels,
        excitations=excitations,
        matrix=matrix,
    )


class TestTouchstone:
    def test_roundtrip_two_port(self, tmp_path):
        res = _complete_result()
        path = tmp_path / "device.s2p"
        res.to_touchstone(path)
        text = path.read_text()
        assert "# Hz S RI R 50" in text
        assert "! port 1 = channel 'p1' mode 0" in text
        data = np.array(
            [
                [float(x) for x in line.split()]
                for line in text.splitlines()
                if line and not line.startswith(("!", "#"))
            ]
        )
        assert data.shape == (5, 1 + 8)
        np.testing.assert_allclose(data[:, 0], res.f_axis)
        # Touchstone two-port column order: S11 S21 S12 S22.
        s11 = data[:, 1] + 1j * data[:, 2]
        s21 = data[:, 3] + 1j * data[:, 4]
        s12 = data[:, 5] + 1j * data[:, 6]
        np.testing.assert_allclose(s11, res.matrix[:, 0, 0], rtol=1e-10)
        np.testing.assert_allclose(s21, res.matrix[:, 1, 0], rtol=1e-10)
        np.testing.assert_allclose(s12, res.matrix[:, 0, 1], rtol=1e-10)

    def test_unexcited_channels_are_reduced_away(self, tmp_path):
        """Excited channels form a fully measured square sub-matrix."""
        res = _multimode_result()
        assert res.export_channels() == (("p1", 0), ("p2", 0))
        out = tmp_path / "x.s2p"
        res.to_touchstone(out)
        data = np.loadtxt(out, comments=("!", "#"))
        assert data.shape == (5, 1 + 8)
        s11 = data[:, 1] + 1j * data[:, 2]
        s21 = data[:, 3] + 1j * data[:, 4]
        s12 = data[:, 5] + 1j * data[:, 6]
        s22 = data[:, 7] + 1j * data[:, 8]
        np.testing.assert_allclose(s11, res.matrix[:, 0, 0], rtol=1e-10)
        np.testing.assert_allclose(s21, res.matrix[:, 3, 0], rtol=1e-10)
        np.testing.assert_allclose(s12, res.matrix[:, 0, 1], rtol=1e-10)
        np.testing.assert_allclose(s22, res.matrix[:, 3, 1], rtol=1e-10)

    def test_one_port_reflection_export(self, tmp_path):
        """A single excited channel is a valid .s1p, not an error."""
        res = _partial_result()
        out = tmp_path / "refl.s1p"
        res.to_touchstone(out)
        data = np.loadtxt(out, comments=("!", "#"))
        assert data.shape == (5, 3)
        np.testing.assert_allclose(data[:, 1] + 1j * data[:, 2], res.matrix[:, 0, 0], rtol=1e-10)

    def test_explicit_channel_selection(self, tmp_path):
        """channels= cuts a chosen sub-network out of a wider result."""
        res = _complete_result()
        out = tmp_path / "p2.s1p"
        res.to_touchstone(out, channels=["p2"])
        data = np.loadtxt(out, comments=("!", "#"))
        np.testing.assert_allclose(data[:, 1] + 1j * data[:, 2], res.matrix[:, 1, 1], rtol=1e-10)

    def test_selecting_an_unexcited_channel_raises(self, tmp_path):
        with pytest.raises(ValueError, match="never excited"):
            _multimode_result().to_touchstone(tmp_path / "x.s2p", channels=[("p1", 1)])

    def test_extension_port_count_must_agree(self, tmp_path):
        with pytest.raises(ValueError, match="declares a 6-port network"):
            _multimode_result().to_touchstone(tmp_path / "x.s6p")
        assert not (tmp_path / "x.s6p").exists()

    def test_missing_extension_is_filled_in(self, tmp_path):
        _multimode_result().to_touchstone(tmp_path / "wr90")
        assert (tmp_path / "wr90.s2p").exists()

    def test_non_touchstone_extension_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not a Touchstone extension"):
            _complete_result().to_touchstone(tmp_path / "x.txt")

    def test_extension_is_case_insensitive(self, tmp_path):
        _complete_result().to_touchstone(tmp_path / "x.S2P")
        assert (tmp_path / "x.S2P").exists()


class TestSkrf:
    def test_to_skrf_reduces_to_excited_channels(self):
        skrf = pytest.importorskip("skrf")
        ntw = _multimode_result().to_skrf()
        assert isinstance(ntw, skrf.Network)
        assert ntw.nports == 2
        assert ntw.port_names == ["p1:0", "p2:0"]

    def test_to_skrf_network(self):
        skrf = pytest.importorskip("skrf")
        res = _complete_result()
        ntw = res.to_skrf(name="dut")
        assert isinstance(ntw, skrf.Network)
        assert ntw.s.shape == (5, 2, 2)
        np.testing.assert_allclose(ntw.s, res.matrix)
        np.testing.assert_allclose(ntw.f, res.f_axis)

    def test_partial_matrix_raises(self):
        pytest.importorskip("skrf")
        with pytest.raises(ValueError, match="never excited"):
            _partial_result().to_skrf()
