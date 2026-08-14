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

    def test_partial_matrix_raises(self, tmp_path):
        with pytest.raises(ValueError, match="never excited"):
            _partial_result().to_touchstone(tmp_path / "x.s2p")


class TestSkrf:
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
