"""A repr says what an object is, how big, in what state — never its arrays.

The helpers in ``magnelio._repr`` render the summaries every result,
run and project prints; the classes are checked here for the one rule
that matters to a user at a prompt: no array content, and nothing
that scrolls.
"""

from __future__ import annotations

import datetime

import numpy as np

from magnelio._repr import (
    fmt_array,
    fmt_db,
    fmt_value,
    html_kv,
    html_table,
    kv_block,
    text_table,
)


class TestValues:
    def test_none_and_numbers(self):
        assert fmt_value(None) == "—"
        assert fmt_value(3.14159265) == "3.142"
        assert fmt_value(np.float32(2.5)) == "2.5"
        assert fmt_value(np.int64(7)) == "7"
        assert fmt_value(True) == "yes"

    def test_arrays_show_shape_not_content(self):
        text = fmt_value(np.zeros((201, 2, 2), dtype=complex))
        assert text == "complex128[201×2×2]"
        assert fmt_array(np.arange(3000.0)) == "float64[3000]"
        assert "0." not in text

    def test_channel_keys_and_stamps(self):
        assert fmt_value(("port1", 0)) == "port1:0"
        assert fmt_value([("p1", 0), ("p2", 1)]) == "p1:0, p2:1"
        stamp = datetime.datetime(2026, 9, 5, 8, 32, 12, tzinfo=datetime.timezone.utc)
        assert fmt_value(stamp) == "2026-09-05 08:32:12"
        assert fmt_db(-31.26) == "-31.3 dB"
        assert fmt_db(None) == "—"


class TestText:
    def test_kv_block_aligns_keys(self):
        text = kv_block("Thing", [("a", 1), ("longer", None)])
        lines = text.splitlines()
        assert lines[0] == "Thing"
        assert lines[1] == "  a       1"
        assert lines[2] == "  longer  —"

    def test_table_has_a_header_and_a_rule(self):
        text = text_table(["run", "steps"], [["p1", 4213], ["port2_mode0", 12]], align="lr")
        lines = text.splitlines()
        assert lines[0].startswith("run")
        assert set(lines[1].replace(" ", "")) == {"─"}
        assert lines[2].endswith(" 4213")
        assert lines[3].endswith("   12")

    def test_empty_table_keeps_its_header(self):
        text = text_table(["run", "steps"], [])
        assert text.splitlines()[0].startswith("run")


class TestHtml:
    def test_kv_and_table_escape_and_tabulate(self):
        page = html_kv("<Thing>", [("k", "<v>")])
        assert "<table" in page
        assert "&lt;Thing&gt;" in page
        assert "&lt;v&gt;" in page
        table = html_table(["run", "dB"], [["p1", -31.2]], caption="Runs", align="lr")
        assert table.count("<tr>") == 2
        assert "text-align:right" in table
        assert "Runs" in table


class TestResultReprs:
    """Every result class prints its size and state, never its arrays."""

    def _trace(self):
        trace = np.empty(5, dtype=[("step", int), ("time", float), ("energy", float)])
        trace["step"] = np.arange(5) * 100
        trace["time"] = trace["step"] * 1e-12
        trace["energy"] = [0.0, 1.0, 0.5, 0.1, 0.001]
        return trace

    def test_td_result(self):
        from magnelio.analysis import TDResult

        result = TDResult(
            excitations=(),
            dt=1e-12,
            n_steps=400,
            signals={},
            excitation_signals={},
            energy_trace=self._trace(),
            elapsed=1.5,
            name="run_1",
        )
        text = repr(result)
        assert text.startswith("TDResult 'run_1'")
        assert "-30.0 dB below peak" in text
        assert "400 (0.4 ns)" in text
        assert "array" not in text
        assert len(text) < 400
        assert "<table" in result._repr_html_()

    def test_scattering_result_and_s_matrix(self):
        from magnelio.analysis import ScatteringTDResult
        from magnelio.analysis.result_interface import RunSettings
        from magnelio.post.sparameter_result import SParameterResult
        from magnelio.signals.signal_1d import Signal1D

        f = np.linspace(8e9, 12e9, 201)
        s_params = SParameterResult(
            f_axis=f,
            channels=(("p1", 0), ("p2", 0)),
            excitations=(("p1", 0),),
            matrix=np.zeros((201, 2, 1), dtype=complex),
        )
        text = repr(s_params)
        assert "complex128[201×2×1]" in text
        assert "8–12 GHz (201 points)" in text
        assert "0.+0.j" not in text
        ref = Signal1D(t=np.arange(10) * 1e-12, values=np.zeros(10), dt=1e-12, label="excitation")
        result = ScatteringTDResult(
            s_params=s_params,
            signals={},
            reference_signal=ref,
            dt=1e-12,
            n_actual_steps=4213,
            port_model_used="modal",
            settings=RunSettings(stop_reason="energy", f_max=12e9),
            energy_traces={("p1", 0): self._trace()},
            elapsed=2.0,
        )
        text = repr(result)
        assert text.startswith("ScatteringTDResult")
        assert "p1:0" in text
        assert "4213" in text
        assert "energy" in text
        assert "-30.0 dB below peak" in text
        assert "array" not in text
        assert len(text) < 500
        assert "<table" in result._repr_html_()

    def test_run_settings_lists_only_what_was_recorded(self):
        from magnelio.analysis.result_interface import RunSettings

        assert repr(RunSettings(f_max=12e9, stop_reason="energy")) == (
            "RunSettings(f_max=12000000000.0, stop_reason='energy')"
        )

    def test_eigenmode_result_html(self):
        from magnelio.solver.eigenmode_result import EigenmodeResult

        result = EigenmodeResult(frequencies=np.array([1.2e9, 3.4e9]), modes=[], mesh=None)
        page = result._repr_html_()
        assert page.count("<tr>") == 3
        assert "3.4" in page
