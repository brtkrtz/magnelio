"""``Project.monitor()``: a widget panel a background thread keeps current."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from magnelio.io.project import ProjectStore, open_project
from magnelio.mesh.grid import GridLines
from magnelio.mesh.mesher import Mesh


def _finished_project(tmp_path):
    grid = GridLines(
        x=np.linspace(0, 6e-3, 7),
        y=np.linspace(0, 4e-3, 5),
        z=np.linspace(0, 4e-3, 5),
    )
    store = ProjectStore.create(tmp_path / "proj", Mesh.from_grid(grid))
    store.register_planned_runs([("port1_mode0", {"excited": ["port1", 0]})])
    store._finalize_run("port1_mode0", 12, "done", stop_reason="energy", elapsed=0.3)
    return open_project(store.path)


def test_panel_shows_the_table_and_a_picture_and_stops(tmp_path):
    ipywidgets = pytest.importorskip("ipywidgets")
    proj = _finished_project(tmp_path)
    panel = proj.monitor(interval=0.05)
    assert isinstance(panel, ipywidgets.VBox)
    table, picture = panel.children
    assert isinstance(table, ipywidgets.HTML)
    assert "<table" in table.value
    assert "port1_mode0" in table.value
    assert picture.value[:8] == b"\x89PNG\r\n\x1a\n"
    # A finished project: the refresh thread delivers the final state and ends.
    panel._magnelio_thread.join(10.0)
    assert not panel._magnelio_thread.is_alive()
    panel.stop()
    panel.stop()


def test_rejects_a_bad_interval(tmp_path):
    pytest.importorskip("ipywidgets")
    with pytest.raises(ValueError, match="interval"):
        _finished_project(tmp_path).monitor(interval=0.0)


def test_names_the_extra_when_ipywidgets_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    with pytest.raises(ImportError, match=r"magnelio\[jupyter\]"):
        _finished_project(tmp_path).monitor()


class TestFollow:
    """``follow()`` redraws the summary in place — on each of its three surfaces."""

    def test_a_log_gets_whole_tables_and_the_project_back(self, tmp_path):
        import io

        proj = _finished_project(tmp_path)
        out = io.StringIO()
        assert proj.follow(interval=0.05, stream=out) is proj
        text = out.getvalue()
        assert text.count("Project ") == 1
        assert "port1_mode0" in text
        assert "\x1b[" not in text

    def test_a_terminal_redraws_over_its_own_lines(self, tmp_path):
        import io

        from magnelio.io.project import _InPlacePainter

        class Tty(io.StringIO):
            def isatty(self):
                return True

        stream = Tty()
        painter = _InPlacePainter(stream)
        proj = _finished_project(tmp_path)
        painter.paint(proj)
        first = stream.getvalue()
        painter.paint(proj)
        second = stream.getvalue()[len(first) :]
        assert second.startswith(f"\x1b[{first.count(chr(10))}A\x1b[J")

    def test_a_notebook_clears_the_cell_and_displays(self, tmp_path, monkeypatch):
        import io
        import types

        from magnelio.io.project import _InPlacePainter

        calls = []
        stub = types.SimpleNamespace(
            clear_output=lambda wait=False: calls.append(("clear", wait)),
            display=lambda obj: calls.append(("display", obj)),
        )
        monkeypatch.setitem(sys.modules, "IPython.display", stub)

        class OutStream(io.StringIO):
            pass

        OutStream.__module__ = "ipykernel.iostream"
        proj = _finished_project(tmp_path)
        painter = _InPlacePainter(OutStream())
        painter.paint(proj)
        assert calls == [("clear", True), ("display", proj)]

    def test_rejects_a_bad_interval(self, tmp_path):
        with pytest.raises(ValueError, match="interval"):
            _finished_project(tmp_path).follow(interval=-1)
