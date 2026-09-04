"""Progress output: who reports, where it goes, and how it is shaped.

The reporting used to be eleven hand-written ``print`` sites with no
test at all, so a refactor could silence the solver without a single
failure.  These tests pin the policy, not the wording: which setting
wins, that a terminal gets one overwritten line and a log file gets
whole ones, and that a worker process stays quiet.
"""

from __future__ import annotations

import io
import time

import pytest

import magnelio as mio
from magnelio import geo, ports
from magnelio._progress import (
    Reporter,
    current_reporter,
    get_verbosity,
    set_verbosity,
)


class FakeTTY(io.StringIO):
    """A stream that claims to be a terminal."""

    def isatty(self) -> bool:
        return True


class NoIsatty:
    """A stream that cannot answer whether it is a terminal."""

    def __init__(self) -> None:
        self.text = ""

    def write(self, s: str) -> int:
        self.text += s
        return len(s)

    def flush(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _isolate_reporter_state():
    """Each test starts from no global setting and no active reporter.

    A reporter registers itself as current for the duration of its
    operation and restores its predecessor when it ends.  A test that
    builds one and never ends it would therefore hand its leftover to
    the next test's ``reset``.
    """
    from magnelio._progress import _current

    before = get_verbosity()
    token = _current.set(None)
    yield
    _current.reset(token)
    set_verbosity(before)


class TestVerbosityResolution:
    def test_local_true_overrides_global_off(self):
        set_verbosity(False)
        assert Reporter("x", True).enabled

    def test_local_false_overrides_global_on(self):
        set_verbosity(True)
        assert not Reporter("x", False).enabled

    def test_none_follows_the_global_setting(self):
        set_verbosity(False)
        assert not Reporter("x").enabled
        set_verbosity(True)
        assert Reporter("x").enabled

    def test_a_worker_process_stays_silent(self, monkeypatch):
        monkeypatch.setattr("magnelio._progress._in_worker_process", lambda: True)
        assert not Reporter("x", True).enabled

    def test_disabled_reporter_writes_nothing(self):
        s = io.StringIO()
        rep = Reporter("x", False, stream=s)
        rep.stage("a")
        rep.step(1, 2)
        rep.tick(3)
        rep.line("l")
        rep.final("f")
        rep.note("n")
        rep.finish("done")
        assert s.getvalue() == ""


@pytest.fixture
def report_every_phase(monkeypatch):
    """Drop the short-phase threshold, so line shape can be tested alone."""
    monkeypatch.setattr("magnelio._progress._MIN_REPORTED", 0.0)


class TestLineShape:
    def test_a_log_gets_whole_lines_and_no_carriage_returns(self, report_every_phase):
        s = io.StringIO()
        rep = Reporter("mesh", True, stream=s)
        rep.stage("grid lines")
        rep.finish("4 x 5 x 6 cells")
        out = s.getvalue()
        assert "\r" not in out
        assert out.endswith("\n")
        assert "mesh | grid lines | done" in out

    def test_a_terminal_gets_one_overwritten_line(self, report_every_phase):
        s = FakeTTY()
        rep = Reporter("mesh", True, stream=s)
        rep.stage("grid lines")
        rep.finish()
        out = s.getvalue()
        assert out.startswith("\r")
        # The closing line replaces the running one rather than being
        # appended below a half-written row.
        assert out.count("\n") == 1

    def test_a_stream_without_isatty_is_treated_as_a_log(self):
        s = NoIsatty()
        rep = Reporter("mesh", True, stream=s)
        rep.stage("a")
        rep.finish()
        assert "\r" not in s.text

    def test_the_label_prefixes_every_line(self):
        s = io.StringIO()
        rep = Reporter("ports", True, stream=s)
        rep.note("hello")
        assert s.getvalue().strip().startswith("ports | ")


class TestCadence:
    def test_a_tight_loop_does_not_print_every_step(self):
        s = io.StringIO()
        rep = Reporter("x", True, stream=s)
        rep.stage("phase")
        for i in range(1, 1000):
            rep.step(i, 1000)  # far below the log interval
        n_progress = sum(1 for ln in s.getvalue().splitlines() if "/1000" in ln)
        assert n_progress <= 2

    def test_the_last_step_always_prints(self):
        s = io.StringIO()
        rep = Reporter("x", True, stream=s)
        rep.stage("phase")
        rep.step(5, 5)
        assert "5/5 (100 %)" in s.getvalue()

    def test_a_terminal_refreshes_more_often_than_a_log(self):
        tty, log = FakeTTY(), io.StringIO()
        for stream in (tty, log):
            rep = Reporter("x", True, stream=stream)
            rep.stage("phase")
            for i in range(3):
                time.sleep(0.11)
                rep.step(i, 100)
        assert tty.getvalue().count("/100") > log.getvalue().count("/100")


class FakeOutStream(io.StringIO):
    """A stream shaped like the one a Jupyter kernel gives its cells."""


# Only the module name marks the kernel's stream; ipykernel is not a
# test dependency.
FakeOutStream.__module__ = "ipykernel.iostream"


class TestNotebookStream:
    """A notebook cell redraws a carriage-returned line like a terminal.

    ipykernel's OutStream is not a TTY, so the first policy filed it as
    a log and a notebook user waited 30 s for a line.  It is now an
    in-place stream with its own cadence between terminal and log.
    """

    def test_is_recognised_by_its_module(self):
        from magnelio._progress import _is_notebook_stream

        assert _is_notebook_stream(FakeOutStream())
        assert not _is_notebook_stream(io.StringIO())
        assert not _is_notebook_stream(FakeTTY())

    def test_writes_in_place(self):
        s = FakeOutStream()
        rep = Reporter("x", True, stream=s)
        rep.stage("phase")
        rep.step(1, 2)
        assert "\r" in s.getvalue()
        rep.finish("summary")
        assert s.getvalue().endswith("summary\n")

    def test_refreshes_faster_than_a_log_and_slower_than_a_terminal(self):
        from magnelio._progress import _LOG_INTERVAL, _NOTEBOOK_INTERVAL, _TTY_INTERVAL

        assert _TTY_INTERVAL < _NOTEBOOK_INTERVAL < _LOG_INTERVAL
        assert Reporter("x", True, stream=FakeOutStream())._interval == _NOTEBOOK_INTERVAL
        assert Reporter("x", True, stream=io.StringIO())._interval == _LOG_INTERVAL


class TestPhaseContext:
    def test_a_raising_phase_closes_its_line(self):
        s = FakeTTY()
        rep = Reporter("x", True, stream=s)
        with pytest.raises(ValueError), rep.phase("boom"):
            raise ValueError("boom")
        assert s.getvalue().endswith("\n")

    def test_a_phase_reports_its_duration(self, report_every_phase):
        s = io.StringIO()
        rep = Reporter("x", True, stream=s)
        with rep.phase("work"):
            pass
        assert "done (" in s.getvalue()


def _waveguide_model():
    a, b, length = 22.86e-3, 10.16e-3, 20e-3
    model = mio.GeometryModel(background="pec", boundary_conditions=mio.BoundaryConditions())
    model.add(geo.Brick(origin=(-a / 2, -b / 2, 0), size=(a, b, length), material="air"))
    model.add_port(ports.PortWaveguide(name="p1", plane="zmin"))
    model.add_port(ports.PortWaveguide(name="p2", plane="zmax"))
    return model, b


class TestOperationsReport:
    def test_mesh_build_names_its_phases(self, capsys):
        model, b = _waveguide_model()
        mio.Mesh.from_geometry(
            model, mio.MeshControl(max_cell_size=b / 4), f_max=12e9, verbose=True
        )
        out = capsys.readouterr().out
        assert "mesh | " in out
        for phase in ("feature planes", "grid lines", "materials"):
            assert phase in out

    def test_mesh_build_is_silent_when_told_to_be(self, capsys):
        model, b = _waveguide_model()
        mio.Mesh.from_geometry(
            model, mio.MeshControl(max_cell_size=b / 4), f_max=12e9, verbose=False
        )
        assert "mesh | " not in capsys.readouterr().out

    def test_mesh_build_follows_the_global_setting(self, capsys):
        model, b = _waveguide_model()
        set_verbosity(False)
        mio.Mesh.from_geometry(model, mio.MeshControl(max_cell_size=b / 4), f_max=12e9)
        assert "mesh | " not in capsys.readouterr().out

    def test_solve_ports_names_the_lanczos_and_each_port(self, capsys):
        model, b = _waveguide_model()
        mesh = mio.Mesh.from_geometry(
            model, mio.MeshControl(max_cell_size=b / 4), f_max=12e9, verbose=False
        )
        mio.AnalysisScatteringTD(mesh=mesh, f_min=8e9, verbose=True).solve_ports()
        out = capsys.readouterr().out
        assert "CFL eigenvalue" in out
        assert "port 'p1' (1/2)" in out
        assert "port 'p2' (2/2)" in out

    def test_analysis_verbose_false_silences_the_setup(self, capsys):
        model, b = _waveguide_model()
        mesh = mio.Mesh.from_geometry(
            model, mio.MeshControl(max_cell_size=b / 4), f_max=12e9, verbose=False
        )
        mio.AnalysisScatteringTD(mesh=mesh, f_min=8e9, verbose=False).solve_ports()
        assert capsys.readouterr().out == ""


class TestShortPhases:
    """A phase too short to matter does not get a line of its own."""

    def test_a_short_phase_reports_no_duration(self):
        s = io.StringIO()
        rep = Reporter("mesh", True, stream=s)
        rep.stage("grid lines")
        rep.finish()
        assert "done (" not in s.getvalue()

    def test_a_short_phase_leaves_no_line_on_a_terminal(self):
        s = FakeTTY()
        rep = Reporter("mesh", True, stream=s)
        rep.stage("grid lines")
        rep.finish()
        # The announcement is erased rather than committed to the
        # scrollback; what remains carries no phase name.
        assert "grid lines" not in s.getvalue().split("\r")[-1]

    def test_a_long_phase_still_reports(self, monkeypatch):
        monkeypatch.setattr("magnelio._progress._MIN_REPORTED", 0.0)
        s = io.StringIO()
        rep = Reporter("mesh", True, stream=s)
        rep.stage("materials")
        rep.finish()
        assert "materials | done (" in s.getvalue()

    def test_the_summary_carries_the_total_when_it_is_worth_it(self, monkeypatch):
        monkeypatch.setattr("magnelio._progress._MIN_REPORTED", 0.0)
        s = io.StringIO()
        rep = Reporter("mesh", True, stream=s)
        rep.finish("14 x 7 x 4 cells")
        assert "s total)" in s.getvalue()

    def test_a_quick_operation_omits_the_total(self):
        s = io.StringIO()
        rep = Reporter("mesh", True, stream=s)
        rep.finish("14 x 7 x 4 cells")
        out = s.getvalue()
        assert "14 x 7 x 4 cells" in out
        assert "total" not in out


class TestOpenEndedCount:
    def test_advance_accumulates_across_calls(self):
        s = io.StringIO()
        rep = Reporter("mesh", True, stream=s)
        rep.stage("conformal cells")
        for _ in range(5):
            rep._last_emit = 0.0  # defeat the cadence for the test
            rep.advance(detail="sections")
        assert "5 sections" in s.getvalue()

    def test_a_new_phase_restarts_the_count(self):
        s = io.StringIO()
        rep = Reporter("mesh", True, stream=s)
        rep.stage("a")
        rep._last_emit = 0.0
        rep.advance(7)
        rep.stage("b")
        rep._last_emit = 0.0
        rep.advance(1)
        assert s.getvalue().rstrip().endswith("| 1")


class TestCurrentReporter:
    """Deep call levels find the reporter without a threaded parameter."""

    def test_the_active_reporter_is_reachable(self):
        rep = Reporter("mesh", True, stream=io.StringIO())
        assert current_reporter() is rep
        rep.finish()

    def test_it_is_none_once_the_operation_ends(self):
        Reporter("mesh", True, stream=io.StringIO()).finish()
        assert current_reporter() is None

    def test_a_disabled_reporter_is_not_offered(self):
        rep = Reporter("mesh", False, stream=io.StringIO())
        assert current_reporter() is None
        rep.finish()

    def test_nesting_restores_the_outer_reporter(self):
        outer = Reporter("refine", True, stream=io.StringIO())
        inner = Reporter("mesh", True, stream=io.StringIO())
        assert current_reporter() is inner
        inner.finish()
        assert current_reporter() is outer
        outer.finish()


class TestLabels:
    def test_sigma_carries_the_frequency_it_targets(self):
        from magnelio.solver._eigenmode_3d import _sigma_label

        label = _sigma_label((2 * 3.141592653589793 * 7.14e9) ** 2)
        assert "7.14 GHz" in label

    def test_a_non_positive_sigma_reports_no_frequency(self):
        from magnelio.solver._eigenmode_3d import _sigma_label

        assert "GHz" not in _sigma_label(0.0)

    def test_a_single_port_is_not_counted(self):
        from magnelio.analysis.time_domain import _port_stage

        assert _port_stage("feed", 1, 1) == "port 'feed'"

    def test_several_ports_are_counted(self):
        from magnelio.analysis.time_domain import _port_stage

        assert _port_stage("p1", 1, 2) == "port 'p1' (1/2)"
