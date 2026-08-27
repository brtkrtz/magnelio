"""Per-example pins of the grid planes and their provenance (DD-200).

Every gallery script is run up to its first ``Mesh.from_geometry`` (the
solver never starts); the resulting ``mesh.planes.as_dict()`` is compared
against ``data/gallery_planes/<dir>__<stem>.json``.  A mesher change that
adds, removes or moves a plane, changes what asked for it, or changes the
node count or cell sizes of an axis fails here with a readable diff — the
review question is then "is the new grid right?", not "which physics test
moved and why".

Regenerate after a deliberate change (and read the diff first)::

    MAGNELIO_UPDATE_PLANE_PINS=1 python -m pytest tests/unit/test_gallery_planes.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("OCC.Core.BRepPrimAPI")

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
PIN_DIR = Path(__file__).parent / "data" / "gallery_planes"
UPDATE = os.environ.get("MAGNELIO_UPDATE_PLANE_PINS") == "1"
UPDATE_CMD = "MAGNELIO_UPDATE_PLANE_PINS=1 python -m pytest tests/unit/test_gallery_planes.py"

# Rule: a script whose first mesh takes longer than ~10 s is excluded, with
# the reason.  Everything else in the gallery is pinned.
EXCLUDE = {
    "19_cassegrain_reflector.py": "first mesh takes ~370 s",
}
SCRIPTS = [
    p
    for p in sorted((EXAMPLES / "tutorials").glob("plot_*.py"))
    + sorted((EXAMPLES / "howto").glob("plot_*.py"))
    if p.name not in EXCLUDE
]


def _pin_path(script: Path) -> Path:
    return PIN_DIR / f"{script.parent.name}__{script.stem}.json"


class _FirstMeshBuilt(Exception):
    def __init__(self, mesh):
        super().__init__("first mesh built")
        self.mesh = mesh


def _first_mesh(script: Path, monkeypatch):
    """Run *script* until its first ``Mesh.from_geometry`` returns."""
    import magnelio.plots as plots
    import magnelio.post.plot_3d as plot_3d
    from magnelio.geo import GeometryModel
    from magnelio.mesh.mesher import Mesh

    original = Mesh.from_geometry.__func__

    def capture(cls, *args, **kwargs):
        raise _FirstMeshBuilt(original(cls, *args, **kwargs))

    def figure(*_args, **_kwargs):
        return plt.subplots()

    monkeypatch.setattr(Mesh, "from_geometry", classmethod(capture))
    # Anything that would open a window or render a section before the
    # mesh exists: the 3D viewer blocks headless, sections cost OCC time.
    monkeypatch.setattr(GeometryModel, "plot", lambda self, *a, **k: None)
    monkeypatch.setattr(GeometryModel, "plot_cross_section", lambda self, *a, **k: figure())
    monkeypatch.setattr(plots, "plot_cross_section", figure)
    monkeypatch.setattr(plots, "show_geometry", lambda *a, **k: None)
    monkeypatch.setattr(plot_3d, "show_geometry", lambda *a, **k: None)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.chdir(script.parent)

    source = script.read_text(encoding="utf-8")
    namespace = {"__name__": "__main__", "__file__": str(script)}
    try:
        exec(compile(source, str(script), "exec"), namespace)  # noqa: S102 — our own gallery
    except _FirstMeshBuilt as built:
        return built.mesh
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"{script.name}: input not available before the first mesh ({exc})")
    finally:
        plt.close("all")
    pytest.skip(f"{script.name}: builds no mesh")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _mm(v: float) -> str:
    return f"{v * 1e3:.6g} mm"


def _rel_differs(a, b, rtol=1e-3) -> bool:
    if a is None or b is None:
        return a is not b
    return abs(a - b) > rtol * max(abs(a), abs(b), 1e-300)


def _diff_planes(axis: str, expected: list, actual: list, gap: float) -> list[str]:
    tol = 0.5 * gap
    out = []
    unmatched_actual = list(range(len(actual)))
    unmatched_expected = []
    for e in expected:
        best, best_d = None, tol
        for j in unmatched_actual:
            d = abs(actual[j]["position"] - e["position"])
            if d <= best_d:
                best, best_d = j, d
        if best is None:
            unmatched_expected.append(e)
            continue
        unmatched_actual.remove(best)
        a = actual[best]
        where = f"{axis} = {_mm(e['position'])}"
        if e["sources"] != a["sources"]:
            out.append(f"  sources at {where}: {e['sources']} -> {a['sources']}")
        for key in ("singular", "domain_end"):
            if e.get(key) != a.get(key):
                out.append(f"  {key} at {where}: {e.get(key)} -> {a.get(key)}")
        if _rel_differs(e.get("h_fine"), a.get("h_fine")):
            out.append(f"  h_fine at {where}: {_mm(e['h_fine'])} -> {_mm(a['h_fine'])}")
        em, am = e.get("moved_to"), a.get("moved_to")
        if (em is None) != (am is None) or (em is not None and abs(em - am) > tol):
            out.append(f"  moved_to at {where}: {em} -> {am}")
    # unmatched: moved when an unmatched pair sits within ten gaps
    for e in unmatched_expected:
        near = [
            j for j in unmatched_actual if abs(actual[j]["position"] - e["position"]) <= 10 * gap
        ]
        if near:
            j = min(near, key=lambda j: abs(actual[j]["position"] - e["position"]))
            unmatched_actual.remove(j)
            a = actual[j]
            delta = (a["position"] - e["position"]) * 1e6
            out.append(
                f"  ~ {axis} {_mm(e['position'])} -> {_mm(a['position'])} "
                f"({delta:+.3g} µm) {a['sources']}"
            )
        else:
            out.append(f"  - {axis} {_mm(e['position'])} {e['sources']}")
    for j in unmatched_actual:
        a = actual[j]
        out.append(f"  + {axis} {_mm(a['position'])} {a['sources']}")
    return out


def _diff_leftovers(axis: str, key: str, expected: list, actual: list, gap: float) -> list[str]:
    def _key(r):
        return (round(r["position"] / gap) if gap else r["position"], tuple(r["sources"]))

    e_set = {_key(r): r for r in expected}
    a_set = {_key(r): r for r in actual}
    out = []
    for k in e_set.keys() - a_set.keys():
        out.append(f"  - {key} {axis} {_mm(e_set[k]['position'])} {list(k[1])}")
    for k in a_set.keys() - e_set.keys():
        out.append(f"  + {key} {axis} {_mm(a_set[k]['position'])} {list(k[1])}")
    return out


def diff_pins(expected: dict, actual: dict) -> list[str]:
    """Human-readable differences between two ``as_dict()`` records."""
    gap = float(expected["feature_gap"])
    out = []
    if _rel_differs(gap, float(actual["feature_gap"])):
        out.append(f"  feature_gap {gap:.6g} -> {actual['feature_gap']:.6g}")
    if expected["n_nodes"] != actual["n_nodes"]:
        out.append(f"  n_nodes {expected['n_nodes']} -> {actual['n_nodes']}")
    if expected["pml_cells"] != actual["pml_cells"]:
        out.append(f"  pml_cells {expected['pml_cells']} -> {actual['pml_cells']}")
    for axis in ("x", "y", "z"):
        e_ax, a_ax = expected["axes"][axis], actual["axes"][axis]
        out += _diff_planes(axis, e_ax["planes"], a_ax["planes"], gap)
        eb, ab = e_ax["h_bulk"], a_ax["h_bulk"]
        if len(eb) != len(ab):
            out.append(f"  h_bulk {axis}: {len(eb)} -> {len(ab)} intervals")
        else:
            for k, (h0, h1) in enumerate(zip(eb, ab)):
                if _rel_differs(h0, h1):
                    out.append(f"  h_bulk {axis}[{k}]: {_mm(h0)} -> {_mm(h1)}")
        for key in ("dropped", "absorbed", "unplaced"):
            out += _diff_leftovers(axis, key, e_ax[key], a_ax[key], gap)
    return out


# ---------------------------------------------------------------------------
# The pins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_gallery_planes_pinned(script: Path, monkeypatch):
    mesh = _first_mesh(script, monkeypatch)
    assert mesh.planes is not None
    actual = mesh.planes.as_dict()
    pin = _pin_path(script)
    if UPDATE:
        PIN_DIR.mkdir(parents=True, exist_ok=True)
        pin.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return
    if not pin.exists():
        pytest.fail(f"no pin for {script.name}; generate it with\n    {UPDATE_CMD}")
    expected = json.loads(pin.read_text(encoding="utf-8"))
    lines = diff_pins(expected, actual)
    assert not lines, (
        f"grid planes of {script.parent.name}/{script.name} changed "
        f"({len(lines)} difference(s)):\n" + "\n".join(lines) + f"\nIf intended: {UPDATE_CMD}"
    )


def test_every_pin_belongs_to_a_script():
    """A renamed or deleted script must not leave a stale pin behind."""
    if not PIN_DIR.exists():
        pytest.skip("no pins yet")
    expected = {_pin_path(s).name for s in SCRIPTS}
    stale = sorted(p.name for p in PIN_DIR.glob("*.json") if p.name not in expected)
    assert not stale, f"stale pins without a script: {stale}"


class TestDiff:
    """The diff names what changed; identical records give no lines."""

    @staticmethod
    def _rec(planes, gap=1e-6, n=None):
        return {
            "feature_gap": gap,
            "n_nodes": n or {"x": len(planes), "y": 2, "z": 2},
            "pml_cells": {},
            "axes": {
                "x": {
                    "planes": planes,
                    "h_bulk": [1e-3] * max(len(planes) - 1, 0),
                    "dropped": [],
                    "absorbed": [],
                    "unplaced": [],
                },
                "y": {"planes": [], "h_bulk": [], "dropped": [], "absorbed": [], "unplaced": []},
                "z": {"planes": [], "h_bulk": [], "dropped": [], "absorbed": [], "unplaced": []},
            },
        }

    @staticmethod
    def _plane(pos, sources=("face #0 Brick(air)",), **kw):
        d = {
            "position": pos,
            "sources": list(sources),
            "singular": False,
            "domain_end": False,
            "h_fine": 1e-4,
            "moved_to": None,
            "node": 0,
        }
        d.update(kw)
        return d

    def test_identical_is_silent(self):
        rec = self._rec([self._plane(0.0), self._plane(5e-3)])
        assert diff_pins(rec, json.loads(json.dumps(rec))) == []

    def test_added_removed_moved_and_sources(self):
        a = self._rec([self._plane(0.0), self._plane(2e-3), self._plane(5e-3)])
        b = self._rec(
            [
                self._plane(0.0, ("face #0 Brick(air)", "forced")),
                self._plane(2e-3 + 3e-6),
                self._plane(3e-3, ("edge #1 Cylinder(pec)",)),
                self._plane(5e-3),
            ],
            n={"x": 4, "y": 2, "z": 2},
        )
        lines = diff_pins(a, b)
        text = "\n".join(lines)
        assert "n_nodes" in text
        assert "sources at x = 0 mm" in text
        assert "~ x 2 mm -> 2.003 mm (+3 µm)" in text
        assert "+ x 3 mm ['edge #1 Cylinder(pec)']" in text
        assert "h_bulk x: 2 -> 3 intervals" in text

    def test_float_noise_is_ignored_but_a_snap_is_not(self):
        a = self._rec([self._plane(0.0), self._plane(5e-3)])
        b = self._rec([self._plane(1e-13), self._plane(5e-3 + 1e-13)])
        assert diff_pins(a, b) == []
        c = self._rec([self._plane(0.0), self._plane(5e-3 + 1e-6)])
        assert any(line.startswith("  ~ x 5 mm") for line in diff_pins(a, c))
