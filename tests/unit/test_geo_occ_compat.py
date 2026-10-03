"""Compatibility with pythonocc output-argument bindings."""

from types import SimpleNamespace

import pytest

from magnelio.geo._occ_compat import connect_edges_to_wires


def _edges(segments):
    pytest.importorskip("OCC.Core.BRepBuilderAPI")
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.TopTools import TopTools_HSequenceOfShape

    sequence = TopTools_HSequenceOfShape()
    for start, end in segments:
        sequence.Append(BRepBuilderAPI_MakeEdge(gp_Pnt(*start), gp_Pnt(*end)).Edge())
    return sequence


def _wire_edge_counts(wires):
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    counts = []
    for index in range(1, wires.Length() + 1):
        wire = wires.Value(index)
        assert wire.ShapeType() == TopAbs_WIRE
        explorer = TopExp_Explorer(wire, TopAbs_EDGE)
        count = 0
        while explorer.More():
            count += 1
            explorer.Next()
        ordered = BRepTools_WireExplorer(topods.Wire(wire))
        visited = 0
        while ordered.More():
            visited += 1
            ordered.Next()
        assert visited == count
        counts.append(count)
    return sorted(counts)


def test_wire_connection_with_installed_binding():
    edges = _edges([((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, 1, 0))])
    assert _wire_edge_counts(connect_edges_to_wires(edges, 1e-7)) == [2]


def test_three_argument_wire_connection_binding(monkeypatch):
    shape_analysis = pytest.importorskip("OCC.Core.ShapeAnalysis")
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
    from OCC.Core.TopTools import TopTools_HSequenceOfShape

    edges = _edges([((0, 0, 0), (1, 0, 0))])
    wires = TopTools_HSequenceOfShape()
    wires.Append(BRepBuilderAPI_MakeWire(edges.Value(1)).Wire())
    calls = []

    def connect(actual_edges, tolerance, shared):
        calls.append((actual_edges, tolerance, shared))
        return wires

    monkeypatch.setattr(
        shape_analysis, "ShapeAnalysis_FreeBounds", SimpleNamespace(ConnectEdgesToWires=connect)
    )
    assert connect_edges_to_wires(edges, 1e-7) is wires
    assert calls == [(edges, 1e-7, False)]


def test_edge_echo_fallback_preserves_loops_and_disconnected_chains(monkeypatch):
    shape_analysis = pytest.importorskip("OCC.Core.ShapeAnalysis")
    edges = _edges(
        [
            ((1, 0, 0), (1, 1, 0)),
            ((4, 0, 0), (5, 0, 0)),
            ((0, 1, 0), (0, 0, 0)),
            ((1, 1, 0), (0, 1, 0)),
            ((5, 0, 0), (5, 1, 0)),
            ((0, 0, 0), (1, 0, 0)),
        ]
    )
    monkeypatch.setattr(
        shape_analysis,
        "ShapeAnalysis_FreeBounds",
        SimpleNamespace(ConnectEdgesToWires=lambda actual, _tol, _shared: actual),
    )
    assert _wire_edge_counts(connect_edges_to_wires(edges, 1e-7)) == [2, 4]


def test_edge_echo_fallback_splits_branches(monkeypatch):
    shape_analysis = pytest.importorskip("OCC.Core.ShapeAnalysis")
    edges = _edges(
        [
            ((0, 0, 0), (1, 0, 0)),
            ((1, 0, 0), (2, 0, 0)),
            ((1, 0, 0), (1, 1, 0)),
        ]
    )
    monkeypatch.setattr(
        shape_analysis,
        "ShapeAnalysis_FreeBounds",
        SimpleNamespace(ConnectEdgesToWires=lambda actual, _tol, _shared: actual),
    )
    assert _wire_edge_counts(connect_edges_to_wires(edges, 1e-7)) == [1, 1, 1]


def test_edge_echo_fallback_respects_shared_vertices(monkeypatch):
    shape_analysis = pytest.importorskip("OCC.Core.ShapeAnalysis")
    edges = _edges([((0, 0, 0), (1, 0, 0)), ((1, 0, 0), (1, 1, 0))])
    monkeypatch.setattr(
        shape_analysis,
        "ShapeAnalysis_FreeBounds",
        SimpleNamespace(ConnectEdgesToWires=lambda actual, _tol, _shared: actual),
    )
    assert _wire_edge_counts(connect_edges_to_wires(edges, 1e-7, shared=True)) == [1, 1]


@pytest.mark.parametrize(("gap", "counts"), [(9e-8, [2]), (2e-7, [1, 1])])
def test_edge_echo_fallback_uses_connection_tolerance(monkeypatch, gap, counts):
    shape_analysis = pytest.importorskip("OCC.Core.ShapeAnalysis")
    edges = _edges([((0, 0, 0), (1, 0, 0)), ((1 + gap, 0, 0), (1 + gap, 1, 0))])
    monkeypatch.setattr(
        shape_analysis,
        "ShapeAnalysis_FreeBounds",
        SimpleNamespace(ConnectEdgesToWires=lambda actual, _tol, _shared: actual),
    )
    assert _wire_edge_counts(connect_edges_to_wires(edges, 1e-7)) == counts
