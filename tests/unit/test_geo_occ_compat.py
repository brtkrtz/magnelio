"""Compatibility with pythonocc output-argument bindings."""

from types import SimpleNamespace

import pytest

from magnelio.geo._occ_compat import connect_edges_to_wires


def test_three_argument_wire_connection_binding(monkeypatch):
    shape_analysis = pytest.importorskip("OCC.Core.ShapeAnalysis")
    edges, wires = object(), object()
    calls = []

    def connect(actual_edges, tolerance, shared):
        calls.append((actual_edges, tolerance, shared))
        return wires

    monkeypatch.setattr(
        shape_analysis, "ShapeAnalysis_FreeBounds", SimpleNamespace(ConnectEdgesToWires=connect)
    )
    assert connect_edges_to_wires(edges, 1e-7) is wires
    assert calls == [(edges, 1e-7, False)]
