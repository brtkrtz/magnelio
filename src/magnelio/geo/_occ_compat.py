"""Adapters for pythonocc bindings with different output-argument conventions."""


def connect_edges_to_wires(edges, tolerance, shared=False):
    """Return connected wires with either supported pythonocc signature."""
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCC.Core.TopTools import TopTools_HSequenceOfShape

    connect = ShapeAnalysis_FreeBounds.ConnectEdgesToWires
    try:
        return connect(edges, tolerance, shared)
    except TypeError as exc:
        if "expected 4 arguments, got 3" not in str(exc):
            raise
    wires = TopTools_HSequenceOfShape()
    result = connect(edges, tolerance, shared, wires)
    return wires if result is None else result
