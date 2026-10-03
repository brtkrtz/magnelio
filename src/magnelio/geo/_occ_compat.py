"""Adapters for pythonocc bindings with different output-argument conventions."""

from math import floor


def connect_edges_to_wires(edges, tolerance, shared=False):
    """Return connected wires with either supported pythonocc signature."""
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_WIRE
    from OCC.Core.TopTools import TopTools_HSequenceOfShape

    connect = ShapeAnalysis_FreeBounds.ConnectEdgesToWires
    try:
        wires = connect(edges, tolerance, shared)
    except TypeError as exc:
        if "expected 4 arguments, got 3" not in str(exc):
            raise
        wires = TopTools_HSequenceOfShape()
        result = connect(edges, tolerance, shared, wires)
        if result is not None:
            wires = result

    kinds = {wires.Value(i).ShapeType() for i in range(1, wires.Length() + 1)}
    if kinds <= {TopAbs_WIRE} and (wires.Length() or not edges.Length()):
        return wires
    if kinds == {TopAbs_EDGE} and wires.Length() == edges.Length():
        # pythonocc 8.0.1 returns the input edges instead of the output wires.
        return _connect_edges_without_binding(edges, tolerance, shared)
    raise RuntimeError("The CAD kernel returned an invalid wire sequence.")


def _connect_edges_without_binding(edges, tolerance, shared):
    """Build non-branching wires from endpoint-connected edges."""
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    from OCC.Core.ShapeExtend import ShapeExtend_WireData
    from OCC.Core.ShapeFix import ShapeFix_Wire
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer, topexp
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import TopTools_HSequenceOfShape

    edge_list = [topods.Edge(edges.Value(i)) for i in range(1, edges.Length() + 1)]
    vertices = [
        vertex
        for edge in edge_list
        for vertex in (topexp.FirstVertex(edge, True), topexp.LastVertex(edge, True))
    ]
    points = [BRep_Tool.Pnt(vertex) for vertex in vertices]
    parent = list(range(len(vertices)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    if shared:
        for i, vertex in enumerate(vertices):
            for j in range(i):
                if vertex.IsSame(vertices[j]):
                    parent[root(i)] = root(j)
                    break
    else:
        cell = max(tolerance, 1e-12)
        buckets = {}
        for i, point in enumerate(points):
            key = tuple(floor(coord / cell) for coord in (point.X(), point.Y(), point.Z()))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                            if point.Distance(points[j]) <= tolerance:
                                parent[root(i)] = root(j)
            buckets.setdefault(key, []).append(i)

    ends = [(root(2 * i), root(2 * i + 1)) for i in range(len(edge_list))]
    incident = {}
    for i, (start, end) in enumerate(ends):
        incident.setdefault(start, []).append((i, 0))
        incident.setdefault(end, []).append((i, 1))

    output = TopTools_HSequenceOfShape()
    unused = set(range(len(edge_list)))
    while unused:
        seed = min(unused)
        start = next((node for node in ends[seed] if len(incident[node]) != 2), ends[seed][0])
        path = []
        node = start
        while True:
            next_edges = [(i, side) for i, side in incident[node] if i in unused]
            if not next_edges:
                break
            i, side = min(next_edges)
            unused.remove(i)
            path.append(topods.Edge(edge_list[i].Reversed()) if side else edge_list[i])
            node = ends[i][1 - side]
            if node == start or len(incident[node]) != 2:
                break

        maker = BRepBuilderAPI_MakeWire()
        for edge in path:
            maker.Add(edge)
            if not maker.IsDone():
                break
        if maker.IsDone():
            wire = maker.Wire()
        else:
            data = ShapeExtend_WireData()
            for edge in path:
                data.Add(edge)
            fixer = ShapeFix_Wire()
            fixer.Load(data)
            fixer.SetPrecision(tolerance)
            fixer.SetMaxTolerance(tolerance)
            if not fixer.FixConnected():
                raise RuntimeError("The CAD kernel could not connect the edges into a wire.")
            wire = fixer.Wire()
        explorer = TopExp_Explorer(wire, TopAbs_EDGE)
        count = 0
        while explorer.More():
            count += 1
            explorer.Next()
        if count != len(path):
            raise RuntimeError("The CAD kernel lost edges while building a wire.")
        ordered = BRepTools_WireExplorer(topods.Wire(wire))
        traversed = 0
        while ordered.More():
            traversed += 1
            ordered.Next()
        if traversed != count:
            raise RuntimeError("The CAD kernel returned a non-traversable wire.")
        output.Append(wire)
    return output
