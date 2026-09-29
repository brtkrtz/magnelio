"""Independent regions and intersection curves from CAD partitioning."""

from __future__ import annotations

from magnelio.geo._axes import normalize_axis
from magnelio.geo._scaling import choose_scale, union_boxes
from magnelio.geo._validate import finite
from magnelio.geo.shape import Solid


def _arguments(source, cutter, normal, position, operation):
    from magnelio.geo._sheet import Sheet

    if not isinstance(source, (Solid, Sheet)):
        raise TypeError(f"{operation} requires a Solid or Sheet receiver.")
    if cutter is not None:
        if normal is not None or position is not None:
            raise ValueError(f"{operation} accepts a cutter or a plane, not both.")
        if not isinstance(cutter, (Solid, Sheet)):
            raise TypeError(f"{operation} cutter must be a Solid or Sheet.")
        return cutter, None
    if normal is None or position is None:
        raise ValueError(f"{operation} needs a cutter or both normal and position.")
    return None, (normalize_axis(normal, f"{operation} normal"), finite(position, "position"))


def _plane(plane, scale):
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCC.Core.gp import gp_Dir, gp_Pln, gp_Pnt

    normal, position = plane
    return BRepBuilderAPI_MakeFace(
        gp_Pln(gp_Pnt(*(position * scale * n for n in normal)), gp_Dir(*normal))
    ).Face()


def _pieces(shape, kind):
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods

    from magnelio.geo.topology import _kind

    explorer = TopExp_Explorer(shape, _kind(kind))
    pieces = []
    while explorer.More():
        pieces.append(getattr(topods, kind.title())(explorer.Current()))
        explorer.Next()
    return pieces


class _PartitionSolid(Solid):
    def __init__(self, shape, scale, bounds, source, cutter, plane, selections):
        self._shape = shape
        self._source_scale = scale
        self._bounds = bounds
        self._source = source
        self._cutter = cutter
        self._plane = plane
        self.material = source.material
        self.name = source.name
        self.color = getattr(source, "color", None)
        self._occ_shape_cache = {scale: shape}
        if selections:
            self._topology_inputs = (source,)
            self._topology_names_cache = {scale: selections}

    def _occ_shape(self, scale=1.0):
        if scale not in self._occ_shape_cache:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
            from OCC.Core.gp import gp_Pnt, gp_Trsf

            from magnelio.geo.topology import TopologyEvolutionError, _cast

            transform = gp_Trsf()
            transform.SetScale(gp_Pnt(0, 0, 0), scale / self._source_scale)
            builder = BRepBuilderAPI_Transform(self._shape, transform, True)
            self._occ_shape_cache[scale] = builder.Shape()
            if hasattr(self, "_topology_inputs"):
                selections = {}
                for key, (members, plural) in self._topology_names_cache[
                    self._source_scale
                ].items():
                    kind, _ = key
                    moved = []
                    for member in members:
                        successors = builder.Modified(member)
                        if successors.Size() != 1:
                            raise TopologyEvolutionError(
                                "partition: named topology has no unique scaled successor."
                            )
                        moved.append(_cast(successors.First(), kind))
                    selections[key] = (tuple(moved), plural)
                self._topology_names_cache[scale] = selections
        return self._occ_shape_cache[scale]

    def _analytic_bbox(self):
        return self._bounds


def partition(source, cutter=None, *, normal=None, position=None):
    """Split a solid or sheet into independently owned connected regions."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Splitter

    from magnelio.geo._occ_backend import _shape_list, bounding_box, keep_operands_intact
    from magnelio.geo._sheet import Profile
    from magnelio.geo._topology_history import _evolve, has_names, names
    from magnelio.geo.surfaces import Surface
    from magnelio.geo.topology import TopologyEvolutionError, _detached_class, _index, _is_planar

    cutter, plane = _arguments(source, cutter, normal, position, "partition")
    bounds = source._analytic_bbox()
    if cutter is not None:
        bounds = union_boxes((bounds, cutter._analytic_bbox()))
    scale = choose_scale(*bounds)
    source_shape = source._occ_shape(scale)
    tool_shape = cutter._occ_shape(scale) if cutter is not None else _plane(plane, scale)
    builder = BRepAlgoAPI_Splitter()
    builder.SetArguments(_shape_list((source_shape,)))
    builder.SetTools(_shape_list((tool_shape,)))
    keep_operands_intact(builder)
    builder.Build()
    if not builder.IsDone():
        raise RuntimeError("partition: the CAD kernel could not split the geometry.")
    kind = "solid" if isinstance(source, Solid) else "face"
    raw_pieces = _pieces(builder.Shape(), kind)
    if not raw_pieces:
        raise RuntimeError("partition: the CAD kernel returned no source regions.")
    selections = {}
    if has_names(source):
        selections = _evolve(
            source, builder.Shape(), [(source, names(source, scale))], [(builder, builder.Shape())]
        )
    output = []
    for raw in raw_pieces:
        piece_bounds = bounding_box(raw, scale=scale)
        if kind == "solid":
            member_names = {}
            for key, (members, plural) in selections.items():
                index = _index(raw, key[0])
                retained = tuple(member for member in members if index.Contains(member))
                if retained:
                    member_names[key] = (retained, plural)
            output.append(
                _PartitionSolid(raw, scale, piece_bounds, source, cutter, plane, member_names)
            )
        else:
            category = Profile if _is_planar(raw) else Surface
            output.append(_detached_class(category)(raw, scale, piece_bounds, source.material))
    if selections:
        for key, (_, plural) in selections.items():
            if (
                not plural
                and sum(
                    key in getattr(piece, "_topology_names_cache", {}).get(scale, {})
                    for piece in output
                )
                != 1
            ):
                raise TopologyEvolutionError(
                    "partition: a singular named selection has no unique owner region."
                )
    return tuple(output)


def section(source, cutter=None, *, normal=None, position=None, filled=False):
    """Return independent intersection curves or explicit planar profiles."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Section
    from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCC.Core.TopTools import TopTools_HSequenceOfShape

    from magnelio.geo._occ_backend import bounding_box, keep_operands_intact
    from magnelio.geo._sheet import Profile
    from magnelio.geo.curves import Curve
    from magnelio.geo.topology import _cast, _detached_class

    cutter, plane = _arguments(source, cutter, normal, position, "section")
    if filled and (plane is None or not isinstance(source, Solid)):
        raise TypeError("A filled section requires a Solid and an explicit plane.")
    bounds = source._analytic_bbox()
    if cutter is not None:
        bounds = union_boxes((bounds, cutter._analytic_bbox()))
    scale = choose_scale(*bounds)
    source_shape = source._occ_shape(scale)
    tool_shape = cutter._occ_shape(scale) if cutter is not None else _plane(plane, scale)
    if _has_coincident_face(source_shape, tool_shape, scale):
        raise ValueError("section: source and cutter share a face region.")
    if filled:
        common = BRepAlgoAPI_Common(source_shape, tool_shape)
        keep_operands_intact(common)
        common.Build()
        if not common.IsDone():
            raise RuntimeError("section: the CAD kernel could not construct filled sections.")
        return tuple(
            _detached_class(Profile)(face, scale, bounding_box(face, scale=scale), source.material)
            for face in _pieces(common.Shape(), "face")
        )
    builder = BRepAlgoAPI_Section(source_shape, tool_shape)
    keep_operands_intact(builder)
    builder.Build()
    if not builder.IsDone():
        raise RuntimeError("section: the CAD kernel could not construct intersection curves.")
    edges = TopTools_HSequenceOfShape()
    for edge in _pieces(builder.Shape(), "edge"):
        edges.Append(edge)
    wires = TopTools_HSequenceOfShape()
    wires = ShapeAnalysis_FreeBounds.ConnectEdgesToWires(edges, 1e-7, False, wires)
    from magnelio.geo.topology import _rescale_copy

    return tuple(
        Curve(
            _build=lambda target, wire=wire: _rescale_copy(wire, scale, target),
            _bounds=bounding_box(wire, scale=scale),
            _ends=_wire_ends(wire, scale),
        )
        for wire in (_cast(wires.Value(i), "wire") for i in range(1, wires.Length() + 1))
    )


def _wire_ends(wire, scale):
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.BRepTools import BRepTools_WireExplorer
    from OCC.Core.TopExp import topexp
    from OCC.Core.TopoDS import topods

    explorer = BRepTools_WireExplorer(wire)
    edges = []
    while explorer.More():
        edges.append(topods.Edge(explorer.Current()))
        explorer.Next()
    if not edges:
        raise RuntimeError("section: the CAD kernel returned an empty curve.")
    ends = []
    for vertex in (topexp.FirstVertex(edges[0], True), topexp.LastVertex(edges[-1], True)):
        point = BRep_Tool.Pnt(vertex)
        ends.append((point.X() / scale, point.Y() / scale, point.Z() / scale))
    return tuple(ends)


def _has_coincident_face(source, tool, scale):
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCC.Core.BRepGProp import brepgprop
    from OCC.Core.GeomAbs import GeomAbs_Plane
    from OCC.Core.GProp import GProp_GProps

    from magnelio.geo._occ_backend import bounding_box

    tool_faces = _pieces(tool, "face")
    tool_boxes = [bounding_box(face, scale=scale) for face in tool_faces]
    for face in _pieces(source, "face"):
        source_box = bounding_box(face, scale=scale)
        source_surface = BRepAdaptor_Surface(face)
        for tool_face, tool_box in zip(tool_faces, tool_boxes):
            if any(
                source_box[1][i] < tool_box[0][i] - 1e-7 / scale
                or tool_box[1][i] < source_box[0][i] - 1e-7 / scale
                for i in range(3)
            ):
                continue
            tool_surface = BRepAdaptor_Surface(tool_face)
            if (
                source_surface.GetType() == GeomAbs_Plane
                and tool_surface.GetType() == GeomAbs_Plane
            ):
                a = source_surface.Plane()
                b = tool_surface.Plane()
                if (
                    abs(a.Axis().Direction().Dot(b.Axis().Direction())) < 1.0 - 1e-10
                    or a.Distance(b) > 1e-7
                ):
                    continue
            common = BRepAlgoAPI_Common(face, tool_face)
            common.Build()
            if not common.IsDone():
                continue
            props = GProp_GProps()
            brepgprop.SurfaceProperties(common.Shape(), props)
            if props.Mass() > 1e-14:
                return True
    return False
