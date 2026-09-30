"""Directed boundary imprinting of a solid by standalone geometry."""

from __future__ import annotations

import math

from magnelio.geo._scaling import choose_scale, union_boxes
from magnelio.geo.shape import Solid


class _ImprintedSolid(Solid):
    def __init__(self, shape, scale, receiver, cutter, selections):
        self._shape = shape
        self._source_scale = scale
        self._receiver = receiver
        self._cutter = cutter
        self._bounds = receiver._analytic_bbox()
        self.material = receiver.material
        self.name = receiver.name
        self.color = getattr(receiver, "color", None)
        self._occ_shape_cache = {scale: shape}
        if selections:
            self._topology_inputs = (receiver,)
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
                                "imprint: named topology has no unique scaled successor."
                            )
                        moved.append(_cast(successors.First(), kind))
                    selections[key] = (tuple(moved), plural)
                self._topology_names_cache[scale] = selections
        return self._occ_shape_cache[scale]

    def _analytic_bbox(self):
        return self._bounds


def imprint(receiver, cutter):
    """Return a solid whose boundary is split at the cutter intersection."""
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepFeat import BRepFeat_SplitShape
    from OCC.Core.TopAbs import TopAbs_EDGE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Core.TopTools import TopTools_SequenceOfShape

    from magnelio.geo._occ_backend import _shape_list, keep_operands_intact, occ_volume
    from magnelio.geo._sheet import Sheet
    from magnelio.geo._topology_history import _evolve, has_names, names

    if not isinstance(receiver, Solid):
        raise TypeError("imprint requires a Solid receiver.")
    if not isinstance(cutter, (Solid, Sheet)):
        raise TypeError("imprint cutter must be a Solid or Sheet.")
    scale = choose_scale(*union_boxes((receiver._analytic_bbox(), cutter._analytic_bbox())))
    source_shape = receiver._occ_shape(scale)
    cutter_shape = cutter._occ_shape(scale)
    section = BRepAlgoAPI_Section()
    section.SetArguments(_shape_list((source_shape,)))
    section.SetTools(_shape_list((cutter_shape,)))
    keep_operands_intact(section)
    section.Build()
    if not section.IsDone():
        raise RuntimeError("imprint: the CAD kernel could not find intersection edges.")
    edges = TopTools_SequenceOfShape()
    explorer = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
    while explorer.More():
        edges.Append(topods.Edge(explorer.Current()))
        explorer.Next()
    builder = BRepFeat_SplitShape(source_shape)
    if edges.Length() and not builder.Add(edges):
        raise RuntimeError("imprint: intersection edges could not be assigned to faces.")
    builder.Build()
    if not builder.IsDone() or not BRepCheck_Analyzer(builder.Shape()).IsValid():
        raise RuntimeError("imprint: the CAD kernel did not produce a valid solid.")
    built = builder.Shape()
    if not math.isclose(abs(occ_volume(built)), abs(occ_volume(source_shape)), rel_tol=1e-9):
        raise RuntimeError("imprint: the CAD kernel changed the receiver volume.")
    result = _ImprintedSolid(built, scale, receiver, cutter, {})
    if has_names(receiver):
        selections = _evolve(
            result,
            built,
            [(receiver, names(receiver, scale))],
            [(builder, built)],
        )
        result._topology_inputs = (receiver,)
        result._topology_names_cache = {scale: selections}
    return result
