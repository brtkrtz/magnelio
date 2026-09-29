"""Caching helpers for OCC shape evaluation.

OCC operations (Boolean cuts, primitive builds) are expensive — Boolean
operations on the order of milliseconds each.  Mesh generation calls
``_occ_shape()`` once per cell-face touching a material boundary, so a
shape that is naively re-evaluated on every call dominates the mesh
build time.

This module exposes a single decorator that gives any ``_occ_shape``
method a per-instance lazy cache, keyed by the DD-120 scale factor.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable


def cached_occ_shape(method: Callable[..., Any]) -> Callable[..., Any]:
    """Cache the result of an ``_occ_shape(scale)`` method on the instance.

    The cache lives under ``instance.__dict__["_occ_shape_cache"]`` as a
    dict keyed by the scale factor, so it is per-instance and independent
    of inheritance.  Keying by scale needs no invalidation logic: a
    changed model scale is simply a different key, and because the scale
    is a power of two shared by a whole model, at most a couple of
    entries ever exist per shape.

    The wrapped method should have signature
    ``(self, scale) -> TopoDS_Shape``.
    """

    @wraps(method)
    def wrapper(self, scale: float = 1.0):
        cache = self.__dict__.setdefault("_occ_shape_cache", {})
        key = float(scale)
        cached = cache.get(key)
        if cached is not None:
            return cached
        from magnelio.geo.topology import TopologyRef

        refs = []
        for field in ("_edges", "_opening_face_near", "_face_near", "_face_near_a", "_face_near_b"):
            value = getattr(self, field, None)
            for item in value if isinstance(value, (tuple, list)) else (value,):
                if isinstance(item, TopologyRef):
                    refs.append(item)
        if refs and key != refs[0]._scale:
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
            from OCC.Core.gp import gp_Pnt, gp_Trsf

            from magnelio.geo._topology_history import has_names, names

            source_scale = refs[0]._scale
            source = self._occ_shape(source_scale)
            transform = gp_Trsf()
            transform.SetScale(gp_Pnt(0, 0, 0), key / source_scale)
            builder = BRepBuilderAPI_Transform(source, transform, True)
            result = builder.Shape()
            if has_names(self):
                self.__dict__.setdefault("_topology_names_cache", {})[key] = {
                    label: (tuple(builder.ModifiedShape(s) for s in members), plural)
                    for label, (members, plural) in names(self, source_scale).items()
                }
        elif hasattr(self, "_topology_inputs"):
            from magnelio.geo._topology_history import evaluate

            result = evaluate(self, key, method)
        else:
            result = method(self, key)
        cache[key] = result
        return result

    return wrapper
