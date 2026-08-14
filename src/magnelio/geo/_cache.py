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
        result = method(self, key)
        cache[key] = result
        return result

    return wrapper
