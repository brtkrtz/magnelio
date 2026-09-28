"""Scoped OCC history capture for named topology (DD-275)."""

from __future__ import annotations

from contextvars import ContextVar

from magnelio.geo.shape import Shape

_CAPTURE = ContextVar("magnelio_topology_history", default=None)
_INPUT_FIELDS = (
    "_inner",
    "_profile",
    "_shape_a",
    "_shape_b",
    "shape_a",
    "shape_b",
    "base",
    "shapes",
    "tools",
    "sections",
    "_spine",
    "_curve",
)


def inputs(owner):
    result = []
    for field in _INPUT_FIELDS:
        value = getattr(owner, field, ())
        for shape in value if isinstance(value, (list, tuple)) else (value,):
            if isinstance(shape, Shape) and not any(shape is previous for previous in result):
                result.append(shape)
    return result


def has_names(owner):
    return bool(getattr(owner, "_registration", None) or getattr(owner, "_topology_inputs", ()))


def finish(owner):
    """Validate named evolution at the public construction call."""
    sources = tuple(source for source in inputs(owner) if has_names(source))
    if sources or getattr(owner, "_registration", None):
        from magnelio.geo.topology import _scale

        owner._topology_inputs = sources
        names(owner, _scale(owner))
    return owner


def recording():
    return _CAPTURE.get() is not None


def result(builder):
    """Return a built shape and retain its builder only in the active scope."""
    built = builder.Shape()
    capture = _CAPTURE.get()
    if capture is not None:
        capture.append((builder, built))
    return built


def evaluate(owner, scale, build):
    """Evaluate one named owner, isolating nested builders and scale caches."""
    source_names = [(source, names(source, scale)) for source in owner._topology_inputs]
    # Build every direct source before capture. An untagged tool may have
    # its own lazy fillet/transform history, unrelated to the result names.
    for source in inputs(owner):
        source._occ_shape(scale)
    histories = []
    token = _CAPTURE.set(histories)
    try:
        built = build(owner, scale)
    finally:
        _CAPTURE.reset(token)
    mapped = _evolve(owner, built, source_names, histories)
    owner.__dict__.setdefault("_topology_names_cache", {})[scale] = mapped
    return built


def _evolve(owner, built, sources, histories):
    from magnelio.geo.topology import TopologyEvolutionError, _cast, _index, _kind

    label = {
        "_ChamferedShape": "chamfered()",
        "_FilletedShape": "filleted()",
        "_ExtrudedFaceShape": "extruded()",
        "_LoftedShape": "lofted()",
        "_RevolvedShape": "revolved()",
        "_SweptShape": "swept()",
        "_ShelledShape": "shelled()",
        "TransformedSolid": "affine placement",
    }.get(type(owner).__name__, type(owner).__name__)
    operation = label
    maps = {}
    merged = {}
    intermediate_maps = {}
    for source, selections in sources:
        for key, (members, plural) in selections.items():
            kind, label = key
            if kind not in maps:
                maps[kind] = _index(built, kind)
            final = maps[kind]
            evolved = []
            for member in members:
                candidates = [member]
                reported_deleted = False
                for step, (builder, intermediate) in enumerate(histories):
                    # Histories can branch (e.g. shell offset then cut).
                    # Keep predecessors until final owner membership is known.
                    next_candidates = list(candidates)
                    for candidate in candidates:
                        modified = []
                        generated = []
                        try:
                            modified = list(builder.Modified(candidate))
                            generated = list(builder.Generated(candidate))
                            deleted = builder.IsDeleted(candidate)
                        except (RuntimeError, KeyError):
                            deleted = False
                        successors = [
                            s for s in modified + generated if s.ShapeType() == _kind(kind)
                        ]
                        # OCC fillet/chamfer IsDeleted is incomplete for edges
                        # and vertices. Actual identity in the result is proof.
                        map_key = (step, kind)
                        if map_key not in intermediate_maps:
                            intermediate_maps[map_key] = _index(intermediate, kind)
                        contained = intermediate_maps[map_key]
                        if contained.Contains(candidate):
                            successors.insert(0, contained.FindKey(contained.FindIndex(candidate)))
                        next_candidates.extend(successors)
                        reported_deleted |= deleted and not successors
                    candidates = _unique(next_candidates)
                surviving = [
                    _cast(final.FindKey(final.FindIndex(s)), kind)
                    for s in candidates
                    if final.Contains(s)
                ]
                if not surviving:
                    raise TopologyEvolutionError(
                        f"{operation}: named {kind} {label!r} "
                        + ("was deleted." if reported_deleted else "has no provable successor.")
                    )
                evolved.extend(surviving)
            evolved = tuple(_unique(evolved))
            if not plural and len(evolved) != 1:
                raise TopologyEvolutionError(
                    f"{operation}: named {kind} {label!r} "
                    f"split into {len(evolved)} successors; "
                    "register a deliberate set before construction."
                )
            if key in merged:
                previous, previous_plural = merged[key]
                if (
                    previous_plural != plural
                    or len(previous) != len(evolved)
                    or any(not any(s.IsSame(p) for p in previous) for s in evolved)
                ):
                    raise TopologyEvolutionError(
                        f"{operation}: conflicting {kind} "
                        f"selection name {label!r} from different operands."
                    )
            merged[key] = (evolved, plural)
    return merged


def _unique(shapes):
    from OCC.Core.TopTools import TopTools_IndexedMapOfShape

    index = TopTools_IndexedMapOfShape()
    for shape in shapes:
        index.Add(shape)
    return [index.FindKey(i) for i in range(1, index.Size() + 1)]


def names(owner, scale):
    from magnelio.geo.topology import TopologyEvolutionError, select

    if not has_names(owner):
        return {}
    cache = owner.__dict__.setdefault("_topology_names_cache", {})
    if scale not in cache:
        owner._occ_shape(scale)
    mapped = cache[scale]
    registration = getattr(owner, "_registration", None)
    if registration is not None:
        kind, label, selectors, plural, count = registration
        key = (kind, label)
        if key not in mapped:
            ref = select(owner, kind, plural=plural, _model_scale=scale, **selectors)
            members = ref._shape if plural else (ref._shape,)
            if len(members) != count:
                raise TopologyEvolutionError(
                    f"Named {kind} {label!r}: rebuilt selection cardinality "
                    f"changed from {count} to {len(members)}."
                )
            mapped[key] = (members, plural)
    return mapped
