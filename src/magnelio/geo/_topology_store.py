"""Semantic origins plus construction replay for project topology names.

No kernel enumeration IDs or geometric-nearest retargeting are persisted.
Opaque untagged leaves use exact BREP snapshots; named branches retain their
operation graph. Legacy projects continue to load their final BREP only.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory

from magnelio.geo._topology_history import finish, has_names, names
from magnelio.geo.shape import Shape, Solid
from magnelio.geo.topology import TopologyEvolutionError, _scale


def _brep_text(shape):
    from OCC.Core.BRepTools import breptools

    with TemporaryDirectory() as directory:
        path = Path(directory) / "snapshot.brep"
        if not breptools.Write(shape, str(path)):
            raise RuntimeError("Writing a topology-origin BREP snapshot failed.")
        return path.read_text()


def _read_snapshot(text):
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopoDS import TopoDS_Shape

    with TemporaryDirectory() as directory:
        path = Path(directory) / "snapshot.brep"
        path.write_text(text)
        shape = TopoDS_Shape()
        if not breptools.Read(shape, str(path), BRep_Builder()) or shape.IsNull():
            raise TopologyEvolutionError("Reading a topology-origin BREP snapshot failed.")
        return shape


def to_recipe(owner):
    if not has_names(owner):
        return None
    from magnelio.geo import Curve, Profile, Solid, Surface
    from magnelio.geo.imprint import _ImprintedSolid
    from magnelio.geo.partition import _PartitionSolid
    from magnelio.geo.topology import _TaggedSolid
    from magnelio.geo.transforms import _TransformedShape
    from magnelio.io.project import _material_to_dict

    nodes = []
    seen = {}

    def encode(shape):
        if id(shape) in seen:
            return {"node": seen[id(shape)]}
        index = len(nodes)
        seen[id(shape)] = index
        nodes.append(None)
        material = (
            _material_to_dict(shape.material)
            if getattr(shape, "material", None) is not None
            else None
        )
        metadata = {
            "material": material,
            "name": getattr(shape, "name", None),
            "color": getattr(shape, "color", None),
        }
        if isinstance(shape, _TaggedSolid):
            node = {
                "operation": "tag",
                "source": encode(shape._inner),
                "registration": shape._registration,
            }
        elif has_names(shape) and isinstance(shape, _TransformedShape):
            node = {
                "operation": "transform",
                "source": encode(shape._inner),
                "matrix": shape._transform.matrix,
            }
        elif isinstance(shape, _PartitionSolid) and has_names(shape):
            scale = _scale(shape)
            node = {
                "operation": "partition_piece",
                "source": encode(shape._source),
                "cutter": encode(shape._cutter) if shape._cutter is not None else None,
                "plane": shape._plane,
                "scale": scale,
                "brep": _brep_text(shape._occ_shape(scale)),
            }
        elif isinstance(shape, _ImprintedSolid) and has_names(shape):
            scale = _scale(shape)
            node = {
                "operation": "imprint",
                "receiver": encode(shape._receiver),
                "cutter": encode(shape._cutter),
                "scale": scale,
                "brep": _brep_text(shape._occ_shape(scale)),
            }
        elif has_names(shape):
            node = {
                "operation": type(shape).__name__,
                "arguments": {
                    field.name: value(getattr(shape, field.name))
                    for field in fields(shape)
                    if field.name not in ("material", "_material", "name")
                },
            }
        else:
            scale = _scale(shape)
            category = next(
                name
                for cls, name in (
                    (Curve, "curve"),
                    (Profile, "profile"),
                    (Surface, "surface"),
                    (Solid, "solid"),
                )
                if isinstance(shape, cls)
            )
            node = {
                "operation": "snapshot",
                "category": category,
                "scale": scale,
                "brep": _brep_text(shape._occ_shape(scale)),
                "bounds": shape._analytic_bbox(),
            }
        nodes[index] = {**node, **metadata}
        return {"node": index}

    def reference(ref):
        origin = ref._origin
        if origin is None:
            raise TopologyEvolutionError("A construction reference has no replayable origin.")
        if "parent" in origin:
            data = {"parent": reference(origin["parent"])}
            if "connection" in origin:
                data["connection"] = origin["connection"]
            else:
                data["member_brep"] = _brep_text(origin["member_shape"])
        else:
            data = {**origin, "owner": encode(ref.owner)}
        return {"reference": data, "scale": ref._scale}

    def value(obj):
        from magnelio.geo.topology import TopologyRef

        if isinstance(obj, TopologyRef):
            return reference(obj)
        if isinstance(obj, Shape):
            return encode(obj)
        if isinstance(obj, (tuple, list)):
            return [value(item) for item in obj]
        return obj

    root = encode(owner)
    expected = [
        {"kind": kind, "name": label, "count": len(members), "plural": plural}
        for (kind, label), (members, plural) in names(owner, _scale(owner)).items()
    ]
    return {"version": 1, "nodes": nodes, "root": root, "expected": expected}


def from_recipe(recipe):
    from magnelio.geo import Curve, Profile, Surface, modifications, operations
    from magnelio.geo.partition import partition
    from magnelio.geo.topology import _cast, _detached_class, _TaggedSolid
    from magnelio.geo.transforms import Transform
    from magnelio.io.project import _material_from_dict

    if recipe.get("version") != 1:
        raise TopologyEvolutionError("Unsupported named-topology recipe version.")
    classes = {
        name: getattr(module, name)
        for module, entries in (
            (operations, ("Union", "Intersection", "Difference", "_InsertRegion")),
            (
                modifications,
                (
                    "_ChamferedShape",
                    "_FilletedShape",
                    "_ExtrudedFaceShape",
                    "_LoftedShape",
                    "_RevolvedShape",
                    "_SweptShape",
                    "_ShelledShape",
                    "_ThickenedSheet",
                    "_ProfileBlend",
                ),
            ),
        )
        for name in entries
    }
    cache = {}
    active = set()

    def decode(reference):
        index = reference["node"]
        if index in cache:
            return cache[index]
        if index in active:
            raise TopologyEvolutionError("Cyclic named-topology construction recipe.")
        active.add(index)
        node = recipe["nodes"][index]
        material = _material_from_dict(node["material"]) if node["material"] is not None else None
        op = node["operation"]
        if op == "snapshot":
            raw = _read_snapshot(node["brep"])
            scale = node["scale"]
            category = node["category"]
            if category == "solid":
                shape = _SnapshotSolid(
                    raw, scale, node["bounds"], material, node["name"], node["color"]
                )
            elif category == "curve":
                from magnelio.geo.topology import _rescale_copy

                shape = Curve(
                    _build=lambda target: _rescale_copy(raw, scale, target),
                    name=node["name"],
                    _bounds=node["bounds"],
                )
            else:
                shape = _detached_class(Profile if category == "profile" else Surface)(
                    _cast(raw, "face"), scale, node["bounds"], material
                )
        elif op == "tag":
            kind, label, selectors, plural, count = node["registration"]
            shape = finish(
                _TaggedSolid(decode(node["source"]), kind, label, selectors, plural, count)
            )
        elif op == "transform":
            shape = Transform(node["matrix"]) @ decode(node["source"])
        elif op == "partition_piece":
            source = decode(node["source"])
            cutter = decode(node["cutter"]) if node["cutter"] is not None else None
            plane = node["plane"]
            pieces = (
                partition(source, cutter)
                if cutter is not None
                else partition(source, normal=plane[0], position=plane[1])
            )
            matches = [
                piece
                for piece in pieces
                if _brep_text(piece._occ_shape(node["scale"])) == node["brep"]
            ]
            if len(matches) != 1:
                raise TopologyEvolutionError(
                    "A partition region has no unique exact construction match."
                )
            shape = matches[0]
        elif op == "imprint":
            receiver = decode(node["receiver"])
            cutter = decode(node["cutter"])
            shape = receiver.imprint(cutter)
            if _brep_text(shape._occ_shape(node["scale"])) != node["brep"]:
                raise TopologyEvolutionError("An imprinted body has no exact construction match.")
        elif op in classes:
            arguments = {key: value(obj) for key, obj in node["arguments"].items()}
            cls = classes[op]
            if op == "Union":
                shape = cls(*arguments["shapes"], material=material, name=node["name"])
            elif op == "Difference":
                shape = cls(
                    arguments["base"], *arguments["tools"], material=material, name=node["name"]
                )
            elif op == "_InsertRegion":
                shape = cls(arguments["base"], *arguments["tools"])
            elif op == "Intersection":
                shape = cls(**arguments, material=material, name=node["name"])
            else:
                if any(field.name == "_material" for field in fields(cls)):
                    arguments["_material"] = material
                shape = finish(cls(**arguments))
        else:
            raise TopologyEvolutionError(f"Unsupported topology-history operation {op!r}.")
        active.remove(index)
        cache[index] = shape
        return shape

    def reference(data, scale):
        from magnelio.geo.topology import select

        if "owner" in data:
            return select(
                decode(data["owner"]),
                data["kind"],
                data["name"],
                plural=data["plural"],
                _model_scale=scale,
                **data["selectors"],
            )
        parent_node = data["parent"]
        parent = reference(parent_node["reference"], parent_node["scale"])
        if "connection" in data:
            if data["connection"] != "edges":
                raise TopologyEvolutionError("Unsupported reference connection.")
            return parent.edges
        matches = [ref for ref in parent if _brep_text(ref._shape) == data["member_brep"]]
        if len(matches) != 1:
            raise TopologyEvolutionError(
                "A connected reference has no unique exact snapshot match."
            )
        return matches[0]

    def value(obj):
        if isinstance(obj, dict) and "reference" in obj:
            return reference(obj["reference"], obj["scale"])
        if isinstance(obj, dict) and "node" in obj:
            return decode(obj)
        if isinstance(obj, list):
            return tuple(value(item) for item in obj)
        return obj

    owner = decode(recipe["root"])
    actual = {
        (kind, label): (len(members), plural)
        for (kind, label), (members, plural) in names(owner, _scale(owner)).items()
    }
    expected = {
        (item["kind"], item["name"]): (item["count"], item["plural"]) for item in recipe["expected"]
    }
    if actual != expected:
        raise TopologyEvolutionError(
            "Named topology cardinality differs from the stored construction history."
        )
    return owner


class _SnapshotSolid(Solid):
    def __init__(self, shape, scale, bounds, material, name, color):
        self._shape = shape
        self._source_scale = scale
        self._bounds = bounds
        self.material = material
        self.name = name
        self.color = color
        self._occ_shape_cache = {scale: shape}

    def _occ_shape(self, scale=1.0):
        from magnelio.geo.topology import _rescale_copy

        if scale not in self._occ_shape_cache:
            self._occ_shape_cache[scale] = _rescale_copy(self._shape, self._source_scale, scale)
        return self._occ_shape_cache[scale]

    def _analytic_bbox(self):
        return self._bounds
