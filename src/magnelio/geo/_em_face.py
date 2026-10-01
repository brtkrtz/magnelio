"""Strict conversion of an owned CAD face to a supported EM plane."""

from __future__ import annotations

import math


def rectangular_face(face):
    """Return the axis and exact bounds of a rectangular planar FaceRef."""
    from magnelio.geo._occ_backend import _shape_wires
    from magnelio.geo.topology import FaceRef, _type

    if not isinstance(face, FaceRef):
        raise TypeError(f"Expected a singular FaceRef; got {type(face).__name__}.")
    if not face.is_planar:
        raise ValueError(
            "The selected face is curved; this EM consumer needs an axis-normal plane."
        )
    normal = face.normal
    axis = max(range(3), key=lambda i: abs(normal[i]))
    if not math.isclose(abs(normal[axis]), 1.0, abs_tol=1e-10):
        raise ValueError(
            "The selected face is oblique; this EM consumer needs an axis-normal plane."
        )
    if len(_shape_wires(face._shape)) != 1:
        raise ValueError("The selected face has holes; this EM consumer needs a full rectangle.")
    lo, hi = face.bounding_box()
    tangential = [i for i in range(3) if i != axis]
    widths = [hi[i] - lo[i] for i in tangential]
    if min(widths) <= 0:
        raise ValueError("The selected face has no two-dimensional sampling region.")
    area = widths[0] * widths[1]
    if not math.isclose(face.area, area, rel_tol=1e-9, abs_tol=0.0):
        raise ValueError(
            "The selected face is not an exact rectangle; "
            "its bounding box is not a sampling region."
        )
    edge_tolerance = max(widths) * 1e-10
    for edge in face.edges:
        on_rectangle_side = any(
            abs(edge.start[i] - bound) <= edge_tolerance
            and abs(edge.end[i] - bound) <= edge_tolerance
            for i in tangential
            for bound in (lo[i], hi[i])
        )
        if _type(edge._shape, "edge") != "line" or not on_rectangle_side:
            raise ValueError(
                "The selected face is not an exact rectangle; "
                "its bounding box is not a sampling region."
            )
    if not math.isclose(lo[axis], hi[axis], rel_tol=0, abs_tol=max(widths) * 1e-10):
        raise ValueError("The selected face does not lie in one axis-normal plane.")
    return axis, normal[axis], lo, hi
