"""
STL geometry export via pythonocc for ParaView visualization.
"""

from __future__ import annotations

from pathlib import Path


def export_stl(
    path: str | Path,
    geometry: list,
    deflection: float | None = None,
    binary: bool = True,
) -> None:
    """Export CSG geometry shapes to an STL file.

    Parameters
    ----------
    path : str or Path
        Output file path (``.stl``).
    geometry : list
        List of geometry shape objects (each must have a ``.shape()``
        method returning a TopoDS_Shape, or be a TopoDS_Shape directly).
    deflection : float, optional
        Tessellation accuracy [m].  Smaller = finer mesh.  Default:
        0.1 % of the model bounding-box diagonal (scale-independent);
        raw ``TopoDS_Shape`` inputs fall back to 1e-3 m.
    binary : bool
        Write binary STL (smaller files).  Set False for ASCII.
    """
    from OCC.Core.BRep import BRep_Builder  # noqa: PLC0415
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh  # noqa: PLC0415
    from OCC.Core.StlAPI import StlAPI_Writer  # noqa: PLC0415
    from OCC.Core.TopoDS import TopoDS_Compound  # noqa: PLC0415

    path = Path(path)

    # DD-120: build at the model scale when every entry supports it
    # (raw TopoDS / .shape() entries are meter-space and force 1.0),
    # then transform back so the STL file is in meters at any scale.
    geometry = list(geometry)
    geo_scale = 1.0
    if geometry and all(hasattr(obj, "_analytic_bbox") for obj in geometry):
        from magnelio.geo._scaling import (  # noqa: PLC0415
            analytic_bbox,
            box_diagonal,
            model_scale,
            union_boxes,
        )

        geo_scale = model_scale(geometry)
        if deflection is None:
            diag = box_diagonal(union_boxes([analytic_bbox(o) for o in geometry]))
            deflection = 1e-3 * diag if diag > 0.0 else 1e-3
    if deflection is None:
        deflection = 1e-3

    # Build a compound of all shapes
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)

    for obj in geometry:
        if hasattr(obj, "_occ_shape"):
            topo_shape = obj._occ_shape(geo_scale)
        elif hasattr(obj, "shape") and callable(obj.shape):
            topo_shape = obj.shape()
        else:
            topo_shape = obj
        builder.Add(compound, topo_shape)

    # Back to meters BEFORE tessellating: the triangulation must be
    # computed on the shape the writer sees.
    if geo_scale != 1.0:
        from magnelio.geo._occ_backend import occ_scale  # noqa: PLC0415

        compound = occ_scale(compound, 1.0 / geo_scale, (0.0, 0.0, 0.0))

    # Tessellate
    BRepMesh_IncrementalMesh(compound, deflection)

    # Write
    writer = StlAPI_Writer()
    writer.SetASCIIMode(not binary)
    writer.Write(compound, str(path))
