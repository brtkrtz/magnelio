"""
XDMF + HDF5 helpers for ParaView visualization.

Generates an XDMF descriptor (XML) that references field data stored in a
companion HDF5 file.  ParaView reads the XDMF file natively and displays
rectilinear grid data with time-series animation.

Each monitor produces a temporal collection in the XDMF file.  The grid
coordinates are taken from the monitor's resolved sub-region, so 2D plane
monitors export as thin slabs that ParaView renders correctly.

ParaView-export helper for the project store's field-time monitors;
the session generator (:mod:`magnelio.io.paraview`)
additionally writes one descriptor per monitor for its pre-built
pipelines.
"""

# Design: DD-070 WP-S9 (project-store field-time monitors), DD-115 (session
# generator).

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def _indent(elem: ET.Element, level: int = 0) -> None:
    """Add pretty-print whitespace to an ElementTree (in-place)."""
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for i, child in enumerate(elem):
            _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent + "  " if i < len(elem) - 1 else indent
    if not elem.tail or not elem.tail.strip():
        elem.tail = indent


def _add_geometry(grid_el, mon_name, h5_filename, nx, ny, nz):
    """Add Topology + Geometry elements for a 3DRectMesh."""
    node_dims = f"{nz + 1} {ny + 1} {nx + 1}"
    ET.SubElement(
        grid_el,
        "Topology",
        TopologyType="3DRectMesh",
        Dimensions=node_dims,
    )
    geom = ET.SubElement(grid_el, "Geometry", GeometryType="VXVYVZ")
    for axis, n_nodes in [("x", nx + 1), ("y", ny + 1), ("z", nz + 1)]:
        di = ET.SubElement(
            geom,
            "DataItem",
            Format="HDF",
            Dimensions=str(n_nodes),
        )
        di.text = f"{h5_filename}:/monitors/{mon_name}/grid_{axis}"


def _hyperslab_dataitem(
    parent,
    h5_filename,
    mon_name,
    ds_name,
    step_idx,
    n_steps,
    nz,
    ny,
    nx,
):
    """Add a HyperSlab DataItem selecting one time/freq step of a dataset."""
    hs = ET.SubElement(
        parent,
        "DataItem",
        ItemType="HyperSlab",
        Dimensions=f"{nz} {ny} {nx}",
        Type="HyperSlab",
    )
    sel = ET.SubElement(hs, "DataItem", Dimensions="3 4", Format="XML")
    sel.text = (
        f"\n            {step_idx} 0 0 0\n"
        f"            1 1 1 1\n"
        f"            1 {nz} {ny} {nx}\n          "
    )
    src = ET.SubElement(
        hs,
        "DataItem",
        Dimensions=f"{n_steps} {nz} {ny} {nx}",
        Format="HDF",
    )
    src.text = f"{h5_filename}:/monitors/{mon_name}/{ds_name}"


def _add_hyperslab_attribute(
    grid_el,
    name,
    h5_filename,
    mon_name,
    ds_name,
    step_idx,
    n_steps,
    nz,
    ny,
    nx,
):
    """Add a scalar attribute with HyperSlab selection of one time/freq step."""
    attr = ET.SubElement(
        grid_el,
        "Attribute",
        Name=name,
        AttributeType="Scalar",
        Center="Cell",
    )
    _hyperslab_dataitem(
        attr,
        h5_filename,
        mon_name,
        ds_name,
        step_idx,
        n_steps,
        nz,
        ny,
        nx,
    )


def _add_vector_attribute(
    grid_el,
    name,
    h5_filename,
    mon_name,
    ds_names,
    step_idx,
    n_steps,
    nz,
    ny,
    nx,
):
    """Add a 3-component vector attribute JOINing per-component datasets.

    The components live in separate HDF5 datasets; an XDMF ``JOIN``
    function stacks them into an interleaved ``(nz, ny, nx, 3)`` vector
    array so ParaView offers the field for Glyph / streamline filters.
    """
    attr = ET.SubElement(
        grid_el,
        "Attribute",
        Name=name,
        AttributeType="Vector",
        Center="Cell",
    )
    join = ET.SubElement(
        attr,
        "DataItem",
        ItemType="Function",
        Function="JOIN($0, $1, $2)",
        Dimensions=f"{nz} {ny} {nx} 3",
    )
    for ds_name in ds_names:
        _hyperslab_dataitem(
            join,
            h5_filename,
            mon_name,
            ds_name,
            step_idx,
            n_steps,
            nz,
            ny,
            nx,
        )


def _vector_groups(components) -> list[tuple[str, list[str]]]:
    """Complete ``E``/``H`` triples among *components*, as (name, ds_names).

    Handles both time-monitor names (``Ex``) and frequency-monitor
    real/imaginary pairs (``Ex_re`` → vector ``E_re``).
    """
    comps = set(components)
    out = []
    for group in ("E", "H"):
        for suffix in ("", "_re", "_im"):
            triple = [f"{group}{ax}{suffix}" for ax in "xyz"]
            if all(c in comps for c in triple):
                out.append((f"{group}{suffix}", triple))
    return out


def write_run_xdmf(
    xdmf_path: str | Path,
    h5_filename: str,
    monitor_specs: list,
) -> None:
    """Write an XDMF descriptor for the project store's streamed monitors.

    Takes plain descriptors instead of live monitor
    objects: the run sink streams monitor snapshots to ``results.h5`` and
    knows their layout without holding the data in RAM.  The referenced
    HDF5 holds, per monitor, ``grid_x/y/z`` node coordinates and each
    component in row-major ``(n_steps, nz, ny, nx)`` order — the same
    layout the ParaView helpers above already emit.

    Parameters
    ----------
    xdmf_path : str or Path
        Output ``fields.xdmf`` path (typically the run directory).
    h5_filename : str
        HDF5 filename relative to the XDMF file (``"results.h5"``).
    monitor_specs : list of dict
        One ``{name, n, nx, ny, nz, components, times}`` per field-time
        monitor: ``n`` recorded snapshots, region cell counts, component
        names, and the recorded time points.
    """
    # Design: WP-S9 (run-sink monitor streaming).
    xdmf_root = ET.Element("Xdmf", Version="2.2")
    domain = ET.SubElement(xdmf_root, "Domain")
    for spec in monitor_specs:
        name = spec["name"]
        nx, ny, nz = spec["nx"], spec["ny"], spec["nz"]
        n_steps = int(spec["n"])
        if n_steps == 0:
            continue
        collection = ET.SubElement(
            domain,
            "Grid",
            Name=name,
            GridType="Collection",
            CollectionType="Temporal",
        )
        for ti, t_val in enumerate(spec["times"]):
            grid_el = ET.SubElement(
                collection,
                "Grid",
                Name=f"t={float(t_val):.4e}s",
                GridType="Uniform",
            )
            ET.SubElement(grid_el, "Time", Value=f"{float(t_val):.6e}")
            _add_geometry(grid_el, name, h5_filename, nx, ny, nz)
            for comp in spec["components"]:
                _add_hyperslab_attribute(
                    grid_el,
                    comp,
                    h5_filename,
                    name,
                    comp,
                    ti,
                    n_steps,
                    nz,
                    ny,
                    nx,
                )
            for vec_name, triple in _vector_groups(spec["components"]):
                _add_vector_attribute(
                    grid_el,
                    vec_name,
                    h5_filename,
                    name,
                    triple,
                    ti,
                    n_steps,
                    nz,
                    ny,
                    nx,
                )
    _indent(xdmf_root)
    ET.ElementTree(xdmf_root).write(str(Path(xdmf_path)), xml_declaration=True, encoding="utf-8")
