#!/usr/bin/env python
"""Benchmark: board import, stage by stage, against layout density.

Where a board import spends its time is not obvious from the code: the
readers are linear in the file, but merging a copper layer is a Boolean
whose cost depends on how much of the layer actually touches.  The two
extremes are generated here on purpose:

* a grid of isolated pads — every pad its own cluster, so the merge is
  skipped entirely and the cost is the face building;
* a serpentine track and a ground zone — one connected cluster, so the
  whole layer enters a single N-ary fuse, which is the worst case the
  clustering cannot avoid.

If the fuse stage turns out superlinear on the one-cluster case, the
fix is to fuse in tiles and then fuse the tiles: fusing is associative,
so the result is the same shape, and each tile stays small.

Usage:
    mamba run --no-capture-output -n mio python \
        benchmarks/bench_pcb_import.py
"""

import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from magnelio.geo._scaling import fine_detail_scale
from magnelio.io import _pcb_geom as geometry
from magnelio.io._gerber import parse_gerber
from magnelio.io.pcb import import_pcb

# Board and feature sizes of an ordinary two-layer design.
PITCH = 1.0  # mm between pads
PAD_D = 0.6
TRACK_W = 0.25
COPPER_T, CORE_T = 0.035, 1.53


def _mm(value: float) -> str:
    return f"{round(value * 1e6):d}"


def _gerber(*body: str) -> str:
    return "\n".join(["%FSLAX46Y46*%", "%MOMM*%", "G01*", "G75*", *body, "M02*"]) + "\n"


def _pad_grid(n: int) -> list[str]:
    """``n`` by ``n`` isolated pads: the many-clusters case."""
    body = [f"%ADD10C,{PAD_D}*%", "D10*"]
    for row in range(n):
        for column in range(n):
            x = (column + 1) * PITCH
            y = (row + 1) * PITCH
            body.append(f"X{_mm(x)}Y{_mm(y)}D03*")
    return body


def _serpentine(n: int, side: float) -> list[str]:
    """One track folded back and forth: the single-cluster case."""
    body = [f"%ADD11C,{TRACK_W}*%", "D11*"]
    x_lo, x_hi = PITCH, side - PITCH
    body.append(f"X{_mm(x_lo)}Y{_mm(PITCH)}D02*")
    for index in range(n):
        y = PITCH + index * PITCH
        left, right = (x_lo, x_hi) if index % 2 == 0 else (x_hi, x_lo)
        body.append(f"X{_mm(left)}Y{_mm(y)}D01*")
        body.append(f"X{_mm(right)}Y{_mm(y)}D01*")
    return body


def _zone(side: float) -> list[str]:
    corners = [
        (0.5, 0.5),
        (side - 0.5, 0.5),
        (side - 0.5, side - 0.5),
        (0.5, side - 0.5),
        (0.5, 0.5),
    ]
    body = ["G36*", f"X{_mm(corners[0][0])}Y{_mm(corners[0][1])}D02*"]
    body += [f"X{_mm(x)}Y{_mm(y)}D01*" for x, y in corners[1:]]
    return [*body, "G37*"]


def _outline(side: float) -> str:
    corners = [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side), (0.0, 0.0)]
    body = ["%ADD10C,0.05*%", "D10*", f"X{_mm(0)}Y{_mm(0)}D02*"]
    body += [f"X{_mm(x)}Y{_mm(y)}D01*" for x, y in corners[1:]]
    return _gerber(*body)


def _write_board(root: Path, n: int, *, connected: bool) -> Path:
    side = (n + 2) * PITCH
    root.mkdir(parents=True, exist_ok=True)
    top = _serpentine(n, side) + _zone(side) if connected else _pad_grid(n)
    (root / "bench-F_Cu.gbr").write_text(_gerber(*top))
    (root / "bench-B_Cu.gbr").write_text(_gerber(*_zone(side)))
    (root / "bench-Edge_Cuts.gbr").write_text(_outline(side))
    (root / "bench.gbrjob").write_text(
        json.dumps(
            {
                "GeneralSpecs": {"ProjectId": {"Name": "bench"}},
                "FilesAttributes": [
                    {"Path": "bench-F_Cu.gbr", "FileFunction": "Copper,L1,Top"},
                    {"Path": "bench-B_Cu.gbr", "FileFunction": "Copper,L2,Bot"},
                    {"Path": "bench-Edge_Cuts.gbr", "FileFunction": "Profile,NP"},
                ],
                "MaterialStackup": [
                    {"Type": "Copper", "Thickness": COPPER_T, "Name": "F.Cu"},
                    {
                        "Type": "Dielectric",
                        "Thickness": CORE_T,
                        "Material": "FR4",
                        "DielectricConstant": "4.5",
                        "Name": "core",
                    },
                    {"Type": "Copper", "Thickness": COPPER_T, "Name": "B.Cu"},
                ],
            }
        )
    )
    return root


def _stages(root: Path) -> dict:
    """Time the import stage by stage on the top copper layer."""
    text = (root / "bench-F_Cu.gbr").read_text()

    start = time.perf_counter()
    layer = parse_gerber(text, source="bench-F_Cu.gbr")
    parse = time.perf_counter() - start

    profile = parse_gerber((root / "bench-Edge_Cuts.gbr").read_text())
    points = [obj.start for _, obj in profile.objects if hasattr(obj, "start")]
    lo = (min(p[0] for p in points), min(p[1] for p in points), 0.0)
    hi = (max(p[0] for p in points), max(p[1] for p in points), 0.0)
    scale = fine_detail_scale(lo, hi)

    start = time.perf_counter()
    faces = [geometry.object_shape(obj, scale, {}) for _, obj in layer.objects]
    build = time.perf_counter() - start

    start = time.perf_counter()
    merged = geometry.merge_faces(faces, layer.resolution * scale)
    fuse = time.perf_counter() - start

    start = time.perf_counter()
    geometry.extrude(merged, 0.0, COPPER_T * 1e-3, scale)
    prism = time.perf_counter() - start

    return {
        "objects": len(layer.objects),
        "clusters": len(geometry._clusters(faces, layer.resolution * scale)),
        "parse": parse,
        "faces": build,
        "fuse": fuse,
        "prism": prism,
    }


def main() -> None:
    print(
        f"{'case':>12} {'n':>4} {'objects':>8} {'clusters':>9} "
        f"{'parse':>8} {'faces':>8} {'fuse':>9} {'prism':>8} {'whole':>8}"
    )
    for connected in (False, True):
        case = "one cluster" if connected else "isolated"
        for n in (10, 20, 40, 60):
            with TemporaryDirectory() as folder:
                root = _write_board(Path(folder) / "fab", n, connected=connected)
                stages = _stages(root)
                start = time.perf_counter()
                import_pcb(root)
                whole = time.perf_counter() - start
            print(
                f"{case:>12} {n:>4} {stages['objects']:>8} {stages['clusters']:>9} "
                f"{stages['parse']:>8.3f} {stages['faces']:>8.3f} "
                f"{stages['fuse']:>9.3f} {stages['prism']:>8.3f} {whole:>8.3f}"
            )


if __name__ == "__main__":
    main()
