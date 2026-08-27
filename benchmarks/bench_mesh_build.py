#!/usr/bin/env python
"""Benchmark: mesh-build time per mesher pass on production geometry classes.

Where a mesh build spends its time depends on the geometry class, not on
the cell count alone.  The sub-cell passes scale along two independent
axes — the number of faces every grid plane has to be cut against, and
the number of grid edges that run close to metal — and a decision such
as "parallelise the edge pass" has to be made per class.  This script
builds three parametric geometries through the public API, meshes them
(no solver run) and times each mesher pass from outside the library by
wrapping the pass entry points, so the numbers are what production pays.

Families (``--family``), each with a size ladder (``--sizes``):

* ``lange`` — a row of *n* interdigitated 3-dB Lange couplers on one
  254 µm alumina carrier (the how-to's coupler: 12.6 µm fingers, ribbon
  bonds, right-angle leads): fine grid, many edges next to metal.
* ``array`` — an *n* × *n* microstrip patch array at 10 GHz with a
  corporate feed (λ/4 transformers, H-tree): hundreds of faces per
  plane, a large grid.
* ``posts`` — a row of *n* small PEC posts in an air-filled box, the
  fixture class behind the section pool's admission rule.

Pool arms (``--pool``): ``off`` disables the section process pool
(``MAGNELIO_SECTION_WORKERS=0``), ``auto`` is production, ``forced``
lowers the admission gates so the pool fires whenever there is any
delegated work — a *policy* change, so its numbers are not what
production would see, only what the pool could do.  Pooled arms turn
the pool's fallback warning into an error, so a silently failed pool
cannot pose as a fast sequential arm.

Pass times share caches inside one build (the section cache feeds both
the ε and the µ pipeline), so they are order-dependent and do not sum to
the total — the total is the number to compare across arms, the passes
say where it went.

Output: a table per family on stdout and, with ``--json``, a report at
``benchmarks/results/bench_mesh_build.json``.

Usage:
    mamba run --no-capture-output -n mio python benchmarks/bench_mesh_build.py \\
        --family posts --sizes 60 240 --pool off
    mamba run --no-capture-output -n mio python benchmarks/bench_mesh_build.py \\
        --family all --pool off auto --json
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import magnelio as mio  # noqa: E402
from magnelio import geo  # noqa: E402
from magnelio.constants import C0  # noqa: E402
from magnelio.geo import _filling, _occ_backend, _subcell  # noqa: E402
from magnelio.mesh import _conformal  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results" / "bench_mesh_build.json"

# ── Pass timing ───────────────────────────────────────────────────────────


class PassTimer:
    """Wrap the mesher's pass entry points and accumulate wall time per pass."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self.extra: dict[str, object] = {}
        self._patched: list[tuple[object, str, object]] = []

    def wrap(self, module, name, key=None, key_fn=None, after=None) -> None:
        orig = getattr(module, name)
        timer = self

        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            result = None
            try:
                result = orig(*args, **kwargs)
                return result
            finally:
                dt = time.perf_counter() - t0
                k = key_fn(args, kwargs) if key_fn else (key or name)
                timer.seconds[k] = timer.seconds.get(k, 0.0) + dt
                timer.calls[k] = timer.calls.get(k, 0) + 1
                if after is not None:
                    after(timer, args, kwargs, result)

        setattr(module, name, wrapped)
        self._patched.append((module, name, orig))

    def restore(self) -> None:
        for module, name, orig in reversed(self._patched):
            setattr(module, name, orig)
        self._patched.clear()


def _areas_key(args, kwargs):
    prop = kwargs.get("prop", args[4] if len(args) > 4 else "epsilon")
    return f"areas_{prop}"


def _edge_stats(timer, args, kwargs, result):
    edges = kwargs.get("edges", args[1] if len(args) > 1 else None)
    timer.extra["edges"] = int(len(edges)) if edges is not None else None
    timer.extra["edge_stats"] = dict(_occ_backend._EDGE_FRACTION_STATS)


def _pool_admitted(timer, args, kwargs, result):
    # ``_sample_and_admit`` runs only when workers are available; it
    # returns the queries handed to the pool (empty = declined).
    timer.extra["pool_admitted"] = timer.extra.get("pool_admitted", 0) + len(result or ())


def instrument() -> PassTimer:
    t = PassTimer()
    t.wrap(_occ_backend, "extract_critical_planes_per_shape", key="planes_material")
    t.wrap(_occ_backend, "extract_feature_planes_per_shape", key="planes_edges")
    t.wrap(_occ_backend, "extract_singular_edge_planes", key="planes_singular")
    t.wrap(_occ_backend, "batch_cross_sections", key="classify_sections")
    t.wrap(_filling, "classify_cells_from_cross_sections", key="classify_fill")
    t.wrap(_occ_backend, "build_effective_pec_solid", key="pec_fuse")
    t.wrap(_subcell, "compute_subcell_data", key="pass_edges_eps")
    t.wrap(_subcell, "compute_subcell_data_mu", key="pass_faces_mu")
    t.wrap(_occ_backend, "compute_edge_pec_fractions", key="edge_fractions", after=_edge_stats)
    t.wrap(_occ_backend, "compute_face_material_areas", key_fn=_areas_key)
    t.wrap(_occ_backend, "cross_section_polygons", key="section_calls")
    t.wrap(_occ_backend, "_parallel_section_prefill", key="pool_prefill")
    t.wrap(_occ_backend, "_sample_and_admit", key="pool_admission", after=_pool_admitted)
    t.wrap(_conformal, "rasterize_thin_sheet_footprint", key="sheets")
    # Every N-ary fuse, wherever it runs: the model's own Unions and the
    # tool fuses of Differences (CSG evaluation), but also the fuses
    # inside ``pec_fuse`` and the edge pass — nested, so not top-level.
    t.wrap(_occ_backend, "boolean_union", key="fuse")
    return t


# ── Geometry families ─────────────────────────────────────────────────────


def microstrip_width(z0: float, eps_r: float, h: float) -> float:
    """Wheeler/Hammerstad synthesis of a microstrip width for *z0* on (eps_r, h)."""
    a = z0 / 60 * np.sqrt((eps_r + 1) / 2) + (eps_r - 1) / (eps_r + 1) * (0.23 + 0.11 / eps_r)
    b = 377 * np.pi / (2 * z0 * np.sqrt(eps_r))
    w_h = 8 * np.exp(a) / (np.exp(2 * a) - 2)
    if w_h > 2:
        w_h = (2 / np.pi) * (
            b
            - 1
            - np.log(2 * b - 1)
            + (eps_r - 1) / (2 * eps_r) * (np.log(b - 1) + 0.39 - 0.61 / eps_r)
        )
    return float(w_h * h)


def microstrip_eps_eff(w: float, eps_r: float, h: float) -> float:
    u = w / h
    return float((eps_r + 1) / 2 + (eps_r - 1) / 2 / np.sqrt(1 + 12 / u))


# -- Lange bank -------------------------------------------------------------

LANGE = dict(
    eps_r=9.8,
    h=254e-6,  # substrate
    t=5e-6,  # gold
    w=12.6e-6,  # finger width (the how-to's design on this grid)
    s=25.4e-6,  # finger gap
    length=3.135e-3,  # quarter wave at 10 GHz
    w_lead=240e-6,  # 50 Ω lead
    overlap=50e-6,  # lead over the outer finger's end
    feed=2.0e-3,  # lead length from the outer finger to the wall
    ribbon_w=25e-6,  # ribbon bond width
    ribbon_h=60e-6,  # ribbon bond height above the substrate
    h_box=2.0e-3,
    margin=2.0e-3,  # housing beyond the leads along x (housing resonance above the band)
)
ALUMINA = mio.Material.from_isotropic(name="alumina", epsilon=LANGE["eps_r"])


def lange_coupler(x0: float, p: dict) -> list:
    """The metal of one Lange coupler whose fingers start at ``x0`` (the how-to's layout).

    Four fingers along x centred on y = 0, ribbon bonds joining fingers
    {1, 3} and {2, 4} at both ends, and a 50 Ω lead at each end of both
    outer fingers leaving at a right angle toward the ymin (line 1) or
    ymax (line 2) wall.
    """
    w, s, L, t, h = p["w"], p["s"], p["length"], p["t"], p["h"]
    pitch = w + s
    ys = [(i - 1.5) * pitch for i in range(4)]
    metal = [
        geo.Brick(origin=(x0, y - w / 2, h), size=(L, w, t), material="pec", name=f"finger{i}")
        for i, y in enumerate(ys)
    ]
    rw, rh = p["ribbon_w"], p["ribbon_h"]
    # The two bonds of one end are staggered along the fingers — at one
    # position and one height their beams would cross, i.e. short.
    for end, (a, b) in ((-1, (0, 2)), (-1, (1, 3)), (+1, (0, 2)), (+1, (1, 3))):
        slot = 0.5 if a == 0 else 2.5
        x = x0 + slot * rw if end < 0 else x0 + L - slot * rw
        for y in (ys[a], ys[b]):
            metal.append(
                geo.Brick(origin=(x - rw / 2, y - w / 2, h), size=(rw, w, rh), material="pec")
            )
        metal.append(
            geo.Brick(
                origin=(x - rw / 2, ys[a] - w / 2, h + rh - t),
                size=(rw, ys[b] - ys[a] + w, t),
                material="pec",
            )
        )
    w_lead, overlap, y_wall = p["w_lead"], p["overlap"], lange_half_width(p)
    for end in (-1, +1):
        x_end = x0 if end < 0 else x0 + L
        xl = x_end - w_lead + overlap if end < 0 else x_end - overlap
        for side in (-1, +1):
            y_finger = ys[0] if side < 0 else ys[3]
            y_lo = -y_wall if side < 0 else y_finger - w / 2
            y_hi = y_finger + w / 2 if side < 0 else y_wall
            metal.append(
                geo.Brick(origin=(xl, y_lo, h), size=(w_lead, y_hi - y_lo, t), material="pec")
            )
    return metal


def lange_half_width(p: dict) -> float:
    return 1.5 * (p["w"] + p["s"]) + p["w"] / 2 + p["feed"]


def lange_pitch(p: dict) -> float:
    return p["length"] + 2 * (p["w_lead"] + p["margin"])


def build_lange(n: int, p: dict = LANGE) -> mio.GeometryModel:
    """A row of *n* Lange couplers along x on one carrier, ports on the y walls."""
    h, y_wall, px = p["h"], lange_half_width(p), lange_pitch(p)
    x_first = p["w_lead"] + p["margin"]
    x_min, x_max = 0.0, n * px
    substrate = geo.Brick(
        origin=(x_min, -y_wall, 0.0), size=(x_max - x_min, 2 * y_wall, h), material=ALUMINA
    )
    air = geo.Brick(
        origin=(x_min, -y_wall, h), size=(x_max - x_min, 2 * y_wall, p["h_box"] - h), material="air"
    )
    metal = []
    for i in range(n):
        metal += lange_coupler(x_first + i * px, p)
    model = mio.GeometryModel(background="pec")
    model.add(substrate)
    model.add(geo.Difference(air, *metal))
    for piece in metal:
        model.add(piece)
    return model


def control_lange(n: int) -> tuple[mio.MeshControl, float]:
    return (
        mio.MeshControl(
            min_nodes_per_wavelength=30,
            max_cell_size=0.3e-3,
            min_cell_size=6e-6,
            singularity_refinement=8,
        ),
        14e9,
    )


# -- Patch array ------------------------------------------------------------

ARRAY = dict(eps_r=2.2, h=0.787e-3, t=35e-6, f0=10e9, h_box=8e-3, margin=8e-3)


def build_array(n: int, p: dict = ARRAY) -> mio.GeometryModel:
    """n x n patches on a corporate H-tree feed (n a power of two)."""
    if n & (n - 1):
        raise ValueError("array size must be a power of two")
    eps_r, h, t, f0 = p["eps_r"], p["h"], p["t"], p["f0"]
    lam0 = C0 / f0
    patch_w = lam0 / 2 * np.sqrt(2 / (eps_r + 1))
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 / np.sqrt(1 + 12 * h / patch_w)
    d_l = 0.412 * h * (eps_eff + 0.3) * (patch_w / h + 0.264)
    d_l /= (eps_eff - 0.258) * (patch_w / h + 0.8)
    patch_l = lam0 / (2 * np.sqrt(eps_eff)) - 2 * d_l
    pitch = 0.75 * lam0
    w50 = microstrip_width(50.0, eps_r, h)
    w70 = microstrip_width(70.7, eps_r, h)
    w100 = microstrip_width(100.0, eps_r, h)
    lam4_70 = lam0 / np.sqrt(microstrip_eps_eff(w70, eps_r, h)) / 4

    strips = []

    def strip(x0, y0, x1, y1, width):
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        strips.append(
            geo.Brick(
                origin=(x0 - width / 2, y0 - width / 2, h),
                size=(x1 - x0 + width, y1 - y0 + width, t),
                material="pec",
            )
        )

    def tree(cx, cy, span, along_x, depth):
        """Feed point (cx, cy) splitting into two sub-trees ``span`` apart."""
        if depth == 0:
            # The patch, fed at the centre of its lower edge by a 100 Ω stub.
            strips.append(
                geo.Brick(
                    origin=(cx - patch_w / 2, cy, h),
                    size=(patch_w, patch_l, t),
                    material="pec",
                    name="patch",
                )
            )
            strip(cx, cy - 0.4 * pitch, cx, cy, w100)
            return
        # T-junction: two 100 Ω arms of half a span, then a λ/4 transformer
        # (70.7 Ω) into the 50 Ω node of the next level.
        if along_x:
            strip(cx - span / 2, cy, cx + span / 2, cy, w50)
            children = ((cx - span / 2, cy), (cx + span / 2, cy))
        else:
            strip(cx, cy - span / 2, cx, cy + span / 2, w50)
            children = ((cx, cy - span / 2), (cx, cy + span / 2))
        for x, y in children:
            if along_x:
                strip(x, y, x, y - lam4_70, w70)
                strip(x, y - lam4_70, x, y - 2 * lam4_70, w100)
                tree(x, y - 2 * lam4_70, span / 2, not along_x, depth - 1)
            else:
                strip(x, y, x - lam4_70, y, w70)
                strip(x - lam4_70, y, x - 2 * lam4_70, y, w100)
                tree(x - 2 * lam4_70, y, span / 2, not along_x, depth - 1)

    depth = 2 * int(np.log2(n))
    if n == 1:
        tree(0.0, 0.0, pitch, True, 0)
    else:
        tree(0.0, 0.0, n * pitch / 2, True, depth)
    xs = np.array([s.origin[0] for s in strips] + [s.origin[0] + s.size[0] for s in strips])
    ys = np.array([s.origin[1] for s in strips] + [s.origin[1] + s.size[1] for s in strips])
    m = p["margin"]
    x0, x1 = xs.min() - m, xs.max() + m
    y0, y1 = ys.min() - m, ys.max() + m
    substrate_mat = mio.Material.from_isotropic(name="duroid", epsilon=eps_r)
    model = mio.GeometryModel(background="pec", allow_overlaps=True)
    model.add(geo.Brick(origin=(x0, y0, 0.0), size=(x1 - x0, y1 - y0, h), material=substrate_mat))
    air = geo.Brick(origin=(x0, y0, h), size=(x1 - x0, y1 - y0, p["h_box"] - h), material="air")
    copper = geo.Union(*strips, material="pec")
    model.add(geo.Difference(air, copper))
    model.add(copper)
    return model


def control_array(n: int) -> tuple[mio.MeshControl, float]:
    return mio.MeshControl(min_nodes_per_wavelength=15, min_cell_size=0.3e-3), 12e9


# -- Post row ---------------------------------------------------------------


def build_posts(n: int) -> mio.GeometryModel:
    """A row of *n* PEC posts (r = 0.5 mm, 4 mm tall) in an air-filled box."""
    pitch, r, hp = 2.0e-3, 0.5e-3, 4.0e-3
    length = (n + 1) * pitch
    box = geo.Brick(origin=(0.0, -5e-3, 0.0), size=(length, 10e-3, 6e-3), material="air")
    post = geo.Cylinder(origin=(pitch, 0.0, 0.0), radius=r, height=hp, axis="z", material="pec")
    posts = post.translated((pitch, 0.0, 0.0), repeat=n - 1, copy=True, unite=True)
    model = mio.GeometryModel(background="pec")
    model.add(geo.Difference(box, posts))
    model.add(posts)
    return model


def control_posts(n: int) -> tuple[mio.MeshControl, float]:
    return mio.MeshControl(min_nodes_per_wavelength=12), 10e9


FAMILIES = {
    "lange": (build_lange, control_lange, (1, 2, 4, 8, 16)),
    "array": (build_array, control_array, (2, 4, 8)),
    "posts": (build_posts, control_posts, (60, 240)),
}

# ── Measurement ───────────────────────────────────────────────────────────


def count_faces(model: mio.GeometryModel) -> int:
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer

    n = 0
    for shape in model:
        explorer = TopExp_Explorer(shape._occ_shape(1.0), TopAbs_FACE)
        while explorer.More():
            n += 1
            explorer.Next()
    return n


class PoolArm:
    """Configure the section pool for one benchmark arm."""

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self._saved = {}

    def __enter__(self):
        self._env = os.environ.get("MAGNELIO_SECTION_WORKERS")
        if self.arm == "off":
            os.environ["MAGNELIO_SECTION_WORKERS"] = "0"
        else:
            os.environ.pop("MAGNELIO_SECTION_WORKERS", None)
        if self.arm == "forced":
            for name, value in (
                ("_SECTION_PARALLEL_MIN_QUERIES", 1),
                ("_SECTION_PARALLEL_MIN_FACE_WORK", 0),
                ("_SECTION_POOL_STARTUP_S", 0.0),
            ):
                self._saved[name] = getattr(_occ_backend, name)
                setattr(_occ_backend, name, value)
        return self

    def __exit__(self, *exc):
        if self._env is None:
            os.environ.pop("MAGNELIO_SECTION_WORKERS", None)
        else:
            os.environ["MAGNELIO_SECTION_WORKERS"] = self._env
        for name, value in self._saved.items():
            setattr(_occ_backend, name, value)
        self._saved.clear()


def build_once(family: str, n: int, arm: str) -> dict:
    build, control, _ = FAMILIES[family]
    model = build(n)
    mesh_control, f_max = control(n)
    timer = instrument()
    try:
        with PoolArm(arm), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if arm != "off":
                warnings.filterwarnings("error", message="Parallel cross-section prefill failed")
            t0 = time.perf_counter()
            mesh = mio.Mesh.from_geometry(model, mesh_control, f_max=f_max)
            total = time.perf_counter() - t0
    finally:
        timer.restore()
    row = {
        "family": family,
        "n": n,
        "pool": arm,
        "cells": int(mesh.Nx * mesh.Ny * mesh.Nz),
        "grid": [int(mesh.Nx), int(mesh.Ny), int(mesh.Nz)],
        "shapes": len(model),
        "faces": count_faces(model),
        "edges": timer.extra.get("edges"),
        "edge_stats": timer.extra.get("edge_stats"),
        "total": total,
        "pool_fired": int(timer.extra.get("pool_admitted", 0) > 0),
        "pool_admitted": timer.extra.get("pool_admitted", 0),
        "sections": timer.calls.get("section_calls", 0),
    }
    for key, seconds in sorted(timer.seconds.items()):
        row[key] = seconds
    top_level = (
        "planes_material",
        "planes_edges",
        "planes_singular",
        "classify_sections",
        "classify_fill",
        "pec_fuse",
        "pass_edges_eps",
        "pass_faces_mu",
        "sheets",
    )
    # What the wrapped passes do not cover: CSG evaluation of the
    # model's Booleans (``fuse`` reports the union share of it), plane
    # merging and grid generation, PEC masks.
    row["other"] = total - sum(timer.seconds.get(k, 0.0) for k in top_level)
    return row


def write_report(rows) -> None:
    report = {
        "case": "mesh_build",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "note": (
            "Mesh-build wall time per mesher pass on three geometry classes; "
            "the passes share a section cache inside one build, so they are "
            "order-dependent and do not sum to the total.  Times in seconds, "
            "fastest of --repeat builds, CPU; 'pool' is the section-pool arm "
            "(off / auto = production / forced = admission gates lowered).  "
            "'sheets' is the thin-sheet footprint rasterisation (top-level); "
            "'fuse' is every N-ary boolean_union call, including those nested "
            "inside pec_fuse and the edge pass, so it is not part of 'other'."
        ),
        "cpu_count": os.cpu_count(),
        "rows": rows,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2))


def run(families, sizes, arms, repeat, warm, json_out=False):
    rows = []
    if json_out and RESULTS.exists():
        # Rows of earlier invocations survive; a re-measured point replaces its predecessor.
        rows = json.loads(RESULTS.read_text()).get("rows", [])
    for family in families:
        _, _, default_sizes = FAMILIES[family]
        ladder = sizes or default_sizes
        if warm:
            build_once(family, ladder[0], "off")
        print(f"\n== {family} ==")
        print(
            f"{'n':>5} {'pool':>6} {'cells':>10} {'faces':>6} {'edges':>9} {'total':>8} "
            f"{'edge_fr':>8} {'areas_eps':>9} {'areas_mu':>8} {'classify':>8} "
            f"{'sheets':>7} {'fuse':>6} {'other':>7} {'sect':>6} {'pool':>4}"
        )
        for n in ladder:
            for arm in arms:
                best = None
                for _ in range(repeat):
                    row = build_once(family, n, arm)
                    if best is None or row["total"] < best["total"]:
                        best = row
                rows = [
                    r
                    for r in rows
                    if (r["family"], r["n"], r["pool"]) != (best["family"], best["n"], best["pool"])
                ]
                rows.append(best)
                if json_out:
                    write_report(rows)
                print(
                    f"{n:>5} {arm:>6} {best['cells']:>10} {best['faces']:>6} "
                    f"{best['edges'] or 0:>9} {best['total']:>8.1f} "
                    f"{best.get('edge_fractions', 0.0):>8.1f} "
                    f"{best.get('areas_epsilon', 0.0):>9.1f} {best.get('areas_mu', 0.0):>8.1f} "
                    f"{best.get('classify_sections', 0.0) + best.get('classify_fill', 0.0):>8.1f} "
                    f"{best.get('sheets', 0.0):>7.1f} {best.get('fuse', 0.0):>6.1f} "
                    f"{best['other']:>7.1f} {best['sections']:>6} {best['pool_fired']:>4}",
                    flush=True,
                )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--family", nargs="+", default=["all"], choices=[*FAMILIES, "all"])
    ap.add_argument("--sizes", nargs="+", type=int, default=None, help="ladder (per family)")
    ap.add_argument("--pool", nargs="+", default=["off"], choices=("off", "auto", "forced"))
    ap.add_argument("--repeat", type=int, default=1, help="builds per point, the fastest is kept")
    ap.add_argument("--no-warm", action="store_true", help="skip the warm-up build per family")
    ap.add_argument(
        "--json",
        action="store_true",
        help="write results/bench_mesh_build.json after every point (earlier rows are kept)",
    )
    args = ap.parse_args()
    families = list(FAMILIES) if "all" in args.family else args.family
    run(families, args.sizes, args.pool, args.repeat, not args.no_warm, json_out=args.json)
    if args.json:
        print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
