"""Certificate: the mirrored halves of a ParaView session carry the field.

Reproduces the measurements DD-169 cites, on the pipeline of DD-262.

A symmetry plane is declared once and consumed twice: the monitor plots
continue their data with ``mirror_sign``, and the ParaView session's
per-monitor Python filter builds the mirrored copy in numpy with the
signs the export resolved through the same function.  (Until DD-262
the renderer's reflection filter did it, transforming every
3-component array as a polar vector — right for one pairing of field
and wall type, off by a global minus for the other, in a picture that
looks symmetric either way.)

Three checks, on a cavity carrying one plane of each type:

1.  **The planes reach the filter.**  Every declared plane stands in the
    field filter's own configuration, and the displayed cut hangs off
    that filter rather than off the unmirrored reader.

2.  **Components stay together.**  The single components still equal the
    components of the vector array everywhere: one dataset, one
    lattice, so a glyph is coloured and aimed from the same place.

3.  **The continuation is the monitors'.**  Across each plane the field
    reproduces the sign ``mirror_sign`` prescribes, component by
    component, including on the seam itself.

Needs ``pvpython`` on the PATH; without it the checks are skipped rather
than failed.

Run from the repository root::

    python validation/paraview_symmetry_certificate.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import magnelio as mio
from magnelio.boundaries.boundary_conditions import BoundaryConditions
from magnelio.geo import Brick
from magnelio.io.paraview import _symmetry_config
from magnelio.post._symmetry import mirror_sign

F_MAX = 12e9
#: Sampling of the mirrored volume.  Odd counts put a sample exactly on
#: each wall, which is where an odd component has to vanish.
SAMPLES = [41, 41, 31]
#: Relative to the field's own peak.  The reflection is a coordinate
#: negation and the correction a factor of one, so the halves agree to
#: rounding — the measured spread sits three decades below this.
TOLERANCE = 1e-12


def _build(path: Path):
    """A quarter cavity behind a magnetic and an electric symmetry plane."""
    model = mio.GeometryModel()
    model.add(Brick(origin=(0, 0, 0), size=(30e-3, 20e-3, 15e-3), material=mio.Material.air()))
    model.boundary_conditions = BoundaryConditions(xmin="ForceSymmetryPMC", ymin="ForceSymmetryPEC")
    mesh = mio.Mesh.from_geometry(model, mio.MeshControl(min_nodes_per_wavelength=8), f_max=F_MAX)
    return mio.AnalysisEigenmode(
        mesh=mesh, n_modes=2, verbose=False, project=str(path), geometry=model
    ).run()


_PROBE = """
import ast, json, sys
import numpy as np
if not hasattr(np, "in1d"):
    np.in1d = np.isin
from paraview import simple, servermanager as sm
from vtk.util import numpy_support as ns

session, out_path, samples = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
# Expected continuation factors, resolved by the caller through
# mirror_sign.  Re-deriving them here would put a second copy of the
# rule next to the one under test — and the copy is what gets it wrong.
expected = json.loads(sys.argv[4])
scope = {"__file__": session, "__name__": "session"}
source = open(session).read().split("\\nbuild()")[0]
exec(compile(source, session, "exec"), scope)
scope["build"]()
config = scope["CONFIG"]
planes = config["symmetry"]
name = config["monitors"][0]["name"]

# The monitor's Python filter carries the planes and the signs in its
# own CFG line (DD-262); the slice must hang off that filter.
report = {"planes": planes, "reflections": [], "displayed_from": None}
field = simple.FindSource("%s_field" % name)
cfg = ast.literal_eval(field.Script.splitlines()[0].split("=", 1)[1].strip())
report["reflections"] = [list(p) in [list(q) for q in cfg["symmetry"]] for p in planes]
sl = simple.FindSource("%s_slice" % name)
if sl is not None:
    report["displayed_from"] = sl.Input.SMProxy.GetGlobalID() == field.SMProxy.GetGlobalID()

# Sample the same filter at the certificate's own resolution (odd
# counts put a sample on every wall).
cfg["dims"] = samples
probe = simple.ProgrammableFilter(Input=simple.FindSource(name))
probe.OutputDataSetType = "vtkImageData"
head = "CFG = %r\\nMODE = %r\\n" % (cfg, "section")
probe.Script = head + scope["FIELD_SCRIPT"]
probe.RequestInformationScript = head + scope["INFO_SCRIPT"]
probe.RequestUpdateExtentScript = scope["UPDATE_SCRIPT"]
probe.UpdatePipeline()
image = sm.Fetch(probe)
dims = image.GetDimensions()
data = image.GetPointData()


def array(label):
    return ns.vtk_to_numpy(data.GetArray(label)).reshape(dims[2], dims[1], dims[0], -1)


ghost = data.GetArray("vtkGhostType")
mask = (array("vtkGhostType")[..., 0] == 0) if ghost is not None else np.ones(dims[::-1], bool)
report["components_together"] = {}
report["continuation"] = {}
for field_name in ("E", "H"):
    if data.GetArray(field_name) is None:
        continue
    vector = array(field_name)
    scalars = {axis: array(field_name + axis)[..., 0] for axis in "xyz"}
    # ParaView runs the field filter in this module's namespace and its
    # preamble shadows the builtin reductions with numpy_interface's, so
    # nothing below relies on max().
    scale = float(np.max([np.abs(v)[mask].max() for v in scalars.values()]))
    report["components_together"][field_name] = float(
        np.max([np.abs(scalars[a] - vector[..., k])[mask].max() for k, a in enumerate("xyz")])
    ) / scale
    for axis, wall, at_low, kind in planes:
        flip = 2 - "xyz".index(axis)
        for a in "xyz":
            label = "%s%s %s%s" % (field_name, a, axis, kind)
            want = expected[label]
            both = mask & np.flip(mask, axis=flip)
            gap = np.abs(np.flip(scalars[a], axis=flip) - want * scalars[a])[both].max()
            report["continuation"][label] = float(gap) / scale

open(out_path, "w").write(json.dumps(report))
"""


def _expected_factors(planes) -> dict:
    """Continuation factor of every field component across every plane."""
    return {
        "%s%s %s%s" % (field, name, axis, kind): mirror_sign(field, comp, "xyz".index(axis), kind)
        for axis, _wall, _at_low, kind in planes
        for field in ("E", "H")
        for comp, name in enumerate("xyz")
    }


def _run_probe(session: Path, workdir: Path, expected: dict):
    """Build the session under ``pvpython`` and return its measurements."""
    probe = workdir / "probe.py"
    probe.write_text(_PROBE, encoding="utf-8")
    out = workdir / "report.json"
    proc = subprocess.run(
        [
            shutil.which("pvpython"),
            "--force-offscreen-rendering",
            str(probe),
            str(session),
            str(out),
            json.dumps(SAMPLES),
            json.dumps(expected),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if not out.exists():
        print("  pvpython failed:")
        print("   ", (proc.stderr or proc.stdout or "").strip()[-800:])
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def check_planes_reach_the_renderer(report) -> bool:
    print("\n[1] every declared plane reaches the monitor's field filter")
    for (axis, wall, at_low, kind), built in zip(report["planes"], report["reflections"]):
        print(f"      {axis} at {wall * 1e3:+8.4f} mm  {kind:3s}  in the filter's CFG: {built}")
    print(f"      displayed cut hangs off the field filter: {report['displayed_from']}")
    return all(report["reflections"]) and report["displayed_from"] is True


def check_components_stay_together(report) -> bool:
    print("\n[2] single components still equal the vector's components")
    ok = True
    for field, gap in report["components_together"].items():
        print(f"      {field}: max relative difference {gap:.3e}")
        ok = ok and gap <= TOLERANCE
    return ok and bool(report["components_together"])


def check_continuation(report, expected: dict) -> bool:
    print("\n[3] continuation across each plane matches mirror_sign")
    ok = True
    for label, gap in sorted(report["continuation"].items()):
        print(f"      {label:12s} expected {expected[label]:+.0f}   mismatch {gap:.3e}")
        ok = ok and gap <= TOLERANCE
    return ok and bool(report["continuation"])


def main() -> int:
    print("=" * 72)
    print("ParaView symmetry certificate (DD-169)")
    print("=" * 72)
    if shutil.which("pvpython") is None:
        print("\n  pvpython not on PATH — nothing measured, nothing claimed.")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        project = _build(workdir / "cavity")
        written = project.export_paraview_eigenmodes(bake_state=False)
        expected = _expected_factors(_symmetry_config(project))
        print(f"\n  session: {Path(written['script']).name}")
        report = _run_probe(Path(written["script"]), workdir, expected)
        if report is None:
            print("\n  FAIL")
            return 1

        results = {
            "planes reach renderer": check_planes_reach_the_renderer(report),
            "components together": check_components_stay_together(report),
            "continuation": check_continuation(report, expected),
        }

    print("\n" + "=" * 72)
    for name, passed in results.items():
        print(f"  {name:24s} {'PASS' if passed else 'FAIL'}")
    verdict = all(results.values())
    print(f"\n  {'PASS' if verdict else 'FAIL'}")
    print("=" * 72)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
