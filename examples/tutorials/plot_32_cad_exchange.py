"""
Exchanging solids and free sheets with CAD
==========================================

A mechanical CAD file can contain both volumetric parts and free surfaces.
Here a housing and a zero-thickness conductor outline are exchanged through
STEP. The outline gets a measured copper thickness before it enters the
simulation model. The original sheet can be exported separately for CAD work.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from magnelio import GeometryModel, geo
from magnelio.io import export_brep, export_step, import_brep, import_step

# %%
# Construct a small source file, so this example can run without an external
# mechanical drawing. A Profile is a standalone sheet in the exchange file.
housing = geo.Brick(
    origin=(0.0, 0.0, 0.0),
    size=(20e-3, 10e-3, 3e-3),
    material="pec",
    name="housing",
)
trace = geo.Profile.rectangle(
    center=(10e-3, 5e-3, 3e-3),
    size=(8e-3, 1e-3),
    material="pec",
    name="trace",
)

with TemporaryDirectory() as directory:
    source = Path(directory) / "assembly.step"
    export_step(source, geo.Group(housing, trace), unit="mm")

    # %%
    # STEP carries the millimetre unit, names and display colours. It does
    # not carry electromagnetic material properties, so supply those here.
    parts = import_step(source, {"housing": "pec", "trace": "pec"})
    by_name = {body.name: body for body in parts.members()}
    assert isinstance(by_name["housing"], geo.Solid)
    assert isinstance(by_name["trace"], geo.ImportedSheet)

    # %%
    # The imported sheet remains zero-thickness. Give the conductor its
    # physical thickness before asking a volume mesh to represent it.
    copper = by_name["trace"].thickened(35e-6)
    model = GeometryModel().add((by_name["housing"], copper))

    # %%
    # Export only the chosen CAD geometry. BREP has no unit annotation, so
    # the reader must be told the same unit used at export.
    selected = Path(directory) / "selected.step"
    export_step(selected, (by_name["housing"], by_name["trace"]), unit="mm")
    native = Path(directory) / "selected.brep"
    export_brep(native, (by_name["housing"], by_name["trace"]), unit="mm")
    assert len(list(import_step(selected).members())) == 2
    assert len(list(import_brep(native, unit="mm").members())) == 2
