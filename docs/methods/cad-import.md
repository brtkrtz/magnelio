# CAD import

Geometry does not have to be drawn in Magnelio.  A model that already
exists in a CAD system is read in with
{func}`~magnelio.io.import_step` (or, for the kernel's own dump,
{func}`~magnelio.io.import_brep`), and what comes back is ordinary
geometry: solids and standalone sheets with distinct dimensional
semantics. Solids take part in Boolean operations and go into a
{class}`~magnelio.GeometryModel` like anything built with the
primitives.

## What an exchange format carries, and what it does not

A boundary-representation exchange file records finished bodies.  The
construction history that produced them — the sketch, the parameters,
the feature tree — is not part of the file and cannot be recovered from
it.  An imported solid is therefore not editable the way a
{class}`~magnelio.geo.Cylinder` is: to change a dimension, change it in
the CAD system and export again.  What *is* stable across such a
re-export are the part names, which is why they carry the material
assignment (below).

Two formats are supported, and they differ in their metadata:

| | STEP (`.step`, `.stp`) | BREP (`.brep`) |
|---|---|---|
| Geometry | exact solids and free sheets | exact solids and free sheets |
| Length unit | in the file | **absent** |
| Part names | yes | no |
| Display colours | yes | no |
| Assembly structure | yes; flattened on import | no |
| Material physics and named topology | no | no |

STEP is therefore the format to prefer.  BREP is the geometry kernel's
native dump; it round-trips a shape without any conversion at all, but
because it states no unit, {func}`~magnelio.io.import_brep` demands one
from the caller:

```python
horn = import_brep("horn.brep", unit="mm", material=pec)
```

Getting that argument wrong scales the model by a factor of a thousand
and nothing in the file contradicts it — one more reason to use STEP,
where the unit is read from the file and the geometry arrives in
meters no matter what it was drawn in.

Materials are not carried by either format in any usable form.  What a
CAD system stores under "material" is a name for a parts list, not the
permittivity, permeability and conductivity a field solver needs, so
assigning materials on import is not a shortcoming of the reader — it
is where the physics enters the model.

## Solids and free sheets

Free faces and faces of open shells import as independent
{class}`~magnelio.geo.ImportedSheet` members. Faces that bound a solid
remain part of that solid and are never duplicated as free sheets. A
shell with several faces becomes several sheet members, so each can be
given a separate thickness. Free curves and points are skipped with a
warning; a file containing no solid or free face is rejected.

An imported sheet has **zero physical thickness**. Giving it a material
does not make it directly meshable. Construct a resolved solid first,
for example `sheet.thickened(35e-6, material="pec")` for a conductor
with 35 µm thickness, and add that solid to the model. Curved sheets
need a thickness compatible with their curvature; the CAD kernel may
reject an offset that folds or cannot close.

## Assigning materials by name

`materials` accepts a single material, applied to every solid and free
sheet, or a dict keyed by the body names in the file:

```python
parts = import_step(
    "connector.step",
    {
        "shell": pec,
        "pin": pec,
        "insulator*": ptfe,   # shell wildcards are allowed
    },
)
```

The rules are chosen so that a mapping that no longer fits the file
says so instead of quietly doing something else:

* A **literal name beats a wildcard**, so a general rule plus an
  exception is written directly (`{"*": ptfe, "pin": pec}`).
* Two wildcards claiming the same solid **for different materials**
  are an error — name that solid literally to settle it.
* A key that **matches no solid** is an error, and the message lists
  the names the file actually contains.  A renamed part is a typo the
  first time it happens; without this it would silently lose its
  material.

Solids that no key matched come back as **construction bodies**: they
have no material, which makes them usable as Boolean operands but
rejects them at
{meth}`~magnelio.GeometryModel.add`.  That is deliberate — a
half-mapped assembly should not mesh as if the unmapped parts were
vacuum. Sheets may remain materialless until their thickness is chosen.
Importing without any mapping is the way to see what is in a
file:

```python
for body in import_step("connector.step").members():
    print(type(body).__name__, body.name, body.bounding_box())
```

## Assemblies

An assembly is flattened on import. Every body comes back placed
where it sits in the assembly, as a member of one
{class}`~magnelio.geo.Group`; the tree structure itself is not
reproduced, because a Group is what the rest of the API consumes — it
distributes transformations over its members, keeps each member's
material, and is flattened again when the model takes it.

Names come from the component instance where there is one, and from
the part it refers to otherwise. A part with several bodies is split
into `<name>_1`, `<name>_2`, …; an unnamed body gets a generated
`solid_…` or `sheet_…` name. Repeated instances may share a name;
one material mapping key then applies to all of them. Distinguish
instances by their world positions or give them unique CAD names.

## Colours

A colour read from a STEP file is display information and nothing
else: it selects the hue an imported solid is drawn with in
{meth}`~magnelio.GeometryModel.plot` and in the ParaView export, and
it survives that solid's project store round-trip. An imported sheet
retains its CAD colour while it is a sheet; thickening it creates a new
solid. Colour never touches the physics,
and it never overrides a colour prescribed by the material — a
material with an explicit `color` keeps it.  Opacity stays with the
material as well, so metals remain opaque and dielectrics translucent
whatever the CAD system painted them.

## Healing

CAD files cross kernel boundaries, and what survives the trip is not
always a valid solid: tolerances disagree, faces do not quite close,
orientations flip.  {func}`~magnelio.io.import_step` therefore repairs
each solid by default (`heal=True`), which is close to free on a file
that was clean to begin with.  `import_brep` does not, since a BREP
file comes from this very kernel.

`unify=True` additionally merges neighbouring faces that lie on the
same surface.  Exporters routinely split what is geometrically one
plane into many patches; merging them makes the solid simpler and the
conformal mesh classification cheaper, but it edits the topology, so
it stays opt-in.

If a solid is still invalid after the repair pass, the import warns and
carries on, naming the solid.  Meshing it may or may not give sensible
results — the warning is there so a strange mesh has an explanation.

## Exporting selected geometry

{func}`~magnelio.io.export_step` and {func}`~magnelio.io.export_brep`
write one Solid, Sheet, Group or an iterable of them. They flatten
Groups and preserve each selected body's world placement. Free curves
are not accepted. Both accept an output unit (`"mm"` is the STEP
default; BREP requires `unit=`). STEP records the unit,
names and display colours. BREP records only exact geometry, so keep
the chosen unit beside the file for the next reader. Neither format
records Magnelio materials, construction history or named-selection
replay; use a Magnelio project for those. Face ordering may change
across exchange.

Both exporters refuse to replace an existing path unless
`overwrite=True` is given. They write to a temporary file first, so
a failed write does not truncate the previous file.

```python
from magnelio import GeometryModel
from magnelio.geo import ImportedSheet
from magnelio.io import export_step, import_step

parts = import_step("mixed.step", {"housing": "pec", "trace": "pec"})
housing = next(body for body in parts.members() if body.name == "housing")
trace = next(body for body in parts.members() if body.name == "trace")
assert isinstance(trace, ImportedSheet)
trace_solid = trace.thickened(35e-6)
model = GeometryModel().add((housing, trace_solid))
export_step("selected.step", (housing, trace), unit="mm")
```

The exported `trace` is still a sheet: the `trace_solid` exists only in
the simulation model. Export that solid instead when its resolved
thickness belongs in the CAD handoff.

## Limits

Stitching surfaces into solids, free-curve exchange, and formats other
than STEP and BREP (IGES, mesh formats) are not supported. Sheet
materials have no thin-sheet mesh law; give a sheet resolved thickness
before meshing.

## Raw BREP versus CAD exchange

`read_brep(path)` reads an ordered list of OpenCascade shapes in metres;
`write_brep(shapes, path)` writes that sequence in its given order. This raw
pair carries no material or named-topology metadata. The CAD-exchange pair
`import_brep(path, units=...)` / `export_brep(path, geometry, units=...)`
works with Magnelio geometry and explicit exchange units. Use that pair for
user CAD workflows. The raw writer's `shapes, path` and CAD export's
`path, geometry` argument order are deliberate exceptions.
