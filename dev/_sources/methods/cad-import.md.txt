# CAD import

Geometry does not have to be drawn in Magnelio.  A model that already
exists in a CAD system is read in with
{func}`~magnelio.io.import_step` (or, for the kernel's own dump,
{func}`~magnelio.io.import_brep`), and what comes back is ordinary
geometry: solids that carry a material, take part in Boolean
operations, accept the chainable verbs and go into a
{class}`~magnelio.GeometryModel` like anything built with the
primitives (`io/cad.py`, DD-178).

## What an exchange format carries, and what it does not

A boundary-representation exchange file records finished solids.  The
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
| Geometry | exact | exact |
| Length unit | in the file | **absent** |
| Part names | yes | no |
| Display colours | yes | no |
| Assembly structure | yes | no |

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

## Only solids are imported

A material fills a volume, so only solid bodies can become part of a
model.  Surface bodies in the file (free faces, unstitched shells) are
reported and skipped; if a file contains nothing else, the import
fails rather than returning an empty model.  Turn such bodies into
solids in the CAD system first.

## Assigning materials by name

`materials` accepts a single material, applied to every solid, or a
dict keyed by the names the solids carry in the file:

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
vacuum.  Importing without any mapping is the way to see what is in a
file:

```python
for solid in import_step("connector.step").members():
    print(solid.name, solid.bounding_box())
```

## Assemblies

An assembly is flattened on import.  Every solid comes back placed
where it sits in the assembly, as a member of one
{class}`~magnelio.geo.Group`; the tree structure itself is not
reproduced, because a Group is what the rest of the API consumes — it
distributes transformations over its members, keeps each member's
material, and is flattened again when the model takes it.

Names come from the component instance where there is one, and from
the part it refers to otherwise.  A part that contains several solids
is split into `<name>_1`, `<name>_2`, …; a part with no name at all
becomes `solid_1`, `solid_2`, … in file order.

## Colours

A colour read from a STEP file is display information and nothing
else: it selects the hue a solid is drawn with in
{meth}`~magnelio.GeometryModel.plot` and in the ParaView export, and
it survives a project store round-trip.  It never touches the physics,
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

## Limits

Sheet bodies, stitching surfaces into solids, and formats other than
STEP and BREP (IGES, mesh formats) are not supported.
