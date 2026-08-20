# Board import

A printed circuit board is drawn in a layout tool, not in a field
solver, and it leaves that tool as *fabrication data*: the set of files
a board house is sent.  {func}`~magnelio.io.import_pcb` reads that set
and returns ordinary geometry — one solid per layer of the stackup,
one per plated hole, each carrying a name and a material
(`io/pcb.py`, DD-179).

```python
from magnelio import GeometryModel
from magnelio.io import import_pcb

board = import_pcb("fabrication/")
model = GeometryModel().add(board)
```

## What a fabrication export carries

The set is deliberately the fabrication output rather than any one
tool's project file: Gerber and Excellon are what every layout tool
writes and what every board house reads, so the import is not tied to
the tool the board was drawn in.

| File | What it carries |
|---|---|
| `*.gbr` per copper layer | where the copper is, in two dimensions |
| `*.gbr` profile layer | the board edge, as a line |
| `*.drl` drill files | hole positions, diameters, plating, layer span |
| `*.gbrjob` job file | the stackup: layer order, thicknesses, dielectric |

The job file is required.  Gerber files are flat drawings: nothing in
them says how thick a copper layer is, how far apart two layers sit, or
what the dielectric between them is made of.  Without the job file
there is no third dimension to build, and the import says so rather
than inventing one.  In a layout tool the job file is part of the
Gerber export; the stackup it records has to be filled in first (in
KiCad, *Board Setup → Physical Stackup*), because a stackup without
thicknesses is refused for the same reason.

Solder mask and silkscreen are ignored.  They are thin coatings whose
effect on a board's fields is below the accuracy of everything else in
the model.

## The stackup is taken literally

Each copper and dielectric layer of the stackup becomes one solid at
its own height.  A layer is built as an *area* in a plane first — pads,
tracks and filled zones merged where they overlap, clipped to the board
outline, with the drilled circles removed — and raised into a slab
once, at the end.  Nothing is extruded and then fused, which is what
keeps a Boolean operation from having to resolve thousands of
35 µm-thin fragments against each other.

The origin sits on the **top face of the topmost dielectric**, and the
stack grows downwards:

| Layer | z range |
|---|---|
| `F.Cu` | `0` … `+35 µm` |
| `dielectric_1` | `−1.53 mm` … `0` |
| `B.Cu` | `−1.565 mm` … `−1.53 mm` |

Fixing the origin to the substrate rather than to the outermost copper
keeps the reference plane where the fields are: adding a layer to the
stack does not move the rest of the board.

### What the layers are named

Names are the key materials are assigned against, so they have to be
unique and to survive a re-export.  Copper layers keep the name the
stackup gives them (`F.Cu`, `In1.Cu`, `B.Cu`).  Dielectrics are named
by their position — `dielectric_1`, `dielectric_2`, … — because layout
tools name every core and prepreg after its material, and two layers
called `FR4` would make the assignment ambiguous.  Plated barrels are
`via_1`, `via_2`, … numbered in coordinate order.

```python
board = import_pcb("fabrication/")
print([solid.name for solid in board.members()])
```

### Holes and vias

A plated hole becomes a solid copper cylinder running between the
copper layers its drill file declares — the whole board for a through
hole, two adjacent layers for a blind or buried via.  The same circle
is removed from every layer the hole passes through, so the barrel
fills the void exactly: barrel and pad meet on coincident faces, with
no overlap for the mesher to resolve and no gap between them.

The barrel is solid metal rather than a plated wall around a void.  The
wall is a closed conductor and the space it encloses carries no field,
so the two are the same model with one fewer surface.

Unplated holes and slots are removed from everything they pass through
and leave no solid behind.

### Where the model is not the board

Between the copper features of an inner layer, a real board has
prepreg that flowed into the gaps during lamination.  Here those gaps
fall to the model background instead, because filling them would mean a
35 µm-thin dielectric slab beside the copper — and a slab that thin
forces the grid to resolve it, which is exactly what the thin-sheet
treatment below exists to avoid.  The difference is a 35 µm sliver of
substrate against air, at the height of an inner layer.

## Meshing a board

Copper is thin.  At 35 µm it is two decades below any cell size a board
simulation can afford, and a grid that resolved it would be unusable.
The mesher does not have to: a **perfectly conducting** layer thinner
than the cell-size floor is given a single grid plane on its substrate
side, and its thickness enters through the sub-cell material fractions
of the neighbouring cells instead of through a layer of cells of its
own (see {doc}`meshing-conformal`).

Two conditions have to hold, and both are the caller's to arrange:

1. **The cell-size floor has to be set.**  The mechanism runs only when
   {class}`~magnelio.MeshControl` carries a `min_cell_size`; the floor
   is what "thin" is measured against, and without it the layer is
   meshed as an ordinary solid.
2. **The copper has to be a perfect conductor.**  This is the default
   for an imported board.  Giving `copper=` a finite conductivity turns
   the layers into ordinary lossy solids, which the grid then has to
   resolve.

```python
from magnelio import MeshControl
from magnelio.mesh.mesher import Mesh

mesh = Mesh.from_geometry(
    model,
    MeshControl(min_nodes_per_wavelength=20, min_cell_size=200e-6),
    f_max=5e9,
)
```

The floor has to be larger than the copper thickness — that is the
comparison the detection makes — and there is no reason to put it close
to it: its purpose is to keep the grid from following the metal, so a
floor several times the thickness, chosen from the smallest feature the
*fields* need resolved, is the useful setting.  200 µm on 35 µm copper
is a typical starting point.

Because a copper layer arrives as **one** solid spanning the whole
board, this works regardless of how narrow the individual tracks are:
what is compared against the floor is the layer's thickness, and the
copper's actual footprint is classified afterwards, feature by feature.

Conductor loss on such a layer is not modelled by giving the metal a
conductivity — see {doc}`conductor-losses` for the surface-impedance
route, which applies to a perfectly conducting boundary.

## Materials

Copper layers and barrels default to a perfect electric conductor.
Dielectrics take the permittivity the job file states; a dielectric
whose permittivity is missing arrives **without** a material — a
construction body that the model refuses to accept — rather than
silently as vacuum.

Anything can be overridden by name:

```python
from magnelio import Material

board = import_pcb(
    "fabrication/",
    {"dielectric_1": Material(name="RO4350B", epsilon=(3.66,) * 3)},
)
```

Keys may use shell wildcards (`"via_*"`), a literal name beats a
wildcard, and a key that matches no solid is an error rather than a
silent no-op.

### Loss tangent

A stackup states a loss tangent, and the import reports it but does not
model it.  A loss tangent is a single number, and the job file does not
record the frequency it was measured at; a loss tangent that does not
vary with frequency violates Kramers–Kronig, so there is nothing in it
alone to build a causal material from.

Supplying that missing frequency is the caller's decision, and once it
is supplied the standard causal substrate model
({doc}`dispersive-materials`) does the rest:

```python
from magnelio import Material
from magnelio.materials.dispersion import DispersionModel

fr4 = Material.dispersive(
    name="FR4",
    model=DispersionModel.djordjevic_sarkar(
        eps_r=4.5, tan_delta=0.02, f_ref=1e9
    ),
)
board = import_pcb("fabrication/", {"dielectric_1": fr4})
```

## Limits

The reader refuses, with the file and line, anything it cannot turn
into copper rather than dropping it silently — a missing pad in a board
is not something a caller can be expected to notice.  Currently
refused: step-and-repeat blocks, negative images, the deprecated image
transformations, single-quadrant arc mode, tracks drawn with a
rectangular aperture, and the moiré and thermal aperture macro
primitives.  In each case the remedy is to re-export the layer with the
construct resolved, which every layout tool can do.
