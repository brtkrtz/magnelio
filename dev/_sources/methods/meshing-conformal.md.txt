# Mesh generation and conformal geometry

## Geometry kernel and material filling

Solid geometry is authored through a CSG layer (`geometry/`) backed by
the Open CASCADE kernel via `pythonocc-core` (DD-003, DD-016).
Material assignment on the grid uses exact boundary-representation
queries (solid classification, 3D face–solid intersection, planar
cross-sections) rather than voxel sampling.  This is engineering
infrastructure on top of a third-party kernel, not a numerical-methods
contribution.

## Graded Cartesian mesh

The mesh generator (`mesh/mesher.py`) produces a graded (non-uniform)
Cartesian tensor-product grid: geometry-derived fixpoints ("anchors",
plane clustering, DD-059…DD-062) plus feature-based two-scale
refinement (`h_fine` near features, `h_coarse` in bulk, geometric
grading between them, DD-028).  Graded Cartesian meshes and the
accuracy trade-offs of local grading are standard FDTD/FIT practice
{cite}`taflovehagness2005`; the specific fixpoint,
plane-clustering and thin-sheet heuristics are in-house engineering.

### Which wavelength sets the bulk cell size

The bulk cell size is `λ / min_nodes_per_wavelength`, and on a
tensor-product grid the question is *which* λ.  A grid line spans the
whole domain, so the finest sensible resolution is per *slab*: each
interval between two grid planes on an axis is a slab of the domain,
and the densest material whose bounding box reaches into that slab
sets the slab's wavelength (DD-192, the default
`MeshControl(wavelength_rule="local")`).  The air box around a small
ceramic is meshed at the air wavelength on every axis interval the
ceramic does not reach; the slabs the ceramic occupies — and every
slab of an axis the ceramic spans entirely, such as the in-plane
axes of a full-width substrate — stay at the ceramic's wavelength.
The background material fills whatever no solid covers and counts in
every slab.  The bounding box is exact for bricks and conservative
for curved or rotated bodies, so a slab is never meshed coarser than
the material in it.  `wavelength_rule="global"` restores the older
rule — the densest material anywhere sets one bulk size for the
whole domain.

The two rules differ only far from material interfaces.  Feature
refinement (`min_cells_per_feature`), the geometric grading from an
interface into the bulk, the DD-107 domain-face buffer and the edge
floor below are the same under both; the edge floor keeps the
densest material's wavelength as its reference, because it bounds
the time step, and the time step follows the smallest cell anywhere.
This is the rule hex-mesh generators apply as per-material mesh
settings; the slab-wise form is its consequence on a tensor grid.

### Which geometry gets a grid plane

Grid planes come from two passes over the CAD model.  The *face* pass
places a plane on every planar face with an axis-parallel normal and
on the axis-normal tangent positions of cylinders and spheres — the
material boundaries.  The *edge* pass (DD-191) places a plane wherever
a B-rep edge lies flat in an axis-normal plane: the circle where a
chamfer cone meets a cylinder, the straight line where a fillet leaves
a box face, the section curves of a loft, the iris and equator
circles of a revolved profile.  These are the positions where a
body's cross-section changes character along an axis, and they are
invisible to the face pass (a chamfer is a cone; a fillet is a quarter
cylinder whose tangent positions lie outside its trimmed extent).
Only sharp edges count: seam edges, degenerated edges and the split
lines a Boolean leaves between two faces of one surface contribute
nothing — a cylinder's seam is a straight line through its axis and
would put a phantom plane there.

Edge planes are a *soft* class.  They never move or outrank a material
plane, and they ask for one cell across the interval they bound —
enough for the cell's midplane to see the feature — rather than the
`min_cells_per_feature` a material gap gets.  They are floored:
an edge plane whose cell would be smaller than
`h_max / max_edge_refinement` (default 4), or than `min_cell_size`,
is dropped and reported by a warning that names the coarsest dropped
position, the cell it would have created and the ratio that keeps it.  The ratio
bounds the time-step cost of resolving small edges (the explicit
loop takes one step bounded by the smallest cell anywhere);
`max_edge_refinement=0` switches the edge pass off.

### Refining the conductor edges

Where a conductor forms a wedge of less than 180° — the edges of a
strip, a patch, an iris — the field and the surface current are
singular: $r^{-1/3}$ at a 90° edge, $r^{-1/2}$ at a knife edge.  A
grid cannot represent that, and everything that integrates the edge
field — the line impedance, the effective permittivity and with them
the phase of $S_{21}$ and the resonant frequency of a patch —
converges only about first order in the cell that holds the edge,
however fine the bulk is.  Measured on a 50 Ω microstrip: the
impedance the port solver reads off the grid is 51.5, 52.2 and
52.6 Ω for edge cells of 50, 25 and 12.5 µm (limit about 52.9 Ω), and
three grids with the same 12.5 µm edge cell agree within 0.1 Ω
although their cell counts differ by almost two.

`MeshControl(singularity_refinement=k)` starts the grading at the
planes that hold such an edge at `h_fine / k` instead of `h_fine`, on
both sides of the plane, and grows by `growth_factor` from there.
Which edges count is read from the CAD model: a convex edge of a
metal body (PEC or lossy metal), or a concave edge of a non-metal
body whose surroundings at the edge are metal — a vacuum body cut
out of a PEC background, where the iris rim is the sharp metal wedge.
The corners of a cavity are concave metal edges and regular; a
fillet's onset is tangential; dielectric edges are much weaker and
not refined.  The refinement never adds a plane (the edge's plane is
a material face or an edge plane already), never touches the domain's
own end planes, and stops at `min_cell_size`.

The factor is off by default, and the reason is the time step.  The
edge cell bounds it, so a factor of 2 halves the step and adds the
cells of the two ramps: on the mesh-convergence how-to's structure
the factors 1, 2 and 3 lie on one cost-versus-error curve — the
factor moves resolution from the bulk to the edges, it does not buy
accuracy for free.  It pays where the impedance or the effective
permittivity is the quantity of interest, where the time step is
bound by a `min_cell_size` floor or by another axis anyway (then the
edge cells are free), and where memory rather than time is the
limit: at equal edge cell the refined grids of the microstrip need a
third fewer cells.  Model-wide refinement is still what
`min_nodes_per_wavelength` is for.

Why a feature below the grid is *silent* rather than approximately
represented is a property of the conformal material matrices, next.

## Conformal sub-cell material matrices (partially filled cells)

Material boundaries that cut through grid cells are represented by
**area/length-weighted averaging in the mass matrices** instead of
staircasing: per primal edge the classifier stores an averaged
$\bar\varepsilon$, a free (non-PEC) length fraction and a free dual-face
area fraction; per dual face a corresponding $\bar\mu$ and free-area
data (unified per-edge/per-face sub-cell classification, DD-051).
This family of techniques — retaining the standard leapfrog update and
encoding sub-cell geometry purely in the material matrices — was
introduced for FIT by Krietenstein, Schuhmann, Thoma and Weiland
{cite}`krietenstein1998`.

The average is taken over the dual face *transverse* to the edge; it
does not integrate along the edge.  A boundary that crosses the grid
edges is therefore resolved continuously — moving a cylinder's radius
by a twentieth of a cell moves a resonance by a proportionate,
linearly scaling amount — while a feature that varies *along* the
edges inside one cell layer has no lever at all until it reaches the
layer's midplane.  A chamfer or a shallow recess on a face that is
smaller than half a cell contributes exactly nothing, and then
switches on in one step when it crosses the midplane.  This is the
reason for the edge pass above: with a grid plane at the chamfer's
onset the chamfer occupies a cell layer of its own, whose dual faces
see it.  Where the edge floor drops that plane, the mesher says so.

For perfectly conducting boundaries the classifier additionally
shortens partially-PEC edges (free-length weighting), which is the
conformal-PEC idea of Dey and Mittra {cite}`deymittra1997`
(DD-036, since generalised into the unified classifier of DD-051).

Two refinements are in-house:

- **LC-consistent pair coupling** (DD-053, `couple_face_material_pairs`):
  on dual faces with a locally translation-invariant ladder direction,
  the averaged $\bar\mu$ is replaced by the value that makes the
  co-located product $M_\varepsilon M_\mu$ equal the exact
  transmission-line value $\varepsilon_0\mu_0\,\varepsilon\mu\,d\tilde d$,
  so a discrete travelling wave on a uniform line is exact
  (derivation in `design-decisions.md` DD-053).
- **Enlarged-cell donor** (DD-058, implemented but dormant — measured
  neutral): stabilising strongly cut cells by borrowing area from the
  uncut neighbour.  The published antecedent is the family of
  uniformly stable conformal schemes / enlarged-cell techniques, e.g.
  Zagorodnov, Schuhmann and Weiland {cite}`zagorodnov2003`.

## Thin conducting sheets

Zero-thickness or sub-cell metallisation is detected before gridding
(DD-035, DD-059) and represented as PEC edge masks on the primal grid
(`apply_thin_pec_sheet`, DD-017) — the standard thin-sheet treatment
in Cartesian time-domain solvers {cite}`taflovehagness2005`
(subcell thin-sheet models are ch. 10 there; the
detection pipeline itself is in-house).

## Thin-wire sub-cell model

`ThinWire(curve, radius)` embeds a conductor thinner than a cell as a
PEC edge chain with corrected surrounding material matrices
(`mesh/thin_wire.py`, DD-080).  The model is the classic thin-wire
sub-cell treatment of Holland and Simpson {cite}`hollandsimpson1981`,
realised in the paired $(m, 1/m)$ encoding of
Noda and Yokoyama {cite}`nodayokoyama2002`:
the four encircling
dual faces scale $M_\mu$ by
$m = \ln(\delta/a)/\ln(\delta/r_0)$ and the co-located radial edges
scale $M_\varepsilon$ by $1/m$, so the wire presents the physical
per-length inductance $L' = (\mu/2\pi)\ln(\delta/a)$ while the pair
product — and hence the wave speed and the CFL bound — is untouched.
The bare-grid equivalent radius $r_0 = \kappa_0\,\delta$ with
$\kappa_0 = e^{-\gamma}/2^{3/2} \approx 0.1985$ comes from the
square-lattice Green's function, as given in the thin-wire literature
{cite}`nodayokoyama2002`.

## Mesh quality safeguards

Hard minimum cell size with floor-aware refits and a longitudinal
series-$\varepsilon$ correction (DD-060), per-axis fine resolution
(DD-061), a permanent 30-case stress sentinel (DD-062) and the
reported edge floor (DD-191) are in-house engineering.
