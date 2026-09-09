# Conductor losses

Magnelio offers two routes for wall losses on good conductors: a
perturbative post-processing route (default) and a broadband
time-domain surface-impedance boundary (opt-in).

## Perturbative wall losses (post-processing)

With lossless PEC walls in the field solve, the dissipated power is
evaluated perturbatively from the tangential magnetic field,

$$
P_{\text{loss}}(f) = \tfrac12\, R_s(f) \sum_{\text{wall}}
 w\,|H_{\tan}|^2, \qquad R_s = \sqrt{\pi f \mu / \sigma},
$$

(`monitors/wall_loss.py`, `postprocessing/wall_loss.py`, DD-082).
This is the classical power-loss perturbation method of microwave
engineering {cite}`pozar2012,jackson1999`.  Two
accuracy refinements are in-house (DD-087): exact conformal
wall areas on curved conductors (removing the $4/\pi$ staircase
over-count) and a conformal tangential-H sampling rule
(uncut-face booking with a normal-direction walk).

## The surface current

On a conductor the tangential magnetic field at the wall *is* the
surface current density, $\mathbf J_s = \mathbf n \times \mathbf H$
[A/m], with $\mathbf n$ the outward normal.  The normal component of H
vanishes there, so the cross product with the full vector is the same
thing as with its tangential part, and $|\mathbf J_s| = |H_\text{tan}|$.

`recording.surface_current(mesh)` — on a field monitor's recording or
spectrum, or on a single `FieldState` — returns one value per wall patch
of the mesh: its position, its outward normal, its conducting area and
the current on it.  The field must carry all three magnetic components
over the whole grid (`fields=["H"]`, no `corners`), because a patch the
monitor does not cover has no current to state.

The patches are the same enumeration the wall loss is booked from, which
is what makes the two consistent.  The **magnitude** is that booking
split by patch — $|\mathbf J_s|^2 A = \sum w |H|^2$ over the samples the
patch booked — so `power_loss(R_s)` reproduces `MonitorWallLoss`
exactly, not approximately.  The **direction** is $\mathbf n \times
\mathbf H$ with the weight-averaged H of those samples and the patch's
own normal, which on a curved conductor is the direction of the sub-cell
wall vector rather than a staircase axis.

Two things to know before reading numbers off it.

**The absolute current is as accurate as the wall sampling**, which is
calibrated on the loss.  The wall samples sit a small step off the
surface and their weights (with the curvature pullback) compensate that
for $|H|^2$; the current is linear in H, and the residual displacement
has opposite sign on a convex and a concave wall.  Measured on an air
coax: the inner conductor's total current lands within 1.3 % of
$\sqrt{P/Z_0}$ and the shield within 5.0 %, and neither refines away.
The *distribution* — where the current runs, and how it crowds at an
edge — is far better resolved than that integral, and it is what a
current picture is read for.

**A port plane is not a wall.**  Where a conductor leaves the model
through a boundary, that face holds the feed's cross-section: the
structure continues through it, and counting it as surface overstates
both current and loss (9.5 % on the coax above).  The mesh cannot tell
that face from one where the domain simply ends inside metal — a port
sits on a PEC face and is substituted at run time — so the call asks
once, and `exclude_faces=("zmin", "zmax")` names the port planes while
`exclude_faces=()` says there are none.

`js.show(geometry)` draws it: one arrow per patch over the model,
coloured by $|\mathbf J_s|$, with `density=` thinning a fine mesh out so
the arrows stay readable.

## Surface roughness

Roughness enters the perturbative chain as one real,
frequency-dependent multiplier $K(f)$ on the surface resistance,
$R_{s,\text{rough}} = K(f)\,R_{s,\text{smooth}}$
(`materials/roughness.py`, DD-088).  Implemented models:

- **Hammerstad** — the classical RMS-height curve fit of Hammerstad
  and Jensen {cite}`hammerstadjensen1980`.
- **Huray "snowball"** — the physics-based sphere-cluster model of
  Huray et al. {cite}`huray2007`; the
  loss-factor form implemented is Bracken's eq. (5)
  {cite}`bracken2012`.
- **Cannonball-Huray parameterisation** — sphere radius and coverage
  from a single $R_z$ datasheet number via close packing,
  after Simonovich {cite}`simonovich2015`.

A real $K(f)$ is non-causal as a time-domain impedance (noted by
Bracken {cite}`bracken2012`); this is admissible here
because the perturbative chain evaluates power per frequency bin and
never forms a time-domain impedance.

## Broadband time-domain SIBC 

SIBC is currently an opt-in: `wall_model="sibc"`.
The surface-impedance boundary condition realises the **Leontovich
condition** $E_{\tan} = Z_s(\omega)\,(\hat n \times \mathbf H)$
{cite}`leontovich1948,senior1960` directly in the
leapfrog update (`solver/sibc.py`, DD-091):

- $Z_s(\omega)$ — smooth-metal $\sqrt{j\omega\mu/\sigma}$ or the
  causally completed rough impedance — is fitted as a
  **Foster/Stieltjes ladder** $c_0 + \sum_p c_p s/(s+b_p)$ with
  non-negative coefficients by **NNLS** on fixed log-spaced poles
  (`materials/surface_impedance.py`).  Foster's canonical positive-
  real ladder form is classical network synthesis
  {cite}`foster1924`; NNLS is Lawson and
  Hanson {cite}`lawsonhanson1974`.  Because
  every branch is
  elementarily passive, the time-domain recursion is dissipative by
  construction — stability is unconditional at the unchanged lossless
  CFL, independent of fit accuracy — an in-house result (internal derivation
  dossier `investigations/sibc/DERIVATION.md`, kept outside the public
  repository).
- The causal reactance of a rough surface is completed from the real
  roughness excess $(K-1)R_s$ by a subtracted **Kramers–Kronig**
  quadrature {cite}`kronig1926,kramers1927`.
- The per-branch states are advanced with the trapezoidal rule and
  folded into the H update like a magnetic surface conductivity.

Approximating a surface impedance by a low-order rational function
and convolving it recursively in FDTD is an established technique:
Maloney and Smith {cite}`maloneysmith1992`, Beggs,
Luebbers, Yee and Kunz {cite}`beggs1992`, and the
first-order-section approach of Oh and Schutt-Ainé
{cite}`ohschuttaine1995` are the closest published
antecedents; Magnelio's specific NNLS-Foster construction with its
unconditional dissipation identity is in-house.

The conformal booking of SIBC faces (which faces carry the damping
term and with which geometric weight $G_f = A_f/l^2_{\text{dual}}$)
reuses the DD-087 conformal wall-area machinery.
