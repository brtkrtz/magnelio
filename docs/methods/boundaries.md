# Boundary conditions

## Absorbing boundary: CPML

Open-region truncation uses the **convolutional perfectly matched
layer (CPML)** of Roden and Gedney {cite}`rodengedney2000`
(`boundaries/cpml.py`, chosen over the uniaxial PML in
DD-001).  The implementation carries the full **complex
frequency-shifted (CFS)** stretching function

$$
s(\omega) = \kappa + \frac{\sigma}{\alpha + j\omega\varepsilon_0}
$$

introduced by Kuzuoglu and Mittra {cite}`kuzuoglumittra1996`,
realised recursively through per-face auxiliary
memory variables $\psi$ with the standard $(b, c)$ update
coefficients.  Profiles are the customary polynomial grading
$\sigma(\rho) = \sigma_{\max}\rho^m$,
$\kappa(\rho) = 1 + (\kappa_{\max}-1)\rho^m$ and a linearly decreasing
$\alpha$; grading choices follow the CPML literature
{cite}`rodengedney2000,taflovehagness2005`.
The PML
concept itself originates with Bérenger {cite}`berenger1994`;
the uniaxial variant used for comparison in DD-001 is
Gedney's {cite}`gedney1996`.

## PEC, PMC and periodic walls

- **PEC** (`boundaries/pec.py`): tangential-E edge zeroing after each
  E update; in the eigenmode solver, PEC is imposed by degree-of-
  freedom elimination (DD-009).  Standard practice
  {cite}`taflovehagness2005`.
- **PMC** (`boundaries/pmc.py`, DD-065): realised as the *natural*
  boundary of the FIT update (the missing exterior circulation terms
  are simply absent), which is the discretely exact magnetic wall on
  the dual grid.  Standard FIT/FDTD practice
  {cite}`weiland1996`.
- **Periodic** (`boundaries/periodic.py`): direct field wrap-around of
  the curl stencil at opposing faces in the time-domain solver (zero
  phase advance), standard practice {cite}`taflovehagness2005`.  The
  eigenmode solver imposes the same pairing with an arbitrary Bloch
  phase advance, see [eigenmode analysis](eigenmode-analysis.md).

## Symmetry planes

A face may be declared a **symmetry plane**
(`boundaries/boundary_conditions.py`, DD-154): physically one of the
walls above, plus the statement that the mirror image of the model
exists beyond it.  Which wall applies follows from the field, not from
the geometry — on an electric wall the electric field stands
perpendicular to the plane and the magnetic field lies in it, on a
magnetic wall it is the other way round.  A structure that is
mirror-symmetric under an excitation that is *not* leaves no symmetry
to exploit.

Two spellings, differing only in what the mesher does with the
geometry:

```python
GeometryModel(boundary_conditions={"xmin": "SymmetryPMC"})           # clip at x = 0
GeometryModel(boundary_conditions={"xmin": ("SymmetryPMC", 1.5e-3)}) # clip at x = 1.5 mm
GeometryModel(boundary_conditions={"xmin": "ForceSymmetryPMC"})      # geometry already halved
```

The `Symmetry…` forms let the full geometry stand and simply never
mesh the discarded half; `ForceSymmetry…` takes the domain as built.
At most one plane per axis — two parallel mirrors would describe an
infinite image chain rather than a finite full model.

Everything the declaration implies is derived from it, so a half model
reports full-model quantities throughout:

- A port window cut by the plane is solved on its half.  A magnetic
  wall halves the window capacitance and puts the two halves in
  parallel ($z_\text{full} = z_\text{half}/2$), an electric wall puts
  them in series ($z_\text{full} = 2 z_\text{half}$).  The modes are
  power-normalised on the half window, so full-model wave amplitudes
  carry $\sqrt2$ per cutting plane and excitations $1/\sqrt2$ — a
  declared injected power stays a full-model watt (DD-155).
- Registered wall losses and flux integrals are scaled by the mirrored
  share in the same way.
- A lumped port or element cut by the plane is declared as the full
  device and internally halved or doubled to the meshed branch — see
  the lumped-elements chapter for the case rules.
- Field monitors and port-mode plots are mirrored back across the
  plane before display, so the pictures show the full cross-section
  while only half of it was solved.

The cost is spectral: a symmetry wall admits only the field
distributions of matching parity, so every mode of the opposite parity
is absent from the model.  For a driven problem whose excitation
respects the plane those modes carry no energy anyway; for eigenmode
work the omission is the point of the exercise, but it has to be
intended.

## Boundary-condition interaction with ports

Waveguide ports are not PML-backed (a PML-terminated port was
evaluated and rejected, DD-031/DD-043): port faces carry their own
transparent terminations, described in the [ports chapter](ports.md).

A port *window* may sit in an absorbing face (DD-198) — the way a horn
or an open-ended guide is fed from the wall of an open box.  The window
must be the cross-section of a conductor-enclosed guide reaching the
face; behind it the absorber is switched off over its whole depth
(``σ = 0, κ = 1`` in those columns), so the guided wave meets the
port's own termination, while the rest of the face keeps absorbing.
The lateral edge of that switch-off falls on the guide's walls, which
is why the enclosure is required rather than recommended.  The mesher
continues a conductor that touches an absorbing face through the
absorber layer — cell materials and, since DD-198, the conformal
sub-cell classification alike — so the guide is uniform up to the
port plane.
