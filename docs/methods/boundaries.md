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
  the curl stencil at opposing faces (no phase shift / Floquet
  variant implemented).  Standard practice
  {cite}`taflovehagness2005`.

## Boundary-condition interaction with ports

Waveguide ports are not PML-backed (a PML-terminated port was
evaluated and rejected, DD-031/DD-043): port faces carry their own
transparent terminations, described in the [ports chapter](ports.md).
