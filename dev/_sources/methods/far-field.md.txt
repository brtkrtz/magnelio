# Far-field computation

## Surface equivalence on a Huygens box

The radiated far field of a time-domain run is obtained by the
frequency-domain near-to-far-field transform: on a closed surface
around the radiator, the tangential fields define equivalent surface
currents $\mathbf J = \hat n \times \mathbf H$ and
$\mathbf M = -\hat n \times \mathbf E$, whose radiation vectors

$$
\mathbf N(\hat r) = \oint \mathbf J\, e^{+jk\,\hat r\cdot\mathbf r'}\,dS',
\qquad
\mathbf L(\hat r) = \oint \mathbf M\, e^{+jk\,\hat r\cdot\mathbf r'}\,dS'
$$

give the far-zone field per direction,

$$
E_\theta = -\frac{jk}{4\pi}\,\bigl(\eta N_\theta + L_\varphi\bigr),
\qquad
E_\varphi = -\frac{jk}{4\pi}\,\bigl(\eta N_\varphi - L_\theta\bigr).
$$

This is the standard surface-equivalence formulation of FDTD/FIT
post-processing, introduced by Umashankar and
Taflove {cite}`umashankartaflove1982` and covered in textbook form by
Taflove and Hagness {cite}`taflovehagness2005`; the antenna-side
definitions follow Balanis {cite}`balanis2016`.

`monitors.MonitorFarField(freqs=[...])` records everything needed
during the run: it places a closed box a few grid cells inside the
physical domain (the absorber layers are excluded automatically, the
clearance is `margin_cells`) and accumulates a running DFT of the
tangential fields on its faces at the requested frequencies.  Each
face lies on a grid-node plane; the fields are interpolated from the
two adjacent cell layers onto that plane, which keeps the surface
exactly closed and second-order accurate on graded grids.  The
memory cost is one complex sample per frequency and surface cell —
negligible next to a volume monitor.

After the run, `monitor.result(f)` performs the transform and returns
a `FarFieldResult` with the complex patterns $E_\theta$, $E_\varphi$
on a spherical grid (ISO convention: $\theta$ from the $+z$ axis,
$\varphi$ from $+x$ in the $xy$-plane), evaluated at any angular
resolution without re-running the solver.

## Ground planes, walls and symmetry planes

A domain face closed with an electric or magnetic wall — a monopole's
ground plane, for instance — makes a closed box impossible.  Such
faces are handled by image theory: the box is left open there and
every recorded surface patch acquires a mirror image with the field
signs of the corresponding wall type.  Two situations share the same
mechanics but differ in meaning:

- **A real boundary** (a plain `PEC`/`PMC` face): the model is a
  half-space problem.  The pattern is masked behind the plane, the
  radiated power integrates over the physical half sphere, and
  directivity refers to it — a quarter-wave monopole on a ground
  plane reports its textbook ~5.2 dBi, not the dipole's 2.15 dBi.
- **A symmetry plane** (`SymmetryPEC`/`SymmetryPMC`, or the as-built
  `ForceSymmetry…` forms): the mirror half exists physically.  The
  image expansion reconstructs the full-model pattern over the whole
  sphere, and no additional power factor applies anywhere — the
  full-model excitation convention of the boundary chapter already
  makes one declared watt a full-model watt.

Periodic boundaries have no radiated field in this sense and are
rejected.  On a magnetic wall the natural wall of the staggered grid
sits a fraction of a cell outside the outermost grid line; the
mirrored surface inherits that sub-cell gap, a second-order effect on
the pattern.

## Normalisation, gain and radiated power

All frequency-domain quantities of the library are effective (RMS)
phasors normalised per √W of incident power, and the far field is no
exception: the radiation intensity is
$U = \bigl(|E_\theta|^2 + |E_\varphi|^2\bigr)/\eta_0$ with no further
factor, and

- `realized_gain` $= 4\pi U / 1\,\mathrm W$ — referenced to the
  incident power, mismatch included; this is the directly measured
  quantity,
- `gain` $= 4\pi U / P_\mathrm{acc}$ — the IEEE gain, using the
  accepted power $1 - \sum|S|^2$ the scattering run wires into the
  result,
- `directivity` $= 4\pi U / P_\mathrm{rad}$,
- `radiation_efficiency` $= P_\mathrm{rad} / P_\mathrm{acc}$ — 1 for
  a lossless model, and a useful closure check: for a lossless
  antenna $P_\mathrm{rad}$ must reproduce $1 - |S_{11}|^2$.

The radiated power integrates the smooth full-sphere pattern and
scales by the physical solid-angle fraction, which the image
symmetry makes exact.

## Plots

```python
pattern = ff.result(2.45e9)
pattern.plot_cut(plane="phi", angle=0.0)        # polar E-plane cut
pattern.plot_cut(plane="theta", angle=np.pi/2)  # azimuth cut
pattern.plot_3d()                               # 3D radiation surface
```

`plots.plot_pattern_cut` and `plots.plot_pattern_3d` are the free
functions behind these methods.  Polar cuts follow the antenna
convention (zero angle up, clockwise) with a dB floor as the radial
minimum; the 3D surface maps the dB value to the radius so nulls stay
visible as indentations.
