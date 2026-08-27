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

## Feed guides crossing the box

A waveguide-fed antenna — a horn, an open-ended guide, a coax entering
the domain — has its port on an absorbing face (see the ports chapter),
and the guide runs from that wall into the box.  The Huygens surface
cannot avoid it.  The monitor treats the crossing the way such antennas
are usually handled (in-house convention): the box face the guide
crosses is sampled at the absorber interface itself, so as little of
the guide as possible lies outside the box; the patches inside the
guide's cross-section are left out — the guided wave there is the
feed, not an external source, and the equivalent surface closes over
the guide's outer wall instead; and patches whose sampled cells are
conductor on both sides carry nothing.  What the surface cannot see
are the currents on the guide's outer wall *beyond* the box face,
inside the absorber, where they decay with the absorber profile.
Measured on an open-ended 20 × 10 mm tube at 10 GHz, the radiated
power balances the accepted power to 3 %; the same tube with an
infinite flange (a PEC face, exact image theory) balances to 9 %,
because the port window then punches a hole into the image plane —
the absorbing face is the better model of an unflanged feed.

## Normalisation, gain and radiated power

All frequency-domain quantities of the library are effective (RMS)
phasors normalised per √W of incident power, and the far field is no
exception.  The reference is the incident power wave the run actually
launched: for lumped, TEM and quasi-TEM feeds that is the excitation
waveform itself, for a TE/TM feed — whose wave impedance varies across
the band — the incident wave $a(f)$ that the S-parameter extraction
separates at the port, so a horn's gain does not inherit the shape of
$Z_{\mathrm{TE}}(f)$.  The radiation intensity is
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

## Power balance: how close the box may sit to the radiator

The recorded surface fields carry a definite real power out of the
box, $P_\mathrm{surf} = \mathrm{Re}\oint (\mathbf E \times \mathbf
H^*)\cdot\hat n\,dS$, and for a lossless exterior the pattern must
radiate exactly that power.  `FarFieldResult.surface_power` carries
the flux and `power_balance` the ratio $P_\mathrm{rad}/P_\mathrm{surf}$;
`monitor.result(f)` warns when the two differ by more than 5 %.  The
flux itself is a robust quantity — it reproduces the accepted power
$1 - |S_{11}|^2$ of a lossless model to a percent wherever the box is
placed — so a shortfall means the transform, not the solver: the box
samples the radiator's near zone too closely, and the discrete near
field there is not the free-space outgoing field the transform
assumes.  The pattern amplitude is then low by that factor, and with
it realized gain and gain; directivity, normalised to $P_\mathrm{rad}$
itself, is unaffected.

The box sits at the absorbing faces, so its distance from the
radiator is the model's clearance to the boundary — and the face that
matters is the one carrying most of the flux.  Measured in-house on a
microstrip patch at 10 GHz on a 0.25 mm floor: with the domain top
0.3 λ above the copper the balance reads 0.93, at 0.7 λ and beyond
0.98–1.02, independent of the lateral clearance (0.3 λ or 0.7 λ) and of
whether the substrate reaches the boundary; halving the cells at 0.3 λ
brings it to 0.97.  A wire dipole or monopole balances to 1 % once the
box is a dozen cells away.  Half a wavelength of clearance in the
direction of the main beam is a safe rule for printed radiators over
a ground plane; the warning tells you when a model needs more.  With
a feed guide crossing the box (previous section) the flux itself
misses the few percent that run along the guide's outer wall beyond
the box, so $P_\mathrm{surf}$ and $P_\mathrm{rad}$ both sit that far
below the accepted power there while the balance still closes.

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
