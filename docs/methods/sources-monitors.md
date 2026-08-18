# Sources, monitors and post-processing

## Plane-wave source (TF/SF)

Plane-wave illumination uses the **total-field/scattered-field
(TF/SF)** technique: the domain is split by a virtual box; consistency
corrections on the six box faces inject the incident wave into the
total-field region while the exterior carries only scattered field
(`sources/plane_wave.py`, DD-013).  The TF/SF formulation is due to
Merewether, Fisher and Smith {cite}`merewether1980` and
Umashankar and Taflove {cite}`umashankartaflove1982`;
textbook treatment in {cite}`taflovehagness2005`.
The incident samples are converted to FIT grid quantities per
edge/face, so amplitudes are physical (V/m) on any grid — an
in-house calibration (DD-085).

## Field, flux and frequency monitors

- **MonitorFieldTime** — time snapshots of E/H in a region, streamed
  to the on-disk store.
- **MonitorFieldFrequency** — running (accumulated) discrete Fourier
  transform of the fields at selected frequencies during the march;
  the running-DFT-during-timestepping technique is standard practice
  in time-domain solvers {cite}`taflovehagness2005`.
- **MonitorFluxTime** — Poynting flux through a plane,
  $\sum \hat e \cdot \hat h$ in the FIT pairing (physical Watt after
  DD-085).
- **MonitorWallLoss** — see the
  [conductor-losses chapter](conductor-losses.md).

All monitors return physical SI units; the calibration (C = 1 pinned
at the excitation source) is in-house bookkeeping (DD-085).

A plotted field plane is one *layer* of cells, sampled at their
centres, not a mathematical plane — the plane coordinate printed in the
title is the cell-centre coordinate the request snapped to.  Geometry
overlays follow the same rule: thin wires, discrete ports and lumped
elements are drawn when they lie inside the displayed layer, so a wire
declared on the grid nodes half a cell away still appears in the
picture of the field around it (DD-175).

## Signal processing

Excitation waveforms are Gaussian-family pulses with prescribed
spectral occupancy; S-parameters divide recorded spectra by the
excitation spectrum (standard practice
{cite}`taflovehagness2005`).  CW measurements use
lock-in phasor extraction over an integer number of periods after
settling (`cw_lockin_phasors`) — standard signal processing.

A frequency monitor is divided by the same spectrum, and for the same
reason.  Its running sum $\sum_n F(t_n)\,e^{+j\omega t_n}\,\Delta t$ is
the transient folded with the excitation, so it carries an extra factor
of time and the pulse's own spectral shape.  Since the excitation
waveform *is* the incident power-wave amplitude $a(t)$ in $\sqrt{\rm W}$
(DD-078), dividing it out leaves the field of a **1 W CW excitation** at
each monitor frequency — E in V/m, H in A/m, per $\sqrt{\rm W}$ of
incident power.  A run performs that division on its own monitors, so
`.data` is in those units from the moment the run returns; `.data_raw`
exposes the undivided bins for callers who want the transient itself.

## Project store, checkpointing, resume

Runs stream results append-only into an HDF5-based on-disk project
store (SWMR single-writer/multi-reader), with periodic checkpoints
and bit-exact resume (DD-070).  File formats: HDF5, XDMF/VTK for
field visualisation.  This is engineering infrastructure, not a
research method; the formats are community standards.
