# Numerical precision

Magnelio runs its time loop in **IEEE-754 single precision by
default** and offers double precision as an opt-in.  This chapter says
what the switch reaches, what single precision buys, what it costs,
and — the part that is easy to miss — how to recognise a number that
is limited by the word length rather than by the mesh.

## The switch

Precision is a per-analysis argument:

```python
analysis = mio.AnalysisScatteringTD(mesh=mesh, precision="double")
```

`precision` takes `"single"`, `"double"` or `None`.  `None` — the
unspecified default — first consults the `MAGNELIO_PRECISION`
environment variable (`"single"` / `"double"`), which is the
deterministic override for test suites and batch farms, and otherwise
resolves to `"single"`.  An explicit argument always wins over the
environment, so a script that asks for double gets double wherever it
runs.  The resolved value — never the `None` sentinel — is stored in
the project recipe, so a resumed run continues in the precision it
started in.

Precision and compute backend are **orthogonal axes**: any precision
runs on any backend, and `precision=` says nothing about `backend=`.

## What the switch reaches

The switch sets the scalar type of the **time-loop state**: both field
vectors, the α/β update coefficients, the material diagonals, the CPML
auxiliary variables, and the auxiliary states of the dispersive-material
and surface-impedance recursions.  In single precision the whole marched
state is `float32`.

Three groups stay in double precision regardless of the switch, because
none of them is a per-cell-per-step cost and all of them accumulate:

- **the frequency-domain accumulators** — the running DFT sums behind
  S-parameters, field monitors and wall-loss integrals, which run over
  $10^5$–$10^6$ steps and would be the textbook naive-summation
  catastrophe in single;
- **the port machinery** — the 2D mode solve, the reference impedances
  and the transparent-boundary convolution state;
- **geometry and meshing** — Boolean robustness is not negotiable at
  reduced word length.

This mixed layout is ordinary engineering practice in production
time-domain field solvers, not a research method (DD-094).

## What single precision buys

**Memory: exactly half, and the law is flat.**  The dtype-carrying
time-loop state is 24 scalars per cell — six field components, twelve
update coefficients, six material-diagonal entries — so it costs

$$
192\ \text{bytes/cell in double},\qquad 96\ \text{bytes/cell in single}.
$$

Measured in-house on a plain air-filled line, the ratio is 0.5000 at
every grid size, approaching the asymptotic values from above (the
excess at small counts is the extra surface layer of the staggered
component shapes):

| cells | double | single |
|---|---|---|
| 4 096 | 213.4 B/cell | 106.7 B/cell |
| 32 768 | 202.6 B/cell | 101.3 B/cell |
| 110 592 | 199.0 B/cell | 99.5 B/cell |
| 262 144 | 197.3 B/cell | 98.6 B/cell |

Ports, recorders and DFT accumulators are *not* in this budget and are
not halved.  On a small port-heavy fixture they dominate the process
footprint, so the saving a user actually observes is well below 50 %;
on the large field-dominated meshes single precision is meant for, it
approaches the full halving.

**Throughput: the win is bandwidth, so it appears at a threshold.**
The leapfrog is a memory-bound stencil.  Halving the word length halves
the traffic, but that only pays once the working set no longer fits in
the last-level cache — below that threshold the loop is latency-bound
and the word length is nearly free.  Measured in-house on the bare time
loop (no ports, no monitors) on a desktop CPU with 96 MiB of last-level
cache:

| cells | double | single | speed-up |
|---|---|---|---|
| 32 768 | 0.138 ms/step | 0.123 ms/step | 1.12× |
| 262 144 | 0.412 ms/step | 0.354 ms/step | 1.16× |
| 884 736 | 1.956 ms/step | 0.936 ms/step | 2.09× |
| 2 097 152 | 9.805 ms/step | 3.908 ms/step | 2.51× |

The two small grids fit in cache in both precisions and gain about a
tenth.  At 884 736 cells the double state (~175 MB) spills while the
single state (~87 MB) still fits, which is why the ratio *overshoots*
two.  Beyond that both spill and the ratio settles into the bandwidth
regime.  The threshold moves with the machine; the shape of the law
does not.

The same effect drives the GPU case, where consumer cards additionally
run FP64 at a fraction of the FP32 rate: the fused update kernels
measure 1.25× at 97 k cells and 2.43× at 373 k on the reference card
(DD-094).

Whole-run speed-ups are smaller than kernel speed-ups, and on a small
port-heavy model can vanish entirely — the double-only parts above do
not shrink, and Amdahl's law does the rest.  The end-to-end figure for
the 262 144-cell fixture with two ports and a full DFT is 1.03×.

## What single precision costs

Not the transmitted quantities.  On the same fixture, single against
double over the whole band:

| quantity | difference |
|---|---|
| max ⏐ΔS₁₁⏐ (linear) | 5.6 · 10⁻⁷ |
| max ⏐ΔS₂₁⏐ (linear) | 7.7 · 10⁻⁷ |
| ⏐S₂₁⏐ at band centre | identical to four decimals in dB |

For comparison, the *discretisation* error of the same run — the
deviation of the computed insertion loss from the closed-form value —
is on the order of $10^{-2}$ dB, three to four orders of magnitude
larger.  For everything a converged mesh is capable of resolving,
the word length is irrelevant.

What single precision costs is **dynamic range**.  The one quantity
that visibly moves is the reflection floor: on that fixture ⏐S₁₁⏐
bottoms out at −171.4 dB in double and −145.2 dB in single.  More
sharply: a single-precision run cannot show a reflection floor much
below

$$
20 \log_{10}(2 \cdot 10^{-6}) \approx -114\ \text{dB},
$$

the rounding floor of `float32` fields, no matter how good the mesh,
the port or the absorber is.

### The floor moves with the length of the run

This is the property most likely to surprise, and it has no analogue in
double precision.  In single precision the *worst-case* reflection
floor of a run degrades as the record grows, because per-step rounding
of the marched state accumulates while the signal being measured
decays.  Measured in-house on the same fixture and an exact
transparent port, ⏐S₁₁⏐ over the band:

| steps | single, worst | single, median | double, worst | double, median |
|---|---|---|---|---|
| 2 000 | −125.8 dB | −139.4 dB | −166.5 dB | −166.5 dB |
| 4 000 | −120.0 dB | −139.4 dB | −166.4 dB | −166.5 dB |
| 8 000 | −113.0 dB | −139.5 dB | −166.0 dB | −166.5 dB |
| 16 000 | −112.6 dB | −139.6 dB | −166.0 dB | −166.5 dB |

Three things to read out of it:

1. **In single the worst-case floor erodes at about 6–7 dB per
   doubling of the record; in double it is flat** to a few tenths of a
   dB over the same range.  The mechanism is the word length, not the
   absorber and not the port.
2. **It saturates** at the `float32` field floor — here −112.6 dB.
   Once a run has arrived there, further steps cost no further dynamic
   range.
3. **The median does not move.**  Over eight times the record it
   travels 0.2 dB.  It is the single worst frequency point that
   erodes, so a band-averaged figure of merit shows nothing at all
   while the worst point loses 13 dB.

Point 3 is the reason this belongs in the documentation rather than in
a release note: the degradation is invisible in exactly the summary
number most users look at.

## Recognising a word-length-limited result

Signs that a number is limited by precision rather than by the model:

- a reflection or isolation floor that settles near −110 to −120 dB and
  refuses to improve when the mesh is refined;
- a floor that gets *worse* when the run is made longer, or that
  changes when a stop criterion is loosened;
- a high-Q resonance whose Q saturates: `float32` coefficient
  resolution limits the achievable Q from roughly $10^4$–$10^5$
  upward;
- run-to-run or CPU-versus-GPU differences at the $10^{-6}$ level,
  which are the expected `float32` field floor and not a defect.

**The operational test is one line.**  Re-run the case with
`precision="double"`:

```python
analysis = mio.AnalysisScatteringTD(mesh=mesh, precision="double")
```

If the number of interest moves, it was limited by the word length.  If
it stays, the limit is physical or discretisation-related and double
precision will not help — refine the mesh instead.  Because the two
axes are independent, this test costs one run and changes nothing else
about the model.

## Choosing

Stay with the default `"single"` for the standard workflow: broadband
S-parameters on a converged mesh, far fields, radiation patterns,
matching and coupling studies — anything whose accuracy is set by the
discretisation, which is everything down to roughly −60 dB of dynamic
range.  Large 3D models benefit twice, in memory and in throughput.

Switch to `"double"` for:

- **high-Q work** — resonators and filters at $Q \gtrsim 10^4$,
  ring-down and eigenfrequency extraction from long records;
- **high dynamic range** — isolation, cross-talk, directivity or
  return-loss floors below about −100 dB, where the numbers in this
  chapter show single precision is the limit;
- **long runs whose deepest points matter**, per the length law above;
- **reference and certification runs**, where a floor is the result
  being reported rather than a byproduct.

Any reported floor should name the precision it was taken at; the same
model, mesh and port yield floors tens of decibels apart between the
two settings, and a floor quoted without its word length is not
reproducible.
