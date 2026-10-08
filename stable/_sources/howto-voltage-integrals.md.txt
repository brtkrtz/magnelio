# Measure voltage along a field path

Use `circuit.integrate_E` to read a voltage along a declared path from
a recorded electric field. A time-domain frame gives a real voltage;
a frequency-domain frame gives a complex voltage phasor. A phasor
retains its magnitude and phase without selecting a real snapshot first.

## Record the complete path

Add a `MonitorFieldFrequency` that records `fields=["E"]` over a region
containing the entire path, then run the analysis. The following recipe
assumes the monitor is named `gap`, records 5 GHz, and spans the example
7 mm path. The same access works for a result loaded from a project:

```python
import numpy as np
from magnelio import circuit, geo

field = result.monitors["gap"].spectrum.at_frequency(5e9)
path = geo.Curve.polyline([(0.0, 0.0, 0.0), (7e-3, 0.0, 0.0)])
voltage = circuit.integrate_E(field, path, field.grid)

magnitude = abs(voltage)
phase_deg = np.angle(voltage, deg=True)
```

For a uniform `Ex = 3 + 4j` V/m along this path, the result is
`0.021 + 0.028j` V: magnitude 35 mV and phase about 53.13 degrees.
Reversing the two points negates the phasor. For a real time recording,
select a frame with `monitor.recording.at_time(t)` and pass that field
in the same way.

## Interpret the result

The field is in physical V/m, not the solver's edge-voltage units.
Always use the selected frame's own `grid`, especially for a monitor
covering only part of the model. Frequency frames normalized per
sqrt(watt) give a voltage per sqrt(watt); a field scaled to a physical
excitation gives the corresponding physical voltage. Integration does
not change the recording's normalization.

The direction defines the sign of `integral E dot dl`. This is a path
measurement, not a path-independent voltage between arbitrary points
in a time-varying magnetic field. The rasteriser snaps endpoints and
follows grid edges; keep the whole path inside the recorded region and
resolve the geometry adequately. A path shorter than the grid can
resolve is rejected. See {ref}`edge-path-rasterisation`
for the discretization and phase convention.
