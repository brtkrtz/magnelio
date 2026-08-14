"""Physical constants of free space (SI, CODATA 2018).

``C0`` is exact by the SI definition of the metre; ``MU0`` carries the
CODATA 2018 measured value.  ``EPS0`` and ``ETA0`` are *derived* from
those two so that the free-space relations

    EPS0 * MU0 * C0**2 == 1        ETA0 == MU0 * C0

hold exactly in floating point — the discretized Maxwell operators and
every mode solver then agree on the same wave speed and impedance to the
last bit.
"""

C0 = 299_792_458.0  # speed of light in vacuum [m/s], exact
MU0 = 1.25663706212e-6  # vacuum permeability [H/m], CODATA 2018
EPS0 = 1.0 / (MU0 * C0 * C0)  # vacuum permittivity [F/m], derived
ETA0 = MU0 * C0  # free-space wave impedance [Ohm], derived

__all__ = ["C0", "EPS0", "MU0", "ETA0"]
