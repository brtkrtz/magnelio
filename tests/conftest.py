"""Suite-wide test configuration.

Pins the array backend to NumPy for every test that does not choose
one explicitly: with the ``backend="auto"`` production default the
whole suite would silently run on the GPU on CUDA machines, moving
every bit-exactness gate onto GPU rounding.  Dedicated GPU tests
request ``backend="cupy"`` explicitly, which bypasses this override.

Likewise pins the time-loop precision to double: the production default
is single (float32, DD-094), but the existing accuracy/bit-exactness
gates were written against double.  Tests that exercise single precision
pass ``precision="single"`` explicitly, which wins over this env pin
(an explicit value bypasses MAGNELIO_PRECISION in resolve_precision).
"""

import os

os.environ.setdefault("MAGNELIO_BACKEND", "numpy")
os.environ.setdefault("MAGNELIO_PRECISION", "double")
# No pvpython state-bake subprocesses during tests (DD-115): the light
# ParaView artefacts (vtm/xdmf/vtr/script) still exercise the exporter;
# the bake itself is covered by one dedicated, gated test.
os.environ.setdefault("MAGNELIO_PVSM_BAKE", "0")
