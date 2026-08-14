"""CUDA-graph capture of the FIT-TD device phases (WP-G3).

After WP-G1/WP-G2 removed the per-step host round trips, each leapfrog
step has exactly two contiguous device-only segments:

* **E phase** — ADE/SIBC ``save_field`` stashes → fused E kernel →
  ``dispersion.update_field`` → BC E-corrections (both passes) →
  optional global PEC re-enforcement.
* **H phase** — fused H kernel → ``dispersion_mu``/``sibc``
  ``update_field`` → CPML H-corrections → PMC enforcement.

Everything in between (sources, port hooks, recorder staging,
monitors, energy checks) stays host-driven eager work.  Capturing each
segment as one CUDA graph replaces its ~10–40 kernel launches with a
single graph launch per phase per step — replaying the *identical*
kernels on the *identical* pointers, so the marched fields are
bit-identical to the eager path by construction.

Temp-buffer stability
---------------------
CuPy elementwise ops allocate temporaries from the current memory
pool.  Pointers recorded at capture time are baked into the graph, so
a temp block that the main pool later hands to unrelated eager work
(port staging, energy checks) would be silently corrupted by the next
replay.  Capture therefore runs under a **private memory pool**: all
temps recorded into the graph come from a pool that is never used for
anything else afterwards — its freed blocks stay reserved on the
pool's free list for the lifetime of the graph, and replays own them
exclusively.  (Reuse of a freed block *within* one capture is safe:
stream order at replay equals recorded order, exactly as in eager
pool reuse.)

Failure policy
--------------
Capture runs once per phase, after :data:`WARMUP_STEPS` eager
iterations (kernel compilation and pool blocks must be stable).  Any
capture failure logs one warning and leaves the whole run on the
eager path — behaviour, never results, changes.
``MAGNELIO_GPU_GRAPHS=0`` disables capture entirely (the deterministic
anchor, mirroring ``MAGNELIO_BACKEND``).
"""

from __future__ import annotations

import os
import warnings

#: Eager warm-up iterations per phase before capture (kernel JIT and
#: memory-pool blocks must be stable across replays).
WARMUP_STEPS = 2


def graphs_enabled() -> bool:
    """Whether CUDA-graph capture is enabled (``MAGNELIO_GPU_GRAPHS``)."""
    return os.environ.get("MAGNELIO_GPU_GRAPHS", "1").strip() != "0"


class CudaGraphPhases:
    """Capture-and-replay dispatcher for the per-step device phases.

    One instance per ``run()`` on the CuPy backend.  The solver calls
    :meth:`run_phase` at each phase site; the dispatcher marches the
    phase eagerly during warm-up, captures it into a CUDA graph once
    (executing that step via an immediate graph launch), and replays
    the graph afterwards.  On any capture failure it disables itself
    for the rest of the run and keeps marching eagerly.
    """

    def __init__(self) -> None:
        self.failed = False
        self._graphs: dict = {}
        self._pools: dict = {}
        self._warmup: dict = {}

    @property
    def ready(self) -> bool:
        """True once at least one phase replays from a captured graph."""
        return bool(self._graphs) and not self.failed

    def run_phase(self, name: str, fn) -> None:
        """Execute phase ``name`` — replay, capture-now, or eager."""
        if self.failed:
            fn()
            return
        graph = self._graphs.get(name)
        if graph is not None:
            graph.launch()
            return
        seen = self._warmup.get(name, 0)
        if seen < WARMUP_STEPS:
            self._warmup[name] = seen + 1
            fn()
            return
        if not self._capture(name, fn):
            fn()

    def _capture(self, name: str, fn) -> bool:
        """Capture ``fn()`` as graph ``name`` and launch it once.

        Returns True when the phase was executed via the new graph;
        False when capture failed (nothing was executed — the caller
        runs the eager phase; ``failed`` is set for the whole run).
        """
        import cupy  # noqa: PLC0415 — only reachable on the CuPy backend

        pool = cupy.cuda.MemoryPool()
        stream = cupy.cuda.Stream(non_blocking=True)
        old_alloc = cupy.cuda.get_allocator()
        try:
            cupy.cuda.set_allocator(pool.malloc)
            with stream:
                stream.begin_capture()
                fn()
                graph = stream.end_capture()
        except Exception as exc:
            self.failed = True
            self._graphs.clear()
            self._pools.clear()
            warnings.warn(
                f"CUDA-graph capture of the {name} phase failed "
                f"({exc}); continuing on the eager kernel path.  Set "
                f"MAGNELIO_GPU_GRAPHS=0 to silence this probe.",
                stacklevel=2,
            )
            return False
        finally:
            cupy.cuda.set_allocator(old_alloc)

        self._graphs[name] = graph
        self._pools[name] = pool  # keeps temp blocks reserved
        # Perform the captured step's work now, ordered on the ambient
        # stream (capture only records — it does not execute).
        graph.launch()
        return True
