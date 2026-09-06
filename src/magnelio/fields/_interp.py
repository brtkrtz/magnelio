"""Cell-centre averaging of the staggered field components (DD-085).

The solver states are FIT grid quantities — ``e = E·l_primal`` [V] on
the primal edges, ``h = H·l_dual`` [A] on the dual edges — and every
sample sits on its own Yee position.  A picture wants one vector per
cell: each sample is converted to the physical field at its own
position and then averaged over its staggered neighbours onto the cell
centre.  This is a low-pass filter and it is not undone; it is applied
at *access* time (DD-259), never when a field is stored.

The module lives with the public field containers because they are its
first consumers; the monitors and the ParaView export import it from
here.
"""

from __future__ import annotations

import numpy as np

from magnelio._fields.field_arrays import FieldArrays
from magnelio.mesh.grid import GridLines


def _solver_dual_widths(d: np.ndarray) -> np.ndarray:
    """Dual widths in the SOLVER convention: boundary = full end cell.

    Mirrors ``operators.material_matrices._build_avg_d`` — the h states
    at domain-boundary nodes carry the full first/last cell as their
    dual length (DD-082 B1 finding), so converting ``h -> H`` must
    divide by the same widths.
    """
    n = d.size
    out = np.empty(n + 1)
    out[0] = d[0]
    if n > 1:
        out[1:n] = 0.5 * (d[:-1] + d[1:])
    out[n] = d[-1]
    return out


def _interp_to_cell_centres(
    fields: FieldArrays, components: list[str], ix: slice, iy: slice, iz: slice, grid: GridLines
) -> dict[str, np.ndarray]:
    """Physical fields at cell centres within *ix, iy, iz* (DD-085).

    The solver states are FIT grid quantities (``e = E·l_primal`` [V],
    ``h = H·l_dual`` [A]); each staggered sample is converted to the
    physical field at its own position (``E = e/l``, ``H = h/l_dual``
    with the solver dual convention) and then averaged over its
    staggered neighbours so that the result lives at cell centres
    ``(xc[i], yc[j], zc[k])``.

    Parameters
    ----------
    fields : FieldArrays
        Current field snapshot (grid-quantity states).
    components : list[str]
        Subset of ``["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]``.
    ix, iy, iz : slice
        Cell-index slices (stop-exclusive) defining the sub-region.
    grid : GridLines
        Simulation grid providing the per-edge/per-face lengths.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping component name -> cell-centred array of shape
        ``(ix.stop - ix.start, iy.stop - iy.start, iz.stop - iz.start)``;
        E in [V/m], H in [A/m].
    """
    result = {}
    i0, i1 = ix.start, ix.stop
    j0, j1 = iy.start, iy.stop
    k0, k1 = iz.start, iz.stop

    # GPU backend: the field arrays are device arrays, which refuse
    # implicit mixing with NumPy operands.  Interpolate on the device
    # (the edge-length vectors are tiny host->device transfers) and
    # move only the region-sized results back at the end — for plane
    # monitors that is orders of magnitude cheaper than syncing the
    # full field state every recorded step.
    xp = np
    if type(getattr(fields, components[0])).__module__.partition(".")[0] == "cupy":
        import cupy as xp  # noqa: PLC0415

    dx_h = np.asarray(grid.dx, dtype=float)
    dy_h = np.asarray(grid.dy, dtype=float)
    dz_h = np.asarray(grid.dz, dtype=float)
    dx, dy, dz = xp.asarray(dx_h), xp.asarray(dy_h), xp.asarray(dz_h)

    need_h = any(c.startswith("H") for c in components)
    if need_h:
        dxa = xp.asarray(_solver_dual_widths(dx_h))
        dya = xp.asarray(_solver_dual_widths(dy_h))
        dza = xp.asarray(_solver_dual_widths(dz_h))

    for comp in components:
        arr = getattr(fields, comp)
        if comp == "Ex":
            # Ex[i,j,k] at (xc[i], y[j], z[k]) — average over j,k
            # neighbours; all four samples share the edge length dx[i]
            result[comp] = (
                0.25
                * (
                    arr[i0:i1, j0:j1, k0:k1]
                    + arr[i0:i1, j0 + 1 : j1 + 1, k0:k1]
                    + arr[i0:i1, j0:j1, k0 + 1 : k1 + 1]
                    + arr[i0:i1, j0 + 1 : j1 + 1, k0 + 1 : k1 + 1]
                )
                / dx[i0:i1, None, None]
            )
        elif comp == "Ey":
            # Ey[i,j,k] at (x[i], yc[j], z[k]) — average over i,k neighbours
            result[comp] = (
                0.25
                * (
                    arr[i0:i1, j0:j1, k0:k1]
                    + arr[i0 + 1 : i1 + 1, j0:j1, k0:k1]
                    + arr[i0:i1, j0:j1, k0 + 1 : k1 + 1]
                    + arr[i0 + 1 : i1 + 1, j0:j1, k0 + 1 : k1 + 1]
                )
                / dy[None, j0:j1, None]
            )
        elif comp == "Ez":
            # Ez[i,j,k] at (x[i], y[j], zc[k]) — average over i,j neighbours
            result[comp] = (
                0.25
                * (
                    arr[i0:i1, j0:j1, k0:k1]
                    + arr[i0 + 1 : i1 + 1, j0:j1, k0:k1]
                    + arr[i0:i1, j0 + 1 : j1 + 1, k0:k1]
                    + arr[i0 + 1 : i1 + 1, j0 + 1 : j1 + 1, k0:k1]
                )
                / dz[None, None, k0:k1]
            )
        elif comp == "Hx":
            # Hx[i,j,k] at (x[i], yc[j], zc[k]) — the two staggered
            # samples sit at different x nodes: convert each first
            result[comp] = 0.5 * (
                arr[i0:i1, j0:j1, k0:k1] / dxa[i0:i1, None, None]
                + arr[i0 + 1 : i1 + 1, j0:j1, k0:k1] / dxa[i0 + 1 : i1 + 1, None, None]
            )
        elif comp == "Hy":
            # Hy[i,j,k] at (xc[i], y[j], zc[k]) — average over j neighbours
            result[comp] = 0.5 * (
                arr[i0:i1, j0:j1, k0:k1] / dya[None, j0:j1, None]
                + arr[i0:i1, j0 + 1 : j1 + 1, k0:k1] / dya[None, j0 + 1 : j1 + 1, None]
            )
        elif comp == "Hz":
            # Hz[i,j,k] at (xc[i], yc[j], z[k]) — average over k neighbours
            result[comp] = 0.5 * (
                arr[i0:i1, j0:j1, k0:k1] / dza[None, None, k0:k1]
                + arr[i0:i1, j0:j1, k0 + 1 : k1 + 1] / dza[None, None, k0 + 1 : k1 + 1]
            )
    if xp is not np:
        result = {comp: arr.get() for comp, arr in result.items()}
    return result
