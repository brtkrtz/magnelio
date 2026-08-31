"""Modal port operator — per-mode boundary termination.

Two termination branches, selected per mode at construction:

**Exact DTBC (WP-R2 TEM, WP-R3 TE/TM).**  On a uniform feed line the
modal amplitude obeys the exact 1D leapfrog Klein-Gordon chain
(DD-052/DD-053 separation), whose modal Courant number is read off
the co-located pair product

    r = dt / sqrt(M_eps[e] * M_mu[h_partner])        (per pair)

— the pair identity ``M_eps * M_mu = eps*mu * dz * dz~`` with
``dz = dz~ = normal_dx`` on the flattened port slab — and whose mass
``q = omega_c * dt`` is the exact discrete 2D eigenvalue of the
3D-restricted transversal operator (zero for TEM; numerical-path
TE/TM modes only — analytical modes carry the continuum cut-off and
stay on Mur).  A mode whose weighted pair-product spread certifies a
uniform chain is terminated by the exact discrete transparent
boundary condition
(:class:`~magnelio.ports._modal.dtbc.DTBCTermination`): ghost-relation
convolution over the boundary history, exact for propagating *and*
evanescent content, with incident-wave injection prescribed at the
ghost plane (no fractional-delay interpolation, no velocity
approximation — dispersive modes launch exactly).  See
DD-054/DD-055 and the ``dtbc`` module docstring (the plan
``REFLECTION_FREE_PLAN.md`` is retired to git history).

**Modal Mur-1st-order (analytical-path modes; inhomogeneous QTEM
until WP-R4).**  The 1D Mur-1st ABC applied per mode in the
modal-coefficient space.  For each mode ``m`` and each TD step,

    V_m,port^{n+1}  =  V_m,interior^n
                       + r_m · (V_m,interior^{n+1} − V_m,port^n)

with the per-mode reflection coefficient

    r_m  =  (v_p,m · dt − dx_n) / (v_p,m · dt + dx_n)

where ``v_p,m = ω_calc / β_m(ω_calc)`` is the phase velocity of mode m
at the user-supplied mode-calculation frequency, and ``dx_n`` is the
distance between the port plane and the one-cell-inside companion plane
(``plane.normal_dx``).

This is the proven-stable absorbing condition standard in TD-FIT codes
approach.  It supersedes the V/I-instantaneous formulation that
proved unstable on FIT-evolved fields with Yee half-cell stagger
between E (at port plane) and H (half a cell inside).

The operator hooks into the FIT solver via the unified
:meth:`update_e(fields, t, dt)` Port-protocol method (see
:class:`magnelio.ports.base.Port`): ``fields.e_flat`` is read at both
the port and interior planes, the Mur-corrected ``V_m,port`` is
computed, and ``e`` at port-plane edges is overwritten to enforce that
value.  ``fields.h_flat`` is unused by the absorber.

Excitation: TF/SF decomposition
-------------------------------

When :meth:`set_excitation` activates a source on one of the modes,
the operator treats the *injected* incident wave separately from any
*scattered* (reflected) field at the source port.  Mur is applied **only to the
scattered component**, and the incident is re-added at the port plane.

The incident value at the interior plane is the source value retarded by
the propagation delay ``τ_m = dx_n / v_p,m``:

    V_inc(z = z_int, t)  =  s(t − τ_m)
    V_inc(z = z_port, t) =  s(t)

A short ring buffer of past source samples is interpolated linearly to
provide the fractional-delay look-up.  The Mur correction acts on
``V_scat = V_total − V_inc``, and the operator writes back
``V_port_total = s(t) + V_scat,port``.

A naive additive Δs injection is *not* compatible with Mur-1: during
the source ramp-up the interior projection trails behind, so Mur drains
the just-injected wave and only ``Δs`` survives at the port — the bug
observed in commit 12d7ee4 (|S₂₁| at the
FFT round-off floor, |S₁₁| ≈ 0 dB).

V/I calibration
---------------

``discretize_modes`` orthonormalises ``ê`` in the M_ε inner product and
applies the same scalar ``α`` to ``ĥ``.  That keeps the analytical
``E/H = Z`` ratio of the *raw* fields, but the M-weighted projection
``V_m = Σ M_ε·ê·e``, ``I_m = Σ M_μ·ĥ·h`` does *not* satisfy
``V_m/I_m = Z_modal`` automatically — the M_ε-vs-M_μ asymmetry in dual-
edge lengths and primal-vs-dual face areas leaves a mode- and mesh-
dependent factor between V_m and I_m.

We close that gap with a per-mode post-Gram-Schmidt rescale of the H
profile.  Two branches (WP7.2):

**Numerical path** (``field_evaluator is None`` — TEM/QTEM Laplace and
TE/TM eigsh modes): the discrete profiles are the edge/face voltages of
the exact discrete travelling wave up to one scalar
(:func:`~magnelio.ports._modal.tem_laplace.travelling_wave_h_profiles`),
so the calibration test field *is* the profile itself and V/I is
measured directly in the operator's own M metric:

```
V_test = Σ M_ε[p]·ê_p²          (= 1 by M_ε-orthonormality)
I_test = Σ M_μ[p]·ĥ_p²
γ      = (V_test / I_test) / Re Z_modal
ĥ      ← γ · ĥ        (per mode)
```

After this, ``I(TW) = V(TW)/Z_modal`` holds *by construction* for the
discrete travelling wave — on uniform *and* graded transversal grids.
Note the formula is not scale-invariant in ĥ (the profile enters as
projection weight and as test field); it is correct precisely because
the travelling-wave form fixes the per-face shape *and* the physical
V-to-I voltage ratio up to the one scalar that γ removes.

**Analytical path** (``field_evaluator`` present): the test field is
the analytical mode sampled at edge midpoints,

```
V_test = Σ M_ε[u-edge p]·ê_u·E_u_phys(p)·L_primal_u(p)
       + Σ M_ε[v-edge p]·ê_v·E_v_phys(p)·L_primal_v(p)
I_test = μ₀·normal_dx · ( Σ L_primal_u·ĥ_v·H_v_phys
                        + Σ L_primal_v·ĥ_u·H_u_phys )
γ      = V_test / (I_test · Re Z_modal)
ĥ      ← ĥ / γ        (per mode)
```

where the ``I_test`` formula uses the FIT identity
``M_μ[h_v at u-edge]·L_dual_for_h_v = μ₀·normal_dx·L_primal_u``
(and the v-edge analogue), which holds for any bbox-aligned port face —
it lets us bypass the (un-exposed) dual-edge lengths entirely.

After calibration, ``V_m/I_m = Z_modal`` holds for the mode's own
travelling wave and *approximately* (modulo discretisation error) for
the FIT-evolved field, restoring the standard
``a = (V/√Z + √Z·I)/2`` power-wave decomposition of
:func:`compute_s_parameters`.  Evanescent modes (purely imaginary
``Z_modal``) cannot be calibrated by a real scalar rescale and are
left at their post-Gram-Schmidt scaling.

Physical √W amplitude convention (DD-078)
-----------------------------------------

The M_ε-orthonormal basis fixes only the *shape* of the recorded V/I;
their absolute scale is mesh- and aperture-dependent and carries no
physical meaning (see ``discretize.py``).  To make power waves
physically commensurate across ports (heterogeneous modal↔modal pairs
and lumped↔modal mixes), the calibration additionally computes, per
mode, the physical Poynting power ``P₁`` of the unit-coefficient
discrete travelling wave,

    P₁ = Σ_u ê_u·H_v·dA_u  −  Σ_v ê_v·H_u·dA_v,
    dA = primal edge length × geometric dual node spacing,

with ``H`` the wave's physical field: the pre-γ ĥ profile directly on
the analytical path (sampled A/m), and ``ĥ·M_μ/(μ₀·normal_dx)`` on the
numerical path (undoing the dual-voltage convention of
``travelling_wave_h_profiles``), and derives

    record_scale κ_m  = √(|P₁|·Re Z_modal)     [V per basis unit]
    source_scale      = √(Re Z_modal) / κ_m    [basis units per √W]

The **recorder** multiplies the projected V/I by ``record_scale``
(never the internal Mur/DTBC projections — the termination dynamics
stay in basis units, bit-identical), and :meth:`set_excitation`
multiplies the user waveform by ``source_scale`` so that the waveform
is the incident power-wave amplitude ``a(t)`` in ``√W`` (default pulse
peak ≈ 1 → 1 W peak instantaneous incident power).  With that,
``|a|²``/``|b|²`` are watts at every port and S-parameters between
arbitrary port types satisfy lossless unitarity.  Uncalibrated modes
(evanescent at ω_calc, or ``calibrate=False`` CW true-mode ports)
keep ``κ = 1`` — their recorded units are unchanged.

Physical volume states (DD-085)
-------------------------------

The DD-078 convention pinned the *recorded* V/I; the volume states
still carried the M_ε-basis scale (measured: a global constant C with
``C ≈ 1/dy``-class grid dependence).  DD-085 pins C = 1 at the source:
``_calibrate_v_i`` additionally computes the state scale of the
unit-coefficient wave from the calibration guarantee ``I = V/Z``
(see the DD entry for the formula) and folds it inversely into
``source_scale`` / ``record_scale``.  The injected states are thereby
the physical FIT grid quantities ``e = E·l_primal`` of the incident
power wave; recorded V/I and S-parameters are analytically unchanged.
The measured pre-pin scale is kept per mode in ``state_scale``.
"""

from __future__ import annotations

import collections
import importlib
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np

from magnelio._fields.field_arrays import FieldState
from magnelio.constants import C0, EPS0, MU0
from magnelio.ports._modal.discrete import DiscreteMode
from magnelio.ports._modal.dtbc import DTBCTermination
from magnelio.ports._modal.mode import ModeType
from magnelio.ports._modal.port_plane import PortPlane, magnetic_window_ends
from magnelio.ports._modal.port_report import PortOperatorReport

if TYPE_CHECKING:
    from magnelio._operators.material_matrices import PairCouplingProvenance

# Weighted-RMS tolerance on the per-pair modal Courant number: the
# chain is uniform enough for the exact termination, or the mode falls
# back to Mur.
#
# Derived from the reflection budget, not from a category split
# (DD-229).  The termination is exact for the weighted-mean chain, so
# what a spread costs is the residual mismatch, measured through the
# production chain on three fixtures and two perturbation shapes
# (``validation/dtbc_chain_spread_floor.py``):
#
#     TEM plate  ramp   |G| = 1.03 d^2.00     TM11 ramp   43.5 d^2.00
#     TEM plate  spike  |G| = 0.143 d^0.99    TM11 spike  1.34 d^1.89
#     TE10 ramp  |G| = 0.99 d^2.01            TE10 spike  0.21 d^1.94
#
# A smooth tilt is antisymmetric against a symmetric mode, so its
# first-order overlap vanishes and the reflection is second order; a
# localised defect keeps first order.  The worst case is therefore the
# localised TEM one, and the gate is set against the deliberately
# crude bound ``|Gamma| <= spread`` (a 17 dB margin on that measured
# 0.143 coefficient, which is itself geometry-dependent).
#
# At this value the chain mismatch contributes at most -114 dB by that
# bound and -129 dB by the measured law — 14 to 29 dB below the
# -100 dB acceptance line, while the previous classifier threshold of
# 1e-8 was protecting a -160 dB floor and rejecting ordinary conformal
# B-Rep tolerance (measured up to 1e-6) onto a -30 dB absorber.
_DTBC_PAIR_SPREAD_TOL = 2e-6

# Upper edge of the band the gate *reports* on (DD-228).  Failing the
# gate is two different events.  Just above it, the cross-section was
# meant to be a uniform chain and numerical jitter cost it the exact
# termination — the surprising case, worth a sentence: nothing a user
# models deliberately deviates by parts in ten thousand, since
# materials and cell sizes differ by percents.  In dB (DD-229) this
# band is a chain floor of -114 to -80 dB: the region where the
# acceptance line itself comes into view.  Far above it, the
# cross-section is genuinely not a uniform chain — an inhomogeneous
# QTEM line deviates at the material-contrast level (measured 0.22 and
# 0.33 on the shielded-microstrip fixtures) and was never eligible for
# the scalar chain; that is the model the user built, and the answer
# is a different port model (DD-056 CW, DD-057 band), not a mesh fix.
# Warning about it on every run would be a model judgement.  The
# decision stays readable either way on ``ModeReport.termination``.
_DTBC_PAIR_MARGINAL_TOL = 1e-4

# Pairs carrying less than this fraction of the peak modal weight are
# excluded from the spread statistics (masked / PEC edges hold zero
# profile; their pair product is meaningless for the chain).
_DTBC_WEIGHT_FLOOR = 1e-12

# Certificate stage 2 (DD-067): tolerance on the maximum relative
# slab deviation of the mass entries feeding the port's 2D mode
# solve across the first feed cells (port plane vs. slab 1 vs.
# slab 2, computed by the factory).  The transversal pair-product
# gate above is structurally blind to the *normal-face* M_mu (it
# enters the TE transversal operator but no co-located pair) — the
# conformal-coax finding of DD-066: 36 % boundary-slab Hz-M_mu
# deviation certified as "uniform" yet reflected at -42 dB.  A
# z-translation-invariant feed measures ~1e-15 here; any defect
# above the gate sends every channel of the port to Mur.
_DTBC_SLAB_DEFECT_TOL = 1e-8


def _gather_host(arr, idx):
    """Gather ``arr[idx]`` as a NumPy array.

    The modal recursion (Mur / DTBC histories, source buffers) is pure
    host-side scalar work on a few hundred port-plane values — on the
    CuPy backend the port-edge subset is pulled to the host once per
    call (one small D2H transfer) instead of porting the recursion to
    the GPU.  No-op pass-through on NumPy.
    """
    out = arr[idx]
    return out.get() if hasattr(out, "get") else out


def _dual_spacings(nodes: np.ndarray) -> np.ndarray:
    """Dual (half-open Voronoi) spacings of a sorted 1D node array.

    Interior node i owns ``(nodes[i+1] - nodes[i-1]) / 2``; the two end
    nodes own the half-cell towards their single neighbour.  Used by the
    DD-078 physical-power patches; a single node owns zero width.
    """
    n = nodes.size
    if n < 2:
        return np.zeros(n, dtype=float)
    d = np.empty(n, dtype=float)
    d[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])
    d[0] = 0.5 * (nodes[1] - nodes[0])
    d[-1] = 0.5 * (nodes[-1] - nodes[-2])
    return d


def _patch_duals(
    plane: PortPlane,
    magnetic_ends: tuple[bool, bool, bool, bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-edge dual node spacings of the DD-078 Poynting patches.

    Returns ``(dv_dual_u, du_dual_v)``: the dual spacing perpendicular
    to each e_u edge (evaluated at its v node) and to each e_v edge
    (at its u node).  The patch area of an edge is
    ``primal length × this spacing``.

    ``magnetic_ends`` are the ``(u_lo, u_hi, v_lo, v_hi)`` flags from
    :func:`~magnelio.ports._modal.port_plane.magnetic_window_ends`: at
    a PMC bbox window end the magnetic wall sits half the outer cell
    beyond the end node, so its dual extends to the full end cell —
    the same wall-position convention the TEM capacitance quadrature
    books, keeping the physical-power patches and ``z_line``
    consistent.
    """
    u_nodes = np.unique(plane.v_edge_uv[:, 0])
    v_nodes = np.unique(plane.u_edge_uv[:, 1])
    dv_nodes = _dual_spacings(v_nodes)
    du_nodes = _dual_spacings(u_nodes)
    if magnetic_ends is not None:
        u_lo, u_hi, v_lo, v_hi = magnetic_ends
        if u_nodes.size >= 2:
            if u_lo:
                du_nodes[0] = u_nodes[1] - u_nodes[0]
            if u_hi:
                du_nodes[-1] = u_nodes[-1] - u_nodes[-2]
        if v_nodes.size >= 2:
            if v_lo:
                dv_nodes[0] = v_nodes[1] - v_nodes[0]
            if v_hi:
                dv_nodes[-1] = v_nodes[-1] - v_nodes[-2]
    dv = dv_nodes[np.searchsorted(v_nodes, plane.u_edge_uv[:, 1])]
    du = du_nodes[np.searchsorted(u_nodes, plane.v_edge_uv[:, 0])]
    return dv, du


def conformal_flux_patch_scale(
    plane: PortPlane,
    mesh,
    m_eps_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-edge conformality factor χ for the flux surrogate (DD-095).

    The physical-power calibration (``_calibrate_v_i``: ``p_one`` and
    ``s_h``) integrates the constructed basis pair over *geometric*
    Voronoi patches.  At conformal cut edges (sub-cell category 2) the
    true discrete wave's local admittance follows the reduced M_ε, so
    the geometric patch over-counts the transported flux — the s² > 1
    power defect of conformal port planes (internal dossier
    ``investigations/port_power/DERIVATION.md`` §4-5).  The corrected
    patch weight is ``χ_e · A_geo,e`` with

    ``χ_e = 1``                                    (categories 0, 1, 3)
    ``χ_e = M_ε,e·l_e / (ε₀·eps_pair,e·A_geo,e)``  (category 2)

    where ``eps_pair = eps_avg / f_A`` is the free-part permittivity of
    the **first interior slab's** edge classification — the slab whose
    M_ε the port-plane flatten copied, so ``M_ε,e`` is exactly the
    port slice of the (flattened) ``m_eps_flat``.  Dividing the slab
    M_ε by its own dielectric content leaves only the conformal
    geometry content (area cut × L_free amplification × enlarged-cell
    borrowing): the correction is conformality-only, layered dielectric
    staircase planes are untouched (χ ≡ 1).

    One category-0/1 edge does move: the receiver of an enlarged-cell
    donation.  Its M_ε carries the parked mass of a short curved-PEC
    neighbour whose own edge is masked and therefore contributes no
    flux at all, so the patch has to transport that neighbour's share
    as well.  The ratio is taken against the edge's *own* staircase
    mass, recovered by subtracting the donation back out,

    ``χ_d = (M_ε,d·l_d/ε₀) / (M_ε,d·l_d/ε₀ − Σ_s borrowed_s)``,

    which needs no assumption about the two edges' materials and is
    exactly 1 without a donation — the conformality-only invariant
    survives.  (Category-2 receivers need no separate treatment: their
    M_ε ratio already carries the donation.)

    Returns ``(χ_u, χ_v)`` aligned with ``plane.e_u/v_indices``, or
    ``None`` when the mesh carries no sub-cell edge classification.
    """
    em = getattr(mesh, "edge_material", None)
    if em is None:
        return None
    grid = mesh.grid
    n_sizes = (grid.Nx, grid.Ny, grid.Nz)

    def _stride(comp_axis: int, along_axis: int) -> int:
        shape = [n_sizes[ax] + (0 if ax == comp_axis else 1) for ax in range(3)]
        s = 1
        for ax in range(along_axis + 1, 3):
            s *= shape[ax]
        return s

    face = plane.face
    n_ax = face.normal_axis
    shift_u = face.inward_sign * _stride(face.u_axis, n_ax)
    shift_v = face.inward_sign * _stride(face.v_axis, n_ax)
    idx_u = plane.e_u_indices + shift_u
    idx_v = plane.e_v_indices + shift_v

    dv_dual, du_dual = _patch_duals(
        plane,
        magnetic_window_ends(plane, grid, getattr(mesh, "boundary_conditions", None)),
    )
    dn = plane.normal_dx

    def _chi(me_port, lens, d_t, idx):
        cat = em.category[idx]
        chi = np.ones(idx.size, dtype=float)
        # f_A == 0 edges would divide 0/0 below; their chi stays 1
        # either way (the isfinite guard rejects the quotient), so
        # exclude them up front instead of warning.
        m2 = (cat == 2) & (em.f_A[idx] > 0.0)
        if np.any(m2):
            eps_pair = em.eps_avg[idx[m2]] / em.f_A[idx[m2]]
            a_geo = d_t[m2] * dn
            val = me_port[m2] * lens[m2] / (EPS0 * eps_pair * a_geo)
            chi[m2] = np.where(np.isfinite(val) & (val > 0.0), val, 1.0)
        return chi, cat

    me_u = np.asarray(m_eps_flat, dtype=float)[plane.e_u_indices]
    me_v = np.asarray(m_eps_flat, dtype=float)[plane.e_v_indices]
    chi_u, cat_u = _chi(me_u, plane.u_edge_lengths, dv_dual, idx_u)
    chi_v, cat_v = _chi(me_v, plane.v_edge_lengths, du_dual, idx_v)

    donor = em.enlarged_cell_donor
    shorts = np.nonzero(donor >= 0)[0]
    if shorts.size:
        for idx, cat, chi, me, lens in (
            (idx_u, cat_u, chi_u, me_u, plane.u_edge_lengths),
            (idx_v, cat_v, chi_v, me_v, plane.v_edge_lengths),
        ):
            pos = {int(e): i for i, e in enumerate(idx)}
            borrowed = np.zeros(idx.size, dtype=float)
            for s in shorts:
                p = pos.get(int(donor[s]))
                if p is None or cat[p] not in (0, 1):
                    continue
                borrowed[p] += em.enlarged_cell_area[s]
            recv = np.nonzero(borrowed > 0.0)[0]
            if recv.size:
                # ε·A of the receiver, donation included, from its M_ε.
                total = me[recv] * lens[recv] / EPS0
                base = total - borrowed[recv]
                val = np.divide(total, base, out=np.ones_like(total), where=base > 0.0)
                chi[recv] = np.where(np.isfinite(val) & (val > 0.0), val, 1.0)
    return chi_u, chi_v


class PortOperatorModal:
    """Modal absorber on one face of the FIT mesh (modal Mur-1st-order).

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"port1"``).
    plane : PortPlane
        Port-plane geometry, built via :meth:`PortPlane.from_mesh`.
    discrete_modes : list[DiscreteMode]
        Modes already sampled and B-orthonormalised onto ``plane``.
    m_eps_flat : np.ndarray
        Diagonal of the FIT ``M_eps`` matrix indexed by the flat
        E-vector layout.
    m_mu_flat : np.ndarray
        Diagonal of the FIT ``M_mu`` matrix indexed by the flat
        H-vector layout.  Used for the I projection (recording, future).
    dt : float
        Solver time step [s].  Together with the per-mode phase velocity
        and ``plane.normal_dx`` determines the Mur reflection coefficient.
    omega_calc : float
        Mode-calculation angular frequency [rad/s].  Per-mode phase
        velocity is computed from ``β_m(ω_calc) = Im(γ(ω_calc))``;
        if ``β_m = 0`` (DC TEM or evanescent at this frequency) the
        operator falls back to ``v_p = c₀``.
    termination : str, default "auto"
        ``"auto"`` — numerical-path TEM/TE/TM modes with a
        certified-uniform pair product get the exact DTBC (Klein-
        Gordon mass from the discrete 2D eigenvalue), everything
        else modal Mur-1st (see module docstring).  ``"mur"`` —
        force the legacy Mur termination on every mode (A/B
        measurements, tests).
    chain_overrides : dict[int, tuple[float, float]] or None
        Per-mode ``(r, q)`` chain parameters certified *externally*
        (the WP-R4a frequency-local fit from the zeta pencil,
        DD-056).  Modes present here bypass the pair-product gate
        and run the DTBC with the given parameters; ``z0`` is not
        set (the CW postprocessing uses exact per-frequency phasors
        instead of the closed-form wave impedance).
    dual_e_profiles : list of (np.ndarray, np.ndarray) or None
        Per-mode dual-basis projection profiles ``(d_u, d_v)`` used
        by ``project_V`` / ``project_V_interior`` instead of the
        stored primal profiles.  Required when the mode profiles are
        not M_eps-orthogonal (per-frequency true hybrid modes): the
        Gram-inverse projectors remove channel cross-talk while the
        port-plane reconstruction stays primal.  ``None`` entries
        (or ``None`` overall) fall back to the primal profile.
    calibrate : bool, default True
        Run the V/I calibration rescale (module docstring).  The CW
        true-mode path passes ``False`` — its stored profiles must
        stay exactly as built, since the a/b decomposition uses
        exact phasors computed against them.
    chain_slab_defect : float or None, default None
        Certificate stage 2 (DD-067): the maximum relative slab
        deviation of the mass entries feeding the 2D mode solve
        across the first feed cells, computed by the factory
        (``_port_chain_slab_defect``).  Values above
        ``_DTBC_SLAB_DEFECT_TOL`` veto the exact DTBC on every
        channel (Mur fallback) — the transversal pair-product gate
        cannot see normal-face M_mu deviations (DD-066 conformal-coax
        finding).  ``None`` skips the check (spec-level callers that
        build operators without the factory).
    pair_provenance : PairCouplingProvenance or None, default None
        Certificate stage 1 provenance (DD-228): the meshing-time
        pair-coupling record restricted to this port plane's
        transversal H faces, from ``mesh._pair_coupling``.  Names the
        mesh-side cause when the pair-product gate below withholds the
        exact DTBC; ``None`` only costs the warning its cause clause.
    flux_patch : tuple of np.ndarray or None, default None
        DD-095 conformality factors ``(χ_u, χ_v)`` for
        the physical-power patches of ``_calibrate_v_i``, from
        :func:`conformal_flux_patch_scale`.  ``None`` means identity
        (staircase planes, meshes without sub-cell classification,
        direct construction in tests).
    """

    def __init__(
        self,
        name: str,
        plane: PortPlane,
        discrete_modes: list[DiscreteMode],
        m_eps_flat: np.ndarray,
        m_mu_flat: np.ndarray,
        dt: float,
        omega_calc: float,
        port_report: PortOperatorReport | None = None,
        termination: str = "auto",
        chain_overrides: dict[int, tuple[float, float]] | None = None,
        dual_e_profiles: (list[tuple[np.ndarray, np.ndarray] | None] | None) = None,
        calibrate: bool = True,
        chain_slab_defect: float | None = None,
        pair_provenance: "PairCouplingProvenance | None" = None,
        flux_patch: tuple[np.ndarray, np.ndarray] | None = None,
        complement_absorber: (tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None) = None,
        magnetic_patch_ends: tuple[bool, bool, bool, bool] | None = None,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if omega_calc <= 0.0:
            raise ValueError("omega_calc must be positive")

        self.name = name
        self.plane = plane
        self.port_report = port_report
        self.discrete_modes = list(discrete_modes)
        self._n_modes = len(self.discrete_modes)
        self._dt = dt
        # Excited modes (DD-224): ``{mode_idx: (waveform_fn, source-history
        # ring buffer)}``.  Several modes of one port may be driven in one
        # run, each with its own waveform and TF/SF retardation buffer.
        self._excitations: dict[int, tuple] = {}

        # Pre-extract M_eps slices (port + interior planes) and M_mu slice.
        self._me_u_port = np.asarray(m_eps_flat[plane.e_u_indices], dtype=float)
        self._me_v_port = np.asarray(m_eps_flat[plane.e_v_indices], dtype=float)
        self._me_u_int = np.asarray(
            m_eps_flat[plane.e_u_indices_interior],
            dtype=float,
        )
        self._me_v_int = np.asarray(
            m_eps_flat[plane.e_v_indices_interior],
            dtype=float,
        )
        self._mh_u = np.asarray(m_mu_flat[plane.h_u_indices], dtype=float)
        self._mh_v = np.asarray(m_mu_flat[plane.h_v_indices], dtype=float)

        # V/I calibration — see module docstring "V/I calibration".  After
        # the M_ε Gram-Schmidt in ``discretize_modes`` the H profiles inherit
        # only the ε-side normalisation; the ratio ``V_m/I_m`` measured on
        # an analytical mode at the port plane is therefore *not*
        # ``Z_modal``.  Here we rescale ``ĥ`` per mode by a scalar ``γ`` so
        # that ``V_m/I_m = Z_modal`` holds for an analytical FIT-discretised
        # field at unit amplitude.  Skipped for evanescent modes (imaginary
        # ``Z_modal`` cannot be matched by a real scalar rescale).
        if dual_e_profiles is not None and len(dual_e_profiles) != self._n_modes:
            raise ValueError(
                "dual_e_profiles must have one entry per mode "
                f"({self._n_modes}), got {len(dual_e_profiles)}",
            )
        self._dual_e_profiles = dual_e_profiles
        self._chain_slab_defect = chain_slab_defect
        self._pair_provenance = pair_provenance
        # PMC bbox window ends (magnetic_window_ends): the Poynting
        # patches there extend to the magnetic wall half a cell beyond
        # the end node — must match the TEM capacitance booking.
        self._magnetic_patch_ends = magnetic_patch_ends
        # DD-095 conformality patch for the physical-power surrogate
        # (see conformal_flux_patch_scale); identity when the factory
        # supplies none (staircase planes, meshes without sub-cell
        # classification, direct test construction).
        if flux_patch is not None:
            self._flux_chi_u = np.asarray(flux_patch[0], dtype=float)
            self._flux_chi_v = np.asarray(flux_patch[1], dtype=float)
        else:
            self._flux_chi_u = np.ones_like(self._me_u_port)
            self._flux_chi_v = np.ones_like(self._me_v_port)
        # Physical √W amplitude scales (DD-078, module docstring).
        # Defaults (identity) apply to uncalibrated operators (CW
        # true-mode ports) and are overwritten by _calibrate_v_i.
        self.record_scale = np.ones(self._n_modes, dtype=float)
        self._source_scale = np.ones(self._n_modes, dtype=float)
        # Volume-state scale C of the unit-coefficient basis wave
        # relative to physical FIT grid quantities (DD-085).  After the
        # C = 1 pin the injected states ARE grid quantities; the value
        # is kept per mode for introspection and gates.
        self.state_scale = np.ones(self._n_modes, dtype=float)
        if calibrate:
            self.discrete_modes = self._calibrate_v_i(omega_calc)

        # Per-mode phase velocity, Mur reflection coefficient, and TF/SF
        # propagation delay τ_m = dx_n / v_p,m (used by the soft source
        # for incident-field look-up at the interior plane).
        self._v_p = np.empty(self._n_modes, dtype=float)
        self._mur_r = np.empty(self._n_modes, dtype=float)
        self._tau_m = np.empty(self._n_modes, dtype=float)
        for m, dm in enumerate(self.discrete_modes):
            gamma = dm.mode.gamma(omega_calc)
            beta = abs(gamma.imag)
            v_p = (omega_calc / beta) if beta > 0.0 else C0
            self._v_p[m] = v_p
            self._mur_r[m] = (v_p * dt - plane.normal_dx) / (v_p * dt + plane.normal_dx)
            self._tau_m[m] = plane.normal_dx / v_p

        # Mur state — V at port and interior at the previous step.
        # Default (zero) is correct when simulation starts with zero field.
        # For non-trivial initial conditions, call ``initialize_state(e)``
        # before the first FIT step to capture the IC's V at both planes.
        self._V_port_prev = np.zeros(self._n_modes, dtype=float)
        self._V_interior_prev = np.zeros(self._n_modes, dtype=float)

        # Complement absorber (DD-096, WP-M2): per-edge scalar Mur-1 on
        # the port-unrepresented remainder of the plane field.  Without
        # it, the modal wipe pins every unrepresented transverse family
        # to zero at the plane (Dirichlet); cut-off-trapped families
        # then close a growth loop through the oblique dual projection
        # (measured |lambda| up to 1 + 8.6e-5 per step on the WP-M0
        # fixture).  ``complement_absorber = (r_u, r_v, live_u,
        # live_v)``: per-edge Mur coefficients from the local effective
        # permittivity and a 0/1 mask that keeps residual-PEC edges
        # (sub-face window frames) pinned.  Active only while at least
        # one channel runs on Mur (DTBC-certified ports keep their
        # exact, bit-identical path).
        if complement_absorber is not None:
            r_u, r_v, live_u, live_v = complement_absorber
            self._comp_r_u = np.asarray(r_u, dtype=float)
            self._comp_r_v = np.asarray(r_v, dtype=float)
            self._comp_live_u = np.asarray(live_u, dtype=float)
            self._comp_live_v = np.asarray(live_v, dtype=float)
            self._comp_int_prev_u = np.zeros(plane.e_u_indices.size)
            self._comp_int_prev_v = np.zeros(plane.e_v_indices.size)
            self._comp_port_prev_u = np.zeros(plane.e_u_indices.size)
            self._comp_port_prev_v = np.zeros(plane.e_v_indices.size)
        else:
            self._comp_r_u = None

        # WP-G2 fused port-plane transfers: concatenated gather/scatter
        # index arrays, built once.  On a device-array backend one
        # gather (D2H) and one scatter (H2D) per port per step replace
        # the former per-array round trips; the host math consumes the
        # split halves unchanged (bit-identical numbers).  Device-
        # resident copies are cached lazily by ``_fused_indices``.
        self._g_port_e = np.concatenate([plane.e_u_indices, plane.e_v_indices])
        self._g_int_e = np.concatenate([plane.e_u_indices_interior, plane.e_v_indices_interior])
        self._g_port_h = np.concatenate([plane.h_u_indices, plane.h_v_indices])
        self._dev_idx: dict | None = None

        # |V| envelope accumulator for the solver's port-signal stop
        # criterion (DD-096); polled and reset via poll_signal_absmax.
        self._V_absmax = 0.0

        # Per-mode DTBC termination (WP-R2).  ``None`` entries stay on
        # modal Mur-1.
        if termination not in ("auto", "mur"):
            raise ValueError(
                f"termination must be 'auto' or 'mur', got {termination!r}",
            )
        self._dtbc: list[DTBCTermination | None] = []
        self._dtbc_r: list[float | None] = []
        self._dtbc_q: list[float | None] = []
        self._dtbc_z0: list[float | None] = []
        self._dtbc_pair_spread: list[float | None] = []
        overrides = chain_overrides or {}
        for m, dm in enumerate(self.discrete_modes):
            if m in overrides:
                # Externally certified chain (WP-R4a frequency-local
                # fit): bypass the pair gate; the CW postprocessing
                # carries the exact phasors, so no z0 is derived.
                r_mode, q_mode = overrides[m]
                spread = None
            else:
                r_mode, q_mode, spread = self._chain_params(dm)
            self._dtbc_pair_spread.append(spread)
            use = termination == "auto" and r_mode is not None
            self._dtbc_r.append(r_mode if use else None)
            self._dtbc_q.append(q_mode if use else None)
            self._dtbc_z0.append(
                self._chain_z0(dm, r_mode) if use and q_mode > 0.0 and m not in overrides else None
            )
            self._dtbc.append(DTBCTermination(r_mode, q_mode) if use else None)
        self._termination = termination
        self._report_withheld_dtbc()

    def _report_withheld_dtbc(self) -> None:
        """Warn about channels the uniform-chain gate sent back to Mur.

        Certificate stage 1 (DD-228, closing KB-022).  The gate is a
        quality decision the user never asked for and cannot see in
        the result: a channel that fails it keeps working, but trades
        a 1e-14 termination for a -30 dB-class reflection floor, and
        the port next to it on the same model may well keep the exact
        one.  Stage 2 has warned since DD-067; stage 1 was quiet.

        Only a *marginal* failure is reported — one inside
        ``(_DTBC_PAIR_SPREAD_TOL, _DTBC_PAIR_MARGINAL_TOL]``, where a
        cross-section that was meant to be uniform missed it by
        jitter.  A cross-section that is genuinely inhomogeneous
        deviates by orders of magnitude more and is the model the user
        built, not a defect (see the constant).

        Two more non-events arrive as ``spread is None``: a mode that
        is ineligible for the exact DTBC by construction (analytical
        field evaluator), and a stage-2 veto, which has already warned
        with its own cause.  An explicit ``termination="mur"`` is the
        user's own choice.
        """
        if self._termination != "auto":
            return
        rejected = [
            (m, self._dtbc_pair_spread[m])
            for m in range(self._n_modes)
            if self._dtbc[m] is None
            and self._dtbc_pair_spread[m] is not None
            and _DTBC_PAIR_SPREAD_TOL < self._dtbc_pair_spread[m] <= _DTBC_PAIR_MARGINAL_TOL
        ]
        if not rejected:
            return

        names = ", ".join(f"{self.discrete_modes[m].mode.name} ({s:.2e})" for m, s in rejected)
        many = len(rejected) > 1
        cause = (
            "Typical causes: a feed cross-section that is not "
            "translation-invariant along the port normal, or a mesh "
            "whose cell sizes vary across the feed."
        )
        prov = self._pair_provenance
        if prov is not None and prov.faces.size:
            cause = (
                f"Mesh-side cause: {self._g_port_h.size} transversal faces "
                f"feed this gate, {prov.faces.size} of them with a magnetic "
                f"mass derived from a conformal ladder that agreed only to "
                f"{prov.worst:.1e} — above the {prov.certify_rtol:.0e} the "
                f"gate certifies at.  Geometric tolerance in the feed solid "
                f"is the usual source, and a mirrored or unioned body "
                f"inflates it."
            )
        warnings.warn(
            f"Modal port {self.name!r}: the feed cross-section is not a "
            f"uniform discrete chain for "
            f"{'channels' if many else 'channel'} {names} — weighted-RMS "
            f"pair-product spread against the "
            f"{_DTBC_PAIR_SPREAD_TOL:.0e} gate — so the exact transparent "
            f"boundary condition is withheld there and "
            f"{'those channels fall' if many else 'that channel falls'} "
            f"back to modal Mur-1st, a reflection floor of order -30 dB "
            f"instead of -100 dB and below.  {cause}",
            stacklevel=2,
        )

    def _chain_params(
        self,
        dm: DiscreteMode,
    ) -> tuple[float | None, float | None, float | None]:
        """Certified 1D-chain parameters ``(r, q, spread)`` of a mode.

        On a uniform feed line ``M_eps[e] * M_mu[h_partner] =
        eps*mu * normal_dx^2`` per co-located transversal pair (the
        DD-053 pair identity with the flattened port slab supplying
        the uniform-continuation ``M_eps``), so ``r = dt / sqrt(pair
        product)`` — no continuum velocity enters.  ``q`` is the
        Klein-Gordon mass: 0 for TEM; ``omega_c * dt`` for TE/TM
        modes on the numerical path, where ``omega_c`` is the exact
        discrete 2D eigenvalue of the 3D-restricted transversal
        operator (``build_2d_curl_curl`` / ``build_2d_tm_curl_curl``,
        WP-R3 pre-check).  Analytical-path modes carry the continuum
        cut-off instead and stay on Mur.

        Returns ``(r, q, spread)`` with ``spread`` the modal-weighted
        RMS deviation of the per-pair ``r``; ``(None, None, spread)``
        when the mode is ineligible, the spread exceeds the
        uniformity gate, or ``r`` falls outside (0, 1].
        """
        mode = dm.mode
        if mode.mode_type is ModeType.TEM:
            q_mode = 0.0
        elif mode.field_evaluator is None:
            q_mode = mode.omega_c * self._dt
        else:
            return None, None, None

        # Certificate stage 2 (DD-067): the factory-measured slab
        # consistency of the masses feeding the 2D mode solve.  The
        # pair products below cannot see normal-face M_mu, so a
        # boundary-slab deviation there certifies as "uniform" while
        # the volume propagates a different transversal operator.
        if self._chain_slab_defect is not None and self._chain_slab_defect > _DTBC_SLAB_DEFECT_TOL:
            return None, None, None

        pair = np.concatenate(
            [
                self._me_u_port * self._mh_v,
                self._me_v_port * self._mh_u,
            ]
        )
        weight = np.concatenate(
            [
                self._me_u_port * dm.e_u_profile**2,
                self._me_v_port * dm.e_v_profile**2,
            ]
        )
        w_max = float(weight.max()) if weight.size else 0.0
        active = weight > _DTBC_WEIGHT_FLOOR * w_max
        if w_max <= 0.0 or not np.any(active) or np.any(pair[active] <= 0.0):
            return None, None, None

        r_pairs = self._dt / np.sqrt(pair[active])
        w = weight[active]
        r_mean = float(np.dot(w, r_pairs) / w.sum())
        spread = float(math.sqrt(np.dot(w, (r_pairs - r_mean) ** 2) / w.sum()) / r_mean)
        if spread > _DTBC_PAIR_SPREAD_TOL or not (0.0 < r_mean <= 1.0):
            return None, None, spread
        return r_mean, q_mode, spread

    def _chain_z0(self, dm: DiscreteMode, r_mode: float) -> float:
        """Static impedance constant ``z0`` of a certified KG channel.

        The discrete travelling wave's V/I ratio through the
        production projections is ``dtbc_wave_impedance(w*dt, q, z0,
        kind)`` with (WP-R3 pre-check, verified against the coupled
        chain symbol to 1e-15)

            z0 = r * nV * c_pair / (dt * nI),

        where ``nV`` / ``nI`` are the M_eps / M_mu norms of the
        stored profiles and ``c_pair = |h_hat * M_mu / e_hat_partner|``
        is the per-pair dual-voltage ratio — constant across the
        cross-section because the H profile carries the
        travelling-wave form ``h ∝ e_partner / M_mu`` (DD-052).  The
        constant is covariant under the V/I calibration rescale, so
        the a/b decomposition stays exact regardless of calibration
        state.
        """
        num = np.concatenate(
            [
                self._mh_v * dm.h_v_profile,
                self._mh_u * dm.h_u_profile,
            ]
        )
        den = np.concatenate([dm.e_u_profile, dm.e_v_profile])
        weight = np.concatenate(
            [
                self._me_u_port * dm.e_u_profile**2,
                self._me_v_port * dm.e_v_profile**2,
            ]
        )
        active = weight > _DTBC_WEIGHT_FLOOR * float(weight.max())
        c_pairs = np.abs(num[active] / den[active])
        w = weight[active]
        c_pair = float(np.dot(w, c_pairs) / w.sum())

        n_v = float(
            np.dot(self._me_u_port, dm.e_u_profile**2) + np.dot(self._me_v_port, dm.e_v_profile**2)
        )
        n_i = float(np.dot(self._mh_u, dm.h_u_profile**2) + np.dot(self._mh_v, dm.h_v_profile**2))
        return r_mode * n_v * c_pair / (self._dt * n_i)

    def _calibrate_v_i(self, omega_calc: float) -> list[DiscreteMode]:
        """Rescale each mode's H profile so that ``V_m/I_m = Z_modal``.

        See module docstring "V/I calibration".  Modes with non-real
        ``Z_modal(omega_calc)`` (evanescent: |Im Z| > |Re Z|) or with
        a degenerate ``I_test == 0`` are passed through unchanged.

        Returns
        -------
        list[DiscreteMode]
            New list of length ``self._n_modes``.  Calibrated entries
            are fresh ``DiscreteMode`` instances (the dataclass is
            frozen, so we cannot mutate in place).
        """
        plane = self.plane
        u_lens = plane.u_edge_lengths
        v_lens = plane.v_edge_lengths
        normal_dx = plane.normal_dx

        # Port-plane area patches for the physical Poynting flux (DD-078):
        # each co-located (E, H) pair integrates over primal-edge length ×
        # dual node spacing, both taken geometrically from the tensor-
        # product plane (edge midpoints of the perpendicular family sit
        # on the nodes; PEC-masked edges carry zero profile, so their
        # patch value is inert).  The DD-095 conformality factor χ
        # rescales the patch at conformal cut edges to the
        # M_ε-consistent area (conformal_flux_patch_scale) — without it
        # the surrogate over-counts the flux at exactly the cells whose
        # profile norm used reduced areas (s² = 1.072 on the round
        # PTFE coax; internal dossier investigations/port_power §4-5).
        dv_dual, du_dual = _patch_duals(plane, self._magnetic_patch_ends)
        area_u = u_lens * dv_dual * self._flux_chi_u
        area_v = v_lens * du_dual * self._flux_chi_v

        out: list[DiscreteMode] = []
        for m_idx, dm in enumerate(self.discrete_modes):
            if dm.mode.field_evaluator is not None:
                # Phase-1 path: sample analytical evaluator at edge midpoints.
                E_u_raw, _, _, H_v_raw = dm.mode.field_evaluator(
                    plane.u_edge_uv[:, 0],
                    plane.u_edge_uv[:, 1],
                )
                _, E_v_raw, H_u_raw, _ = dm.mode.field_evaluator(
                    plane.v_edge_uv[:, 0],
                    plane.v_edge_uv[:, 1],
                )
                E_u_raw = np.asarray(E_u_raw, dtype=float)
                E_v_raw = np.asarray(E_v_raw, dtype=float)
                H_u_raw = np.asarray(H_u_raw, dtype=float)
                H_v_raw = np.asarray(H_v_raw, dtype=float)

                V_test = float(np.dot(self._me_u_port, dm.e_u_profile * E_u_raw * u_lens)) + float(
                    np.dot(self._me_v_port, dm.e_v_profile * E_v_raw * v_lens)
                )
                I_test = (
                    MU0
                    * normal_dx
                    * (
                        float(np.dot(u_lens, dm.h_v_profile * H_v_raw))
                        + float(np.dot(v_lens, dm.h_u_profile * H_u_raw))
                    )
                )
            else:
                # Phase-2 numerical path (WP7.2): the discrete profiles
                # are the edge/face voltages of the discrete travelling
                # wave up to one scalar, so V/I on the travelling wave
                # is measured directly in the operator's own M metric —
                # no edge-length identities needed.
                V_test = float(np.dot(self._me_u_port, dm.e_u_profile**2)) + float(
                    np.dot(self._me_v_port, dm.e_v_profile**2)
                )
                I_test = float(np.dot(self._mh_u, dm.h_u_profile**2)) + float(
                    np.dot(self._mh_v, dm.h_v_profile**2)
                )
            Z = dm.mode.z_modal(omega_calc)
            Z_real_part = float(Z.real)

            # Physical power of the unit-coefficient travelling wave
            # (DD-078).  The analytical path stores pre-γ ĥ as sampled
            # physical H (A/m); the numerical path stores the wave's
            # dual voltages h = e·(μ₀·ndx/Z)/M_μ — undo that convention
            # to recover the pointwise field before integrating.
            if dm.mode.field_evaluator is not None:
                h_v_phys = dm.h_v_profile
                h_u_phys = dm.h_u_profile
            else:
                h_v_phys = dm.h_v_profile * self._mh_v / (MU0 * normal_dx)
                h_u_phys = dm.h_u_profile * self._mh_u / (MU0 * normal_dx)
            p_one = float(np.dot(dm.e_u_profile * h_v_phys, area_u)) - float(
                np.dot(dm.e_v_profile * h_u_phys, area_v)
            )

            calibrated = False
            gamma = 0.0
            if I_test != 0.0 and abs(Z.imag) <= abs(Z.real) and Z_real_part != 0.0:
                # γ = (V_test / I_test) / Z_real chosen so that after
                # ``ĥ_new = γ · ĥ_old`` the projection scales as
                # ``I_m_new = γ · I_m_old`` and therefore
                # ``V_m / I_m_new = (V_test/I_test) / γ = Z_real``.
                gamma = (V_test / I_test) / Z_real_part
                if math.isfinite(gamma) and gamma != 0.0:
                    out.append(
                        DiscreteMode(
                            mode=dm.mode,
                            e_u_profile=dm.e_u_profile,
                            e_v_profile=dm.e_v_profile,
                            h_u_profile=dm.h_u_profile * gamma,
                            h_v_profile=dm.h_v_profile * gamma,
                        )
                    )
                    calibrated = True

            if calibrated and p_one != 0.0 and math.isfinite(p_one):
                kappa = math.sqrt(abs(p_one) * abs(Z_real_part))

                # DD-085: volume-state scale of the unit-coefficient
                # wave.  The physical A/m shape of the wave's H:
                if dm.mode.field_evaluator is not None:
                    H_v_shape = dm.h_v_profile  # sampled A/m
                    H_u_shape = dm.h_u_profile
                else:
                    # undo the dual-voltage convention per edge
                    # (h = H·l_dual; l_dual = μ₀·ndx·l_partner/M_μ)
                    H_v_shape = dm.h_v_profile * self._mh_v / (MU0 * normal_dx * u_lens)
                    H_u_shape = dm.h_u_profile * self._mh_u / (MU0 * normal_dx * v_lens)

                # Grid-quantity Poynting sum (E = ê/l over dA = l·dual
                # spacing — the primal lengths cancel):
                s_h = float(np.dot(dm.e_u_profile * H_v_shape, area_u / u_lens)) - float(
                    np.dot(dm.e_v_profile * H_u_shape, area_v / v_lens)
                )
                # Projection-metric partner ⟨ĥ_post-γ, H·l_dual⟩_Mμ
                # (M_μ·l_dual = μ₀·ndx·l_partner):
                hq = (
                    gamma
                    * MU0
                    * normal_dx
                    * (
                        float(np.dot(dm.h_v_profile * H_v_shape, u_lens))
                        + float(np.dot(dm.h_u_profile * H_u_shape, v_lens))
                    )
                )
                c_state = 0.0
                if hq != 0.0:
                    ratio = s_h / hq
                    if math.isfinite(ratio) and ratio > 0.0:
                        c_state = math.sqrt(ratio) / kappa

                if c_state > 0.0 and math.isfinite(c_state):
                    # Pin C = 1 at the source: the injected profile per
                    # √W shrinks by C, the recorder compensates — the
                    # volume states become physical grid quantities
                    # while recorded V/I stay analytically unchanged.
                    self.state_scale[m_idx] = c_state
                    self.record_scale[m_idx] = kappa * c_state
                    self._source_scale[m_idx] = math.sqrt(abs(Z_real_part)) / kappa / c_state
                else:
                    self.record_scale[m_idx] = kappa
                    self._source_scale[m_idx] = math.sqrt(abs(Z_real_part)) / kappa

            if not calibrated:
                out.append(dm)

        return out

    def initialize_state(self, e: np.ndarray) -> None:
        """Capture ``V_port`` and ``V_interior`` from the initial-condition E.

        Required when the simulation starts with non-zero fields (e.g.
        a wave packet IC for the silence test): without this call, the
        Mur formula at step 0 uses ``V_*_prev = 0`` while ``V_*_new``
        reflects the IC, and ``r·(V_int_new − 0)`` introduces a sign-
        flipped correction at the boundary.

        Parameters
        ----------
        e : np.ndarray
            Flat E vector at the initial time ``t = 0``.  Typically
            ``solver._fields.e_flat`` after the user injects the IC.
        """
        self._V_port_prev[:] = self.project_V(e)
        self._V_interior_prev[:] = self.project_V_interior(e)
        for m, term in enumerate(self._dtbc):
            if term is not None:
                term.initialize(float(self._V_port_prev[m]))
        if self._comp_r_u is not None:
            pl = self.plane
            for idx_u, idx_v, V, dst_u, dst_v in (
                (
                    pl.e_u_indices_interior,
                    pl.e_v_indices_interior,
                    self._V_interior_prev,
                    "_comp_int_prev_u",
                    "_comp_int_prev_v",
                ),
                (
                    pl.e_u_indices,
                    pl.e_v_indices,
                    self._V_port_prev,
                    "_comp_port_prev_u",
                    "_comp_port_prev_v",
                ),
            ):
                cu = np.asarray(_gather_host(e, idx_u), dtype=float).copy()
                cv = np.asarray(_gather_host(e, idx_v), dtype=float).copy()
                for m, dm in enumerate(self.discrete_modes):
                    cu -= V[m] * dm.e_u_profile
                    cv -= V[m] * dm.e_v_profile
                setattr(self, dst_u, cu)
                setattr(self, dst_v, cv)

    # ------------------------------------------------------------------
    # Checkpoint / resume (DD-070, WP-S6)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Checkpoint the Mur previous-values, the per-mode TF/SF
        source-history ring buffers, and every per-mode DTBC convolution
        history.

        The excitation waveforms themselves are *not* stored (they are
        re-set on resume); everything here is state the leapfrog update
        mutates and a bit-exact continuation must restore.
        """
        sd = {
            "V_port_prev": self._V_port_prev.copy(),
            "V_interior_prev": self._V_interior_prev.copy(),
            "src_buffers": {
                str(m): np.array(buf, dtype=float) for m, (_, buf) in self._excitations.items()
            },
            "src_maxlens": {
                str(m): (0 if buf.maxlen is None else int(buf.maxlen))
                for m, (_, buf) in self._excitations.items()
            },
            "dtbc": {
                str(m): term.state_dict() for m, term in enumerate(self._dtbc) if term is not None
            },
        }
        if self._comp_r_u is not None:
            sd["complement"] = {
                "int_u": self._comp_int_prev_u.copy(),
                "int_v": self._comp_int_prev_v.copy(),
                "port_u": self._comp_port_prev_u.copy(),
                "port_v": self._comp_port_prev_v.copy(),
            }
        return sd

    def load_state_dict(self, sd: dict) -> None:
        """Restore state written by :meth:`state_dict` (bit-exact resume)."""
        self._V_port_prev[:] = np.asarray(sd["V_port_prev"], dtype=float)
        self._V_interior_prev[:] = np.asarray(sd["V_interior_prev"], dtype=float)
        # The waveforms were re-bound by the caller (set_excitation before
        # the load); only the retardation buffers are restored, per mode.
        for m_str, buf in sd["src_buffers"].items():
            m = int(m_str)
            if m not in self._excitations:
                raise ValueError(
                    f"checkpoint carries a source-history buffer for mode {m} "
                    f"of port {self.name!r}, which the rebuilt run does not excite",
                )
            fn = self._excitations[m][0]
            self._excitations[m] = (
                fn,
                collections.deque(
                    np.asarray(buf, dtype=float).tolist(),
                    maxlen=int(sd["src_maxlens"][m_str]),
                ),
            )
        for m_str, tsd in sd["dtbc"].items():
            term = self._dtbc[int(m_str)]
            if term is None:
                raise ValueError(
                    f"checkpoint carries DTBC state for mode {m_str} but "
                    f"the rebuilt port {self.name!r} terminates it on Mur",
                )
            term.load_state_dict(tsd)
        comp = sd.get("complement")
        if comp is not None:
            if self._comp_r_u is None:
                raise ValueError(
                    f"checkpoint carries complement-absorber state but "
                    f"the rebuilt port {self.name!r} has no absorber "
                    f"configured",
                )
            self._comp_int_prev_u = np.asarray(comp["int_u"], dtype=float)
            self._comp_int_prev_v = np.asarray(comp["int_v"], dtype=float)
            self._comp_port_prev_u = np.asarray(comp["port_u"], dtype=float)
            self._comp_port_prev_v = np.asarray(comp["port_v"], dtype=float)
        elif self._comp_r_u is not None:
            # Pre-DD-096 checkpoint: the absorber starts from rest (the
            # complement state re-fills within one step from the field).
            self._comp_int_prev_u = np.zeros(self.plane.e_u_indices.size)
            self._comp_int_prev_v = np.zeros(self.plane.e_v_indices.size)
            self._comp_port_prev_u = np.zeros(self.plane.e_u_indices.size)
            self._comp_port_prev_v = np.zeros(self.plane.e_v_indices.size)

    # ------------------------------------------------------------------
    # Read-only inspection
    # ------------------------------------------------------------------

    @property
    def n_modes(self) -> int:
        return self._n_modes

    @property
    def mur_r(self) -> np.ndarray:
        """Per-mode Mur reflection coefficient ``r_m``."""
        return self._mur_r.copy()

    @property
    def h_dual_lengths(self) -> tuple[np.ndarray, np.ndarray]:
        """Dual edge lengths of the port plane's H faces, ``(l_u, l_v)``.

        The mode solvers store H as the dual voltage
        ``ĥ = H · l_dual`` with ``l_dual = μ₀ · normal_dx · l_partner /
        M_μ`` (the convention the Poynting sum in
        :meth:`_calibrate_v_i` undoes).  Dividing by these lengths is
        the only way back to A/m — on a graded transversal grid the two
        differ per face.  ``l_u`` is co-located with the v-edges (like
        ``h_u_profile``), ``l_v`` with the u-edges.

        Faces frozen inside a conductor carry ``M_μ = 0`` and
        ``ĥ = 0``; their entry is ``0.0`` so a consumer can mask on it
        instead of dividing by zero.
        """
        scale = MU0 * float(self.plane.normal_dx)
        l_u = np.divide(
            scale * self.plane.v_edge_lengths,
            self._mh_u,
            out=np.zeros_like(self._mh_u),
            where=self._mh_u > 0.0,
        )
        l_v = np.divide(
            scale * self.plane.u_edge_lengths,
            self._mh_v,
            out=np.zeros_like(self._mh_v),
            where=self._mh_v > 0.0,
        )
        return l_u, l_v

    @property
    def dtbc_line_params(
        self,
    ) -> dict[int, tuple[float, float, float | None]]:
        """Per-mode discrete line parameters ``(r, q, z0)`` of DTBC modes.

        Keys are mode indices terminated by the exact DTBC; modes on
        Mur are absent.  Consumed by ``compute_s_parameters`` for the
        discrete-exact de-stagger of the I sampling plane and — for
        Klein-Gordon channels (``q > 0``, ``z0`` set) — the exact
        discrete wave impedance
        (:func:`~magnelio.ports._modal.dtbc.dtbc_wave_impedance`).
        ``z0`` is ``None`` on TEM channels, whose calibrated V/I is
        frequency-flat (DD-054).
        """
        return {
            m: (r, self._dtbc_q[m], self._dtbc_z0[m])
            for m, r in enumerate(self._dtbc_r)
            if r is not None
        }

    @property
    def termination_kinds(self) -> list[str]:
        """Per-mode termination branch: ``"dtbc"`` or ``"mur"``."""
        return ["dtbc" if term is not None else "mur" for term in self._dtbc]

    @property
    def chain_spreads(self) -> list[float | None]:
        """Per-mode uniform-chain measurement behind the termination.

        The weighted-RMS pair-product spread of the feed cross-section,
        the quantity the gate compares against.  ``None`` where no
        spread was measured: an analytical-path mode (ineligible by
        construction) or a stage-2 veto that decided before stage 1
        ran.  Publishing it lets a caller distinguish a cross-section
        that missed the gate by jitter from one that is genuinely
        inhomogeneous, which are different conversations.
        """
        return list(self._dtbc_pair_spread)

    # ------------------------------------------------------------------
    # Excitation lifecycle
    # ------------------------------------------------------------------

    def set_excitation(
        self,
        mode_idx: int,
        waveform_fn,
    ) -> None:
        """Activate the TF/SF source on mode ``mode_idx``.

        Allocates the mode's source-history ring buffer sized for its
        propagation delay ``τ_m = dx_n / v_p,m`` plus a 3-sample safety
        margin.  Each mode carries its own excitation: a second call on
        another mode drives both modes simultaneously (DD-224), a second
        call on the same mode replaces its waveform.
        :meth:`clear_excitation` drops every excitation.

        Parameters
        ----------
        mode_idx : int
            Index in ``[0, n_modes)``.
        waveform_fn : Callable[[float], float]
            Source amplitude as a function of time [s].

        Raises
        ------
        ValueError
            If ``mode_idx`` is out of range.
        """
        if not (0 <= mode_idx < self._n_modes):
            raise ValueError(
                f"mode_idx {mode_idx} out of range [0, {self._n_modes})",
            )
        # DD-078: the user waveform is the incident power-wave amplitude
        # a(t) in √W; convert to the operator's internal basis units so
        # that record_scale·V projects back to physical volts.
        src_scale = float(self._source_scale[mode_idx])
        if src_scale != 1.0:

            def _scaled_wf(t: float, _fn=waveform_fn, _s=src_scale) -> float:
                return _s * _fn(t)

            waveform_fn = _scaled_wf

        tau_exc = self._tau_m[mode_idx]
        buf_len = max(
            int(math.ceil((tau_exc + self._dt) / self._dt)) + 3,
            5,
        )
        self._excitations[mode_idx] = (waveform_fn, collections.deque(maxlen=buf_len))

    def clear_excitation(self) -> None:
        """Deactivate every TF/SF source — operator becomes a passive absorber."""
        self._excitations = {}

    @property
    def excited_modes(self) -> tuple[int, ...]:
        """Mode indices currently driven through :meth:`set_excitation`."""
        return tuple(sorted(self._excitations))

    def excitation_waveform(self, mode_idx: int):
        """The (basis-scaled) waveform bound to ``mode_idx``, or ``None``."""
        entry = self._excitations.get(mode_idx)
        return None if entry is None else entry[0]

    # ------------------------------------------------------------------
    # Projections
    # ------------------------------------------------------------------

    def _fused_indices(self, arr) -> dict:
        """Device-resident concatenated index arrays (WP-G2), cached.

        Built on the first call with a device array; ``xp`` is resolved
        from the array itself (NumPy for ndarray subclasses that mimic
        the device interface — the unit-test path without CUDA).
        """
        if self._dev_idx is None:
            xp = (
                np
                if isinstance(arr, np.ndarray)
                else (importlib.import_module(type(arr).__module__.split(".")[0]))
            )
            self._dev_idx = {
                "xp": xp,
                "port_e": xp.asarray(self._g_port_e),
                "int_e": xp.asarray(self._g_int_e),
                "port_h": xp.asarray(self._g_port_h),
            }
        return self._dev_idx

    def project_V(self, e: np.ndarray) -> np.ndarray:
        """``V_m`` at the port plane: ``Σ_p M_ε[p] · ê_m,p · e_pp,p``."""
        if hasattr(e, "get"):  # WP-G2: ONE fused gather round trip
            return self.project_V_samples(_gather_host(e, self._fused_indices(e)["port_e"]))
        return self._project_V_at(
            e,
            self.plane.e_u_indices,
            self.plane.e_v_indices,
            self._me_u_port,
            self._me_v_port,
        )

    def project_V_interior(self, e: np.ndarray) -> np.ndarray:
        """``V_m`` at the one-cell-inside companion plane."""
        if hasattr(e, "get"):  # WP-G2: ONE fused gather round trip
            s = _gather_host(e, self._fused_indices(e)["int_e"])
            n_u = self.plane.e_u_indices_interior.size
            return self._V_from_samples(s[:n_u], s[n_u:], self._me_u_int, self._me_v_int)
        return self._project_V_at(
            e,
            self.plane.e_u_indices_interior,
            self.plane.e_v_indices_interior,
            self._me_u_int,
            self._me_v_int,
        )

    def project_I(self, h: np.ndarray) -> np.ndarray:
        """``I_m = ⟨ĥ_m, h⟩_Mμ`` at the port plane's dual edges."""
        if hasattr(h, "get"):  # WP-G2: ONE fused gather round trip
            return self.project_I_samples(_gather_host(h, self._fused_indices(h)["port_h"]))
        h_u = _gather_host(h, self.plane.h_u_indices)
        h_v = _gather_host(h, self.plane.h_v_indices)
        return self._I_from_samples(h_u, h_v)

    # ------------------------------------------------------------------
    # Staged-recording interface (WP-G1): the recorder gathers the raw
    # port-plane samples into a device ring buffer (no per-step sync on
    # the CuPy backend) and later feeds the drained host blocks through
    # the SAME dot products as project_V / project_I — bit-identical by
    # construction, since a gather-then-split reproduces the per-array
    # gathers element for element.
    # ------------------------------------------------------------------

    @property
    def record_gather_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """``(e_indices, h_indices)`` flat gather indices for staged V/I.

        ``e_indices = concat(e_u, e_v)`` port-plane E indices and
        ``h_indices = concat(h_u, h_v)`` dual-edge H indices;
        ``project_V_samples`` / ``project_I_samples`` consume one
        gathered row each and split at the stored u-count.
        """
        return self._g_port_e, self._g_port_h

    def project_V_samples(self, e_samples: np.ndarray) -> np.ndarray:
        """Port-plane ``V_m`` from a pre-gathered host sample row."""
        n_u = self.plane.e_u_indices.size
        return self._V_from_samples(
            e_samples[:n_u],
            e_samples[n_u:],
            self._me_u_port,
            self._me_v_port,
        )

    def project_I_samples(self, h_samples: np.ndarray) -> np.ndarray:
        """Port-plane ``I_m`` from a pre-gathered host sample row."""
        n_u = self.plane.h_u_indices.size
        return self._I_from_samples(h_samples[:n_u], h_samples[n_u:])

    def _I_from_samples(
        self,
        h_u: np.ndarray,
        h_v: np.ndarray,
    ) -> np.ndarray:
        I = np.empty(self._n_modes)
        for m, dm in enumerate(self.discrete_modes):
            I[m] = float(np.dot(self._mh_u, dm.h_u_profile * h_u)) + float(
                np.dot(self._mh_v, dm.h_v_profile * h_v)
            )
        return I

    def _project_V_at(
        self,
        e: np.ndarray,
        e_u_idx: np.ndarray,
        e_v_idx: np.ndarray,
        me_u: np.ndarray,
        me_v: np.ndarray,
    ) -> np.ndarray:
        e_u = _gather_host(e, e_u_idx)
        e_v = _gather_host(e, e_v_idx)
        return self._V_from_samples(e_u, e_v, me_u, me_v)

    def _V_from_samples(
        self,
        e_u: np.ndarray,
        e_v: np.ndarray,
        me_u: np.ndarray,
        me_v: np.ndarray,
    ) -> np.ndarray:
        V = np.empty(self._n_modes)
        for m, dm in enumerate(self.discrete_modes):
            dual = self._dual_e_profiles[m] if self._dual_e_profiles is not None else None
            p_u, p_v = (
                dual
                if dual is not None
                else (
                    dm.e_u_profile,
                    dm.e_v_profile,
                )
            )
            V[m] = float(np.dot(me_u, p_u * e_u)) + float(np.dot(me_v, p_v * e_v))
        return V

    def poll_signal_absmax(self) -> float:
        """Return and reset the |V| envelope since the last poll.

        Feeds the solver's ``port_signal_stop_db`` criterion (DD-096):
        the maximum over all channels of ``|V_port_corr|`` accumulated
        by ``update_e`` between polls.  In DD-078 physical units the
        scale is channel-consistent; only the ratio to the run peak is
        ever used.
        """
        m = self._V_absmax
        self._V_absmax = 0.0
        return m

    @property
    def _complement_active(self) -> bool:
        """Complement absorber configured AND >= 1 channel on Mur.

        Fully DTBC-certified ports keep the exact wipe path
        (bit-identical to pre-DD-096 behaviour); the trapped-family
        growth loop is a property of the Mur recursion (WP-M1 dossier),
        so the absorber ships with the Mur fallback only.
        """
        return self._comp_r_u is not None and any(term is None for term in self._dtbc)

    # ------------------------------------------------------------------
    # FIT-solver hook
    # ------------------------------------------------------------------

    def update_e(self, fields: FieldState, t: float, dt: float) -> None:
        """Apply the per-mode boundary termination (and optional source).

        DTBC modes take one exact chain step (ghost-relation
        convolution; incident prescribed at the ghost plane when
        excited); the remaining modes run modal Mur-1st as described
        below.

        Implements the unified :class:`magnelio.ports.base.Port` protocol
        E-side hook.  Called after ``update_E``, PEC enforcement, CPML
        E-corrections, and source injections — i.e. as the *last* E-side
        step of the leapfrog.  ``fields.e_flat`` is at ``t^{n+1}``;
        ``fields.h_flat`` is at ``t^{n+1/2}`` (unused by the absorber).

        Non-excited modes
        -----------------
        Standard modal Mur on the total field:

            V_port_corr = V_int_prev + r_m · (V_int_new − V_port_prev)

        Excited mode (TF/SF, see module docstring)
        -------------------------------------------
        Mur acts on the *scattered* field; the *incident* is reinstated:

            V_inc,port  = s(t^{n+1})
            V_inc,int   = s(t^{n+1} − τ_m)            (linearly interpolated)
            V_scat      = V_total − V_inc
            V_scat,port_corr = V_scat,int_prev
                               + r_m · (V_scat,int_new − V_scat,port_prev)
            V_port_corr = V_inc,port + V_scat,port_corr

        Finally ``e`` at port-plane primal edges is overwritten with
        ``Σ_m V_port_corr[m] · ê_m,p``.  With the complement absorber
        active (DD-096; configured by the factory, live while >= 1
        channel runs on Mur) the port-unrepresented remainder of the
        interior plane is advanced by a per-edge scalar Mur-1 and added
        on top, so unrepresented transverse families see an absorbing
        plane instead of a Dirichlet wall (the WP-M1-identified
        trapped-family growth loop).  Without the absorber the plane
        lies purely in the modal basis, as before.
        """
        del dt  # operator carries its own ``self._dt`` from build time

        e = fields.e_flat
        comp_on = self._complement_active
        if comp_on:
            # One interior-plane gather serves both the modal projection
            # and the complement extraction (same numbers as
            # ``project_V_interior`` — a gather-then-split reproduces
            # the per-array gathers element for element).
            if hasattr(e, "get"):
                s_int = _gather_host(e, self._fused_indices(e)["int_e"])
            else:
                s_int = np.concatenate(
                    [
                        _gather_host(e, self.plane.e_u_indices_interior),
                        _gather_host(e, self.plane.e_v_indices_interior),
                    ]
                )
            n_u_int = self.plane.e_u_indices_interior.size
            e_u_int = np.asarray(s_int[:n_u_int], dtype=float)
            e_v_int = np.asarray(s_int[n_u_int:], dtype=float)
            V_int_new = self._V_from_samples(
                e_u_int,
                e_v_int,
                self._me_u_int,
                self._me_v_int,
            )
            # Port-unrepresented remainder at the interior plane.  The
            # dual-basis projection makes the subtraction exact:
            # <w_c, comp> = 0 for every port channel c.
            comp_u_int = e_u_int.copy()
            comp_v_int = e_v_int.copy()
            for m in range(self._n_modes):
                dm = self.discrete_modes[m]
                comp_u_int -= V_int_new[m] * dm.e_u_profile
                comp_v_int -= V_int_new[m] * dm.e_v_profile
        else:
            V_int_new = self.project_V_interior(e)

        # Naive Mur on total — correct for non-excited modes; will be
        # overridden below for the excited mode via TF/SF.
        V_port_corr = self._V_interior_prev + self._mur_r * (V_int_new - self._V_port_prev)

        for m_exc, (exc_fn, src_buffer) in self._excitations.items():
            if self._dtbc[m_exc] is not None:
                continue  # DTBC modes take the incident at the ghost plane below
            src_val_new = float(exc_fn(t))
            src_buffer.append(src_val_new)

            tau = self._tau_m[m_exc]
            v_inc_int_now = _interp_delayed(
                src_buffer,
                tau,
                self._dt,
            )
            v_inc_int_prev = _interp_delayed(
                src_buffer,
                tau + self._dt,
                self._dt,
            )
            v_inc_face_prev = _interp_delayed(
                src_buffer,
                self._dt,
                self._dt,
            )

            scat_int_now = V_int_new[m_exc] - v_inc_int_now
            scat_int_prev = self._V_interior_prev[m_exc] - v_inc_int_prev
            scat_face_prev = self._V_port_prev[m_exc] - v_inc_face_prev

            V_port_corr[m_exc] = (
                src_val_new + scat_int_prev + self._mur_r[m_exc] * (scat_int_now - scat_face_prev)
            )

        # DTBC modes: one exact chain step of the boundary plane
        # (overrides the naive Mur value).  ``_V_interior_prev`` holds
        # the interior amplitude at the boundary's own time level
        # (projected from the field one solver step ago).  For an
        # excited DTBC mode the incident is prescribed at the ghost
        # plane at chain time ``t^n = t - dt``; the same kernel then
        # propagates it into the domain as the exact discrete incoming
        # wave (see the dtbc module docstring).
        for m, term in enumerate(self._dtbc):
            if term is None:
                continue
            src = 0.0
            exc = self._excitations.get(m)
            if exc is not None:
                src = float(exc[0](t - self._dt))
            V_port_corr[m] = term.advance(
                float(self._V_interior_prev[m]),
                src,
            )

        # Reconstruct e at port-plane edges from the modal expansion
        # (zero + add per-mode contribution).  This wipes any non-modal
        # component the FIT update injected and forces e_pp to lie
        # purely in the span of the modal basis.  Critical for proper
        # absorption — a "diff subtract" approach leaves non-modal
        # numerical artefacts at the boundary that prevent energy decay.
        # Built host-side in the NumPy branch's op order (bit-identical
        # on the CPU backend), then written in ONE scatter per axis —
        # a single small H2D transfer on the CuPy backend.
        e_u_new = np.zeros(self.plane.e_u_indices.size)
        e_v_new = np.zeros(self.plane.e_v_indices.size)
        for m in range(self._n_modes):
            dm = self.discrete_modes[m]
            e_u_new += V_port_corr[m] * dm.e_u_profile
            e_v_new += V_port_corr[m] * dm.e_v_profile
        if comp_on:
            # Complement absorber: advance the remainder to the plane
            # with the per-edge Mur-1 and ADD it to the modal write
            # (the plane no longer acts as a Dirichlet wall for
            # unrepresented families).  Residual-PEC edges stay pinned
            # via the live mask.
            comp_u_port = self._comp_live_u * (
                self._comp_int_prev_u + self._comp_r_u * (comp_u_int - self._comp_port_prev_u)
            )
            comp_v_port = self._comp_live_v * (
                self._comp_int_prev_v + self._comp_r_v * (comp_v_int - self._comp_port_prev_v)
            )
            e_u_new += comp_u_port
            e_v_new += comp_v_port
            self._comp_int_prev_u = comp_u_int
            self._comp_int_prev_v = comp_v_int
            self._comp_port_prev_u = comp_u_port
            self._comp_port_prev_v = comp_v_port
        if hasattr(e, "get"):  # WP-G2: ONE fused H2D scatter
            idx = self._fused_indices(e)
            e[idx["port_e"]] = idx["xp"].asarray(np.concatenate([e_u_new, e_v_new]))
        else:
            e[self.plane.e_u_indices] = e_u_new
            e[self.plane.e_v_indices] = e_v_new

        # Save TOTAL state for next step.  ``V_port_corr`` is the
        # operator-corrected port amplitude (incident + scattered for the
        # excited mode, pure Mur otherwise).
        self._V_port_prev[:] = V_port_corr
        self._V_interior_prev[:] = V_int_new
        # |V| envelope since the last solver poll (DD-096 port-signal
        # stop criterion): a windowed max cannot be faked low by a zero
        # crossing at poll time.  ``record_scale`` puts the channels on
        # the common physical scale (DD-078) so the cross-channel max
        # is meaningful.
        m = float(np.max(np.abs(V_port_corr) * self.record_scale))
        if m > self._V_absmax:
            self._V_absmax = m


def _interp_delayed(
    buffer: collections.deque,
    delay: float,
    dt: float,
) -> float:
    """Linear interpolation for fractional delay in a ring buffer.

    The buffer stores past source samples with the most recently appended
    value last (``buffer[-1]`` = source at ``t^{n+1}``).  ``delay`` is the
    desired look-back time; returns the linearly interpolated value at
    ``t^{n+1} − delay``.  Out-of-range look-backs return ``0.0`` so that
    early steps (buffer not yet filled) treat the source as quiescent.
    """
    n = len(buffer)
    if n == 0 or dt <= 0.0:
        return 0.0
    idx_frac = delay / dt
    i0 = int(math.floor(idx_frac))
    alpha = idx_frac - i0
    if n - 1 - i0 < 0:
        return 0.0
    v0 = buffer[n - 1 - i0]
    v1 = buffer[n - 2 - i0] if n - 2 - i0 >= 0 else 0.0
    return (1.0 - alpha) * v0 + alpha * v1
