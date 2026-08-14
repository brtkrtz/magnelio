"""
CFS-PML (Complex Frequency Shifted Perfectly Matched Layer) boundary condition.

Implements the full CFS stretching function:

    s(ω) = κ + σ / (α + jωε₀)

with three graded profiles:
    σ(ρ) = σ_max · ρ^m          (conductivity — primary absorption)
    κ(ρ) = 1 + (κ_max − 1) · ρ^m  (coordinate stretching — evanescent waves)
    α(ρ) = α_max · (1 − ρ)       (frequency shift — low-frequency stability)

Auxiliary ψ variables implement the convolutional update (z-face example):

    ψ_{Ex,z}^{n+1} = b · ψ_{Ex,z}^n + c · ΔHy
    Ex_correction   = β_E · [(1 − 1/κ) · ΔHy + ψ_{Ex,z}]

    b = exp(−(σ/κ + α) · dt / ε₀)
    c = σ / (κσ + κ²α) · (1 − b)     (= 0 where σ = 0)

"""

# Design: DD-001/DD-031 (CPML formulation; see spec.md and design-decisions.md).

from __future__ import annotations

import numpy as np

from magnelio._backend.array_api import get_xp
from magnelio.constants import C0, EPS0  # noqa: E402


class CPMLBoundary:
    """CFS-PML absorbing boundary on one face of the domain.

    Parameters
    ----------
    face : str
        ``'xmin'`` | ``'xmax'`` | ``'ymin'`` | ``'ymax'`` |
        ``'zmin'`` | ``'zmax'``
    grid : GridLines
        Simulation grid.
    thickness_cells : int
        Number of PML cells (default 8).
    m : int
        Polynomial grading order (default 3).
    R_target : float
        Target reflection coefficient (default 1e-8).
    kappa_max : float
        Maximum coordinate stretching factor (default 7).
        κ = 1 means no stretching (classic CPML).
    alpha_max : float
        Maximum frequency-shift parameter (default 0.02).
        Improves low-frequency absorption and late-time stability.
    """

    _FACE_AXIS = {
        "xmin": ("x", "min"),
        "xmax": ("x", "max"),
        "ymin": ("y", "min"),
        "ymax": ("y", "max"),
        "zmin": ("z", "min"),
        "zmax": ("z", "max"),
    }

    def __init__(
        self,
        face: str,
        grid,
        thickness_cells: int = 8,
        m: int = 3,
        R_target: float = 1e-8,
        kappa_max: float = 7.0,
        alpha_max: float = 0.02,
    ) -> None:
        if face not in self._FACE_AXIS:
            raise ValueError(f"Unknown face: {face!r}")
        self.face = face
        self.grid = grid
        self.thickness_cells = thickness_cells
        self.m = m
        self.R_target = R_target
        self.kappa_max = kappa_max
        self.alpha_max = alpha_max

        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, dt: float, xp=None, dtype=None) -> None:
        """Pre-compute CPML coefficients and allocate ψ arrays.

        Must be called once before the time-stepping loop.  ``xp`` is
        the solver's array backend (NumPy or CuPy); ``None`` falls back
        to the module-global ``get_xp()`` for standalone use.  ``dtype``
        is the field scalar precision (WP1b) — the ψ recursion state and
        the device-side b/c/ck coefficients follow it so the CPML
        correction ``β·ψ`` stays a same-dtype op (no float64 penalty on a
        float32 GPU run); ``None`` keeps float64.
        """
        if dtype is None:
            dtype = float
        grid = self.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        axis_name, side = self._FACE_AXIS[self.face]

        d_arr = getattr(grid, f"d{axis_name}")  # dx, dy, or dz
        N = len(d_arr)
        n_pml = min(self.thickness_cells, N)

        # Cell indices inside PML region
        if side == "min":
            self._pml_idx = list(range(n_pml))
        else:
            self._pml_idx = list(range(N - n_pml, N))

        d_phys = float(sum(d_arr[k] for k in self._pml_idx)) or 1.0
        sigma_max = -(self.m + 1) * C0 * EPS0 * np.log(self.R_target) / (2 * d_phys)

        # Build per-cell σ, κ, α arrays (sampled at cell centres)
        cumulative = 0.0
        sigma = np.zeros(n_pml)
        kappa = np.ones(n_pml)
        alpha = np.zeros(n_pml)
        for local_i, cell_idx in enumerate(self._pml_idx):
            half = d_arr[cell_idx] / 2.0
            if side == "min":
                rho = 1.0 - (cumulative + half) / d_phys
            else:
                rho = (cumulative + half) / d_phys
            cumulative += d_arr[cell_idx]
            sigma[local_i] = sigma_max * rho**self.m
            kappa[local_i] = 1.0 + (self.kappa_max - 1.0) * rho**self.m
            alpha[local_i] = self.alpha_max * (1.0 - rho)

        # CFS-PML coefficients (computed on CPU, kept as 1D for inspection)
        self._b = np.exp(-(sigma / kappa + alpha) * dt / EPS0)
        self._c = np.where(
            sigma > 0.0,
            sigma / (kappa * sigma + kappa**2 * alpha) * (1.0 - self._b),
            0.0,
        )
        self._ck = 1.0 - 1.0 / kappa
        self._sigma_arr = sigma
        b, c, ck = self._b, self._c, self._ck

        self._dt = dt
        self._axis = axis_name
        self._side = side
        self._n_pml = n_pml
        self._d_arr = d_arr
        self._k0 = self._pml_idx[0]

        # Transfer coefficients to active device and reshape for broadcasting
        if xp is None:
            xp = get_xp()
        if axis_name == "x":
            self._b_3d = xp.asarray(b[:, None, None], dtype=dtype)
            self._c_3d = xp.asarray(c[:, None, None], dtype=dtype)
            self._ck_3d = xp.asarray(ck[:, None, None], dtype=dtype)
        elif axis_name == "y":
            self._b_3d = xp.asarray(b[None, :, None], dtype=dtype)
            self._c_3d = xp.asarray(c[None, :, None], dtype=dtype)
            self._ck_3d = xp.asarray(ck[None, :, None], dtype=dtype)
        else:  # z
            self._b_3d = xp.asarray(b[None, None, :], dtype=dtype)
            self._c_3d = xp.asarray(c[None, None, :], dtype=dtype)
            self._ck_3d = xp.asarray(ck[None, None, :], dtype=dtype)

        # Allocate ψ arrays on the active device
        if axis_name == "x":
            self._psi_Ey = xp.zeros((n_pml, Ny, Nz + 1), dtype=dtype)
            self._psi_Ez = xp.zeros((n_pml, Ny + 1, Nz), dtype=dtype)
            self._psi_Hy = xp.zeros((n_pml, Ny + 1, Nz), dtype=dtype)
            self._psi_Hz = xp.zeros((n_pml, Ny, Nz + 1), dtype=dtype)
        elif axis_name == "y":
            self._psi_Ex = xp.zeros((Nx, n_pml, Nz + 1), dtype=dtype)
            self._psi_Ez = xp.zeros((Nx + 1, n_pml, Nz), dtype=dtype)
            self._psi_Hx = xp.zeros((Nx + 1, n_pml, Nz), dtype=dtype)
            self._psi_Hz = xp.zeros((Nx, n_pml, Nz + 1), dtype=dtype)
        else:  # z
            self._psi_Ex = xp.zeros((Nx, Ny + 1, n_pml), dtype=dtype)
            self._psi_Ey = xp.zeros((Nx + 1, Ny, n_pml), dtype=dtype)
            self._psi_Hx = xp.zeros((Nx + 1, Ny, n_pml), dtype=dtype)
            self._psi_Hy = xp.zeros((Nx, Ny + 1, n_pml), dtype=dtype)

        # PEC air masks — set via set_pec_mask(), None = no masking
        self._air_E1 = None
        self._air_E2 = None
        self._air_H1 = None
        self._air_H2 = None

        self._initialized = True

    def state_dict(self) -> dict:
        """Checkpoint the ψ auxiliary convolution fields.

        Serialises every ``_psi_*`` array — the exact subset depends on
        the boundary face axis.  All other CPML data are constant
        stretching coefficients, re-derived at construction.
        """
        return {name: getattr(self, name).copy() for name in vars(self) if name.startswith("_psi_")}

    def load_state_dict(self, sd: dict) -> None:
        """Restore ψ fields written by :meth:`state_dict` (in place)."""
        from magnelio._backend.array_api import copy_into  # noqa: PLC0415

        for name, arr in sd.items():
            copy_into(getattr(self, name), arr)

    def set_pec_mask(
        self,
        pec_mask_E: np.ndarray,
        Nx: int,
        Ny: int,
        Nz: int,
        material_id: np.ndarray | None = None,
        material_library: dict | None = None,
        xp=None,
    ) -> None:
        """Extract air-edge masks for the PML region from a PEC mask.

        Prevents ψ auxiliary fields from accumulating at PEC edges/faces,
        which would otherwise cause late-time instability when PEC material
        exists inside the PML region (e.g. waveguide walls at port faces).

        Parameters
        ----------
        pec_mask_E : np.ndarray
            Flat boolean PEC mask for all E-field edges.
        Nx, Ny, Nz : int
            Grid cell counts.
        material_id : np.ndarray, optional
            Cell material IDs, shape ``(Nx, Ny, Nz)``.  Used to derive
            H-field masks.
        material_library : dict, optional
            Maps material ID to Material objects.
        """
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)

        pec_Ex = pec_mask_E[:n_Ex].reshape(Nx, Ny + 1, Nz + 1)
        pec_Ey = pec_mask_E[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1)
        pec_Ez = pec_mask_E[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz)

        # Build PEC cell mask from material_id
        pec_cell = np.zeros((Nx, Ny, Nz), dtype=bool)
        if material_id is not None and material_library is not None:
            for mid, mat in material_library.items():
                if getattr(mat, "is_pec", False):
                    pec_cell |= material_id == mid

        # H-field PEC masks: Hx[i,j,k] is masked if the cell at (i,j,k)
        # is PEC (conservative: mask the side closer to PML interior).
        # Hx shape (Nx+1, Ny, Nz), Hy shape (Nx, Ny+1, Nz), Hz shape (Nx, Ny, Nz+1)
        # Use OR of neighbouring cells along H-face normal.
        pec_Hx = np.zeros((Nx + 1, Ny, Nz), dtype=bool)
        pec_Hy = np.zeros((Nx, Ny + 1, Nz), dtype=bool)
        pec_Hz = np.zeros((Nx, Ny, Nz + 1), dtype=bool)
        if pec_cell.any():
            pec_Hx[:-1] |= pec_cell
            pec_Hx[1:] |= pec_cell
            pec_Hy[:, :-1] |= pec_cell
            pec_Hy[:, 1:] |= pec_cell
            pec_Hz[:, :, :-1] |= pec_cell
            pec_Hz[:, :, 1:] |= pec_cell

        # Build air masks on CPU, then transfer to active device
        if xp is None:
            xp = get_xp()

        if self._axis == "z":
            air1 = np.ones(self._psi_Ex.shape)
            air2 = np.ones(self._psi_Ey.shape)
            airH1 = np.ones(self._psi_Hx.shape)
            airH2 = np.ones(self._psi_Hy.shape)
            for li, gi in enumerate(self._pml_idx):
                air1[:, :, li] = ~pec_Ex[:, :, gi]
                air2[:, :, li] = ~pec_Ey[:, :, gi]
                airH1[:, :, li] = ~pec_Hx[:, :, gi]
                airH2[:, :, li] = ~pec_Hy[:, :, gi]

        elif self._axis == "x":
            air1 = np.ones(self._psi_Ey.shape)
            air2 = np.ones(self._psi_Ez.shape)
            airH1 = np.ones(self._psi_Hy.shape)
            airH2 = np.ones(self._psi_Hz.shape)
            for li, gi in enumerate(self._pml_idx):
                air1[li, :, :] = ~pec_Ey[gi, :, :]
                air2[li, :, :] = ~pec_Ez[gi, :, :]
                airH1[li, :, :] = ~pec_Hy[gi, :, :]
                airH2[li, :, :] = ~pec_Hz[gi, :, :]

        elif self._axis == "y":
            air1 = np.ones(self._psi_Ex.shape)
            air2 = np.ones(self._psi_Ez.shape)
            airH1 = np.ones(self._psi_Hx.shape)
            airH2 = np.ones(self._psi_Hz.shape)
            for li, gi in enumerate(self._pml_idx):
                air1[:, li, :] = ~pec_Ex[:, gi, :]
                air2[:, li, :] = ~pec_Ez[:, gi, :]
                airH1[:, li, :] = ~pec_Hx[:, gi, :]
                airH2[:, li, :] = ~pec_Hz[:, gi, :]

        self._air_E1 = xp.asarray(air1)
        self._air_E2 = xp.asarray(air2)
        self._air_H1 = xp.asarray(airH1)
        self._air_H2 = xp.asarray(airH2)

    # ------------------------------------------------------------------
    # Update methods (called every time step)
    # ------------------------------------------------------------------

    def apply(self, fields) -> None:
        """Enforce PEC (E_tangential = 0) at the outer face of the PML.

        Every CPML is backed by a PEC wall at its outermost grid boundary.
        Without this, the tangential E-fields at the domain edge are not
        constrained and cause spurious reflections.
        """
        face = self.face
        if face == "xmin":
            fields.Ey[0, :, :] = 0.0
            fields.Ez[0, :, :] = 0.0
        elif face == "xmax":
            fields.Ey[-1, :, :] = 0.0
            fields.Ez[-1, :, :] = 0.0
        elif face == "ymin":
            fields.Ex[:, 0, :] = 0.0
            fields.Ez[:, 0, :] = 0.0
        elif face == "ymax":
            fields.Ex[:, -1, :] = 0.0
            fields.Ez[:, -1, :] = 0.0
        elif face == "zmin":
            fields.Ex[:, :, 0] = 0.0
            fields.Ey[:, :, 0] = 0.0
        elif face == "zmax":
            fields.Ex[:, :, -1] = 0.0
            fields.Ey[:, :, -1] = 0.0

    def update_E(self, fields, beta_E: np.ndarray) -> None:
        """Update ψ_E and apply additive correction to E-field (PML region only).

        ``beta_E`` is the flat array ``dt / M_eps_eff`` of length n_E
        (as produced by :class:`~magnelio.solver.fit_td.FITTimeDomainSolver`).
        The correction ``Δ E = β_E[edge] · ψ`` is applied in-place.

        All operations are vectorised over the PML slab to minimise
        kernel-launch overhead on GPU.
        """
        if not self._initialized:
            return

        grid = self.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        axis = self._axis
        k0 = self._k0
        n = self._n_pml
        b3, c3, ck3 = self._b_3d, self._c_3d, self._ck_3d

        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)

        bE_Ex = beta_E[:n_Ex].reshape(Nx, Ny + 1, Nz + 1)
        bE_Ey = beta_E[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1)
        bE_Ez = beta_E[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz)

        air1 = self._air_E1
        air2 = self._air_E2

        if axis == "z":
            # Backward H differences along z for all PML layers at once
            if self._side == "min":
                dHy = fields.Hy[:, :, :n].copy()
                dHy[:, :, 1:] -= fields.Hy[:, :, : n - 1]
                dHx = fields.Hx[:, :, :n].copy()
                dHx[:, :, 1:] -= fields.Hx[:, :, : n - 1]
            else:
                dHy = fields.Hy[:, :, k0 : k0 + n] - fields.Hy[:, :, k0 - 1 : k0 + n - 1]
                dHx = fields.Hx[:, :, k0 : k0 + n] - fields.Hx[:, :, k0 - 1 : k0 + n - 1]

            self._psi_Ex *= b3
            self._psi_Ex += c3 * dHy
            if air1 is not None:
                self._psi_Ex *= air1
            fields.Ex[:, :, k0 : k0 + n] += bE_Ex[:, :, k0 : k0 + n] * (ck3 * dHy + self._psi_Ex)

            self._psi_Ey *= b3
            self._psi_Ey += c3 * dHx
            if air2 is not None:
                self._psi_Ey *= air2
            fields.Ey[:, :, k0 : k0 + n] -= bE_Ey[:, :, k0 : k0 + n] * (ck3 * dHx + self._psi_Ey)

        elif axis == "x":
            if self._side == "min":
                dHz = fields.Hz[:n, :, :].copy()
                dHz[1:, :, :] -= fields.Hz[: n - 1, :, :]
                dHy = fields.Hy[:n, :, :].copy()
                dHy[1:, :, :] -= fields.Hy[: n - 1, :, :]
            else:
                dHz = fields.Hz[k0 : k0 + n, :, :] - fields.Hz[k0 - 1 : k0 + n - 1, :, :]
                dHy = fields.Hy[k0 : k0 + n, :, :] - fields.Hy[k0 - 1 : k0 + n - 1, :, :]

            self._psi_Ey *= b3
            self._psi_Ey += c3 * dHz
            if air1 is not None:
                self._psi_Ey *= air1
            fields.Ey[k0 : k0 + n, :, :] += bE_Ey[k0 : k0 + n, :, :] * (ck3 * dHz + self._psi_Ey)

            self._psi_Ez *= b3
            self._psi_Ez += c3 * dHy
            if air2 is not None:
                self._psi_Ez *= air2
            fields.Ez[k0 : k0 + n, :, :] -= bE_Ez[k0 : k0 + n, :, :] * (ck3 * dHy + self._psi_Ez)

        elif axis == "y":
            if self._side == "min":
                dHz = fields.Hz[:, :n, :].copy()
                dHz[:, 1:, :] -= fields.Hz[:, : n - 1, :]
                dHx = fields.Hx[:, :n, :].copy()
                dHx[:, 1:, :] -= fields.Hx[:, : n - 1, :]
            else:
                dHz = fields.Hz[:, k0 : k0 + n, :] - fields.Hz[:, k0 - 1 : k0 + n - 1, :]
                dHx = fields.Hx[:, k0 : k0 + n, :] - fields.Hx[:, k0 - 1 : k0 + n - 1, :]

            self._psi_Ex *= b3
            self._psi_Ex += c3 * dHz
            if air1 is not None:
                self._psi_Ex *= air1
            fields.Ex[:, k0 : k0 + n, :] -= bE_Ex[:, k0 : k0 + n, :] * (ck3 * dHz + self._psi_Ex)

            self._psi_Ez *= b3
            self._psi_Ez += c3 * dHx
            if air2 is not None:
                self._psi_Ez *= air2
            fields.Ez[:, k0 : k0 + n, :] += bE_Ez[:, k0 : k0 + n, :] * (ck3 * dHx + self._psi_Ez)

    def update_H(self, fields, beta_H: np.ndarray) -> None:
        """Update ψ_H and apply additive correction to H-field (PML region only).

        ``beta_H`` is the flat array ``dt / M_mu`` of length n_H
        (as produced by :class:`~magnelio.solver.fit_td.FITTimeDomainSolver`).

        Forward E differences rely on PEC being enforced at the outer face
        before this method is called (tangential E = 0 at PEC boundary).
        """
        if not self._initialized:
            return

        grid = self.grid
        Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
        axis = self._axis
        k0 = self._k0
        n = self._n_pml
        b3, c3, ck3 = self._b_3d, self._c_3d, self._ck_3d

        n_Hx = (Nx + 1) * Ny * Nz
        n_Hy = Nx * (Ny + 1) * Nz

        bH_Hx = beta_H[:n_Hx].reshape(Nx + 1, Ny, Nz)
        bH_Hy = beta_H[n_Hx : n_Hx + n_Hy].reshape(Nx, Ny + 1, Nz)
        bH_Hz = beta_H[n_Hx + n_Hy :].reshape(Nx, Ny, Nz + 1)

        airH1 = self._air_H1
        airH2 = self._air_H2

        if axis == "z":
            # Forward E differences (PEC: E at outer face already zeroed)
            dEy = fields.Ey[:, :, k0 + 1 : k0 + n + 1] - fields.Ey[:, :, k0 : k0 + n]
            dEx = fields.Ex[:, :, k0 + 1 : k0 + n + 1] - fields.Ex[:, :, k0 : k0 + n]

            self._psi_Hx *= b3
            self._psi_Hx += c3 * dEy
            if airH1 is not None:
                self._psi_Hx *= airH1
            fields.Hx[:, :, k0 : k0 + n] -= bH_Hx[:, :, k0 : k0 + n] * (ck3 * dEy + self._psi_Hx)

            self._psi_Hy *= b3
            self._psi_Hy += c3 * dEx
            if airH2 is not None:
                self._psi_Hy *= airH2
            fields.Hy[:, :, k0 : k0 + n] += bH_Hy[:, :, k0 : k0 + n] * (ck3 * dEx + self._psi_Hy)

        elif axis == "x":
            dEz = fields.Ez[k0 + 1 : k0 + n + 1, :, :] - fields.Ez[k0 : k0 + n, :, :]
            dEy = fields.Ey[k0 + 1 : k0 + n + 1, :, :] - fields.Ey[k0 : k0 + n, :, :]

            self._psi_Hy *= b3
            self._psi_Hy += c3 * dEz
            if airH1 is not None:
                self._psi_Hy *= airH1
            fields.Hy[k0 : k0 + n, :, :] -= bH_Hy[k0 : k0 + n, :, :] * (ck3 * dEz + self._psi_Hy)

            self._psi_Hz *= b3
            self._psi_Hz += c3 * dEy
            if airH2 is not None:
                self._psi_Hz *= airH2
            fields.Hz[k0 : k0 + n, :, :] += bH_Hz[k0 : k0 + n, :, :] * (ck3 * dEy + self._psi_Hz)

        elif axis == "y":
            dEz = fields.Ez[:, k0 + 1 : k0 + n + 1, :] - fields.Ez[:, k0 : k0 + n, :]
            dEx = fields.Ex[:, k0 + 1 : k0 + n + 1, :] - fields.Ex[:, k0 : k0 + n, :]

            self._psi_Hx *= b3
            self._psi_Hx += c3 * dEz
            if airH1 is not None:
                self._psi_Hx *= airH1
            fields.Hx[:, k0 : k0 + n, :] += bH_Hx[:, k0 : k0 + n, :] * (ck3 * dEz + self._psi_Hx)

            self._psi_Hz *= b3
            self._psi_Hz += c3 * dEx
            if airH2 is not None:
                self._psi_Hz *= airH2
            fields.Hz[:, k0 : k0 + n, :] -= bH_Hz[:, k0 : k0 + n, :] * (ck3 * dEx + self._psi_Hz)

    @property
    def sigma_per_cell(self) -> np.ndarray:
        """Conductivity σ [S/m] for each PML cell along the normal axis.

        The array has length ``thickness_cells`` and follows the grading
        profile from the interface (index 0) inward.  Requires
        :meth:`initialize` to have been called first.
        """
        if not self._initialized:
            raise RuntimeError("CPMLBoundary.initialize() must be called first.")
        return self._sigma_arr.copy()

    @property
    def pml_axis_indices(self) -> list[int]:
        """Global cell indices along the PML normal axis.

        Requires :meth:`initialize` to have been called first.
        """
        if not self._initialized:
            raise RuntimeError("CPMLBoundary.initialize() must be called first.")
        return list(self._pml_idx)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def __repr__(self) -> str:
        return (
            f"CPMLBoundary(face={self.face!r}, thickness_cells={self.thickness_cells}, "
            f"kappa_max={self.kappa_max}, alpha_max={self.alpha_max})"
        )
