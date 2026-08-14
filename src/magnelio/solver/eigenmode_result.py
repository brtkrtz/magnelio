"""Result container for 3D cavity eigenmode analysis.

Stores resonant frequencies and E/H field patterns as FieldState objects
(one per mode), together with the reference mesh and solver metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from magnelio._fields.field_arrays import FieldState


@dataclass
class EigenmodeResult:
    """Result of a 3D cavity eigenmode analysis.

    Parameters
    ----------
    frequencies : np.ndarray
        Resonant frequencies [Hz], shape ``(n_modes,)``, ascending.
    modes : list[FieldState]
        One FieldState per mode with E and H fields on the Yee grid.
        Field amplitudes are normalised so that ``e^T M_eps e = 1``.
    mesh : Mesh
        Reference mesh (grid, material library).
    solver_info : dict
        Metadata: solver backend, number of iterations, residuals, etc.
    """

    frequencies: np.ndarray
    modes: list[FieldState]
    mesh: object
    solver_info: dict = field(default_factory=dict)

    @property
    def n_modes(self) -> int:
        """Number of physical modes found."""
        return len(self.frequencies)

    def __repr__(self) -> str:
        n = self.n_modes
        if n == 0:
            return "EigenmodeResult(n_modes=0)"
        f_min = self.frequencies[0] / 1e9
        f_max = self.frequencies[-1] / 1e9
        return f"EigenmodeResult(n_modes={n}, f=[{f_min:.4f}, {f_max:.4f}] GHz)"

    def plot(
        self,
        mode: int = 0,
        component: str = "E",
        *,
        normal: str | None = None,
        position: float = 0.0,
        plot_type: str = "vector",
        ax=None,
        scale_mm: bool = True,
        cmap: str | None = None,
        geometry=None,
        flip: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        density: int = 20,
        normalize_arrows: bool = False,
        threshold: float = 0.02,
    ):
        """Plot a mode's field pattern on a slice plane through the cavity.

        The staggered Yee-grid components are interpolated onto cell
        centres and rendered on the plane selected by *normal* and
        *position*.  Mode fields are normalised eigenvectors, so the
        amplitudes are in arbitrary units; the spatial pattern is the
        physical content.

        Parameters
        ----------
        mode : int
            Mode index (ascending in frequency).
        component : str
            ``"E"`` or ``"H"`` for the vector plot / vector magnitude.
            ``"Ex"``, ``"Hy"``, … for a single component (scalar only).
        normal : {"x", "y", "z"}
            Normal axis of the slice plane.
        position : float
            Slice-plane position along *normal* [m]; snapped to the
            nearest cell-centre plane.
        plot_type : str
            ``"vector"``, ``"color"``, or ``"contour"``.
        ax : matplotlib.axes.Axes, optional
        scale_mm : bool
            Axis coordinates in mm instead of m.
        cmap : str or None
            Colourmap (None = auto-select).
        geometry : GeometryModel, optional
            Geometry for a cross-section overlay of the slice plane.
        flip : bool
            Swap horizontal and vertical axes.
        vmin, vmax : float, optional
            Colour limits (scalar) or arrow clipping (vector).
        density : int
            Target arrows per axis (vector mode).
        normalize_arrows : bool
            Unit-length arrows, colour = magnitude (vector mode).
        threshold : float
            Suppress arrows below this fraction of peak (vector mode).

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        from magnelio.monitors.base import (  # noqa: PLC0415
            _AXES,
            _interp_to_cell_centres,
            _resolve_component,
            resolve_plane_view,
            resolve_region,
        )
        from magnelio.post.plot_field import (  # noqa: PLC0415
            CrossSectionOverlay,
            plot_field_scalar,
            plot_field_vector,
        )

        if not 0 <= mode < self.n_modes:
            raise IndexError(f"mode must be in [0, {self.n_modes}); got {mode}")

        grid = self.mesh.grid
        region = resolve_region(None, grid)
        pv = resolve_plane_view(region, normal, position)
        (i0, c0), (i1, c1) = pv.free

        is_magnitude = component in ("E", "H", "|E|", "|H|")
        is_field_group = component in ("E", "H")
        field_group = component.strip("|")[:1]
        valid = {f"{g}{a}" for g in ("E", "H") for a in _AXES}
        if not is_magnitude and component not in valid:
            raise KeyError(f"component must be E/H (or |E|/|H|) or one of {sorted(valid)}.")
        comps = [f"{field_group}{a}" for a in _AXES] if is_magnitude else [component]

        # Interpolate only the requested slab, not the full volume
        slabs = [region.ix, region.iy, region.iz]
        if pv.slice_index is not None:
            base = slabs[pv.normal_idx].start
            slabs[pv.normal_idx] = slice(base + pv.slice_index, base + pv.slice_index + 1)
        data = _interp_to_cell_centres(self.modes[mode], comps, *slabs, grid)
        data = {c: np.squeeze(a, axis=pv.normal_idx) for c, a in data.items()}

        overlay = None
        if geometry is not None:
            overlay = CrossSectionOverlay(
                geometry=geometry,
                normal=_AXES[pv.normal_idx],
                position=pv.normal_pos,
            )

        pos_txt = (
            f"{_AXES[pv.normal_idx]}={pv.normal_pos * 1e3:.3g} mm"
            if scale_mm
            else f"{_AXES[pv.normal_idx]}={pv.normal_pos:.3g} m"
        )
        f_ghz = self.frequencies[mode] / 1e9

        if plot_type == "vector":
            if not is_field_group:
                raise ValueError("Vector plots need component='E' or 'H'.")
            title = f"Mode {mode} ({f_ghz:.3f} GHz) — {field_group}-field, {pos_txt}"
            return plot_field_vector(
                c0,
                c1,
                data[f"{field_group}{_AXES[i0]}"],
                data[f"{field_group}{_AXES[i1]}"],
                w=data[f"{field_group}{_AXES[pv.normal_idx]}"],
                xlabel=_AXES[i0],
                ylabel=_AXES[i1],
                wlabel=_AXES[pv.normal_idx],
                title=title,
                clabel=f"|{field_group}| (arb. units)",
                ax=ax,
                scale_mm=scale_mm,
                cmap=cmap or "viridis",
                density=density,
                normalize_arrows=normalize_arrows,
                vmax=vmax,
                threshold=threshold,
                flip=flip,
                geometry=overlay,
            )

        # Scalar plot (color / contour)
        vals = _resolve_component(data, field_group if is_magnitude else component)
        if is_magnitude:
            label = f"|{field_group}|"
            effective_cmap = cmap or "viridis"
            sym = False
            if vmin is None:
                vmin = 0.0
        else:
            label = component
            effective_cmap = cmap or "RdBu_r"
            sym = True
        title = f"Mode {mode} ({f_ghz:.3f} GHz) — {label}, {pos_txt}"

        return plot_field_scalar(
            c0,
            c1,
            vals,
            xlabel=_AXES[i0],
            ylabel=_AXES[i1],
            title=title,
            clabel=f"{label} (arb. units)",
            ax=ax,
            scale_mm=scale_mm,
            cmap=effective_cmap,
            vmin=vmin,
            vmax=vmax,
            symmetric=sym,
            plot_type=plot_type,
            flip=flip,
            geometry=overlay,
        )

    @staticmethod
    def _modes_from_flat(
        E_modes: np.ndarray,
        H_modes: np.ndarray,
        Nx: int,
        Ny: int,
        Nz: int,
    ) -> list[FieldState]:
        """Build FieldState list from flat E/H mode matrices.

        Parameters
        ----------
        E_modes : np.ndarray, shape (n_E, n_modes)
            E-field eigenvectors in [Ex|Ey|Ez] ordering.
        H_modes : np.ndarray, shape (n_H, n_modes)
            H-field eigenvectors in [Hx|Hy|Hz] ordering.
        Nx, Ny, Nz : int
            Grid cell counts.

        Returns
        -------
        list[FieldState]
        """
        n_Ex = Nx * (Ny + 1) * (Nz + 1)
        n_Ey = (Nx + 1) * Ny * (Nz + 1)
        n_Hx = (Nx + 1) * Ny * Nz
        n_Hy = Nx * (Ny + 1) * Nz

        n_modes = E_modes.shape[1]
        modes = []
        for m in range(n_modes):
            e = E_modes[:, m]
            h = H_modes[:, m]
            fs = FieldState(
                Ex=e[:n_Ex].reshape(Nx, Ny + 1, Nz + 1),
                Ey=e[n_Ex : n_Ex + n_Ey].reshape(Nx + 1, Ny, Nz + 1),
                Ez=e[n_Ex + n_Ey :].reshape(Nx + 1, Ny + 1, Nz),
                Hx=h[:n_Hx].reshape(Nx + 1, Ny, Nz),
                Hy=h[n_Hx : n_Hx + n_Hy].reshape(Nx, Ny + 1, Nz),
                Hz=h[n_Hx + n_Hy :].reshape(Nx, Ny, Nz + 1),
            )
            modes.append(fs)
        return modes
