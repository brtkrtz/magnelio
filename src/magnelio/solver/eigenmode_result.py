"""Result container for 3D cavity eigenmode analysis.

Stores resonant frequencies and E/H field patterns as FieldState objects
(one per mode), together with the reference mesh and solver metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from magnelio._fields.field_arrays import FieldState


def _real_snapshot(data: dict) -> dict:
    """Real part of complex slice data at the instant of maximum energy.

    An eigenvector's global phase is arbitrary.  Rotating by half the
    argument of ``sum(a*a)`` aligns the field so its real part carries
    the maximum of ``sum(|Re a|^2)`` over all instants, which is the
    natural snapshot of a travelling Bloch mode.  Real data passes
    through untouched.
    """
    if not any(np.iscomplexobj(a) for a in data.values()):
        return data
    moment = sum(np.sum(np.asarray(a, dtype=complex) ** 2) for a in data.values())
    rotation = np.exp(-0.5j * np.angle(moment)) if moment != 0 else 1.0
    return {c: np.real(np.asarray(a) * rotation) for c, a in data.items()}


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

    def _repr_html_(self) -> str:
        """A notebook cell shows the modes as a table: index and frequency."""
        from magnelio._repr import html_table  # noqa: PLC0415

        rows = [[i, f"{f / 1e9:.6g}"] for i, f in enumerate(self.frequencies)]
        return html_table(
            ["mode", "f [GHz]"],
            rows,
            caption=f"EigenmodeResult ({self.n_modes} modes)",
            align="rr",
        )

    def field(self, mode: int = 0):
        """The mode's field pattern as a :class:`~magnelio.fields.FieldState`.

        The eigenvector is normalised to ``e^T M_eps e = 1``, i.e. to a
        peak stored electric energy of 0.5 J; the physical amplitude
        follows from that, the spatial pattern is the physical content.
        A Bloch mode with a phase advance other than 0 or 180 degrees is
        complex.

        Parameters
        ----------
        mode : int
            Mode index (ascending in frequency).
        """
        from magnelio.fields import FieldState as _PublicFieldState  # noqa: PLC0415

        if not 0 <= mode < self.n_modes:
            raise IndexError(f"mode must be in [0, {self.n_modes}); got {mode}")
        return _PublicFieldState._from_raw(self.mesh.grid, self.modes[mode])

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
        *position* (:meth:`magnelio.fields.FieldState.plot`).  Mode
        fields are normalised eigenvectors, so the amplitudes are in
        arbitrary units; the spatial pattern is the physical content.
        A complex mode (Bloch phase advance other than 0 or 180
        degrees) is drawn as the real snapshot at the instant of
        maximum field energy on the slice — the global phase of an
        eigenvector is arbitrary, and this choice makes the picture
        independent of it.

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
            resolve_plane_view,
            resolve_region,
        )

        field = self.field(mode)
        pv = resolve_plane_view(resolve_region(None, self.mesh.grid), normal, position)
        pos_txt = (
            f"{_AXES[pv.normal_idx]}={pv.normal_pos * 1e3:.3g} mm"
            if scale_mm
            else f"{_AXES[pv.normal_idx]}={pv.normal_pos:.3g} m"
        )
        f_ghz = self.frequencies[mode] / 1e9
        group = component.strip("|")[:1]
        label = f"{group}-field" if component in ("E", "H") and plot_type == "vector" else None
        if label is None:
            label = f"|{group}|" if component in ("E", "H", "|E|", "|H|") else component
        return field.plot(
            component,
            normal=normal,
            position=position,
            plot_type=plot_type,
            ax=ax,
            scale_mm=scale_mm,
            cmap=cmap,
            geometry=geometry,
            flip=flip,
            vmin=vmin,
            vmax=vmax,
            density=density,
            normalize_arrows=normalize_arrows,
            threshold=threshold,
            title=f"Mode {mode} ({f_ghz:.3f} GHz) — {label}, {pos_txt}",
            unit="arb. units",
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
