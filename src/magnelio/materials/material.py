"""
Material dataclass for electromagnetic properties.

Supports isotropic and diagonal anisotropic materials (εx,εy,εz; μx,μy,μz; σ; σ*).
See spec.md for field layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from magnelio.materials.dispersion import DispersionModel
from magnelio.materials.roughness import SurfaceRoughness


@dataclass
class Material:
    """Electromagnetic material with diagonal anisotropic properties.

    All values are in SI units. Permittivity and permeability are relative
    (dimensionless, relative to ε₀ and μ₀ respectively).

    Wherever the public API expects a material (shape ``material=``,
    ``GeometryModel(background=)``, ...), the parameter-free built-ins
    may be named by string instead: ``"air"``, ``"vacuum"`` or ``"pec"``
    (case-insensitive) resolve to the canonical instances of
    :meth:`air`, :meth:`vacuum` and :meth:`pec`.

    Parameters
    ----------
    name : str
        Human-readable identifier.
    epsilon : tuple of float
        Relative permittivity ``(εrx, εry, εrz)``.  Default
        ``(1, 1, 1)`` = vacuum.
    mu : tuple of float
        Relative permeability ``(μrx, μry, μrz)``.  Default
        ``(1, 1, 1)`` = vacuum.
    sigma : tuple of float
        Electric conductivity ``(σx, σy, σz)`` in S/m.  Default
        ``(0, 0, 0)``.
    sigma_m : tuple of float
        Magnetic loss ``(σ*x, σ*y, σ*z)`` in Ω/m.  Default
        ``(0, 0, 0)``.
    is_pec : bool
        If True, the field solver treats this material as a Perfect
        Electric Conductor (edge masking; σ never enters M_sigma).
        A finite σ > 0 alongside ``is_pec=True`` marks a *lossy
        metal*: identical field solution, but surface-loss models
        consume σ (and μ) for the skin-effect surface resistance.
    dispersion : DispersionModel, optional
        Pole-residue model for frequency-dependent permittivity.
        ``epsilon`` must then equal the model's ``eps_inf`` on all
        three axes (it drives the mass matrix and the CFL limit); use
        :meth:`dispersive` to get this right automatically.
        Incompatible with ``is_pec``.
    dispersion_mu : DispersionModel, optional
        Pole-residue model for frequency-dependent *permeability* —
        the H-side mirror of ``dispersion``, realised by the same
        trapezoidal ADE on the H-faces.  ``DispersionModel`` is reused
        verbatim: it is a relative-units pole-residue form, so its
        ``eps_inf`` field carries μ_inf here, and its passivity
        condition (Im ≥ 0 on the band) is the μ'' ≥ 0 one.  ``mu``
        must then equal that ``eps_inf`` on all three axes; use
        :meth:`dispersive_mu`.  Incompatible with ``is_pec``.
    roughness : SurfaceRoughness, optional
        Surface-roughness model raising the skin-effect surface
        resistance by K(f) in loss models.  Like σ it is consumed only
        by those models, never by the field solve — so it is
        meaningful on a *lossy metal* only (use :meth:`lossy_metal`).
    color : tuple of float, optional
        RGB colour tuple in [0, 1] for visualisation.  ``None`` lets
        the plotting code assign a colour automatically.
    alpha : float
        Opacity in [0, 1] for 3-D rendering (default opaque).
    visible : bool
        Whether to render this material in 3-D visualisations.
    """

    name: str
    epsilon: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mu: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sigma: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sigma_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_pec: bool = False
    dispersion: Optional[DispersionModel] = None
    dispersion_mu: Optional[DispersionModel] = None
    roughness: Optional[SurfaceRoughness] = None
    # Visualisation — not used in EM computations, excluded from equality checks
    color: tuple[float, float, float] | None = field(default=None, compare=False)
    alpha: float = field(default=1.0, compare=False)
    visible: bool = field(default=True, compare=False)

    def __post_init__(self) -> None:
        for attr in ("epsilon", "mu", "sigma", "sigma_m"):
            val = getattr(self, attr)
            if len(val) != 3:
                raise ValueError(f"Material.{attr} must be a 3-tuple, got: {val!r}")
        if self.dispersion is not None:
            if self.is_pec:
                raise ValueError(
                    f"Material {self.name!r}: dispersion and is_pec are mutually exclusive"
                )
            if any(e != self.dispersion.eps_inf for e in self.epsilon):
                raise ValueError(
                    f"Material {self.name!r}: epsilon={self.epsilon} must "
                    f"equal the dispersion model's eps_inf="
                    f"{self.dispersion.eps_inf} on all axes — it drives the "
                    f"mass matrix and the CFL limit (use Material.dispersive)"
                )
        if self.dispersion_mu is not None:
            if self.is_pec:
                raise ValueError(
                    f"Material {self.name!r}: dispersion_mu and is_pec are mutually exclusive"
                )
            if any(m != self.dispersion_mu.eps_inf for m in self.mu):
                raise ValueError(
                    f"Material {self.name!r}: mu={self.mu} must equal the "
                    f"mu-dispersion model's high-frequency limit "
                    f"(eps_inf={self.dispersion_mu.eps_inf} — the model "
                    f"class is relative-units, so that field carries "
                    f"mu_inf) on all axes: it drives the mass matrix and "
                    f"the CFL limit (use Material.dispersive_mu)"
                )
        if self.roughness is not None and not self.is_lossy_metal:
            raise ValueError(
                f"Material {self.name!r}: roughness needs a conductivity to "
                f"act on — it multiplies the skin-effect surface resistance "
                f"R_s(sigma), so it is only meaningful on a lossy metal "
                f"(use Material.lossy_metal(..., roughness=...))"
            )

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def air(cls) -> "Material":
        """Free-space / air material (ε=μ=1, σ=0)."""
        return cls(name="air")

    @classmethod
    def vacuum(cls) -> "Material":
        """Alias for air()."""
        return cls(name="vacuum")

    @classmethod
    def pec(cls) -> "Material":
        """Perfect Electric Conductor."""
        return cls(name="PEC", is_pec=True)

    @classmethod
    def lossy_metal(
        cls,
        name: str,
        sigma: float,
        mu: float = 1.0,
        roughness: Optional[SurfaceRoughness] = None,
    ) -> "Material":
        """Good conductor: PEC for the field solve, finite σ for loss models.

        The field solver treats the material exactly like PEC (edge
        masking — results are identical to :meth:`pec`).  The finite
        conductivity and permeability are consumed only by surface-loss
        models via the skin-effect surface resistance
        ``R_s(ω) = sqrt(ω·μ₀·μ_r / (2σ))``.

        Parameters
        ----------
        name : str
            Human-readable identifier.
        sigma : float
            Electric conductivity in S/m.  Must be finite and > 0.
        mu : float, optional
            Relative permeability (enters R_s), default 1.
        roughness : SurfaceRoughness, optional
            Surface-roughness model.  Multiplies R_s by its
            frequency-dependent factor K(f) >= 1 wherever a loss model
            evaluates this metal's walls; ``None`` (default) is a
            perfectly smooth conductor.
        """
        if not (0.0 < sigma < float("inf")):
            raise ValueError(f"lossy_metal requires finite sigma > 0, got: {sigma!r}")
        return cls(
            name=name,
            mu=(mu, mu, mu),
            sigma=(sigma, sigma, sigma),
            is_pec=True,
            roughness=roughness,
        )

    @classmethod
    def dispersive(
        cls,
        name: str,
        model: DispersionModel,
        mu: float = 1.0,
        sigma: float = 0.0,
    ) -> "Material":
        """Material with a pole-residue dispersive permittivity.

        ``epsilon`` is set to the model's ``eps_inf`` (the high-frequency
        limit that drives the mass matrix and the CFL condition); the
        pole set is realised in the solver by the trapezoidal ADE.  An
        additional static conductivity may be given — it runs through
        the standard semi-implicit σ channel alongside the pole
        currents.

        Parameters
        ----------
        name : str
            Human-readable identifier.
        model : DispersionModel
            Pole-residue permittivity model (passivity-checked at its
            construction).
        mu : float, optional
            Relative permeability, default 1.
        sigma : float, optional
            Static electric conductivity in S/m, default 0.
        """
        return cls(
            name=name,
            epsilon=(model.eps_inf,) * 3,
            mu=(mu, mu, mu),
            sigma=(sigma, sigma, sigma),
            dispersion=model,
        )

    @classmethod
    def dispersive_mu(
        cls,
        name: str,
        model: DispersionModel,
        epsilon: float = 1.0,
        sigma: float = 0.0,
        sigma_m: float = 0.0,
    ) -> "Material":
        """Material with a pole-residue dispersive permeability.

        The H-side mirror of :meth:`dispersive`.  ``mu`` is set to the
        model's high-frequency limit (which drives the mass matrix and
        the CFL condition); the pole set is realised in the solver by
        the same trapezoidal ADE, on the H-faces.  An
        additional magnetic loss σ* may be given — it runs through the
        standard semi-implicit σ* channel alongside the pole currents.

        ``DispersionModel`` is reused verbatim for μ(ω): it is a
        relative-units pole-residue form, so its ``eps_inf`` field
        carries μ_inf and its constructors (``debye``, ``lorentz``,
        ``drude``, …) describe the magnetic analogues.

        Both permittivity and permeability can be dispersive at once —
        pass ``dispersion=`` and ``dispersion_mu=`` to the constructor
        directly (``epsilon`` and ``mu`` must then equal the respective
        high-frequency limits).

        Parameters
        ----------
        name : str
            Human-readable identifier.
        model : DispersionModel
            Pole-residue permeability model (passivity-checked at its
            construction — for μ that check reads μ'' >= 0).
        epsilon : float, optional
            Relative permittivity, default 1.
        sigma : float, optional
            Static electric conductivity in S/m, default 0.
        sigma_m : float, optional
            Magnetic loss σ* in Ω/m, default 0.
        """
        return cls(
            name=name,
            epsilon=(epsilon,) * 3,
            mu=(model.eps_inf,) * 3,
            sigma=(sigma,) * 3,
            sigma_m=(sigma_m,) * 3,
            dispersion_mu=model,
        )

    @classmethod
    def from_isotropic(
        cls,
        name: str,
        epsilon: float = 1.0,
        mu: float = 1.0,
        sigma: float = 0.0,
        sigma_m: float = 0.0,
    ) -> "Material":
        """Create an isotropic material from scalar values."""
        return cls(
            name=name,
            epsilon=(epsilon, epsilon, epsilon),
            mu=(mu, mu, mu),
            sigma=(sigma, sigma, sigma),
            sigma_m=(sigma_m, sigma_m, sigma_m),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_lossless(self) -> bool:
        """True if σ = 0, σ* = 0, not PEC, and not dispersive."""
        return (
            not self.is_pec
            and self.dispersion is None
            and self.dispersion_mu is None
            and all(s == 0.0 for s in self.sigma)
            and all(s == 0.0 for s in self.sigma_m)
        )

    @property
    def is_lossy_metal(self) -> bool:
        """True for a PEC-classified material carrying a finite σ > 0.

        Legacy stores may carry ``sigma = inf`` on plain PEC (written by
        versions that forced it); the finiteness check excludes those.
        """
        return (
            self.is_pec
            and all(s < float("inf") for s in self.sigma)
            and any(s > 0.0 for s in self.sigma)
        )

    @property
    def is_isotropic(self) -> bool:
        """True if ε, μ, σ, σ* are all isotropic (x=y=z)."""
        return (
            len(set(self.epsilon)) == 1
            and len(set(self.mu)) == 1
            and len(set(self.sigma)) == 1
            and len(set(self.sigma_m)) == 1
        )

    def __repr__(self) -> str:
        pec_flag = ", is_pec=True" if self.is_pec else ""
        disp_flag = (
            f", dispersive({len(self.dispersion.poles)} poles)"
            if self.dispersion is not None
            else ""
        )
        disp_mu_flag = (
            f", dispersive_mu({len(self.dispersion_mu.poles)} poles)"
            if self.dispersion_mu is not None
            else ""
        )
        return (
            f"Material(name={self.name!r}, "
            f"epsilon={self.epsilon}, mu={self.mu}"
            f"{pec_flag}{disp_flag}{disp_mu_flag})"
        )


# ----------------------------------------------------------------------
# Built-in material names (DD-185)
# ----------------------------------------------------------------------

_BUILTIN_FACTORIES = {
    "air": Material.air,
    "vacuum": Material.vacuum,
    "pec": Material.pec,
}
_builtin_instances: dict[str, Material] = {}


def resolve_material(value, what: str = "material"):
    """Return *value* as a :class:`Material`, resolving built-in names.

    The public API accepts the parameter-free built-in materials by name
    wherever a material is expected: ``"air"``, ``"vacuum"`` and
    ``"pec"`` (case-insensitive) resolve to canonical instances of
    :meth:`Material.air`, :meth:`Material.vacuum` and
    :meth:`Material.pec` — the same shared instance on every use, so the
    mesher's identity-based material bookkeeping sees one material, not
    one per shape.  ``None`` and :class:`Material` instances pass
    through unchanged; anything else raises at the call site.

    Parameters
    ----------
    value : Material or str or None
        The user-supplied material argument.
    what : str
        Name of the argument for error messages, e.g. ``"Brick.material"``.

    Returns
    -------
    Material or None
    """
    if value is None or isinstance(value, Material):
        return value
    if isinstance(value, str):
        key = value.lower()
        factory = _BUILTIN_FACTORIES.get(key)
        if factory is None:
            names = ", ".join(f'"{n}"' for n in _BUILTIN_FACTORIES)
            raise ValueError(
                f"{what} = {value!r} is not a built-in material name; "
                f"recognised names: {names}. Parameterised materials are "
                f"built explicitly, e.g. Material.from_isotropic(...) or "
                f"Material.lossy_metal(...)."
            )
        instance = _builtin_instances.get(key)
        if instance is None:
            instance = _builtin_instances[key] = factory()
        return instance
    raise TypeError(
        f'{what} takes a Material or a built-in material name ("air", '
        f'"vacuum", "pec"), not a {type(value).__name__}. Build custom '
        f"materials with Material.from_isotropic(...) / "
        f"Material.lossy_metal(...)."
    )
