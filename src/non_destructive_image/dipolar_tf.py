"""Static Thomas--Fermi ansatz for a polarised dipolar condensate.

The density is restricted to the exact inverted-parabola Thomas--Fermi family
described by O'Dell, Giovanazzi and Eberlein.  The three radii are obtained by
minimising the trap, contact and dipolar mean-field energy of a general
triaxial ellipsoid.  Kinetic energy, thermal back-action and beyond-mean-field
terms are deliberately outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.special import elliprd

from .atomic_model import build_thomas_fermi_state


@dataclass(frozen=True)
class DipolarThomasFermiState:
    """Derived quantities for a stationary triaxial dipolar-TF ansatz."""

    atom_number: float
    trap_angular_frequencies_rad_s: NDArray[np.floating]
    geometric_mean_frequency_rad_s: float
    harmonic_oscillator_length_m: float
    axis_harmonic_oscillator_lengths_m: NDArray[np.floating]
    dipole_axis: int
    epsilon_dd: float
    chemical_potential_j: float
    chemical_potential_temperature_k: float
    chemical_potential_to_trap_quantum_ratio: NDArray[np.floating]
    peak_density_m3: float
    radii_m: NDArray[np.floating]
    peak_column_density_m2: NDArray[np.floating]
    atom_number_check: float
    demagnetisation_factors: NDArray[np.floating]
    trap_energy_j: float
    contact_energy_j: float
    dipolar_energy_j: float
    total_energy_j: float
    virial_relative_residual: float
    stationarity_max_abs: float
    optimiser_iterations: int

    @property
    def trap_angular_frequencies(self) -> NDArray[np.floating]:
        """Return the angular-frequency vector through the common TF interface."""

        return self.trap_angular_frequencies_rad_s

    @property
    def geometric_mean_frequency(self) -> float:
        """Return the geometric-mean angular frequency."""

        return self.geometric_mean_frequency_rad_s

    @property
    def harmonic_oscillator_length(self) -> float:
        """Return the geometric-mean harmonic-oscillator length."""

        return self.harmonic_oscillator_length_m

    @property
    def chemical_potential(self) -> float:
        """Return the chemical potential in joules."""

        return self.chemical_potential_j

    @property
    def chemical_potential_temperature(self) -> float:
        """Return the chemical-potential temperature in kelvin."""

        return self.chemical_potential_temperature_k

    @property
    def peak_density(self) -> float:
        """Return the central three-dimensional density in inverse cubic metres."""

        return self.peak_density_m3

    @property
    def radii(self) -> NDArray[np.floating]:
        """Return the Thomas--Fermi radii in metres."""

        return self.radii_m

    @property
    def column_density(self) -> NDArray[np.floating]:
        """Return the three peak line-of-sight column densities in inverse square metres."""

        return self.peak_column_density_m2


def _positive_finite_scalar(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_radii(radii_m: ArrayLike) -> NDArray[np.floating]:
    radii = np.asarray(radii_m, dtype=float)
    if radii.shape != (3,):
        raise ValueError("radii_m must have shape (3,)")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii_m must contain positive finite values")
    return radii


def ellipsoid_demagnetisation_factors(
    radii_m: ArrayLike,
) -> NDArray[np.floating]:
    """Return the three geometric demagnetisation factors of an ellipsoid.

    Carlson's symmetric elliptic integral evaluates

    ``N_i = Rx*Ry*Rz/2 * integral_0^inf ds / ((Ri^2+s)*Delta(s))``.

    The factors are scale invariant, non-negative and sum to one.  For dipoles
    polarised along axis ``i``, the dipolar/contact energy ratio is
    ``epsilon_dd * (3*N_i - 1)`` for the parabolic TF density.
    """

    radii = _positive_radii(radii_m)
    scale = float(np.prod(radii) ** (1.0 / 3.0))
    axes = radii / scale
    prefactor = float(np.prod(axes) / 3.0)
    factors = prefactor * np.asarray(
        [
            elliprd(axes[1] ** 2, axes[2] ** 2, axes[0] ** 2),
            elliprd(axes[0] ** 2, axes[2] ** 2, axes[1] ** 2),
            elliprd(axes[0] ** 2, axes[1] ** 2, axes[2] ** 2),
        ],
        dtype=float,
    )
    if (
        not np.all(np.isfinite(factors))
        or np.any(factors < 0.0)
        or not np.isclose(float(np.sum(factors)), 1.0, rtol=0.0, atol=2e-12)
    ):
        raise RuntimeError("ellipsoid demagnetisation factors are invalid")
    return factors


def _energies_j(
    radii_m: NDArray[np.floating],
    *,
    atom_number: float,
    atomic_mass_kg: float,
    trap_angular_frequencies_rad_s: NDArray[np.floating],
    contact_coupling_j_m3: float,
    epsilon_dd: float,
    dipole_axis: int,
) -> tuple[float, float, float, NDArray[np.floating]]:
    factors = ellipsoid_demagnetisation_factors(radii_m)
    trap = float(
        atom_number
        * atomic_mass_kg
        * np.sum((trap_angular_frequencies_rad_s * radii_m) ** 2)
        / 14.0
    )
    contact = float(
        15.0
        * contact_coupling_j_m3
        * atom_number**2
        / (28.0 * np.pi * np.prod(radii_m))
    )
    dipolar = float(contact * epsilon_dd * (3.0 * factors[dipole_axis] - 1.0))
    return trap, contact, dipolar, factors


def build_dipolar_thomas_fermi_state(
    atom_number: float,
    scattering_length_m: float,
    dipolar_length_m: float,
    trap_frequencies_hz: ArrayLike,
    dipole_axis: int,
    atomic_mass_kg: float,
    hbar_j_s: float,
    boltzmann_constant_j_k: float,
) -> DipolarThomasFermiState:
    """Return the stationary zero-temperature dipolar-TF ellipsoid.

    The contract is limited to repulsive contact interactions and
    ``0 <= epsilon_dd < 1``.  The returned state is a minimum within the
    inverted-parabola scaling family; it is not a kinetic-energy-inclusive GPE
    validation.  All physical inputs remain explicit; the function does not
    read an active configuration or add thermal atoms.
    """

    number = _positive_finite_scalar(atom_number, "atom_number")
    scattering_length = _positive_finite_scalar(
        scattering_length_m, "scattering_length_m"
    )
    dipolar_length = float(dipolar_length_m)
    if not np.isfinite(dipolar_length) or dipolar_length < 0.0:
        raise ValueError("dipolar_length_m must be non-negative and finite")
    epsilon_dd = dipolar_length / scattering_length
    if epsilon_dd >= 1.0:
        raise ValueError("stable dipolar-TF contract requires epsilon_dd < 1")
    if isinstance(dipole_axis, bool) or not isinstance(dipole_axis, (int, np.integer)):
        raise ValueError("dipole_axis must be one of 0, 1 or 2")
    axis = int(dipole_axis)
    if axis not in (0, 1, 2):
        raise ValueError("dipole_axis must be one of 0, 1 or 2")
    mass = _positive_finite_scalar(atomic_mass_kg, "atomic_mass_kg")
    hbar = _positive_finite_scalar(hbar_j_s, "hbar_j_s")
    boltzmann = _positive_finite_scalar(
        boltzmann_constant_j_k, "boltzmann_constant_j_k"
    )
    trap_frequencies = np.asarray(trap_frequencies_hz, dtype=float)
    if trap_frequencies.shape != (3,):
        raise ValueError("trap_frequencies_hz must have shape (3,)")
    if not np.all(np.isfinite(trap_frequencies)) or np.any(trap_frequencies <= 0.0):
        raise ValueError("trap_frequencies_hz must contain positive finite values")

    contact_state = build_thomas_fermi_state(
        number,
        scattering_length,
        trap_frequencies,
        mass,
        hbar,
        boltzmann,
    )
    angular_frequencies = np.asarray(
        contact_state.trap_angular_frequencies, dtype=float
    )
    contact_coupling = 4.0 * np.pi * hbar**2 * scattering_length / mass
    contact_radii = np.asarray(contact_state.radii, dtype=float)

    def energies(
        radii: NDArray[np.floating],
    ) -> tuple[float, float, float, NDArray[np.floating]]:
        return _energies_j(
            radii,
            atom_number=number,
            atomic_mass_kg=mass,
            trap_angular_frequencies_rad_s=angular_frequencies,
            contact_coupling_j_m3=contact_coupling,
            epsilon_dd=epsilon_dd,
            dipole_axis=axis,
        )

    initial_energies = energies(contact_radii)
    energy_scale = float(sum(initial_energies[:3]))
    if not np.isfinite(energy_scale) or energy_scale <= 0.0:
        raise RuntimeError("initial dipolar-TF energy scale is invalid")

    if epsilon_dd == 0.0:
        radii = contact_radii.copy()
        optimiser_iterations = 0
        stationarity_max_abs = 0.0
    else:

        def objective(log_radius_ratio: NDArray[np.floating]) -> float:
            if (
                log_radius_ratio.shape != (3,)
                or not np.all(np.isfinite(log_radius_ratio))
                or np.any(np.abs(log_radius_ratio) > 20.0)
            ):
                return float("inf")
            trial_radii = contact_radii * np.exp(log_radius_ratio)
            return float(sum(energies(trial_radii)[:3]) / energy_scale)

        result = minimize(
            objective,
            np.zeros(3, dtype=float),
            method="Nelder-Mead",
            options={
                "xatol": 2e-11,
                "fatol": 2e-13,
                "maxiter": 3000,
            },
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"dipolar-TF radius optimisation failed: {result.message}")
        radii = contact_radii * np.exp(np.asarray(result.x, dtype=float))
        optimiser_iterations = int(result.nit)
        derivative_step = 1e-5
        basis = np.eye(3, dtype=float)
        gradient = np.asarray(
            [
                (
                    objective(result.x + derivative_step * basis[index])
                    - objective(result.x - derivative_step * basis[index])
                )
                / (2.0 * derivative_step)
                for index in range(3)
            ],
            dtype=float,
        )
        stationarity_max_abs = float(np.max(np.abs(gradient)))
        if not np.isfinite(stationarity_max_abs) or stationarity_max_abs > 2e-6:
            raise RuntimeError("dipolar-TF radius solution failed stationarity check")

    trap_energy, contact_energy, dipolar_energy, factors = energies(radii)
    total_energy = float(trap_energy + contact_energy + dipolar_energy)
    interaction_energy = float(contact_energy + dipolar_energy)
    virial_relative_residual = float(
        (2.0 * trap_energy - 3.0 * interaction_energy) / (2.0 * trap_energy)
    )
    if not np.isfinite(virial_relative_residual) or abs(virial_relative_residual) > 2e-6:
        raise RuntimeError("dipolar-TF radius solution failed the virial check")

    peak_density = float(15.0 * number / (8.0 * np.pi * np.prod(radii)))
    peak_column_density = (4.0 / 3.0) * peak_density * radii
    atom_number_check = float(
        (8.0 * np.pi / 15.0) * peak_density * np.prod(radii)
    )
    chemical_potential = float(7.0 * total_energy / (5.0 * number))
    if not (
        np.isfinite(peak_density)
        and peak_density > 0.0
        and np.all(np.isfinite(peak_column_density))
        and np.all(peak_column_density > 0.0)
        and np.isclose(atom_number_check, number, rtol=2e-13, atol=0.0)
        and np.isfinite(chemical_potential)
        and chemical_potential > 0.0
    ):
        raise RuntimeError("derived dipolar-TF state is invalid")

    return DipolarThomasFermiState(
        atom_number=number,
        trap_angular_frequencies_rad_s=angular_frequencies,
        geometric_mean_frequency_rad_s=float(
            contact_state.geometric_mean_frequency
        ),
        harmonic_oscillator_length_m=float(
            contact_state.harmonic_oscillator_length
        ),
        axis_harmonic_oscillator_lengths_m=np.sqrt(
            hbar / (mass * angular_frequencies)
        ),
        dipole_axis=axis,
        epsilon_dd=float(epsilon_dd),
        chemical_potential_j=chemical_potential,
        chemical_potential_temperature_k=float(chemical_potential / boltzmann),
        chemical_potential_to_trap_quantum_ratio=np.asarray(
            chemical_potential / (hbar * angular_frequencies), dtype=float
        ),
        peak_density_m3=peak_density,
        radii_m=np.asarray(radii, dtype=float),
        peak_column_density_m2=np.asarray(peak_column_density, dtype=float),
        atom_number_check=atom_number_check,
        demagnetisation_factors=np.asarray(factors, dtype=float),
        trap_energy_j=float(trap_energy),
        contact_energy_j=float(contact_energy),
        dipolar_energy_j=float(dipolar_energy),
        total_energy_j=total_energy,
        virial_relative_residual=virial_relative_residual,
        stationarity_max_abs=stationarity_max_abs,
        optimiser_iterations=optimiser_iterations,
    )


def scale_dipolar_thomas_fermi_state(
    reference: DipolarThomasFermiState,
    atom_number: float,
) -> DipolarThomasFermiState:
    """Scale a fixed-geometry dipolar-TF solution to a new condensate number.

    For fixed trap, contact coupling, dipolar ratio and polarisation axis, the
    inverted-parabola TF shape is independent of atom number. Radii scale as
    ``N**(1/5)``, chemical potential and peak density as ``N**(2/5)``, peak
    column density as ``N**(3/5)``, and every energy term as ``N**(7/5)``.
    """

    number = _positive_finite_scalar(atom_number, "atom_number")
    ratio = number / reference.atom_number
    radius_scale = ratio ** (1.0 / 5.0)
    chemical_scale = ratio ** (2.0 / 5.0)
    column_scale = ratio ** (3.0 / 5.0)
    energy_scale = ratio ** (7.0 / 5.0)
    return replace(
        reference,
        atom_number=number,
        chemical_potential_j=reference.chemical_potential_j * chemical_scale,
        chemical_potential_temperature_k=(
            reference.chemical_potential_temperature_k * chemical_scale
        ),
        chemical_potential_to_trap_quantum_ratio=(
            reference.chemical_potential_to_trap_quantum_ratio * chemical_scale
        ),
        peak_density_m3=reference.peak_density_m3 * chemical_scale,
        radii_m=reference.radii_m * radius_scale,
        peak_column_density_m2=(
            reference.peak_column_density_m2 * column_scale
        ),
        atom_number_check=number,
        trap_energy_j=reference.trap_energy_j * energy_scale,
        contact_energy_j=reference.contact_energy_j * energy_scale,
        dipolar_energy_j=reference.dipolar_energy_j * energy_scale,
        total_energy_j=reference.total_energy_j * energy_scale,
    )
