"""Finite-temperature equilibrium in a triaxial dipolar harmonic trap.

The functions in this module keep all physical inputs explicit.  They implement
the critical-point part of the Oxford Supplemental Eqs. S7 and S10--S11.  The
non-saturation population and energy use the same semiclassical Hartree--Fock
distribution.  Together with the dipolar Thomas--Fermi core energy they define
the fixed-number, recoil-energy endpoint closure used by the target-geometry
multiframe model.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import gamma, zeta

from .dipolar_tf import (
    DipolarThomasFermiState,
    build_dipolar_thomas_fermi_state,
    ellipsoid_demagnetisation_factors,
)


@dataclass(frozen=True)
class HarmonicCriticalPoint:
    """Critical-population terms for one harmonic trap and field direction."""

    temperature_k: float
    trap_frequencies_hz: NDArray[np.floating]
    dipole_axis: int
    thermal_wavelength_m: float
    ideal_critical_atoms: float
    dipolar_geometry_factors: NDArray[np.floating]
    mean_field_temperature_shift_fraction: float
    interaction_critical_population_shift_fraction: float
    finite_number_critical_population_shift_fraction: float
    harmonic_critical_population_fraction: float
    harmonic_critical_atoms: float


@dataclass(frozen=True)
class SemiclassicalThermalPopulation:
    """Thermal population and energy produced by one dipolar-TF condensate."""

    temperature_k: float
    condensate_atoms: float
    ideal_critical_atoms: float
    critical_atoms: float
    mean_field_excess_fraction_of_ideal: float
    mean_field_excess_atoms: float
    thermal_atoms: float
    ideal_harmonic_energy_coefficient: float
    ideal_critical_energy_j: float
    critical_baseline_energy_j: float
    mean_field_excess_energy_j: float
    thermal_energy_j: float
    thomas_fermi_state: DipolarThomasFermiState
    minimum_reduced_excitation_energy: float


@dataclass(frozen=True)
class FiniteTemperatureEquilibrium:
    """Partially condensed equilibrium at fixed total atom number and temperature."""

    total_atoms: float
    temperature_k: float
    condensate_atoms: float
    thermal_atoms: float
    number_residual_atoms: float
    condensate_energy_j: float
    thermal_energy_j: float
    total_energy_j: float
    ideal_harmonic_sensitivity_energy_j: float
    critical_point: HarmonicCriticalPoint
    thermal_population: SemiclassicalThermalPopulation


@dataclass(frozen=True)
class FiniteTemperatureEnergyUpdate:
    """One fixed-number equilibrium update after a declared energy increment."""

    initial_equilibrium: FiniteTemperatureEquilibrium
    final_equilibrium: FiniteTemperatureEquilibrium
    energy_model: str
    deposited_energy_j: float
    target_energy_j: float
    energy_residual_j: float
    relative_energy_residual: float
    temperature_bracket_k: tuple[float, float]


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


class _UnsupportedEquilibriumDomain(ValueError):
    """Internal marker for a physically unsupported equilibrium endpoint."""


def _trap_frequencies(value: ArrayLike) -> NDArray[np.floating]:
    frequencies = np.asarray(value, dtype=float)
    if frequencies.shape != (3,):
        raise ValueError("trap_frequencies_hz must have shape (3,)")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("trap_frequencies_hz must be positive and finite")
    return frequencies


def _axis(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("dipole_axis must be 0, 1 or 2")
    result = int(value)
    if result not in (0, 1, 2):
        raise ValueError("dipole_axis must be 0, 1 or 2")
    return result


@lru_cache(maxsize=None)
def _legendre_nodes_weights(
    order: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    nodes, weights = leggauss(order)
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)


def triaxial_dipolar_geometry_factors(
    trap_frequencies_hz: ArrayLike,
) -> NDArray[np.floating]:
    """Return the three axis-aligned factors in Supplemental Eq. S10.

    The Carlson-integral demagnetisation representation is algebraically
    equivalent to the incomplete-elliptic-integral function in Eq. S11.  The
    three factors sum to zero, as required by the traceless dipolar interaction.
    """

    frequencies = _trap_frequencies(trap_frequencies_hz)
    factors = ellipsoid_demagnetisation_factors(1.0 / frequencies)
    geometry = 0.5 * (1.0 - 3.0 * factors)
    if (
        not np.all(np.isfinite(geometry))
        or not np.isclose(float(np.sum(geometry)), 0.0, rtol=0.0, atol=2e-12)
    ):
        raise RuntimeError("triaxial dipolar geometry factors are invalid")
    return np.asarray(geometry, dtype=float)


def ideal_harmonic_critical_atoms(
    temperature_k: float,
    trap_frequencies_hz: ArrayLike,
    *,
    boltzmann_constant_j_k: float,
    hbar_j_s: float,
) -> float:
    """Return the semiclassical ideal-gas critical population."""

    temperature = _positive_finite(temperature_k, "temperature_k")
    boltzmann = _positive_finite(
        boltzmann_constant_j_k, "boltzmann_constant_j_k"
    )
    hbar = _positive_finite(hbar_j_s, "hbar_j_s")
    angular = 2.0 * np.pi * _trap_frequencies(trap_frequencies_hz)
    omega_bar = float(np.prod(angular) ** (1.0 / 3.0))
    return float(zeta(3.0) * (boltzmann * temperature / (hbar * omega_bar)) ** 3)


def thermal_de_broglie_wavelength_m(
    temperature_k: float,
    *,
    atomic_mass_kg: float,
    boltzmann_constant_j_k: float,
    hbar_j_s: float,
) -> float:
    """Return ``sqrt(2*pi*hbar**2/(m*kB*T))``."""

    temperature = _positive_finite(temperature_k, "temperature_k")
    mass = _positive_finite(atomic_mass_kg, "atomic_mass_kg")
    boltzmann = _positive_finite(
        boltzmann_constant_j_k, "boltzmann_constant_j_k"
    )
    hbar = _positive_finite(hbar_j_s, "hbar_j_s")
    return float(np.sqrt(2.0 * np.pi * hbar**2 / (mass * boltzmann * temperature)))


def finite_number_critical_population_shift_fraction(
    ideal_critical_atoms: float,
    trap_frequencies_hz: ArrayLike,
) -> float:
    """Return the positive finite-number correction in Supplemental Eq. S7."""

    critical = _positive_finite(ideal_critical_atoms, "ideal_critical_atoms")
    angular = 2.0 * np.pi * _trap_frequencies(trap_frequencies_hz)
    omega_bar = float(np.prod(angular) ** (1.0 / 3.0))
    omega_mean = float(np.mean(angular))
    return float(2.18 * (omega_mean / omega_bar) * critical ** (-1.0 / 3.0))


def harmonic_dipolar_critical_point(
    *,
    temperature_k: float,
    trap_frequencies_hz: ArrayLike,
    dipole_axis: int,
    scattering_length_m: float,
    dipolar_length_m: float,
    atomic_mass_kg: float,
    boltzmann_constant_j_k: float,
    hbar_j_s: float,
    include_finite_number_correction: bool = True,
) -> HarmonicCriticalPoint:
    """Return the first-order harmonic critical point for one orientation.

    Oxford-specific finite-depth and anharmonic corrections are deliberately
    absent.  Interaction shifts use ``Delta Nc/Nc0 = -3 Delta Tc/Tc0``.
    """

    frequencies = _trap_frequencies(trap_frequencies_hz)
    axis = _axis(dipole_axis)
    scattering = _positive_finite(scattering_length_m, "scattering_length_m")
    dipolar = float(dipolar_length_m)
    if not np.isfinite(dipolar) or dipolar < 0.0:
        raise ValueError("dipolar_length_m must be non-negative and finite")
    critical = ideal_harmonic_critical_atoms(
        temperature_k,
        frequencies,
        boltzmann_constant_j_k=boltzmann_constant_j_k,
        hbar_j_s=hbar_j_s,
    )
    wavelength = thermal_de_broglie_wavelength_m(
        temperature_k,
        atomic_mass_kg=atomic_mass_kg,
        boltzmann_constant_j_k=boltzmann_constant_j_k,
        hbar_j_s=hbar_j_s,
    )
    geometry = triaxial_dipolar_geometry_factors(frequencies)
    temperature_shift = float(
        -3.426 * scattering / wavelength
        + 3.426 * dipolar / wavelength * geometry[axis]
    )
    interaction_population_shift = float(-3.0 * temperature_shift)
    finite_number_shift = (
        finite_number_critical_population_shift_fraction(critical, frequencies)
        if include_finite_number_correction
        else 0.0
    )
    critical_fraction = float(
        1.0 + interaction_population_shift + finite_number_shift
    )
    if not np.isfinite(critical_fraction) or critical_fraction <= 0.0:
        raise RuntimeError("harmonic critical population is not positive")
    return HarmonicCriticalPoint(
        temperature_k=float(temperature_k),
        trap_frequencies_hz=frequencies,
        dipole_axis=axis,
        thermal_wavelength_m=wavelength,
        ideal_critical_atoms=critical,
        dipolar_geometry_factors=geometry,
        mean_field_temperature_shift_fraction=temperature_shift,
        interaction_critical_population_shift_fraction=(
            interaction_population_shift
        ),
        finite_number_critical_population_shift_fraction=finite_number_shift,
        harmonic_critical_population_fraction=critical_fraction,
        harmonic_critical_atoms=float(critical * critical_fraction),
    )


def _bose_function(
    reduced_energy: ArrayLike,
    order: float,
) -> NDArray[np.floating]:
    """Return ``Li_order(exp(-x))`` for the supported half-integer orders.

    The expansion about ``x=0`` resolves the square-root cusp without the
    spatial-grid bias produced by sampling the condensate boundary.  A direct
    exponential series is faster and well conditioned away from that cusp.
    """

    if order not in (1.5, 2.5):
        raise ValueError("order must be 3/2 or 5/2")
    energy = np.asarray(reduced_energy, dtype=float)
    if not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("reduced_energy must be non-negative and finite")

    result = np.empty_like(energy)
    zero = energy == 0.0
    result[zero] = zeta(order)

    small = (energy < 0.8) & ~zero
    if np.any(small):
        values = energy[small]
        expansion = gamma(1.0 - order) * values ** (order - 1.0)
        term = np.ones_like(values)
        for series_order in range(18):
            if series_order:
                term *= -values / series_order
            expansion += zeta(order - series_order) * term
        result[small] = expansion

    large = ~(zero | small)
    if np.any(large):
        values = energy[large]
        series = np.zeros_like(values)
        for index in range(1, 200):
            series += np.exp(-index * values) / index**order
        result[large] = series

    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise RuntimeError("Bose function evaluation failed")
    return result


def bose_function_three_halves(
    reduced_energy: ArrayLike,
) -> NDArray[np.floating]:
    """Return ``Li_(3/2)(exp(-x))`` for finite ``x >= 0``."""

    return _bose_function(reduced_energy, 1.5)


def bose_function_five_halves(
    reduced_energy: ArrayLike,
) -> NDArray[np.floating]:
    """Return ``Li_(5/2)(exp(-x))`` for finite ``x >= 0``."""

    return _bose_function(reduced_energy, 2.5)


def _outside_ellipsoid_tau(
    points_m: NDArray[np.floating],
    radii_m: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return the confocal coordinate of points outside one ellipsoid."""

    squared_points = points_m**2
    squared_radii = radii_m**2
    lower = np.zeros(points_m.shape[0], dtype=float)
    upper = np.sum(squared_points, axis=1)
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        residual = (
            np.sum(
                squared_points / (squared_radii + midpoint[:, np.newaxis]),
                axis=1,
            )
            - 1.0
        )
        lower = np.where(residual > 0.0, midpoint, lower)
        upper = np.where(residual > 0.0, upper, midpoint)
    return 0.5 * (lower + upper)


def _outside_quasi_electrostatic_second_derivative_m5(
    points_m: NDArray[np.floating],
    *,
    radii_m: NDArray[np.floating],
    atom_number: float,
    dipole_axis: int,
    quadrature_order: int,
) -> NDArray[np.floating]:
    """Return ``d_i d_i phi`` outside a parabolic TF ellipsoid.

    The normalised Thomas--Fermi density is
    ``15*N/(8*pi*Rx*Ry*Rz) * (1-sum(x_i**2/R_i**2))``.  Its corresponding
    quasi-electrostatic potential has prefactor ``15*N/(64*pi)``.  The lower
    confocal limit contributes no boundary term because the integrand and its
    first spatial derivative vanish there.
    """

    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_m must have shape (n, 3)")
    if points.shape[0] == 0:
        return np.empty(0, dtype=float)
    radii = _positive_radii_for_thermal(radii_m)
    axis = _axis(dipole_axis)
    if quadrature_order < 16:
        raise ValueError("quadrature_order must be at least 16")

    ellipsoid_coordinate = np.sum((points / radii) ** 2, axis=1)
    if np.any(ellipsoid_coordinate < 1.0 - 2e-12):
        raise ValueError("quasi-electrostatic exterior points entered the ellipsoid")

    tau = _outside_ellipsoid_tau(points, radii)
    nodes, weights = _legendre_nodes_weights(quadrature_order)
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    squared_radii = radii**2
    scale = np.maximum(np.max(squared_radii), tau)[:, np.newaxis]
    sigma = tau[:, np.newaxis] + scale * unit_nodes / (1.0 - unit_nodes)
    jacobian = scale / (1.0 - unit_nodes) ** 2
    denominators = squared_radii[np.newaxis, np.newaxis, :] + sigma[:, :, np.newaxis]
    reduced_radius = np.sum(
        points[:, np.newaxis, :] ** 2 / denominators,
        axis=2,
    )
    delta = np.sqrt(np.prod(denominators, axis=2))
    second_derivative = (
        8.0
        * points[:, np.newaxis, axis] ** 2
        / denominators[:, :, axis] ** 2
        - 4.0 * (1.0 - reduced_radius) / denominators[:, :, axis]
    )
    integral = np.sum(
        unit_weights
        * second_derivative
        * jacobian
        / delta,
        axis=1,
    )
    return np.asarray(15.0 * atom_number * integral / (64.0 * np.pi), dtype=float)


def _positive_radii_for_thermal(value: ArrayLike) -> NDArray[np.floating]:
    radii = np.asarray(value, dtype=float)
    if radii.shape != (3,):
        raise ValueError("radii_m must have shape (3,)")
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii_m must be positive and finite")
    return radii


def semiclassical_dipolar_thermal_population(
    *,
    temperature_k: float,
    condensate_atoms: float,
    critical_atoms: float,
    trap_frequencies_hz: ArrayLike,
    dipole_axis: int,
    scattering_length_m: float,
    dipolar_length_m: float,
    atomic_mass_kg: float,
    boltzmann_constant_j_k: float,
    hbar_j_s: float,
    angular_order: int = 12,
    azimuthal_order: int = 24,
    radial_order: int = 14,
    potential_order: int = 48,
    radial_limit: float = 6.0,
) -> SemiclassicalThermalPopulation:
    """Return the target-trap thermal population at fixed ``T`` and ``N0``.

    Momentum is integrated analytically.  The remaining spatial integral uses
    dimensionless harmonic coordinates, with the radial quadrature split at
    the Thomas--Fermi boundary in every direction.  This resolves the Bose
    cusp and avoids the finite-grid shell error that otherwise biases the
    non-saturation response.

    ``critical_atoms`` supplies the independently calculated transition-point
    population.  The condensate-dependent mean-field excess is then added to
    that intercept, matching the separation made in the Oxford analysis.
    """

    temperature = _positive_finite(temperature_k, "temperature_k")
    number = _positive_finite(condensate_atoms, "condensate_atoms")
    critical = _positive_finite(critical_atoms, "critical_atoms")
    frequencies = _trap_frequencies(trap_frequencies_hz)
    axis = _axis(dipole_axis)
    scattering = _positive_finite(scattering_length_m, "scattering_length_m")
    dipolar = float(dipolar_length_m)
    if not np.isfinite(dipolar) or dipolar < 0.0 or dipolar >= scattering:
        raise ValueError("dipolar_length_m must satisfy 0 <= a_dd < a_s")
    mass = _positive_finite(atomic_mass_kg, "atomic_mass_kg")
    boltzmann = _positive_finite(
        boltzmann_constant_j_k, "boltzmann_constant_j_k"
    )
    hbar = _positive_finite(hbar_j_s, "hbar_j_s")
    if angular_order < 8 or azimuthal_order < 16 or radial_order < 10:
        raise ValueError("spatial quadrature orders are too small")
    if not np.isfinite(radial_limit) or radial_limit < 5.0:
        raise ValueError("radial_limit must be finite and at least 5")

    state = build_dipolar_thomas_fermi_state(
        number,
        scattering,
        dipolar,
        frequencies,
        axis,
        mass,
        hbar,
        boltzmann,
    )
    radii = np.asarray(state.radii_m, dtype=float)
    angular_frequencies = 2.0 * np.pi * frequencies
    thermal_lengths = np.sqrt(2.0 * boltzmann * temperature / mass) / angular_frequencies
    contact_coupling = 4.0 * np.pi * hbar**2 * scattering / mass
    dipolar_coupling = 12.0 * np.pi * hbar**2 * dipolar / mass
    epsilon_dd = dipolar / scattering
    ideal_critical = ideal_harmonic_critical_atoms(
        temperature,
        frequencies,
        boltzmann_constant_j_k=boltzmann,
        hbar_j_s=hbar,
    )

    polar_nodes, polar_weights = _legendre_nodes_weights(angular_order)
    azimuths = (
        np.arange(azimuthal_order, dtype=float) + 0.5
    ) * 2.0 * np.pi / azimuthal_order
    radial_nodes, radial_weights = _legendre_nodes_weights(radial_order)
    number_integral = 0.0
    energy_integral = 0.0
    minimum_energy = float("inf")
    chemical_potential_reduced = state.chemical_potential_j / (
        boltzmann * temperature
    )

    for polar_node, polar_weight in zip(
        polar_nodes, polar_weights, strict=True
    ):
        transverse = np.sqrt(1.0 - polar_node**2)
        directions = np.column_stack(
            (
                transverse * np.cos(azimuths),
                transverse * np.sin(azimuths),
                np.full(azimuthal_order, polar_node),
            )
        )
        boundary = 1.0 / np.sqrt(
            np.sum((thermal_lengths * directions / radii) ** 2, axis=1)
        )

        for direction, boundary_radius in zip(
            directions, boundary, strict=True
        ):
            boundaries = sorted(
                {
                    0.0,
                    float(boundary_radius),
                    max(float(boundary_radius), 0.5),
                    1.5,
                    3.0,
                    float(radial_limit),
                }
            )
            radial_number_integral = 0.0
            radial_energy_integral = 0.0
            for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
                if upper - lower < 1e-13:
                    continue
                radius = (
                    0.5 * (lower + upper)
                    + 0.5 * (upper - lower) * radial_nodes
                )
                weights = 0.5 * (upper - lower) * radial_weights
                points = (
                    radius[:, np.newaxis]
                    * direction[np.newaxis, :]
                    * thermal_lengths[np.newaxis, :]
                )
                inside = radius < boundary_radius
                reduced_excitation = np.empty_like(radius)

                if np.any(inside):
                    reduced_radius = np.sum(
                        (points[inside] / radii) ** 2,
                        axis=1,
                    )
                    density = (
                        15.0
                        * number
                        * (1.0 - reduced_radius)
                        / (8.0 * np.pi * np.prod(radii))
                    )
                    reduced_excitation[inside] = (
                        contact_coupling
                        * (1.0 - epsilon_dd)
                        * density
                        / (boltzmann * temperature)
                    )

                if np.any(~inside):
                    derivative = _outside_quasi_electrostatic_second_derivative_m5(
                        points[~inside],
                        radii_m=radii,
                        atom_number=number,
                        dipole_axis=axis,
                        quadrature_order=potential_order,
                    )
                    reduced_excitation[~inside] = (
                        boltzmann * temperature * radius[~inside] ** 2
                        - dipolar_coupling * derivative
                        - state.chemical_potential_j
                    ) / (boltzmann * temperature)

                minimum_energy = min(
                    minimum_energy, float(np.min(reduced_excitation))
                )
                if np.min(reduced_excitation) < -2e-6:
                    raise RuntimeError(
                        "Hartree-Fock excitation energy became negative"
                    )
                reduced_excitation = np.maximum(reduced_excitation, 0.0)
                actual_three_halves = bose_function_three_halves(
                    reduced_excitation
                )
                ideal_three_halves = bose_function_three_halves(radius**2)
                actual_five_halves = bose_function_five_halves(
                    reduced_excitation
                )
                ideal_five_halves = bose_function_five_halves(radius**2)
                radial_number_integral += float(
                    np.sum(
                        weights
                        * radius**2
                        * (actual_three_halves - ideal_three_halves)
                    )
                )
                radial_energy_integral += float(
                    np.sum(
                        weights
                        * radius**2
                        * (
                            1.5
                            * (actual_five_halves - ideal_five_halves)
                            + (
                                reduced_excitation
                                + chemical_potential_reduced
                            )
                            * actual_three_halves
                            - radius**2 * ideal_three_halves
                        )
                    )
                )
            angular_weight = (
                float(polar_weight)
                * 2.0
                * np.pi
                / azimuthal_order
            )
            number_integral += angular_weight * radial_number_integral
            energy_integral += angular_weight * radial_energy_integral

    phase_space_normalisation = float(np.pi**1.5 * zeta(3.0))
    excess_fraction = float(number_integral / phase_space_normalisation)
    excess_atoms = float(ideal_critical * excess_fraction)
    thermal_atoms = float(critical + excess_atoms)
    ideal_energy_coefficient = float(3.0 * zeta(4.0) / zeta(3.0))
    ideal_critical_energy = float(
        ideal_energy_coefficient * ideal_critical * boltzmann * temperature
    )
    critical_baseline_energy = float(
        ideal_energy_coefficient * critical * boltzmann * temperature
    )
    excess_energy = float(
        ideal_critical
        * boltzmann
        * temperature
        * energy_integral
        / phase_space_normalisation
    )
    thermal_energy = float(critical_baseline_energy + excess_energy)
    if not np.all(
        np.isfinite(
            (
                excess_fraction,
                excess_atoms,
                thermal_atoms,
                ideal_critical_energy,
                critical_baseline_energy,
                excess_energy,
                thermal_energy,
                minimum_energy,
            )
        )
    ) or thermal_atoms <= 0.0 or thermal_energy <= 0.0:
        raise RuntimeError("semiclassical thermal population is invalid")

    return SemiclassicalThermalPopulation(
        temperature_k=temperature,
        condensate_atoms=number,
        ideal_critical_atoms=ideal_critical,
        critical_atoms=critical,
        mean_field_excess_fraction_of_ideal=excess_fraction,
        mean_field_excess_atoms=excess_atoms,
        thermal_atoms=thermal_atoms,
        ideal_harmonic_energy_coefficient=ideal_energy_coefficient,
        ideal_critical_energy_j=ideal_critical_energy,
        critical_baseline_energy_j=critical_baseline_energy,
        mean_field_excess_energy_j=excess_energy,
        thermal_energy_j=thermal_energy,
        thomas_fermi_state=state,
        minimum_reduced_excitation_energy=minimum_energy,
    )


def solve_finite_temperature_equilibrium(
    *,
    total_atoms: float,
    temperature_k: float,
    trap_frequencies_hz: ArrayLike,
    dipole_axis: int,
    scattering_length_m: float,
    dipolar_length_m: float,
    atomic_mass_kg: float,
    boltzmann_constant_j_k: float,
    hbar_j_s: float,
    angular_order: int = 12,
    azimuthal_order: int = 24,
    radial_order: int = 14,
    potential_order: int = 48,
) -> FiniteTemperatureEquilibrium:
    """Solve ``N_total = N0 + Nth(T, N0)`` in the target harmonic trap.

    This routine is deliberately limited to a partially condensed equilibrium.
    It does not evolve a pulse sequence or supply the energy closure required
    for later multiframe calculations.
    """

    total = _positive_finite(total_atoms, "total_atoms")
    critical_point = harmonic_dipolar_critical_point(
        temperature_k=temperature_k,
        trap_frequencies_hz=trap_frequencies_hz,
        dipole_axis=dipole_axis,
        scattering_length_m=scattering_length_m,
        dipolar_length_m=dipolar_length_m,
        atomic_mass_kg=atomic_mass_kg,
        boltzmann_constant_j_k=boltzmann_constant_j_k,
        hbar_j_s=hbar_j_s,
    )
    if total <= critical_point.harmonic_critical_atoms:
        raise ValueError(
            "total_atoms does not define a partially condensed harmonic state"
        )

    cache: dict[float, SemiclassicalThermalPopulation] = {}

    def population(condensate_atoms: float) -> SemiclassicalThermalPopulation:
        key = float(condensate_atoms)
        if key not in cache:
            cache[key] = semiclassical_dipolar_thermal_population(
                temperature_k=temperature_k,
                condensate_atoms=key,
                critical_atoms=critical_point.harmonic_critical_atoms,
                trap_frequencies_hz=trap_frequencies_hz,
                dipole_axis=dipole_axis,
                scattering_length_m=scattering_length_m,
                dipolar_length_m=dipolar_length_m,
                atomic_mass_kg=atomic_mass_kg,
                boltzmann_constant_j_k=boltzmann_constant_j_k,
                hbar_j_s=hbar_j_s,
                angular_order=angular_order,
                azimuthal_order=azimuthal_order,
                radial_order=radial_order,
                potential_order=potential_order,
            )
        return cache[key]

    def residual(condensate_atoms: float) -> float:
        return float(
            condensate_atoms
            + population(condensate_atoms).thermal_atoms
            - total
        )

    lower = max(1e-6, total * 1e-10)
    upper = total - critical_point.harmonic_critical_atoms
    lower_residual = residual(lower)
    upper_residual = residual(upper)
    if lower_residual >= 0.0 or upper_residual <= 0.0:
        raise RuntimeError("finite-temperature population root is not bracketed")
    condensate = float(
        brentq(
            residual,
            lower,
            upper,
            xtol=0.02,
            rtol=1e-10,
        )
    )
    thermal = population(condensate)
    number_residual = float(condensate + thermal.thermal_atoms - total)
    if abs(number_residual) > 0.05:
        raise RuntimeError("finite-temperature equilibrium failed number closure")
    condensate_energy = float(thermal.thomas_fermi_state.total_energy_j)
    total_energy = float(condensate_energy + thermal.thermal_energy_j)
    ideal_harmonic_sensitivity_energy = float(
        condensate_energy
        + thermal.ideal_harmonic_energy_coefficient
        * thermal.thermal_atoms
        * boltzmann_constant_j_k
        * float(temperature_k)
    )
    if not np.all(
        np.isfinite(
            (
                condensate_energy,
                thermal.thermal_energy_j,
                total_energy,
                ideal_harmonic_sensitivity_energy,
            )
        )
    ) or min(
        condensate_energy,
        thermal.thermal_energy_j,
        total_energy,
        ideal_harmonic_sensitivity_energy,
    ) <= 0.0:
        raise RuntimeError("finite-temperature equilibrium energy is invalid")
    return FiniteTemperatureEquilibrium(
        total_atoms=total,
        temperature_k=float(temperature_k),
        condensate_atoms=condensate,
        thermal_atoms=thermal.thermal_atoms,
        number_residual_atoms=number_residual,
        condensate_energy_j=condensate_energy,
        thermal_energy_j=thermal.thermal_energy_j,
        total_energy_j=total_energy,
        ideal_harmonic_sensitivity_energy_j=(
            ideal_harmonic_sensitivity_energy
        ),
        critical_point=critical_point,
        thermal_population=thermal,
    )


def recoil_deposited_energy_j(
    *,
    total_atoms: float,
    scattered_photons_per_atom: float,
    single_photon_recoil_energy_j: float,
) -> float:
    """Return the absorption--emission recoil energy deposited in one pulse.

    The broad probe addresses the complete trapped population.  Each scattered
    photon contributes two recoil energies on average: one from absorption and
    one from spontaneous emission.  This helper does not model atom loss or a
    density-dependent retention fraction.
    """

    total = _positive_finite(total_atoms, "total_atoms")
    scattered = _nonnegative_finite(
        scattered_photons_per_atom, "scattered_photons_per_atom"
    )
    recoil = _positive_finite(
        single_photon_recoil_energy_j, "single_photon_recoil_energy_j"
    )
    return float(2.0 * total * scattered * recoil)


def solve_finite_temperature_energy_update(
    *,
    initial_equilibrium: FiniteTemperatureEquilibrium,
    deposited_energy_j: float,
    minimum_temperature_k: float,
    maximum_temperature_k: float,
    trap_frequencies_hz: ArrayLike,
    dipole_axis: int,
    scattering_length_m: float,
    dipolar_length_m: float,
    atomic_mass_kg: float,
    boltzmann_constant_j_k: float,
    hbar_j_s: float,
    energy_model: str = "semiclassical_primary",
    minimum_tf_chemical_potential_ratio: float = 1.0,
    temperature_step_fraction: float = 0.01,
    angular_order: int = 12,
    azimuthal_order: int = 24,
    radial_order: int = 14,
    potential_order: int = 48,
) -> FiniteTemperatureEnergyUpdate:
    """Return the next fixed-number equilibrium after an energy increment.

    The temperature is the only outer root variable.  At every trial
    temperature, :func:`solve_finite_temperature_equilibrium` first closes atom
    number and then evaluates the paired condensate and thermal energies.  The
    search stops rather than extrapolating if the partially condensed solution
    or the dipolar-TF core leaves its declared domain.
    """

    if not isinstance(initial_equilibrium, FiniteTemperatureEquilibrium):
        raise TypeError(
            "initial_equilibrium must be a FiniteTemperatureEquilibrium"
        )
    deposited = _nonnegative_finite(deposited_energy_j, "deposited_energy_j")
    if energy_model not in (
        "semiclassical_primary",
        "ideal_harmonic_sensitivity",
    ):
        raise ValueError(
            "energy_model must be semiclassical_primary or "
            "ideal_harmonic_sensitivity"
        )
    minimum_temperature = _positive_finite(
        minimum_temperature_k, "minimum_temperature_k"
    )
    maximum_temperature = _positive_finite(
        maximum_temperature_k, "maximum_temperature_k"
    )
    if maximum_temperature <= minimum_temperature:
        raise ValueError(
            "maximum_temperature_k must exceed minimum_temperature_k"
        )
    minimum_tf_ratio = _positive_finite(
        minimum_tf_chemical_potential_ratio,
        "minimum_tf_chemical_potential_ratio",
    )
    step_fraction = _positive_finite(
        temperature_step_fraction, "temperature_step_fraction"
    )
    if step_fraction > 0.1:
        raise ValueError("temperature_step_fraction must not exceed 0.1")
    initial_temperature = float(initial_equilibrium.temperature_k)
    if not minimum_temperature <= initial_temperature <= maximum_temperature:
        raise ValueError(
            "initial equilibrium temperature lies outside the supported range"
        )
    if (
        initial_equilibrium.condensate_atoms <= 0.0
        or initial_equilibrium.thermal_atoms <= 0.0
    ):
        raise ValueError(
            "initial equilibrium lies outside the partially condensed domain"
        )
    initial_tf_ratios = np.asarray(
        initial_equilibrium.thermal_population.thomas_fermi_state
        .chemical_potential_to_trap_quantum_ratio,
        dtype=float,
    )
    if float(np.min(initial_tf_ratios)) < minimum_tf_ratio:
        raise ValueError(
            "initial equilibrium lies outside the dipolar Thomas--Fermi domain"
        )
    if deposited > 0.0 and maximum_temperature <= initial_temperature:
        raise ValueError(
            "maximum_temperature_k must exceed the initial temperature"
        )

    def represented_energy(equilibrium: FiniteTemperatureEquilibrium) -> float:
        if energy_model == "semiclassical_primary":
            return float(equilibrium.total_energy_j)
        return float(equilibrium.ideal_harmonic_sensitivity_energy_j)

    initial_energy = represented_energy(initial_equilibrium)
    target_energy = float(initial_energy + deposited)
    if deposited == 0.0:
        return FiniteTemperatureEnergyUpdate(
            initial_equilibrium=initial_equilibrium,
            final_equilibrium=initial_equilibrium,
            energy_model=energy_model,
            deposited_energy_j=0.0,
            target_energy_j=initial_energy,
            energy_residual_j=0.0,
            relative_energy_residual=0.0,
            temperature_bracket_k=(
                initial_temperature,
                initial_temperature,
            ),
        )

    frequencies = _trap_frequencies(trap_frequencies_hz)
    axis = _axis(dipole_axis)
    total_atoms = _positive_finite(
        initial_equilibrium.total_atoms, "initial_equilibrium.total_atoms"
    )
    cache: dict[float, FiniteTemperatureEquilibrium] = {
        initial_temperature: initial_equilibrium
    }

    def equilibrium_at(temperature: float) -> FiniteTemperatureEquilibrium:
        key = float(temperature)
        if key in cache:
            return cache[key]
        critical_point = harmonic_dipolar_critical_point(
            temperature_k=key,
            trap_frequencies_hz=frequencies,
            dipole_axis=axis,
            scattering_length_m=scattering_length_m,
            dipolar_length_m=dipolar_length_m,
            atomic_mass_kg=atomic_mass_kg,
            boltzmann_constant_j_k=boltzmann_constant_j_k,
            hbar_j_s=hbar_j_s,
        )
        if total_atoms <= critical_point.harmonic_critical_atoms:
            raise _UnsupportedEquilibriumDomain(
                "temperature leaves the partially condensed domain"
            )
        equilibrium = solve_finite_temperature_equilibrium(
            total_atoms=total_atoms,
            temperature_k=key,
            trap_frequencies_hz=frequencies,
            dipole_axis=axis,
            scattering_length_m=scattering_length_m,
            dipolar_length_m=dipolar_length_m,
            atomic_mass_kg=atomic_mass_kg,
            boltzmann_constant_j_k=boltzmann_constant_j_k,
            hbar_j_s=hbar_j_s,
            angular_order=angular_order,
            azimuthal_order=azimuthal_order,
            radial_order=radial_order,
            potential_order=potential_order,
        )
        tf_ratios = np.asarray(
            equilibrium.thermal_population.thomas_fermi_state
            .chemical_potential_to_trap_quantum_ratio,
            dtype=float,
        )
        if float(np.min(tf_ratios)) < minimum_tf_ratio:
            raise _UnsupportedEquilibriumDomain(
                "temperature leaves the dipolar Thomas--Fermi domain"
            )
        cache[key] = equilibrium
        return equilibrium

    lower_temperature = initial_temperature
    lower_equilibrium = initial_equilibrium
    lower_residual = float(represented_energy(lower_equilibrium) - target_energy)
    if lower_residual >= 0.0:
        raise RuntimeError("positive deposited energy did not raise target energy")

    minimum_step = 0.25e-9
    upper_temperature = lower_temperature
    upper_equilibrium: FiniteTemperatureEquilibrium | None = None
    upper_residual = float("nan")

    while upper_temperature < maximum_temperature:
        candidate_temperature = min(
            maximum_temperature,
            max(
                upper_temperature + minimum_step,
                upper_temperature * (1.0 + step_fraction),
            ),
        )
        try:
            candidate_equilibrium = equilibrium_at(candidate_temperature)
        except _UnsupportedEquilibriumDomain as domain_error:
            boundary_lower = upper_temperature
            boundary_equilibrium = lower_equilibrium
            boundary_upper = candidate_temperature
            for _ in range(18):
                midpoint = 0.5 * (boundary_lower + boundary_upper)
                try:
                    midpoint_equilibrium = equilibrium_at(midpoint)
                except _UnsupportedEquilibriumDomain:
                    boundary_upper = midpoint
                else:
                    boundary_lower = midpoint
                    boundary_equilibrium = midpoint_equilibrium
            boundary_residual = float(
                represented_energy(boundary_equilibrium) - target_energy
            )
            if boundary_residual < 0.0:
                raise ValueError(
                    "deposited energy exceeds the supported equilibrium domain"
                ) from domain_error
            upper_temperature = boundary_lower
            upper_equilibrium = boundary_equilibrium
            upper_residual = boundary_residual
            break

        candidate_residual = float(
            represented_energy(candidate_equilibrium) - target_energy
        )
        if represented_energy(candidate_equilibrium) <= represented_energy(
            lower_equilibrium
        ):
            raise RuntimeError(
                "finite-temperature equilibrium energy is not monotonic"
            )
        upper_temperature = candidate_temperature
        upper_equilibrium = candidate_equilibrium
        upper_residual = candidate_residual
        if candidate_residual >= 0.0:
            break
        lower_temperature = candidate_temperature
        lower_equilibrium = candidate_equilibrium

    if upper_equilibrium is None or upper_residual < 0.0:
        raise ValueError(
            "deposited energy is not bracketed below maximum_temperature_k"
        )

    bracket_lower = float(lower_temperature)
    bracket_upper = float(upper_temperature)

    def energy_residual(temperature: float) -> float:
        return float(represented_energy(equilibrium_at(temperature)) - target_energy)

    final_temperature = float(
        brentq(
            energy_residual,
            bracket_lower,
            bracket_upper,
            xtol=1e-14,
            rtol=1e-12,
        )
    )
    final_equilibrium = equilibrium_at(final_temperature)
    energy_residual_j = float(represented_energy(final_equilibrium) - target_energy)
    relative_residual = float(
        energy_residual_j / max(abs(target_energy), abs(deposited))
    )
    if abs(relative_residual) > 2e-8:
        raise RuntimeError("finite-temperature update failed energy closure")
    if abs(final_equilibrium.number_residual_atoms) > 0.05:
        raise RuntimeError("finite-temperature update failed number closure")
    if final_equilibrium.temperature_k <= initial_equilibrium.temperature_k:
        raise RuntimeError("positive recoil did not increase temperature")
    if final_equilibrium.condensate_atoms > initial_equilibrium.condensate_atoms:
        raise RuntimeError("positive recoil increased the condensate population")

    return FiniteTemperatureEnergyUpdate(
        initial_equilibrium=initial_equilibrium,
        final_equilibrium=final_equilibrium,
        energy_model=energy_model,
        deposited_energy_j=deposited,
        target_energy_j=target_energy,
        energy_residual_j=energy_residual_j,
        relative_energy_residual=relative_residual,
        temperature_bracket_k=(bracket_lower, bracket_upper),
    )


__all__ = [
    "HarmonicCriticalPoint",
    "FiniteTemperatureEquilibrium",
    "FiniteTemperatureEnergyUpdate",
    "SemiclassicalThermalPopulation",
    "bose_function_five_halves",
    "bose_function_three_halves",
    "finite_number_critical_population_shift_fraction",
    "harmonic_dipolar_critical_point",
    "ideal_harmonic_critical_atoms",
    "recoil_deposited_energy_j",
    "semiclassical_dipolar_thermal_population",
    "solve_finite_temperature_energy_update",
    "solve_finite_temperature_equilibrium",
    "thermal_de_broglie_wavelength_m",
    "triaxial_dipolar_geometry_factors",
]
