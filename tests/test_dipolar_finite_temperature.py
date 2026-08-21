from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import Boltzmann, atomic_mass, hbar, physical_constants
from scipy.special import zeta

from non_destructive_image.dipolar_finite_temperature import (
    bose_function_five_halves,
    bose_function_three_halves,
    harmonic_dipolar_critical_point,
    ideal_harmonic_critical_atoms,
    recoil_deposited_energy_j,
    semiclassical_dipolar_thermal_population,
    solve_finite_temperature_equilibrium,
    solve_finite_temperature_energy_update,
    triaxial_dipolar_geometry_factors,
)
from non_destructive_image.atomic_model import recoil_quantities


BOHR = physical_constants["Bohr radius"][0]
ERBIUM_166_MASS = 166.0 * atomic_mass


def test_triaxial_geometry_matches_oxford_trap_i() -> None:
    geometry = triaxial_dipolar_geometry_factors((294.0, 14.0, 233.3))
    np.testing.assert_allclose(
        geometry,
        (-0.330583769454121, 0.488737220627285, -0.158153451173163),
        rtol=0.0,
        atol=2e-15,
    )
    assert float(np.sum(geometry)) == pytest.approx(0.0, abs=2e-15)


def test_target_geometry_uses_full_triaxial_factor() -> None:
    geometry = triaxial_dipolar_geometry_factors((151.0, 31.5, 227.0))
    np.testing.assert_allclose(
        geometry,
        (-0.068128201271721, 0.433958727461693, -0.365830526189972),
        rtol=0.0,
        atol=2e-15,
    )
    assert geometry[0] != pytest.approx(-0.25, abs=0.1)


def test_isotropic_trap_has_no_orientation_dependence() -> None:
    geometry = triaxial_dipolar_geometry_factors((100.0, 100.0, 100.0))
    np.testing.assert_allclose(geometry, 0.0, rtol=0.0, atol=2e-15)


def test_target_harmonic_critical_point_reference_values() -> None:
    point = harmonic_dipolar_critical_point(
        temperature_k=245.102e-9,
        trap_frequencies_hz=(151.0, 31.5, 227.0),
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
    )
    assert point.ideal_critical_atoms == pytest.approx(148297.55590525965)
    assert point.mean_field_temperature_shift_fraction == pytest.approx(
        -0.05064368163722689
    )
    assert point.finite_number_critical_population_shift_fraction == pytest.approx(
        0.054799091825230344
    )
    assert point.harmonic_critical_population_fraction == pytest.approx(
        1.206730136736911
    )


def test_axis_permutation_preserves_physical_result() -> None:
    original = harmonic_dipolar_critical_point(
        temperature_k=245.102e-9,
        trap_frequencies_hz=(151.0, 31.5, 227.0),
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
    )
    permuted = harmonic_dipolar_critical_point(
        temperature_k=245.102e-9,
        trap_frequencies_hz=(227.0, 31.5, 151.0),
        dipole_axis=2,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
    )
    assert permuted.harmonic_critical_atoms == pytest.approx(
        original.harmonic_critical_atoms, rel=0.0, abs=1e-9
    )
    assert permuted.mean_field_temperature_shift_fraction == pytest.approx(
        original.mean_field_temperature_shift_fraction, rel=0.0, abs=1e-15
    )


def test_bose_function_resolves_the_zero_energy_cusp() -> None:
    values = bose_function_three_halves((0.0, 1e-8, 0.2, 2.0))
    assert values[0] == pytest.approx(zeta(1.5), rel=0.0, abs=1e-14)
    assert np.all(np.diff(values) < 0.0)

    energy_values = bose_function_five_halves((0.0, 1e-8, 0.2, 2.0))
    assert energy_values[0] == pytest.approx(zeta(2.5), rel=0.0, abs=1e-14)
    assert np.all(np.diff(energy_values) < 0.0)
    direct_series = sum(np.exp(-0.2 * index) / index**2.5 for index in range(1, 500))
    assert energy_values[2] == pytest.approx(direct_series, rel=0.0, abs=2e-14)


def test_contact_only_population_recovers_analytic_small_condensate_limit() -> None:
    temperature = 245.10243205732764e-9
    frequencies = np.asarray((151.0, 31.5, 227.0), dtype=float)
    condensate_atoms = 1.0
    ideal_critical = ideal_harmonic_critical_atoms(
        temperature,
        frequencies,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
    )
    result = semiclassical_dipolar_thermal_population(
        temperature_k=temperature,
        condensate_atoms=condensate_atoms,
        critical_atoms=ideal_critical,
        trap_frequencies_hz=frequencies,
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=0.0,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
    )
    omega_bar = 2.0 * np.pi * float(np.prod(frequencies) ** (1.0 / 3.0))
    oscillator_length = np.sqrt(hbar / (ERBIUM_166_MASS * omega_bar))
    temperature_scale = Boltzmann * temperature / (hbar * omega_bar)
    analytic_slope = (
        zeta(2.0)
        / 2.0
        * temperature_scale**2
        * (15.0 * 72.0 * BOHR / oscillator_length) ** 0.4
        / ideal_critical**0.6
    )
    numerical_slope = result.mean_field_excess_fraction_of_ideal / (
        condensate_atoms / ideal_critical
    ) ** 0.4
    assert numerical_slope == pytest.approx(analytic_slope, rel=6e-3)


def test_semiclassical_anisotropy_regression_uses_declared_y_z_axes() -> None:
    """Internal harmonic regression only; this is not Oxford curve validation."""

    temperature = 244.165959e-9
    frequencies = (294.0, 14.0, 233.3)
    condensate_atoms = 300.0
    ideal_critical = ideal_harmonic_critical_atoms(
        temperature,
        frequencies,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
    )
    slopes = []
    for axis in (1, 2):
        result = semiclassical_dipolar_thermal_population(
            temperature_k=temperature,
            condensate_atoms=condensate_atoms,
            critical_atoms=ideal_critical,
            trap_frequencies_hz=frequencies,
            dipole_axis=axis,
            scattering_length_m=72.0 * BOHR,
            dipolar_length_m=65.4 * BOHR,
            atomic_mass_kg=ERBIUM_166_MASS,
            boltzmann_constant_j_k=Boltzmann,
            hbar_j_s=hbar,
        )
        slopes.append(
            result.mean_field_excess_fraction_of_ideal
            / (condensate_atoms / ideal_critical) ** 0.4
        )
    np.testing.assert_allclose(
        slopes,
        (0.23746638814623347, 0.6060657851929268),
        rtol=0.0,
        atol=2.0e-3,
    )


def test_target_equilibrium_recomputes_populations_in_shared_geometry() -> None:
    total_atoms = 18292.2198430069 + 199133.90988630912
    equilibrium = solve_finite_temperature_equilibrium(
        total_atoms=total_atoms,
        temperature_k=245.10243205732764e-9,
        trap_frequencies_hz=(151.0, 31.5, 227.0),
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
    )
    assert equilibrium.condensate_atoms == pytest.approx(9210.24, abs=0.1)
    assert equilibrium.thermal_atoms == pytest.approx(208215.90, abs=0.1)
    assert equilibrium.number_residual_atoms == pytest.approx(0.0, abs=0.05)
    np.testing.assert_allclose(
        equilibrium.thermal_population.thomas_fermi_state.radii_m * 1e6,
        (2.49526, 8.70680, 0.74036),
        rtol=0.0,
        atol=2e-5,
    )


def test_target_energy_decomposition_uses_paired_semiclassical_integral() -> None:
    equilibrium = solve_finite_temperature_equilibrium(
        total_atoms=289228.8760760734,
        temperature_k=245.10243205732764e-9,
        trap_frequencies_hz=(151.0, 31.5, 227.0),
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
    )
    thermal = equilibrium.thermal_population
    expected_coefficient = float(3.0 * zeta(4.0) / zeta(3.0))
    assert thermal.ideal_harmonic_energy_coefficient == pytest.approx(
        expected_coefficient, rel=0.0, abs=1e-15
    )
    assert thermal.critical_baseline_energy_j == pytest.approx(
        expected_coefficient
        * thermal.critical_atoms
        * Boltzmann
        * equilibrium.temperature_k,
        rel=2e-15,
    )
    assert thermal.thermal_energy_j == pytest.approx(
        thermal.critical_baseline_energy_j
        + thermal.mean_field_excess_energy_j,
        rel=2e-15,
    )
    assert equilibrium.total_energy_j == pytest.approx(
        equilibrium.condensate_energy_j + equilibrium.thermal_energy_j,
        rel=2e-15,
    )
    assert equilibrium.condensate_atoms == pytest.approx(50000.0, abs=0.05)
    assert equilibrium.total_energy_j == pytest.approx(
        2.1261351046373136e-24, rel=2e-8
    )
    assert thermal.thermal_energy_j / (
        thermal.thermal_atoms * Boltzmann * equilibrium.temperature_k
    ) == pytest.approx(2.5876778152, rel=2e-8)
    assert equilibrium.ideal_harmonic_sensitivity_energy_j > equilibrium.total_energy_j


def test_semiclassical_energy_converges_with_spatial_quadrature() -> None:
    common = dict(
        temperature_k=245.10243205732764e-9,
        condensate_atoms=50000.0,
        trap_frequencies_hz=(151.0, 31.5, 227.0),
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
    )
    critical = harmonic_dipolar_critical_point(
        **{key: value for key, value in common.items() if key != "condensate_atoms"}
    ).harmonic_critical_atoms
    coarse = semiclassical_dipolar_thermal_population(
        critical_atoms=critical,
        angular_order=8,
        azimuthal_order=16,
        radial_order=10,
        potential_order=24,
        **common,
    )
    reference = semiclassical_dipolar_thermal_population(
        critical_atoms=critical,
        angular_order=12,
        azimuthal_order=24,
        radial_order=14,
        potential_order=48,
        **common,
    )
    assert coarse.thermal_atoms == pytest.approx(reference.thermal_atoms, rel=5e-5)
    assert coarse.thermal_energy_j == pytest.approx(
        reference.thermal_energy_j, rel=2e-5
    )


def test_semiclassical_population_and_energy_are_axis_permutation_invariant() -> None:
    results = []
    for frequencies, axis in (
        ((151.0, 31.5, 227.0), 0),
        ((227.0, 31.5, 151.0), 2),
    ):
        common = dict(
            temperature_k=245.10243205732764e-9,
            trap_frequencies_hz=frequencies,
            dipole_axis=axis,
            scattering_length_m=72.0 * BOHR,
            dipolar_length_m=65.4 * BOHR,
            atomic_mass_kg=ERBIUM_166_MASS,
            boltzmann_constant_j_k=Boltzmann,
            hbar_j_s=hbar,
        )
        critical = harmonic_dipolar_critical_point(**common)
        results.append(
            semiclassical_dipolar_thermal_population(
                condensate_atoms=50000.0,
                critical_atoms=critical.harmonic_critical_atoms,
                angular_order=8,
                azimuthal_order=16,
                radial_order=10,
                potential_order=24,
                **common,
            )
        )
    assert results[1].thermal_atoms == pytest.approx(
        results[0].thermal_atoms, rel=2e-7
    )
    assert results[1].thermal_energy_j == pytest.approx(
        results[0].thermal_energy_j, rel=6e-6
    )
    assert results[1].thomas_fermi_state.total_energy_j == pytest.approx(
        results[0].thomas_fermi_state.total_energy_j, rel=2e-12
    )


def test_recoil_energy_and_zero_update_contract() -> None:
    equilibrium = solve_finite_temperature_equilibrium(
        total_atoms=289228.8760760734,
        temperature_k=245.10243205732764e-9,
        trap_frequencies_hz=(151.0, 31.5, 227.0),
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
        angular_order=8,
        azimuthal_order=16,
        radial_order=10,
        potential_order=24,
    )
    recoil_energy, recoil_temperature, recoil_velocity = recoil_quantities(
        hbar=hbar,
        wavevector=2.0 * np.pi / 401e-9,
        atomic_mass=ERBIUM_166_MASS,
        boltzmann_constant=Boltzmann,
    )
    assert recoil_energy == pytest.approx(4.952632826240702e-30, rel=2e-15)
    assert recoil_temperature == pytest.approx(3.5871773537232865e-7, rel=2e-15)
    assert recoil_velocity == pytest.approx(0.00599452078943813, rel=2e-15)
    deposited = recoil_deposited_energy_j(
        total_atoms=equilibrium.total_atoms,
        scattered_photons_per_atom=0.01,
        single_photon_recoil_energy_j=recoil_energy,
    )
    assert deposited == pytest.approx(
        2.0 * equilibrium.total_atoms * 0.01 * recoil_energy,
        rel=0.0,
        abs=0.0,
    )
    zero = solve_finite_temperature_energy_update(
        initial_equilibrium=equilibrium,
        deposited_energy_j=0.0,
        minimum_temperature_k=235e-9,
        maximum_temperature_k=404e-9,
        trap_frequencies_hz=(151.0, 31.5, 227.0),
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
    )
    assert zero.final_equilibrium is equilibrium
    assert zero.energy_residual_j == 0.0


def test_positive_energy_update_closes_number_and_energy() -> None:
    common = dict(
        trap_frequencies_hz=(151.0, 31.5, 227.0),
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
        angular_order=8,
        azimuthal_order=16,
        radial_order=10,
        potential_order=24,
    )
    initial = solve_finite_temperature_equilibrium(
        total_atoms=289228.8760760734,
        temperature_k=245.10243205732764e-9,
        **common,
    )
    recoil_energy, _, _ = recoil_quantities(
        hbar=hbar,
        wavevector=2.0 * np.pi / 401e-9,
        atomic_mass=ERBIUM_166_MASS,
        boltzmann_constant=Boltzmann,
    )
    deposited = recoil_deposited_energy_j(
        total_atoms=initial.total_atoms,
        scattered_photons_per_atom=2e-4,
        single_photon_recoil_energy_j=recoil_energy,
    )
    update = solve_finite_temperature_energy_update(
        initial_equilibrium=initial,
        deposited_energy_j=deposited,
        minimum_temperature_k=235e-9,
        maximum_temperature_k=250e-9,
        **common,
    )
    final = update.final_equilibrium
    assert final.temperature_k > initial.temperature_k
    assert final.condensate_atoms < initial.condensate_atoms
    assert final.thermal_atoms > initial.thermal_atoms
    assert final.number_residual_atoms == pytest.approx(0.0, abs=0.05)
    assert update.relative_energy_residual == pytest.approx(0.0, abs=2e-8)
    assert final.total_energy_j == pytest.approx(
        initial.total_energy_j + deposited,
        rel=2e-8,
    )


def test_energy_update_fails_instead_of_clipping_at_temperature_limit() -> None:
    common = dict(
        trap_frequencies_hz=(151.0, 31.5, 227.0),
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
        angular_order=8,
        azimuthal_order=16,
        radial_order=10,
        potential_order=24,
    )
    initial = solve_finite_temperature_equilibrium(
        total_atoms=289228.8760760734,
        temperature_k=245.10243205732764e-9,
        **common,
    )
    with pytest.raises(ValueError, match="not bracketed"):
        solve_finite_temperature_energy_update(
            initial_equilibrium=initial,
            deposited_energy_j=initial.total_energy_j,
            minimum_temperature_k=235e-9,
            maximum_temperature_k=246e-9,
            **common,
        )


def test_energy_update_rejects_initial_temperature_below_validated_range() -> None:
    common = dict(
        trap_frequencies_hz=(151.0, 31.5, 227.0),
        dipole_axis=0,
        scattering_length_m=72.0 * BOHR,
        dipolar_length_m=65.4 * BOHR,
        atomic_mass_kg=ERBIUM_166_MASS,
        boltzmann_constant_j_k=Boltzmann,
        hbar_j_s=hbar,
        angular_order=8,
        azimuthal_order=16,
        radial_order=10,
        potential_order=24,
    )
    initial = solve_finite_temperature_equilibrium(
        total_atoms=289228.8760760734,
        temperature_k=230e-9,
        **common,
    )
    with pytest.raises(ValueError, match="outside the supported range"):
        solve_finite_temperature_energy_update(
            initial_equilibrium=initial,
            deposited_energy_j=0.0,
            minimum_temperature_k=235e-9,
            maximum_temperature_k=404e-9,
            **common,
        )
