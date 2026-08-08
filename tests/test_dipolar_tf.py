from __future__ import annotations

import numpy as np
import pytest

from non_destructive_image.atomic_model import build_thomas_fermi_state
from non_destructive_image.dipolar_tf import (
    build_dipolar_thomas_fermi_state,
    ellipsoid_demagnetisation_factors,
    scale_dipolar_thomas_fermi_state,
)


HBAR_J_S = 1.054571817e-34
BOLTZMANN_J_K = 1.380649e-23
AMU_KG = 1.66053906660e-27
BOHR_RADIUS_M = 5.29177210903e-11
ERBIUM_166_MASS_KG = 166.0 * AMU_KG
SOURCE_EPSILON_DD = 0.902435305062547
DIPOLAR_LENGTH_BOHR = 65.4
DIPOLAR_TF_SCATTERING_LENGTH_BOHR = DIPOLAR_LENGTH_BOHR / SOURCE_EPSILON_DD


def _state(
    *,
    dipolar_length_bohr: float = DIPOLAR_LENGTH_BOHR,
    trap_frequencies_hz: tuple[float, float, float] = (294.0, 14.0, 233.3),
    dipole_axis: int = 1,
):
    return build_dipolar_thomas_fermi_state(
        atom_number=18292.2198430069,
        scattering_length_m=DIPOLAR_TF_SCATTERING_LENGTH_BOHR * BOHR_RADIUS_M,
        dipolar_length_m=dipolar_length_bohr * BOHR_RADIUS_M,
        trap_frequencies_hz=trap_frequencies_hz,
        dipole_axis=dipole_axis,
        atomic_mass_kg=ERBIUM_166_MASS_KG,
        hbar_j_s=HBAR_J_S,
        boltzmann_constant_j_k=BOLTZMANN_J_K,
    )


def test_demagnetisation_factors_are_scale_invariant_and_sum_to_one() -> None:
    sphere = ellipsoid_demagnetisation_factors([1.0, 1.0, 1.0])
    ellipsoid = ellipsoid_demagnetisation_factors([2e-6, 7e-6, 3e-6])
    scaled = ellipsoid_demagnetisation_factors([2.0, 7.0, 3.0])

    np.testing.assert_allclose(sphere, np.full(3, 1.0 / 3.0), rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(ellipsoid, scaled, rtol=2e-15, atol=0.0)
    assert float(np.sum(ellipsoid)) == pytest.approx(1.0, abs=2e-15)


def test_contact_limit_exactly_matches_existing_thomas_fermi_state() -> None:
    actual = _state(dipolar_length_bohr=0.0)
    expected = build_thomas_fermi_state(
        18292.2198430069,
        DIPOLAR_TF_SCATTERING_LENGTH_BOHR * BOHR_RADIUS_M,
        (294.0, 14.0, 233.3),
        ERBIUM_166_MASS_KG,
        HBAR_J_S,
        BOLTZMANN_J_K,
    )

    assert actual.epsilon_dd == 0.0
    assert actual.optimiser_iterations == 0
    np.testing.assert_array_equal(actual.radii_m, expected.radii)
    np.testing.assert_allclose(
        actual.peak_column_density_m2,
        expected.column_density,
        rtol=2e-15,
        atol=0.0,
    )
    assert actual.chemical_potential_j == pytest.approx(
        expected.chemical_potential, rel=2e-15
    )


@pytest.mark.parametrize(
    ("trap_aspect_ratio", "epsilon_dd"),
    [
        (0.5, 0.2),
        (1.0, 0.5),
        (1.4, 0.3),
        (4.0, 0.8),
        (4.0, SOURCE_EPSILON_DD),
    ],
)
def test_axisymmetric_solution_satisfies_odell_aspect_ratio_equation(
    trap_aspect_ratio: float,
    epsilon_dd: float,
) -> None:
    state = build_dipolar_thomas_fermi_state(
        atom_number=2.5e4,
        scattering_length_m=72.0 * BOHR_RADIUS_M,
        dipolar_length_m=epsilon_dd * 72.0 * BOHR_RADIUS_M,
        trap_frequencies_hz=(100.0, 100.0, 100.0 * trap_aspect_ratio),
        dipole_axis=2,
        atomic_mass_kg=ERBIUM_166_MASS_KG,
        hbar_j_s=HBAR_J_S,
        boltzmann_constant_j_k=BOLTZMANN_J_K,
    )
    kappa = float(state.radii_m[0] / state.radii_m[2])
    shape_function = float(1.0 - 3.0 * state.demagnetisation_factors[2])
    residual = (
        3.0
        * kappa**2
        * epsilon_dd
        * (
            (trap_aspect_ratio**2 / 2.0 + 1.0)
            * shape_function
            / (1.0 - kappa**2)
            - 1.0
        )
        + (epsilon_dd - 1.0) * (kappa**2 - trap_aspect_ratio**2)
    )

    assert residual == pytest.approx(0.0, abs=2e-6)


def test_oxford_conditional_ansatz_is_normalised_and_stationary() -> None:
    state = _state()

    assert state.epsilon_dd == pytest.approx(SOURCE_EPSILON_DD, rel=2e-15)
    assert state.dipole_axis == 1
    assert state.atom_number_check == pytest.approx(18292.2198430069, rel=2e-13)
    assert abs(state.virial_relative_residual) < 2e-6
    assert state.stationarity_max_abs < 2e-6
    assert state.dipolar_energy_j < 0.0
    assert state.contact_energy_j + state.dipolar_energy_j > 0.0
    assert state.radii_m[1] > state.radii_m[2] > state.radii_m[0]
    np.testing.assert_allclose(
        state.chemical_potential_to_trap_quantum_ratio,
        np.array([1.25930261, 26.44535477, 1.58694799]),
        rtol=3e-7,
        atol=0.0,
    )
    assert state.chemical_potential_to_trap_quantum_ratio[[0, 2]].max() < 2.0
    assert np.all(state.axis_harmonic_oscillator_lengths_m > 0.0)
    np.testing.assert_allclose(
        state.radii_m * 1e6,
        np.array([0.67897605, 16.79882871, 0.85834103]),
        rtol=3e-7,
        atol=0.0,
    )


def test_fixed_geometry_atom_number_scaling_matches_direct_solution() -> None:
    reference = _state()
    target_atoms = 0.54 * reference.atom_number
    scaled = scale_dipolar_thomas_fermi_state(reference, target_atoms)
    direct = build_dipolar_thomas_fermi_state(
        atom_number=target_atoms,
        scattering_length_m=DIPOLAR_TF_SCATTERING_LENGTH_BOHR * BOHR_RADIUS_M,
        dipolar_length_m=DIPOLAR_LENGTH_BOHR * BOHR_RADIUS_M,
        trap_frequencies_hz=(294.0, 14.0, 233.3),
        dipole_axis=1,
        atomic_mass_kg=ERBIUM_166_MASS_KG,
        hbar_j_s=HBAR_J_S,
        boltzmann_constant_j_k=BOLTZMANN_J_K,
    )

    assert scaled.atom_number == pytest.approx(target_atoms)
    assert scaled.atom_number_check == pytest.approx(target_atoms)
    np.testing.assert_allclose(scaled.radii_m, direct.radii_m, rtol=2e-7, atol=0.0)
    np.testing.assert_allclose(
        scaled.peak_column_density_m2,
        direct.peak_column_density_m2,
        rtol=2e-7,
        atol=0.0,
    )
    assert scaled.chemical_potential_j == pytest.approx(
        direct.chemical_potential_j,
        rel=2e-7,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"atom_number": 0.0},
        {"scattering_length_m": -1.0},
        {"dipolar_length_m": -1.0},
        {
            "dipolar_length_m": (
                DIPOLAR_TF_SCATTERING_LENGTH_BOHR * BOHR_RADIUS_M
            )
        },
        {"trap_frequencies_hz": (294.0, 14.0)},
        {"trap_frequencies_hz": (294.0, np.nan, 233.3)},
        {"dipole_axis": 3},
        {"dipole_axis": True},
    ],
)
def test_invalid_inputs_are_rejected(overrides: dict[str, object]) -> None:
    inputs: dict[str, object] = {
        "atom_number": 18292.2198430069,
        "scattering_length_m": (
            DIPOLAR_TF_SCATTERING_LENGTH_BOHR * BOHR_RADIUS_M
        ),
        "dipolar_length_m": DIPOLAR_LENGTH_BOHR * BOHR_RADIUS_M,
        "trap_frequencies_hz": (294.0, 14.0, 233.3),
        "dipole_axis": 1,
        "atomic_mass_kg": ERBIUM_166_MASS_KG,
        "hbar_j_s": HBAR_J_S,
        "boltzmann_constant_j_k": BOLTZMANN_J_K,
    }
    inputs.update(overrides)

    with pytest.raises(ValueError):
        build_dipolar_thomas_fermi_state(**inputs)


@pytest.mark.parametrize(
    "radii",
    [[1.0, 2.0], [1.0, 0.0, 2.0], [1.0, np.inf, 2.0]],
)
def test_invalid_demagnetisation_radii_are_rejected(radii: list[float]) -> None:
    with pytest.raises(ValueError):
        ellipsoid_demagnetisation_factors(radii)
