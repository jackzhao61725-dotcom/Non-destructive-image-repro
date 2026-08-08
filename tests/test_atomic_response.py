from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from non_destructive_image import (
    OpticalBranch,
    PolarisedOpticalResponse,
    branch_summed_scattered_photons_per_atom,
    complex_column_response,
    dimensionless_detuning,
    polarised_optical_response_from_config,
    residual_optical_depth,
    scalar_phase_shift,
    scattered_photons_per_atom,
)


CONFIG_PATH = Path("configs/model.json")


@pytest.fixture()
def active_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def erbium_response(active_config: dict) -> PolarisedOpticalResponse:
    return polarised_optical_response_from_config(active_config)


def _unit_scalar_response() -> PolarisedOpticalResponse:
    return PolarisedOpticalResponse(
        species="synthetic",
        transition_label="unit scalar regression response",
        ground_j=0.0,
        ground_m_j=0.0,
        excited_j=1.0,
        rank0_factor=1.0,
        rank1_helicity_factor=0.0,
        rank2_factor=0.0,
        probe_axis=(1.0, 0.0, 0.0),
        quantisation_axis=(0.0, 1.0, 0.0),
        polarisation_axis=(0.0, 0.0, 1.0),
        selected_eigenmode="perpendicular",
        branches=(
            OpticalBranch("minus", -1, 0.5, 1.0, "synthetic"),
            OpticalBranch("pi", 0, 0.0, 1.0, "synthetic"),
            OpticalBranch("plus", 1, 0.5, 1.0, "synthetic"),
        ),
        model_status="synthetic unit-strength test response",
    )


def test_active_erbium_rank_branch_and_geometry_contract(
    active_config: dict,
    erbium_response: PolarisedOpticalResponse,
) -> None:
    response = erbium_response
    by_q = {branch.q: branch for branch in response.branches}

    assert response.species == "166Er"
    assert response.ground_j == 6.0
    assert response.ground_m_j == -6.0
    assert response.excited_j == 7.0
    assert response.rank0_factor == pytest.approx(35.0 / 91.0)
    assert response.rank1_helicity_factor == pytest.approx(-45.0 / 91.0)
    assert response.rank2_factor == pytest.approx(-22.0 / 91.0)
    assert response.circular_response_difference == pytest.approx(-90.0 / 91.0)
    assert by_q[-1].relative_line_strength == pytest.approx(1.0)
    assert by_q[0].relative_line_strength == pytest.approx(13.0 / 91.0)
    assert by_q[1].relative_line_strength == pytest.approx(1.0 / 91.0)
    assert response.parallel_line_strength == pytest.approx(13.0 / 91.0)
    assert response.perpendicular_line_strength == pytest.approx(46.0 / 91.0)
    assert response.effective_line_strength == pytest.approx(46.0 / 91.0)
    assert response.perpendicular_line_strength / response.parallel_line_strength == pytest.approx(
        46.0 / 13.0
    )
    np.testing.assert_array_equal(
        np.cross(response.probe_axis, response.quantisation_axis),
        response.polarisation_axis,
    )
    assert active_config["readout_selection"] == {
        "primary": "pci",
        "secondary": "dgi",
        "supported": ["pci", "dgi"],
        "selection_basis": (
            "both use the same selected transverse-projected rank-0 plus rank-2 "
            "eigenmode complex object and differ only in Fourier-plane reference "
            "handling"
        ),
    }
    rank_config = active_config["polarised_atomic_response"]["rank_factors"]
    assert "K_perp = P_k K P_k" in rank_config["transverse_projection_convention"]
    assert "longitudinal component" in rank_config["longitudinal_component_policy"]
    assert active_config["pci"]["phase_plate_transmittance_kind"] == (
        "field_amplitude"
    )
    assert active_config["pci"]["atom_free_intensity_over_incident"] == (
        pytest.approx(0.95**2)
    )
    assert active_config["dgi"]["stop_optical_depth_convention"] == (
        "base10_intensity"
    )
    assert active_config["dgi"]["atom_free_intensity_over_incident"] == (
        pytest.approx(10.0**-4)
    )


def test_response_is_invariant_under_simultaneous_axis_permutation(
    erbium_response: PolarisedOpticalResponse,
) -> None:
    permuted = replace(
        erbium_response,
        probe_axis=(0.0, 1.0, 0.0),
        quantisation_axis=(0.0, 0.0, 1.0),
        polarisation_axis=(1.0, 0.0, 0.0),
    )

    assert permuted.branch_weights == erbium_response.branch_weights
    assert permuted.effective_line_strength == erbium_response.effective_line_strength


def test_tensor_free_synthetic_response_has_equal_linear_eigenvalues() -> None:
    response = _unit_scalar_response()

    assert response.parallel_line_strength == 1.0
    assert response.perpendicular_line_strength == 1.0
    assert response.effective_line_strength == 1.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"probe_axis": (2.0, 0.0, 0.0)}, "unit vector"),
        ({"probe_axis": (0.0, 1.0, 0.0)}, "perpendicular to B"),
        ({"polarisation_axis": (1.0, 0.0, 0.0)}, "transverse"),
        ({"polarisation_axis": (0.0, 1.0, 0.0)}, "selected eigenmode"),
        ({"rank2_factor": 0.0}, "rank/branch"),
        ({"ground_j": 6.1}, "half-integer"),
        ({"ground_m_j": -5.0}, "lowest stretched"),
        ({"excited_j": 6.0}, r"J-to-J\+1"),
        (
            {"ground_j": 5.0, "ground_m_j": -5.0, "excited_j": 6.0},
            "Clebsch--Gordan",
        ),
    ],
)
def test_response_rejects_inconsistent_physical_contract(
    erbium_response: PolarisedOpticalResponse,
    changes: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(erbium_response, **changes)


def test_branch_rejects_invalid_weights_and_quantum_number() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        OpticalBranch("bad", -1, -0.1, 1.0, "invalid")
    with pytest.raises(ValueError, match="positive"):
        OpticalBranch("bad", -1, 0.5, 0.0, "invalid")
    with pytest.raises(ValueError, match="q must"):
        OpticalBranch("bad", 2, 0.5, 1.0, "invalid")


def test_config_loader_rejects_missing_or_inconsistent_contract(active_config: dict) -> None:
    with pytest.raises(ValueError, match="missing"):
        polarised_optical_response_from_config({})

    broken = json.loads(json.dumps(active_config))
    broken["polarised_atomic_response"]["rank_factors"]["rank2_factor"] = 0.0
    with pytest.raises(ValueError, match="rank/branch"):
        polarised_optical_response_from_config(broken)

    coerced_q = json.loads(json.dumps(active_config))
    coerced_q["polarised_atomic_response"]["selected_branches"][0]["q"] = -0.5
    with pytest.raises(ValueError, match="q must"):
        polarised_optical_response_from_config(coerced_q)

    coerced_label = json.loads(json.dumps(active_config))
    coerced_label["polarised_atomic_response"]["selected_branches"][0]["label"] = None
    with pytest.raises(ValueError, match="label"):
        polarised_optical_response_from_config(coerced_label)


def test_common_detuning_complex_response_matches_weighted_scalar_baseline(
    erbium_response: PolarisedOpticalResponse,
) -> None:
    density = np.array([[0.0, 1.0e14], [2.0e14, 5.0e13]])
    detuning_hz = 1.5e9
    cross_section = 7.678673341230136e-14
    linewidth = 2.0 * np.pi * 29.5e6

    result = complex_column_response(
        density,
        detuning_hz,
        cross_section,
        linewidth,
        erbium_response,
    )
    factor = 46.0 / 91.0
    expected_phase = factor * scalar_phase_shift(
        detuning_hz,
        density,
        cross_section,
        linewidth,
    )
    expected_od = factor * residual_optical_depth(
        detuning_hz,
        density,
        cross_section,
        linewidth,
    )

    np.testing.assert_allclose(result.phase_map_rad, expected_phase, rtol=1e-15)
    np.testing.assert_allclose(result.optical_depth_map, expected_od, rtol=1e-15)
    np.testing.assert_allclose(
        result.object_field,
        np.exp(-expected_od / 2.0 + 1j * expected_phase),
        rtol=1e-15,
    )
    assert np.all(result.optical_depth_map >= 0.0)
    assert np.all(np.abs(result.object_field) <= 1.0)
    assert result.object_field[0, 0] == 1.0 + 0.0j


def test_tensor_free_complex_response_reduces_to_unit_scalar_helpers() -> None:
    response = _unit_scalar_response()
    density = np.array([[0.5e14, 1.5e14]])
    detuning_hz = 0.7e9
    cross_section = 7.68e-14
    linewidth = 2.0 * np.pi * 29.5e6
    result = complex_column_response(
        density,
        detuning_hz,
        cross_section,
        linewidth,
        response,
    )

    np.testing.assert_allclose(
        result.phase_map_rad,
        scalar_phase_shift(detuning_hz, density, cross_section, linewidth),
    )
    np.testing.assert_allclose(
        result.optical_depth_map,
        residual_optical_depth(detuning_hz, density, cross_section, linewidth),
    )


def test_complex_response_detuning_parity_and_branch_relation(
    erbium_response: PolarisedOpticalResponse,
) -> None:
    density = np.array([[1.2e14, 0.8e14]])
    cross_section = 7.68e-14
    linewidth = 2.0 * np.pi * 29.5e6
    positive = complex_column_response(
        density,
        1.5e9,
        cross_section,
        linewidth,
        erbium_response,
    )
    negative = complex_column_response(
        density,
        -1.5e9,
        cross_section,
        linewidth,
        erbium_response,
    )

    np.testing.assert_allclose(negative.phase_map_rad, -positive.phase_map_rad)
    np.testing.assert_allclose(negative.optical_depth_map, positive.optical_depth_map)
    delta = dimensionless_detuning(1.5e9, linewidth)
    np.testing.assert_allclose(
        positive.branch_phase_maps_rad,
        delta * positive.branch_optical_depth_maps / 2.0,
    )


def test_exact_coherent_branch_detunings_are_summed_once(
    erbium_response: PolarisedOpticalResponse,
) -> None:
    density = np.array([[1.0e14]])
    cross_section = 7.68e-14
    linewidth = 2.0 * np.pi * 29.5e6
    detunings = {
        "sigma_minus": 1.4977e9,
        "pi": 1.5e9,
        "sigma_plus": 1.5023e9,
    }
    result = complex_column_response(
        density,
        1.5e9,
        cross_section,
        linewidth,
        erbium_response,
        detuning_by_branch_hz=detunings,
    )
    expected_phase = sum(
        branch.weighted_line_strength
        * scalar_phase_shift(
            detunings[branch.label], density, cross_section, linewidth
        )
        for branch in erbium_response.branches
    )

    np.testing.assert_allclose(result.phase_map_rad, expected_phase)
    with pytest.raises(ValueError, match="keys"):
        complex_column_response(
            density,
            1.5e9,
            cross_section,
            linewidth,
            erbium_response,
            detuning_by_branch_hz={"sigma_minus": 1.5e9},
        )


@pytest.mark.parametrize(
    "density",
    [
        np.array([1.0]),
        np.empty((0, 2)),
        np.array([[1.0 + 1.0j]]),
        np.array([[np.nan]]),
        np.array([[-1.0]]),
    ],
)
def test_complex_response_rejects_invalid_column_density(
    density: np.ndarray,
    erbium_response: PolarisedOpticalResponse,
) -> None:
    with pytest.raises(ValueError, match="column_density_m2"):
        complex_column_response(density, 1.0, 1.0, 1.0, erbium_response)


@pytest.mark.parametrize(
    ("cross_section", "linewidth"),
    [(0.0, 1.0), (1.0, 0.0), (np.nan, 1.0), (1.0, np.inf)],
)
def test_complex_response_rejects_invalid_atomic_scalars(
    cross_section: float,
    linewidth: float,
    erbium_response: PolarisedOpticalResponse,
) -> None:
    with pytest.raises(ValueError):
        complex_column_response(
            np.ones((1, 1)),
            1.0,
            cross_section,
            linewidth,
            erbium_response,
        )


def test_branch_scattering_is_nonnegative_and_has_expected_fractions(
    erbium_response: PolarisedOpticalResponse,
) -> None:
    result = branch_summed_scattered_photons_per_atom(
        1.5e9,
        1.0,
        300e-6,
        600.0,
        2.0 * np.pi * 29.5e6,
        24e-3,
        erbium_response,
    )
    by_label = dict(zip(result.branch_labels, result.photons_per_atom_by_branch, strict=True))

    assert all(value >= 0.0 for value in result.photons_per_atom_by_branch)
    assert result.total_photons_per_atom == pytest.approx(
        sum(result.photons_per_atom_by_branch)
    )
    assert by_label["sigma_minus"] / by_label["sigma_plus"] == pytest.approx(91.0)
    assert by_label["sigma_minus"] / result.total_photons_per_atom == pytest.approx(
        91.0 / 92.0
    )
    assert by_label["sigma_plus"] / result.total_photons_per_atom == pytest.approx(
        1.0 / 92.0
    )
    assert by_label["pi"] == 0.0

    common_rate = (
        result.natural_linewidth_rad_s
        * result.pulse_duration_s
        / 2.0
        / (
            1.0
            + sum(result.branch_saturation_parameters)
            + result.dimensionless_detuning**2
        )
    )
    expected_by_branch = np.asarray(result.branch_saturation_parameters) * common_rate
    np.testing.assert_allclose(
        result.photons_per_atom_by_branch,
        expected_by_branch,
        rtol=1e-15,
        atol=0.0,
    )
    assert result.total_photons_per_atom == pytest.approx(
        float(np.sum(expected_by_branch)),
        rel=1e-15,
    )
    assert result.response is erbium_response
    assert result.detuning_hz == 1.5e9
    assert result.probe_power_mw == 1.0
    assert result.pulse_duration_s == 300e-6
    assert result.saturation_intensity_w_m2 == 600.0
    assert result.probe_diameter_m == 24e-3
    assert result.use_peak_intensity is True


@pytest.mark.parametrize("detuning_hz", [0.0, 1.5e9, -1.5e9])
@pytest.mark.parametrize("probe_power_mw", [1e-6, 1.0, 1e5])
def test_unit_scalar_scattering_exactly_matches_existing_helper(
    detuning_hz: float,
    probe_power_mw: float,
) -> None:
    response = _unit_scalar_response()
    duration = 40e-6
    saturation_intensity = 600.0
    linewidth = 2.0 * np.pi * 29.5e6
    diameter = 24e-3
    actual = branch_summed_scattered_photons_per_atom(
        detuning_hz,
        probe_power_mw,
        duration,
        saturation_intensity,
        linewidth,
        diameter,
        response,
    )
    expected = scattered_photons_per_atom(
        detuning_hz,
        probe_power_mw,
        duration,
        saturation_intensity,
        linewidth,
        diameter,
    )

    assert actual.total_photons_per_atom == pytest.approx(expected, rel=1e-15)


def test_erbium_high_saturation_is_not_factor_times_saturated_unit_result(
    erbium_response: PolarisedOpticalResponse,
) -> None:
    parameters = {
        "detuning_hz": 0.0,
        "probe_power_mw": 1e5,
        "pulse_duration_s": 10e-6,
        "saturation_intensity_w_m2": 600.0,
        "natural_linewidth_rad_s": 2.0 * np.pi * 29.5e6,
        "probe_diameter_m": 24e-3,
    }
    result = branch_summed_scattered_photons_per_atom(
        **parameters,
        response=erbium_response,
    )
    old_total = scattered_photons_per_atom(
        parameters["detuning_hz"],
        parameters["probe_power_mw"],
        parameters["pulse_duration_s"],
        parameters["saturation_intensity_w_m2"],
        parameters["natural_linewidth_rad_s"],
        parameters["probe_diameter_m"],
    )

    assert result.total_photons_per_atom != pytest.approx((46.0 / 91.0) * old_total)


def test_scattering_detuning_units_parity_zero_inputs_and_intensity_mode(
    erbium_response: PolarisedOpticalResponse,
) -> None:
    linewidth = 2.0 * np.pi * 29.5e6
    detuning_for_delta_one = linewidth / (4.0 * np.pi)
    common = (1.0, 300e-6, 600.0, linewidth, 24e-3, erbium_response)
    positive = branch_summed_scattered_photons_per_atom(
        detuning_for_delta_one, *common
    )
    negative = branch_summed_scattered_photons_per_atom(
        -detuning_for_delta_one, *common
    )
    average_intensity = branch_summed_scattered_photons_per_atom(
        detuning_for_delta_one,
        *common,
        use_peak_intensity=False,
    )

    assert positive.dimensionless_detuning == pytest.approx(1.0)
    assert negative.total_photons_per_atom == pytest.approx(
        positive.total_photons_per_atom
    )
    assert average_intensity.incident_saturation_parameter == pytest.approx(
        positive.incident_saturation_parameter / 2.0
    )
    assert branch_summed_scattered_photons_per_atom(
        1.0, 0.0, 1.0, 1.0, 1.0, 1.0, erbium_response
    ).total_photons_per_atom == 0.0
    assert branch_summed_scattered_photons_per_atom(
        1.0, 1.0, 0.0, 1.0, 1.0, 1.0, erbium_response
    ).total_photons_per_atom == 0.0


@pytest.mark.parametrize(
    "arguments",
    [
        (np.nan, 1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    ],
)
def test_branch_scattering_rejects_invalid_scalars(
    arguments: tuple[float, ...],
    erbium_response: PolarisedOpticalResponse,
) -> None:
    with pytest.raises(ValueError):
        branch_summed_scattered_photons_per_atom(*arguments, erbium_response)
