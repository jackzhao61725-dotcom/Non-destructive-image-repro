from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from isolated_non_destructive_image import (  # noqa: E402
    load_isolated_non_destructive_image_module,
)


_ATOMIC = load_isolated_non_destructive_image_module(
    "atomic_response", namespace="_ndi_corrected_acquisition_scientific_tests_v1"
)
_LIGHT_ATOM = load_isolated_non_destructive_image_module(
    "light_atom", namespace="_ndi_corrected_acquisition_scientific_tests_v1"
)
ParallelJonesOpticalResponse = _ATOMIC.ParallelJonesOpticalResponse
branch_summed_scattered_photons_per_atom = (
    _ATOMIC.branch_summed_scattered_photons_per_atom
)
parallel_jones_column_response = _ATOMIC.parallel_jones_column_response
parallel_jones_optical_response_from_config = (
    _ATOMIC.parallel_jones_optical_response_from_config
)
polarised_optical_response_from_config = _ATOMIC.polarised_optical_response_from_config
residual_optical_depth = _LIGHT_ATOM.residual_optical_depth
scalar_phase_shift = _LIGHT_ATOM.scalar_phase_shift


JONES_CONTRACT = ROOT / "configs" / "imaging_orientation_contract_v2.json"
SCALAR_CONFIG = ROOT / "configs" / "dissertation_v3_orca_fusion.json"
SIGMA_0_M2 = 7.678673341230137e-14
GAMMA_RAD_S = 2.0 * np.pi * 29.5e6
DETUNING_HZ = 1.5e9


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _response() -> ParallelJonesOpticalResponse:
    return parallel_jones_optical_response_from_config(_load(JONES_CONTRACT))


def test_contract_maps_chomaz_axes_without_changing_the_y_coordinate() -> None:
    config = _load(JONES_CONTRACT)
    three_state = config["geometries"]["three_state_equilibrium"]
    mapping = np.asarray(
        three_state["source_to_thesis_axis_mapping"][
            "matrix_rows_thesis_from_source"
        ],
        dtype=float,
    )

    np.testing.assert_allclose(mapping @ mapping.T, np.eye(3), rtol=0.0, atol=0.0)
    assert np.linalg.det(mapping) == pytest.approx(1.0)
    np.testing.assert_array_equal(mapping @ np.asarray([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(mapping @ np.asarray([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0])
    assert three_state["source_to_thesis_axis_mapping"][
        "mapped_trap_frequency_hz_by_thesis_axis"
    ] == {"x": 151.0, "y": 31.5, "z": 227.0}


def test_parallel_jones_contract_has_the_required_geometry_and_factors() -> None:
    response = _response()

    assert response.probe_axis == (1.0, 0.0, 0.0)
    assert response.quantisation_axis == (1.0, 0.0, 0.0)
    assert response.input_polarisation_axis == (0.0, 0.0, 1.0)
    assert response.faraday_polarisation_axis == (0.0, 1.0, 0.0)
    assert response.effective_line_strength == pytest.approx(46.0 / 91.0)
    assert response.common_phase_factor == pytest.approx(46.0 / 91.0)
    assert response.faraday_rotation_factor == pytest.approx(-45.0 / 91.0)
    by_q = {branch.q: branch for branch in response.branches}
    assert {q: by_q[q].intensity_fraction for q in (-1, 0, 1)} == pytest.approx(
        {-1: 0.5, 0: 0.0, 1: 0.5}
    )
    assert {q: by_q[q].relative_line_strength for q in (-1, 0, 1)} == pytest.approx(
        {-1: 1.0, 0: 1.0 / 7.0, 1: 1.0 / 91.0}
    )


def test_oxford_block_remains_identical_to_the_active_scalar_geometry() -> None:
    contract = _load(JONES_CONTRACT)
    scalar_config = _load(SCALAR_CONFIG)
    oxford = contract["geometries"]["oxford_tfbec"]["geometry"]
    scalar_geometry = scalar_config["polarised_atomic_response"]["geometry"]

    assert oxford["probe_wavevector_unit_vector"] == scalar_geometry[
        "probe_wavevector_unit_vector"
    ]
    assert oxford["quantisation_axis_unit_vector"] == scalar_geometry[
        "quantisation_axis_unit_vector"
    ]
    assert oxford["selected_polarisation_unit_vector"] == scalar_geometry[
        "selected_polarisation_unit_vector"
    ]
    assert oxford["selected_linear_eigenmode"] == scalar_geometry[
        "selected_linear_eigenmode"
    ]
    assert oxford["effective_line_strength"] == pytest.approx(46.0 / 91.0)


def test_standard_circular_basis_reconstructs_the_stored_linear_components() -> None:
    e_y = np.asarray([0.0, 1.0, 0.0], dtype=complex)
    e_z = np.asarray([0.0, 0.0, 1.0], dtype=complex)
    epsilon_plus = -(e_y + 1j * e_z) / np.sqrt(2.0)
    epsilon_minus = (e_y - 1j * e_z) / np.sqrt(2.0)
    t_minus = 0.71 * np.exp(0.43j)
    t_plus = 0.94 * np.exp(-0.17j)

    output = 1j * (
        t_plus * epsilon_plus + t_minus * epsilon_minus
    ) / np.sqrt(2.0)

    assert output[2] == pytest.approx((t_minus + t_plus) / 2.0)
    assert output[1] == pytest.approx(0.5j * (t_minus - t_plus))
    assert output[0] == pytest.approx(0.0 + 0.0j)


def test_circular_transmissions_use_full_line_strengths_before_projection() -> None:
    density = np.asarray([[0.0, 1.2e14], [3.4e14, 0.7e14]])
    result = parallel_jones_column_response(
        density,
        DETUNING_HZ,
        SIGMA_0_M2,
        GAMMA_RAD_S,
        _response(),
    )
    phi_2l = np.asarray(
        scalar_phase_shift(DETUNING_HZ, density, SIGMA_0_M2, GAMMA_RAD_S)
    )
    od_2l = np.asarray(
        residual_optical_depth(DETUNING_HZ, density, SIGMA_0_M2, GAMMA_RAD_S)
    )

    np.testing.assert_allclose(result.branch_phase_maps_rad[0], phi_2l)
    np.testing.assert_allclose(result.branch_phase_maps_rad[1], phi_2l / 91.0)
    np.testing.assert_allclose(result.branch_optical_depth_maps[0], od_2l)
    np.testing.assert_allclose(result.branch_optical_depth_maps[1], od_2l / 91.0)
    np.testing.assert_allclose(result.common_phase_map_rad, (46.0 / 91.0) * phi_2l)
    np.testing.assert_allclose(result.faraday_rotation_map_rad, (-45.0 / 91.0) * phi_2l)
    np.testing.assert_allclose(result.common_optical_depth_map, (46.0 / 91.0) * od_2l)


def test_blank_density_returns_the_blank_linear_field() -> None:
    result = parallel_jones_column_response(
        np.zeros((3, 4)),
        DETUNING_HZ,
        SIGMA_0_M2,
        GAMMA_RAD_S,
        _response(),
    )

    np.testing.assert_array_equal(result.circular_transmission_fields, 1.0 + 0.0j)
    np.testing.assert_array_equal(result.co_polarised_field, 1.0 + 0.0j)
    np.testing.assert_array_equal(result.faraday_orthogonal_field, 0.0 + 0.0j)
    np.testing.assert_array_equal(result.total_intensity_fraction, 1.0)


def test_linear_and_circular_bases_give_the_same_transmitted_intensity() -> None:
    density = np.linspace(0.0, 8.0e14, 20).reshape(4, 5)
    result = parallel_jones_column_response(
        density,
        DETUNING_HZ,
        SIGMA_0_M2,
        GAMMA_RAD_S,
        _response(),
    )
    circular_average = np.mean(
        np.abs(result.circular_transmission_fields) ** 2,
        axis=0,
    )

    np.testing.assert_allclose(
        result.total_intensity_fraction,
        circular_average,
        rtol=2e-14,
        atol=2e-15,
    )


def test_weak_field_derivatives_recover_common_phase_and_faraday_factors() -> None:
    density = np.asarray([[1.0e8]])
    result = parallel_jones_column_response(
        density,
        DETUNING_HZ,
        SIGMA_0_M2,
        GAMMA_RAD_S,
        _response(),
    )
    phi_2l = float(
        scalar_phase_shift(DETUNING_HZ, density, SIGMA_0_M2, GAMMA_RAD_S)[0, 0]
    )

    assert np.imag(result.co_polarised_field[0, 0] - 1.0) / phi_2l == pytest.approx(
        46.0 / 91.0,
        rel=2e-7,
    )
    assert np.real(result.faraday_orthogonal_field[0, 0]) / phi_2l == pytest.approx(
        -45.0 / 91.0,
        rel=2e-7,
    )


def test_jones_geometry_preserves_the_existing_branch_summed_scattering() -> None:
    jones = _response()
    scalar = polarised_optical_response_from_config(_load(SCALAR_CONFIG))
    arguments = {
        "detuning_hz": DETUNING_HZ,
        "probe_power_mw": 1.0,
        "pulse_duration_s": 100e-6,
        "saturation_intensity_w_m2": 600.0,
        "natural_linewidth_rad_s": GAMMA_RAD_S,
        "probe_diameter_m": 160e-6,
    }
    jones_scattering = branch_summed_scattered_photons_per_atom(
        response=jones,
        **arguments,
    )
    scalar_scattering = branch_summed_scattered_photons_per_atom(
        response=scalar,
        **arguments,
    )

    assert jones.branch_weights == pytest.approx(scalar.branch_weights)
    assert jones_scattering.branch_saturation_parameters == pytest.approx(
        scalar_scattering.branch_saturation_parameters
    )
    assert jones_scattering.photons_per_atom_by_branch == pytest.approx(
        scalar_scattering.photons_per_atom_by_branch
    )
    assert jones_scattering.total_photons_per_atom == pytest.approx(
        scalar_scattering.total_photons_per_atom
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"quantisation_axis": (0.0, 1.0, 0.0)}, r"parallel to \+B"),
        ({"input_polarisation_axis": (1.0, 0.0, 0.0)}, "transverse"),
        ({"faraday_polarisation_axis": (0.0, -1.0, 0.0)}, "must equal"),
    ],
)
def test_parallel_jones_response_rejects_inconsistent_geometry(
    changes: dict[str, tuple[float, float, float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_response(), **changes)
