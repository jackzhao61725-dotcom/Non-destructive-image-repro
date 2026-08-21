from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from non_destructive_image.camera import resample_to_camera_pixels
from non_destructive_image.equilibrium_imaging import OpticalTransfer
from non_destructive_image.four_method_jones_imaging import (
    simulate_matched_four_method_jones_images,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads(
    (ROOT / "configs" / "dissertation_v3_orca_fusion.json").read_text(
        encoding="utf-8"
    )
)
ORIENTATION = json.loads(
    (ROOT / "configs" / "imaging_orientation_contract_v2.json").read_text(
        encoding="utf-8"
    )
)


def _simulate(density: np.ndarray, *, camera_pixel_m: float = 0.5e-6):
    axis = (np.arange(density.shape[0], dtype=float) - density.shape[0] // 2) * 0.25e-6
    transfer = OpticalTransfer(
        case_id="unit_test",
        model="unit_pupil",
        evidence_role="unit_test_only",
        transfer=np.ones_like(density),
    )
    return simulate_matched_four_method_jones_images(
        density,
        axis,
        axis,
        model_config=MODEL,
        orientation_config=ORIENTATION,
        optical_transfer=transfer,
        detuning_hz=1.5e9,
        camera_pixel_size_m=camera_pixel_m,
        phase_plate_transmittance=0.95,
        phase_plate_phase_rad=np.pi / 2.0,
        dgi_stop_optical_depth=4.0,
    )


def test_blank_closes_all_four_camera_readouts() -> None:
    result = _simulate(np.zeros((64, 64), dtype=float))

    np.testing.assert_allclose(result.pci.camera_intensity_over_i0, 0.95**2)
    np.testing.assert_allclose(result.pci.pci_signal_over_i0, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.dgi.camera_intensity_over_i0, 1e-4)
    np.testing.assert_allclose(result.dgi.dgi_signal_over_i0, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.dffi_camera_intensity_over_i0, 0.0)
    np.testing.assert_allclose(result.dpfi_h_camera_intensity_over_i0, 0.5)
    np.testing.assert_allclose(result.dpfi_v_camera_intensity_over_i0, 0.5)
    np.testing.assert_allclose(result.dpfi_normalised_difference, 0.0)


def test_camera_level_power_and_faraday_identities_close() -> None:
    axis = (np.arange(64, dtype=float) - 64 // 2) * 0.25e-6
    yy, zz = np.meshgrid(axis, axis)
    density = 2.7e15 * np.exp(
        -(yy / 2.7e-6) ** 2 / 2.0 - (zz / 1.1e-6) ** 2 / 2.0
    )
    result = _simulate(density)

    np.testing.assert_allclose(
        result.pci.camera_intensity_over_i0 - result.pci_co_camera_intensity_over_i0,
        result.dffi_camera_intensity_over_i0,
        rtol=2e-14,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result.dgi.camera_intensity_over_i0 - result.dgi_co_camera_intensity_over_i0,
        result.dffi_camera_intensity_over_i0,
        rtol=2e-14,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        result.dpfi_sum_camera_intensity_over_i0,
        result.open_co_camera_intensity_over_i0
        + result.dffi_camera_intensity_over_i0,
        rtol=2e-14,
        atol=2e-15,
    )
    assert np.max(np.abs(result.dpfi_normalised_difference)) <= 1.0
    assert np.min(result.dpfi_normalised_difference) < 0.0


def test_dpfi_ratio_is_formed_after_both_port_intensities_are_integrated() -> None:
    axis = (np.arange(64, dtype=float) - 64 // 2) * 0.25e-6
    yy, zz = np.meshgrid(axis, axis)
    density = 3.0e15 * np.exp(
        -(yy / 2.1e-6) ** 2 / 2.0 - (zz / 0.9e-6) ** 2 / 2.0
    )
    result = _simulate(density, camera_pixel_m=0.75e-6)

    h_camera = resample_to_camera_pixels(
        result.dpfi_h_optical_grid_intensity_over_i0,
        result.pci.input_pixel_size_m,
        result.pci.camera_pixel_size_m,
        result.pci.camera_intensity_over_i0.shape,
    )
    v_camera = resample_to_camera_pixels(
        result.dpfi_v_optical_grid_intensity_over_i0,
        result.pci.input_pixel_size_m,
        result.pci.camera_pixel_size_m,
        result.pci.camera_intensity_over_i0.shape,
    )
    expected = (h_camera - v_camera) / (h_camera + v_camera)
    np.testing.assert_allclose(
        result.dpfi_normalised_difference,
        expected,
        rtol=2e-14,
        atol=2e-15,
    )

    optical_ratio = (
        result.dpfi_h_optical_grid_intensity_over_i0
        - result.dpfi_v_optical_grid_intensity_over_i0
    ) / (
        result.dpfi_h_optical_grid_intensity_over_i0
        + result.dpfi_v_optical_grid_intensity_over_i0
    )
    wrong_order = resample_to_camera_pixels(
        optical_ratio,
        result.pci.input_pixel_size_m,
        result.pci.camera_pixel_size_m,
        result.pci.camera_intensity_over_i0.shape,
    )
    assert np.max(np.abs(result.dpfi_normalised_difference - wrong_order)) > 1e-6


def test_circular_to_linear_object_intensity_identity_is_retained() -> None:
    axis = (np.arange(64, dtype=float) - 64 // 2) * 0.25e-6
    yy, zz = np.meshgrid(axis, axis)
    density = 2.2e15 * np.exp(
        -(yy / 2.6e-6) ** 2 / 2.0 - (zz / 1.0e-6) ** 2 / 2.0
    )
    result = _simulate(density)

    np.testing.assert_allclose(
        np.abs(result.pci.co_polarised_object_field) ** 2
        + np.abs(result.pci.faraday_orthogonal_object_field) ** 2,
        np.mean(np.abs(result.circular_transmission_fields) ** 2, axis=0),
        rtol=2e-14,
        atol=2e-15,
    )
