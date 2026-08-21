from __future__ import annotations

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


_EQUILIBRIUM = load_isolated_non_destructive_image_module(
    "equilibrium_imaging",
    namespace="_ndi_corrected_acquisition_scientific_tests_v1",
)
_IMAGING = load_isolated_non_destructive_image_module(
    "imaging", namespace="_ndi_corrected_acquisition_scientific_tests_v1"
)
_PARALLEL = load_isolated_non_destructive_image_module(
    "parallel_jones_imaging",
    namespace="_ndi_corrected_acquisition_scientific_tests_v1",
)
OpticalTransfer = _EQUILIBRIUM.OpticalTransfer
simulate_dffi_jones_fields = _IMAGING.simulate_dffi_jones_fields
simulate_dgi_jones_fields = _IMAGING.simulate_dgi_jones_fields
simulate_dgi_object_field = _IMAGING.simulate_dgi_object_field
simulate_dpfi_jones_fields = _IMAGING.simulate_dpfi_jones_fields
simulate_pci_jones_fields = _IMAGING.simulate_pci_jones_fields
simulate_pci_object_field = _IMAGING.simulate_pci_object_field
simulate_matched_jones_dgi_image = _PARALLEL.simulate_matched_jones_dgi_image
simulate_matched_jones_pci_image = _PARALLEL.simulate_matched_jones_pci_image


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


def test_unit_pupil_jones_readouts_match_the_exact_two_polarisation_sum() -> None:
    co = np.array(
        [
            [1.0 + 0.0j, 0.91 + 0.08j],
            [0.87 + 0.13j, 0.96 - 0.04j],
        ]
    )
    faraday = np.array(
        [
            [0.0 + 0.0j, -0.03 - 0.09j],
            [0.02 - 0.12j, 0.01 + 0.04j],
        ]
    )
    pupil = np.ones_like(co)

    pci = simulate_pci_jones_fields(co, faraday, pupil, 0.95, np.pi / 2.0)
    pci_reference = 0.95j
    np.testing.assert_allclose(
        pci,
        np.abs(pci_reference + co - 1.0) ** 2 + np.abs(faraday) ** 2,
        rtol=2e-15,
        atol=2e-15,
    )

    dgi = simulate_dgi_jones_fields(co, faraday, pupil, 4.0)
    dgi_reference = 1e-2
    np.testing.assert_allclose(
        dgi,
        np.abs(dgi_reference + co - 1.0) ** 2 + np.abs(faraday) ** 2,
        rtol=2e-15,
        atol=2e-15,
    )


def test_zero_faraday_component_recovers_the_existing_scalar_readouts() -> None:
    y = np.linspace(-1.0, 1.0, 16)
    yy, zz = np.meshgrid(y, y)
    object_field = np.exp(-0.02 * np.exp(-(yy**2 + zz**2)) + 0.3j * yy)
    faraday = np.zeros_like(object_field)
    pupil = np.exp(-0.1 * (yy**2 + zz**2))

    np.testing.assert_allclose(
        simulate_pci_jones_fields(object_field, faraday, pupil, 0.95, np.pi / 2),
        simulate_pci_object_field(object_field, pupil, 0.95, np.pi / 2),
        rtol=2e-15,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        simulate_dgi_jones_fields(object_field, faraday, pupil, 4.0),
        simulate_dgi_object_field(object_field, pupil, 4.0),
        rtol=2e-15,
        atol=2e-15,
    )


def test_faraday_intensity_is_added_without_reference_interference() -> None:
    co = np.ones((8, 8), dtype=complex)
    faraday = np.full((8, 8), 0.2 - 0.1j)
    pupil = np.ones((8, 8))

    pci = simulate_pci_jones_fields(
        co,
        faraday,
        pupil,
        0.95,
        np.pi / 2,
        return_intermediates=True,
    )
    dgi = simulate_dgi_jones_fields(
        co,
        faraday,
        pupil,
        4.0,
        return_intermediates=True,
    )

    faraday_intensity = np.abs(faraday) ** 2
    np.testing.assert_allclose(
        pci["total_image_intensity"] - np.abs(pci["reference_field"]) ** 2,
        faraday_intensity,
    )
    np.testing.assert_allclose(
        dgi["total_image_intensity"] - np.abs(dgi["reference_field"]) ** 2,
        faraday_intensity,
    )


def test_faraday_readouts_project_the_same_propagated_jones_field() -> None:
    co = np.full((8, 8), 0.9 + 0.2j)
    faraday = np.full((8, 8), -0.08 - 0.16j)
    pupil = np.ones((8, 8))

    dffi = simulate_dffi_jones_fields(co, faraday, pupil)
    dpfi = simulate_dpfi_jones_fields(co, faraday, pupil)

    np.testing.assert_allclose(dffi, np.abs(faraday) ** 2)
    expected_h = np.abs(co + faraday) ** 2 / 2.0
    expected_v = np.abs(co - faraday) ** 2 / 2.0
    np.testing.assert_allclose(dpfi["analyser_h_intensity"], expected_h)
    np.testing.assert_allclose(dpfi["analyser_v_intensity"], expected_v)
    np.testing.assert_allclose(
        dpfi["dual_port_signal"],
        (expected_h - expected_v) / (expected_h + expected_v),
    )


def test_blank_jones_field_gives_dark_dffi_and_balanced_bright_dpfi_ports() -> None:
    co = np.ones((8, 8), dtype=complex)
    faraday = np.zeros((8, 8), dtype=complex)
    pupil = np.ones((8, 8))

    dffi = simulate_dffi_jones_fields(co, faraday, pupil)
    dpfi = simulate_dpfi_jones_fields(co, faraday, pupil)

    np.testing.assert_array_equal(dffi, np.zeros_like(dffi))
    np.testing.assert_allclose(dpfi["analyser_h_intensity"], 0.5)
    np.testing.assert_allclose(dpfi["analyser_v_intensity"], 0.5)
    np.testing.assert_allclose(dpfi["dual_port_signal"], 0.0)


def test_matched_three_state_path_uses_exact_jones_fields_for_both_methods() -> None:
    axis = (np.arange(64, dtype=float) - 64 // 2) * 0.25e-6
    y_grid, z_grid = np.meshgrid(axis, axis)
    density = 2.5e15 * np.exp(
        -(y_grid / 2.5e-6) ** 2 / 2.0 - (z_grid / 1.2e-6) ** 2 / 2.0
    )
    transfer = OpticalTransfer(
        case_id="measured_best",
        model="unit_test_identity",
        evidence_role="unit_test_only",
        transfer=np.ones_like(density),
    )

    pci = simulate_matched_jones_pci_image(
        density,
        axis,
        axis,
        model_config=MODEL,
        orientation_config=ORIENTATION,
        optical_transfer=transfer,
        detuning_hz=1.5e9,
        camera_pixel_size_m=0.25e-6,
        phase_plate_transmittance=0.95,
        phase_plate_phase_rad=np.pi / 2.0,
    )
    dgi = simulate_matched_jones_dgi_image(
        pci,
        stop_optical_depth=4.0,
    )

    assert np.max(pci.common_phase_map_rad) > 0.0
    assert np.min(pci.faraday_rotation_map_rad) < 0.0
    assert np.max(np.abs(pci.faraday_orthogonal_object_field)) > 0.0
    np.testing.assert_allclose(
        pci.total_object_intensity_fraction,
        np.abs(pci.co_polarised_object_field) ** 2
        + np.abs(pci.faraday_orthogonal_object_field) ** 2,
        rtol=2e-14,
        atol=2e-15,
    )
    assert dgi.co_polarised_object_field is pci.co_polarised_object_field
    assert dgi.faraday_orthogonal_object_field is pci.faraday_orthogonal_object_field
    assert dgi.camera_intensity_over_i0.shape == pci.camera_intensity_over_i0.shape
    assert dgi.camera_pixel_size_m == pci.camera_pixel_size_m
    assert pci.atom_free_intensity_over_i0 == pytest.approx(0.95**2)
    assert dgi.atom_free_intensity_over_i0 == pytest.approx(1e-4)
    assert np.isfinite(pci.camera_intensity_over_i0).all()
    assert np.isfinite(dgi.camera_intensity_over_i0).all()

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        simulate_matched_jones_dgi_image(
            pci,
            camera_pixel_size_m=0.5e-6,  # type: ignore[call-arg]
            stop_optical_depth=4.0,
        )


def test_jones_readout_rejects_mismatched_polarisation_shapes() -> None:
    with pytest.raises(ValueError, match="same 2D shape"):
        simulate_pci_jones_fields(
            np.ones((8, 8), dtype=complex),
            np.zeros((7, 8), dtype=complex),
            np.ones((8, 8)),
        )
