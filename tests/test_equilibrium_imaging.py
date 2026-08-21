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


_IMAGING = load_isolated_non_destructive_image_module(
    "equilibrium_imaging",
    namespace="_ndi_corrected_acquisition_scientific_tests_v1",
)
_PROFILES = load_isolated_non_destructive_image_module(
    "equilibrium_profiles",
    namespace="_ndi_corrected_acquisition_scientific_tests_v1",
)
gaussian_amplitude_psf_transfer = _IMAGING.gaussian_amplitude_psf_transfer
hard_circular_coherent_transfer = _IMAGING.hard_circular_coherent_transfer
incident_photoelectrons_per_i0_pixel = _IMAGING.incident_photoelectrons_per_i0_pixel
optical_transfer_from_objective_config = _IMAGING.optical_transfer_from_objective_config
recover_pci_morphology = _IMAGING.recover_pci_morphology
simulate_matched_pci_image = _IMAGING.simulate_matched_pci_image
simulate_noisy_pci_difference = _IMAGING.simulate_noisy_pci_difference
EquilibriumProfileDefinition = _PROFILES.EquilibriumProfileDefinition
build_equilibrium_profile = _PROFILES.build_equilibrium_profile


MODEL = json.loads(Path("configs/dissertation_v3_orca_fusion.json").read_text(encoding="utf-8"))
OBJECTIVE = json.loads(Path("configs/erk_401nm_objective_v1.json").read_text(encoding="utf-8"))


def test_design_and_measured_transfers_have_distinct_models() -> None:
    shape = (256, 256)
    pixel_m = 40e-6 / 256
    design = optical_transfer_from_objective_config(OBJECTIVE, "design", shape, pixel_m)
    measured = optical_transfer_from_objective_config(
        OBJECTIVE,
        "measured_best",
        shape,
        pixel_m,
    )

    assert design.numerical_aperture == pytest.approx(0.31)
    assert design.resolution_m is None
    assert set(np.unique(design.transfer)) <= {0.0, 1.0}
    assert measured.numerical_aperture is None
    assert measured.resolution_m == pytest.approx(0.92e-6)
    assert np.any((measured.transfer > 0.0) & (measured.transfer < 1.0))
    assert not np.array_equal(design.transfer, measured.transfer)


def test_transfer_formulas_fix_cutoff_and_gaussian_resolution_mapping() -> None:
    shape = (128, 128)
    pixel_m = 0.1e-6
    wavelength_m = 401e-9
    pupil = hard_circular_coherent_transfer(
        shape,
        pixel_m,
        wavelength_m=wavelength_m,
        numerical_aperture=0.31,
    )
    frequency = np.fft.fftfreq(shape[0], d=pixel_m)
    nonnegative = np.flatnonzero(frequency >= 0.0)
    inside = nonnegative[
        np.flatnonzero(frequency[nonnegative] <= 0.31 / wavelength_m)[-1]
    ]
    outside = inside + 1
    assert pupil[0, inside] == 1.0
    assert pupil[0, outside] == 0.0

    resolution_m = 0.92e-6
    gaussian = gaussian_amplitude_psf_transfer(
        shape,
        pixel_m,
        resolution_m=resolution_m,
    )
    index = 4
    sigma_m = resolution_m / 2.9039
    expected = np.exp(-2.0 * np.pi**2 * sigma_m**2 * frequency[index] ** 2)
    assert gaussian[0, 0] == pytest.approx(1.0)
    assert gaussian[0, index] == pytest.approx(expected, rel=1e-14)


def _connected_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis_m = (np.arange(256, dtype=float) - 128) * (30e-6 / 256)
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    definition = EquilibriumProfileDefinition(
        state_id="connected_modulated",
        label="Connected density-modulated profile",
        morphology="connected_modulated",
        atom_number=5.0e4,
        component_centres_y_m=(-4e-6, 0.0, 4e-6),
        component_weights=(0.82, 1.0, 0.82),
        component_sigma_y_m=1.35e-6,
        component_sigma_z_m=1.0e-6,
    )
    profile = build_equilibrium_profile(
        definition,
        y_grid_m,
        z_grid_m,
        minimum_peak_distance_m=2.4e-6,
        peak_prominence_fraction=0.05,
    )
    return axis_m, axis_m, profile.column_density_m2


def test_matched_pci_path_reuses_atomic_response_and_physical_camera_sampling() -> None:
    y_axis_m, z_axis_m, density = _connected_profile()
    transfer = optical_transfer_from_objective_config(
        OBJECTIVE,
        "measured_best",
        density.shape,
        float(np.diff(y_axis_m).mean()),
    )
    camera_pixel_m = OBJECTIVE["detector_sampling"]["object_plane_pixel_pitch_m"]
    image = simulate_matched_pci_image(
        density,
        y_axis_m,
        z_axis_m,
        model_config=MODEL,
        optical_transfer=transfer,
        detuning_hz=1.5e9,
        camera_pixel_size_m=camera_pixel_m,
        phase_plate_transmittance=0.95,
        phase_plate_phase_rad=np.pi / 2,
    )

    assert image.phase_map_rad.shape == density.shape
    assert np.max(image.phase_map_rad) > 0.0
    assert np.all(image.optical_depth_map >= 0.0)
    assert image.camera_intensity_over_i0.shape == (
        image.camera_z_m.size,
        image.camera_y_m.size,
    )
    assert image.atom_free_intensity_over_i0 == pytest.approx(0.95**2)
    recovered = recover_pci_morphology(
        image.pci_signal_over_i0,
        image.camera_y_m,
        image.camera_z_m,
        analysis_half_width_y_m=10e-6,
        analysis_half_width_z_m=4e-6,
        minimum_peak_distance_m=2.4e-6,
        peak_prominence_fraction=0.05,
    )
    assert recovered.peak_count == 3
    assert recovered.mean_peak_spacing_m == pytest.approx(4e-6, abs=camera_pixel_m)


def test_fixed_seed_raw_pci_draw_is_replayable() -> None:
    y_axis_m, z_axis_m, density = _connected_profile()
    transfer = optical_transfer_from_objective_config(
        OBJECTIVE,
        "measured_best",
        density.shape,
        float(np.diff(y_axis_m).mean()),
    )
    camera_pixel_m = OBJECTIVE["detector_sampling"]["object_plane_pixel_pitch_m"]
    image = simulate_matched_pci_image(
        density,
        y_axis_m,
        z_axis_m,
        model_config=MODEL,
        optical_transfer=transfer,
        detuning_hz=1.5e9,
        camera_pixel_size_m=camera_pixel_m,
        phase_plate_transmittance=0.95,
        phase_plate_phase_rad=np.pi / 2,
    )
    count_scale = incident_photoelectrons_per_i0_pixel(
        MODEL,
        camera_pixel_size_m=camera_pixel_m,
        probe_power_mw=1.0,
        pulse_duration_s=175e-6,
        quantum_efficiency=0.65,
    )
    first = simulate_noisy_pci_difference(
        image,
        photoelectrons_per_i0_pixel=count_scale,
        read_noise_electrons_rms=0.7,
        seed_components_prefix=(20260812, 1),
        camera_contract_id="camera",
        sampling_contract_id="sampling",
    )
    second = simulate_noisy_pci_difference(
        image,
        photoelectrons_per_i0_pixel=count_scale,
        read_noise_electrons_rms=0.7,
        seed_components_prefix=(20260812, 1),
        camera_contract_id="camera",
        sampling_contract_id="sampling",
    )

    assert count_scale > 0.0
    np.testing.assert_array_equal(first.observed_signal_over_i0, second.observed_signal_over_i0)
    np.testing.assert_allclose(
        first.expected_signal_over_i0,
        image.pci_signal_over_i0,
        rtol=0.0,
        atol=1e-15,
    )
