from __future__ import annotations

import numpy as np

from non_destructive_image.four_method_camera_noise import (
    dffi_frame_noise_model,
    dpfi_frame_noise_model,
    process_dffi_counts,
    process_dpfi_counts,
)


def _constant(value: float) -> np.ndarray:
    return np.full((3, 4), value, dtype=float)


def test_dffi_ideal_blank_has_read_noise_but_no_photon_noise() -> None:
    count_scale = 100.0
    read_noise = 0.7
    model = dffi_frame_noise_model(
        _constant(0.0),
        _constant(0.0),
        _constant(0.0),
        _constant(count_scale),
        _constant(0.0),
        read_noise_electrons_rms=read_noise,
    )

    assert np.all(model.expected_signal == 0.0)
    assert np.allclose(model.variance, 2.0 * read_noise**2 / count_scale**2)
    assert np.all(model.raw_variances["crossed_atom"] == read_noise**2)
    assert np.all(model.raw_variances["crossed_blank"] == read_noise**2)
    assert np.all(model.jacobians["crossed_dark"] == 0.0)


def test_dffi_processing_and_expected_signal_match() -> None:
    model = dffi_frame_noise_model(
        _constant(12.0),
        _constant(2.0),
        _constant(0.0),
        _constant(100.0),
        _constant(0.0),
        read_noise_electrons_rms=0.7,
    )
    processed = process_dffi_counts(
        _constant(12.0),
        _constant(2.0),
        _constant(100.0),
        _constant(0.0),
    )
    assert np.allclose(model.expected_signal, 0.1)
    assert np.array_equal(processed, model.expected_signal)


def test_dpfi_ports_are_combined_only_after_separate_camera_sampling() -> None:
    model = dpfi_frame_noise_model(
        _constant(40.0),
        _constant(60.0),
        _constant(50.0),
        _constant(50.0),
        _constant(0.0),
        _constant(0.0),
        read_noise_electrons_rms=0.7,
    )
    processed = process_dpfi_counts(
        _constant(40.0),
        _constant(60.0),
        _constant(50.0),
        _constant(50.0),
        _constant(0.0),
        _constant(0.0),
    )
    assert np.allclose(model.expected_signal, -0.2)
    assert np.array_equal(processed, model.expected_signal)


def test_dpfi_balanced_mean_does_not_cancel_port_shot_noise() -> None:
    count_scale = 100.0
    read_noise = 0.7
    model = dpfi_frame_noise_model(
        _constant(count_scale / 2.0),
        _constant(count_scale / 2.0),
        _constant(count_scale / 2.0),
        _constant(count_scale / 2.0),
        _constant(0.0),
        _constant(0.0),
        read_noise_electrons_rms=read_noise,
    )

    expected_variance = (count_scale + 4.0 * read_noise**2) / count_scale**2
    assert np.all(model.expected_signal == 0.0)
    assert np.allclose(model.variance, expected_variance)
    atom_port_contribution = (
        model.jacobians["atom_h"] ** 2 * model.raw_variances["atom_h"]
        + model.jacobians["atom_v"] ** 2 * model.raw_variances["atom_v"]
    )
    assert np.all(atom_port_contribution > 0.0)


def test_dpfi_dark_jacobians_include_the_shared_normalisation() -> None:
    model = dpfi_frame_noise_model(
        _constant(40.0),
        _constant(60.0),
        _constant(50.0),
        _constant(50.0),
        _constant(0.0),
        _constant(0.0),
        read_noise_electrons_rms=0.7,
    )
    assert np.allclose(model.jacobians["dark_h"], -0.012)
    assert np.allclose(model.jacobians["dark_v"], 0.008)
    assert model.image_snr > 0.0
