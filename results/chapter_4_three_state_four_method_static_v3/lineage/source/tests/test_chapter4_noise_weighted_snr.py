"""Focused tests for the D4c analytic image-SNR contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from non_destructive_image.chapter4_noise_weighted_snr import (
    diagonal_noise_weighted_snr,
    dgi_frame_noise_model,
    exact_common_grid_indices,
    load_expected_npz,
    oxford_change_noise_model,
    pci_frame_noise_model,
    verify_retained_family,
    verify_scratch_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/chapter4_noise_weighted_image_snr_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _inputs():
    config = _config()
    three = load_expected_npz(ROOT / config["sources"]["three_state_arrays"])
    oxford = load_expected_npz(ROOT / config["sources"]["d4b_arrays"])
    return config, three, oxford


def _oxford_models(method: str):
    config, _, arrays = _inputs()
    read = config["statistic"]["read_noise_electrons_rms"]
    models = []
    for q in (1, 2, 3):
        if method == "pci":
            model = pci_frame_noise_model(
                arrays[f"raw_pci_P_q{q}_expected_electrons"],
                arrays["raw_pci_B_shared_expected_electrons"],
                arrays["raw_pci_D_shared_expected_electrons"],
                np.zeros((35, 111)),
                phase_plate_amplitude_transmittance=0.95,
                read_noise_electrons_rms=read,
            )
        else:
            model = dgi_frame_noise_model(
                arrays[f"raw_dgi_P_s_q{q}_expected_electrons"],
                arrays["raw_dgi_L_s_shared_expected_electrons"],
                arrays["raw_dgi_D_s_shared_expected_electrons"],
                arrays["raw_dgi_B_o_shared_expected_electrons"],
                arrays["raw_dgi_D_o_shared_expected_electrons"],
                np.zeros((35, 111)),
                open_to_stop_scale=1.0,
                read_noise_electrons_rms=read,
            )
        models.append(model)
    return models


def test_common_grid_is_an_exact_centered_subset_without_interpolation() -> None:
    _, three, oxford = _inputs()
    rows, columns = exact_common_grid_indices(
        three["camera_y_m"],
        three["camera_z_m"],
        oxford["camera_y_m"],
        oxford["camera_z_m"],
    )
    assert rows == slice(0, 35)
    assert columns == slice(20, 91)
    assert np.array_equal(oxford["camera_y_m"][columns], three["camera_y_m"])
    assert np.array_equal(oxford["camera_z_m"], three["camera_z_m"])


def test_diagonal_definition_matches_sum_and_is_unit_scaling_invariant() -> None:
    mu = np.array([1.0, -2.0, 0.5])
    variance = np.array([4.0, 9.0, 0.25])
    expected = np.sqrt(np.sum(mu**2 / variance))
    assert diagonal_noise_weighted_snr(mu, variance) == expected
    scale = 37.0
    assert np.isclose(
        diagonal_noise_weighted_snr(scale * mu, scale**2 * variance),
        expected,
        rtol=0.0,
        atol=2e-16,
    )


def test_blank_zero_signal_pixels_do_not_change_snr() -> None:
    mu = np.array([1.0, 2.0])
    variance = np.array([2.0, 3.0])
    baseline = diagonal_noise_weighted_snr(mu, variance)
    padded = diagonal_noise_weighted_snr(
        np.concatenate([mu, np.zeros(10)]),
        np.concatenate([variance, np.linspace(0.1, 10.0, 10)]),
    )
    assert padded == baseline


def test_pci_and_dgi_expected_jacobian_variances_match_manual_examples() -> None:
    read2 = 0.7**2
    pci = pci_frame_noise_model(
        np.array([[12.0]]),
        np.array([[10.0]]),
        np.array([[1.0]]),
        np.array([[0.0]]),
        phase_plate_amplitude_transmittance=0.95,
        read_noise_electrons_rms=0.7,
    )
    tp2 = 0.95**2
    denominator = 9.0
    jac = np.array([tp2 / denominator, -tp2 * 11.0 / denominator**2, tp2 * 2.0 / denominator**2])
    manual = np.sum(jac**2 * np.array([12.0 + read2, 10.0 + read2, 1.0 + read2]))
    assert np.isclose(pci.variance.item(), manual, rtol=1e-15, atol=0.0)

    dgi = dgi_frame_noise_model(
        np.array([[3.0]]),
        np.array([[1.0]]),
        np.array([[0.2]]),
        np.array([[20.0]]),
        np.array([[2.0]]),
        np.array([[0.0]]),
        open_to_stop_scale=1.0,
        read_noise_electrons_rms=0.7,
    )
    expected_signal = 2.0 / 18.0
    jac_dgi = np.array([1 / 18, -1 / 18, 0.0, -expected_signal / 18, expected_signal / 18])
    manual_dgi = np.sum(
        jac_dgi**2
        * np.array([3.0 + read2, 1.0 + read2, 0.2 + read2, 20.0 + read2, 2.0 + read2])
    )
    assert np.isclose(dgi.variance.item(), manual_dgi, rtol=1e-15, atol=0.0)


def test_oxford_shared_references_create_cross_frame_covariance() -> None:
    for method in ("pci", "dgi"):
        first, later, _ = _oxford_models(method)
        change = oxford_change_noise_model(later, first)
        assert np.any(change.cross_covariance > 0.0)
        assert not np.array_equal(change.variance, change.independent_frame_variance)
        atom_expected = (
            later.role_variance_contributions["P" if method == "pci" else "P_s"]
            + first.role_variance_contributions["P" if method == "pci" else "P_s"]
        )
        assert np.array_equal(change.contributions["atom_later"] + change.contributions["atom_q1"], atom_expected)


def test_dgi_change_cancels_leakage_and_stop_dark_but_keeps_open_denominator() -> None:
    first, _, later = _oxford_models("dgi")
    change = oxford_change_noise_model(later, first)
    assert np.count_nonzero(change.contributions["L_s_shared"]) == 0
    assert np.count_nonzero(change.contributions["D_s_shared"]) == 0
    assert np.any(change.contributions["B_o_shared"] > 0.0)
    assert np.any(change.contributions["D_o_shared"] > 0.0)


def test_observed_fixed_draw_arrays_are_structurally_excluded_and_replay_is_exact() -> None:
    config, _, arrays = _inputs()
    assert arrays
    assert not any("observed" in name for name in arrays)
    first = _oxford_models("pci")
    second = _oxford_models("pci")
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.mu, right.mu)
        assert np.array_equal(left.variance, right.variance)
    assert config["statistic"]["observed_fixed_draw_consumed"] is False


def test_all_source_manifests_authenticate_with_hashes_and_bytes() -> None:
    config = _config()
    assert len(verify_retained_family(ROOT, ROOT / config["sources"]["three_state_manifest"])) == 9
    assert len(verify_scratch_manifest(ROOT, ROOT / config["sources"]["d4a_manifest"], 6)) == 6
    assert len(verify_scratch_manifest(ROOT, ROOT / config["sources"]["d4b_manifest"], 7)) == 7
