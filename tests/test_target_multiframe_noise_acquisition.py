from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from isolated_non_destructive_image import load_isolated_non_destructive_image_module


ROOT = Path(__file__).resolve().parents[1]
MODULE = load_isolated_non_destructive_image_module(
    "target_multiframe_noise_acquisition",
    namespace="_test_target_multiframe_noise_acquisition",
)


def _expected(frame_count: int = 3):
    shape = (3, 5)
    dgi = {q: np.full(shape, 0.1 + 0.01 * q) for q in range(1, frame_count + 1)}
    h_port = {q: np.full(shape, 0.55 + 0.01 * q) for q in range(1, frame_count + 1)}
    v_port = {q: np.full(shape, 0.45 - 0.01 * q) for q in range(1, frame_count + 1)}
    return MODULE.expected_sequence_roles(
        dgi_camera_intensity_over_i0=dgi,
        dpfi_h_camera_intensity_over_i0=h_port,
        dpfi_v_camera_intensity_over_i0=v_port,
        photoelectrons_per_i0_pixel=200.0,
    )


def test_full_grid_random_identity_inventory_is_frozen() -> None:
    assert MODULE.random_identity_count(
        duration_count=16, draw_count=64, frame_count=15
    ) == 54_272
    identities = []
    prefix = (20260818, 6317)
    for duration in range(25, 401, 25):
        for draw_id in range(64):
            for q in range(1, 16):
                identities.append(
                    MODULE.seed_tuple(
                        prefix,
                        duration_us=duration,
                        method="dgi",
                        draw_id=draw_id,
                        role="atom_stop",
                        image_q=q,
                    )
                )
                for role in ("atom_h", "atom_v"):
                    identities.append(
                        MODULE.seed_tuple(
                            prefix,
                            duration_us=duration,
                            method="dpfi",
                            draw_id=draw_id,
                            role=role,
                            image_q=q,
                        )
                    )
            for method, roles in (
                ("dgi", ("leakage_stop", "stop_dark", "open_reference", "open_dark")),
                ("dpfi", ("blank_h", "blank_v", "dark_h", "dark_v")),
            ):
                for role in roles:
                    identities.append(
                        MODULE.seed_tuple(
                            prefix,
                            duration_us=duration,
                            method=method,
                            draw_id=draw_id,
                            role=role,
                            image_q=0,
                        )
                    )
    assert len(identities) == 54_272
    assert len(set(identities)) == 54_272


def test_expected_means_close_to_method_processing_without_noise() -> None:
    expected = _expected()
    dgi = MODULE.process_dgi_sequence(expected["dgi"], (1, 2, 3))
    dpfi = MODULE.process_dpfi_sequence(expected["dpfi"], (1, 2, 3))
    for q in (1, 2, 3):
        np.testing.assert_allclose(dgi[q], 0.1 + 0.01 * q - 1.0e-4)
        np.testing.assert_allclose(dpfi[q], 0.1 + 0.02 * q)


def test_one_sequence_samples_shared_roles_once_and_ports_independently() -> None:
    expected = _expected()
    sampled, ledger = MODULE.sample_sequence(
        expected,
        prefix=(20260818, 6317),
        duration_us=200,
        draw_id=7,
        read_noise_electrons_rms=0.7,
    )
    assert len(ledger) == (3 + 4) + (2 * 3 + 4)
    assert len({row["seed"] for row in ledger}) == len(ledger)
    assert sum(row["role"] == "open_reference" for row in ledger) == 1
    assert sum(row["role"] == "blank_h" for row in ledger) == 1
    h_seed = next(row["seed"] for row in ledger if row["role"] == "atom_h" and row["image_q"] == 1)
    v_seed = next(row["seed"] for row in ledger if row["role"] == "atom_v" and row["image_q"] == 1)
    assert h_seed != v_seed
    dgi = MODULE.process_dgi_sequence(sampled["dgi"], (1, 2, 3))
    dpfi = MODULE.process_dpfi_sequence(sampled["dpfi"], (1, 2, 3))
    assert all(values.shape == (3, 5) and np.isfinite(values).all() for values in dgi.values())
    assert all(values.shape == (3, 5) and np.isfinite(values).all() for values in dpfi.values())


def test_fifteen_frame_sequence_has_exactly_fifty_three_roles() -> None:
    expected = _expected(frame_count=15)
    sampled, ledger = MODULE.sample_sequence(
        expected,
        prefix=(20260818, 6317),
        duration_us=200,
        draw_id=0,
        read_noise_electrons_rms=0.7,
    )
    assert len(ledger) == 53
    assert len(sampled["dgi"]) == 19
    assert len(sampled["dpfi"]) == 34
    assert [row["image_q"] for row in ledger if row["role"] == "atom_stop"] == list(
        range(1, 16)
    )
    assert len({row["seed"] for row in ledger}) == 53


def test_axial_profile_sums_camera_z_rows() -> None:
    values = np.arange(15, dtype=float).reshape(3, 5)
    np.testing.assert_array_equal(MODULE.axial_profile(values), np.sum(values, axis=0))


def test_direct_array_apis_reject_empty_camera_arrays() -> None:
    empty = np.empty((0, 5))
    with pytest.raises(ValueError, match="expected camera electrons"):
        MODULE.sample_raw_frame(
            empty,
            read_noise_electrons_rms=0.7,
            seed=(20260818, 6317, 200, 1, 0, 1, 1),
        )
    with pytest.raises(ValueError, match="invalid camera signal"):
        MODULE.axial_profile(empty)


def test_axial_profile_rejects_finite_input_that_overflows() -> None:
    values = np.full((3, 4), np.finfo(float).max)
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="axial profile is non-finite"):
            MODULE.axial_profile(values)


def test_fractional_identity_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="duration_count must be an integer"):
        MODULE.random_identity_count(duration_count=16.5, draw_count=64, frame_count=15)
    with pytest.raises(ValueError, match="duration_us must be an integer"):
        MODULE.seed_tuple(
            (20260818, 6317),
            duration_us=200.5,
            method="dgi",
            draw_id=0,
            role="atom_stop",
            image_q=1,
        )
    with pytest.raises(ValueError, match="seed prefix value must be an integer"):
        MODULE.seed_tuple(
            (20260818, 6317.5),
            duration_us=200,
            method="dgi",
            draw_id=0,
            role="atom_stop",
            image_q=1,
        )


def test_incomplete_role_topology_is_rejected_before_sampling() -> None:
    expected = _expected()
    calibration_only = {
        "dgi": {key: value for key, value in expected["dgi"].items() if not key.startswith("atom_")},
        "dpfi": {key: value for key, value in expected["dpfi"].items() if not key.startswith("atom_")},
    }
    with pytest.raises(ValueError, match="DGI atom-frame inventory is empty"):
        MODULE.sample_sequence(
            calibration_only,
            prefix=(20260818, 6317),
            duration_us=200,
            draw_id=0,
            read_noise_electrons_rms=0.7,
        )


def test_process_rejects_missing_nan_and_mismatched_roles() -> None:
    expected = _expected()
    missing_stop_dark = dict(expected["dgi"])
    del missing_stop_dark["stop_dark"]
    with pytest.raises(ValueError, match="raw-role inventory changed"):
        MODULE.process_dgi_sequence(missing_stop_dark, (1, 2, 3))

    nan_atom = {key: value.copy() for key, value in expected["dgi"].items()}
    nan_atom["atom_stop_q1"][0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid raw-role array"):
        MODULE.process_dgi_sequence(nan_atom, (1, 2, 3))

    bad_shape = {key: value.copy() for key, value in expected["dpfi"].items()}
    bad_shape["atom_h_q2"] = np.ones((1, 5))
    with pytest.raises(ValueError, match="raw-role shape changed"):
        MODULE.process_dpfi_sequence(bad_shape, (1, 2, 3))


def test_expected_roles_require_positive_consecutive_integer_frames() -> None:
    shape = (3, 5)
    with pytest.raises(ValueError, match="image_q must be consecutive from one"):
        MODULE.expected_sequence_roles(
            dgi_camera_intensity_over_i0={1: np.ones(shape), 3: np.ones(shape)},
            dpfi_h_camera_intensity_over_i0={1: np.ones(shape), 3: np.ones(shape)},
            dpfi_v_camera_intensity_over_i0={1: np.ones(shape), 3: np.ones(shape)},
            photoelectrons_per_i0_pixel=200.0,
        )
    with pytest.raises(ValueError, match="image_q must be an integer"):
        MODULE.expected_sequence_roles(
            dgi_camera_intensity_over_i0={1.0: np.ones(shape)},
            dpfi_h_camera_intensity_over_i0={1.0: np.ones(shape)},
            dpfi_v_camera_intensity_over_i0={1.0: np.ones(shape)},
            photoelectrons_per_i0_pixel=200.0,
        )
    empty = np.empty((0, 5))
    with pytest.raises(ValueError, match="invalid expected camera intensity"):
        MODULE.expected_sequence_roles(
            dgi_camera_intensity_over_i0={1: empty},
            dpfi_h_camera_intensity_over_i0={1: empty},
            dpfi_v_camera_intensity_over_i0={1: empty},
            photoelectrons_per_i0_pixel=200.0,
        )
