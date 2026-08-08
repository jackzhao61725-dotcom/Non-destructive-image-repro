"""Contract tests for the code-only reproduction entry points."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import reproduce_forward_model, reproduce_inference
from scripts._common import endpoint_products, load_configs


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_public_configs_build_independent_endpoints() -> None:
    model, reference, reproduction = load_configs(CONFIGS)
    first, second = endpoint_products(model, reference, reproduction)

    assert first.spec.label == "B_parallel_y"
    assert second.spec.label == "B_parallel_z"
    assert first.state is not second.state
    assert first.canonical_operator is not second.canonical_operator
    assert np.all(np.asarray(first.state.radii_m) > 0.0)
    assert np.all(np.asarray(second.state.radii_m) > 0.0)


def test_forward_payload_is_finite_and_tracks_orientation_contrast() -> None:
    payload = reproduce_forward_model.reproduce(CONFIGS)

    contrast = payload["orientation_contrast_um"]
    assert contrast["delta_sigma_y"] > 0.0
    assert contrast["delta_sigma_z"] > 0.0
    assert payload["reference_probe"]["total_scattered_photons_per_atom"] > 0.0
    assert len(payload["conditional_thermodynamic_sequence"]["states"]) >= 2


def test_inference_seed_tree_repeats_one_raw_block() -> None:
    _model, _reference, reproduction = load_configs(CONFIGS)
    mean = tuple(np.full((3, 4), 10.0 + index) for index in range(3))
    first = reproduce_inference._raw_block(
        reproduction,
        expected=mean,
        read_noise_electrons=0.7,
        fluence_index=16,
        draw_id=0,
        endpoint_index=0,
    )
    second = reproduce_inference._raw_block(
        reproduction,
        expected=mean,
        read_noise_electrons=0.7,
        fluence_index=16,
        draw_id=0,
        endpoint_index=0,
    )
    third = reproduce_inference._raw_block(
        reproduction,
        expected=mean,
        read_noise_electrons=0.7,
        fluence_index=16,
        draw_id=1,
        endpoint_index=0,
    )

    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            first.observed_electrons, second.observed_electrons, strict=True
        )
    )
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            first.observed_electrons, third.observed_electrons, strict=True
        )
    )


def test_estimator_contract_has_four_bounded_starts() -> None:
    _model, _reference, reproduction = load_configs(CONFIGS)
    ids, starts, lower, upper = reproduce_inference._parameter_contract(reproduction)

    assert ids == ("neutral", "low_peak", "high_peak", "shifted")
    assert len(starts) == 4
    assert all(np.all(vector >= lower) and np.all(vector <= upper) for vector in starts)
