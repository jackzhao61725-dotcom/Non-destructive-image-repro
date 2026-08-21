from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_target_four_method_noise.py"
OUTPUT = ROOT / ".scratch/target_three_state_four_method_noise_v5"


def _module():
    specification = importlib.util.spec_from_file_location("target_four_method_noise_runner", RUNNER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_noise_contract_freezes_200us_and_independent_dpfi_ports() -> None:
    module = _module()
    config = module.validate_config()
    assert config["display_pulse_duration_us"] == 200
    assert config["randomness"]["expected_random_identities"] == 57
    assert config["acquisition"]["dpfi_port_policy"].startswith("H_and_V_are_camera_sampled")
    assert config["presentation"]["noisy_scale"].startswith("one_zero_centred_scale")
    assert config["detector"]["full_well_gate"] == "maximum_expected_plus_6sigma_below_full_well"


def test_generated_noise_candidate_replays_when_present() -> None:
    if not OUTPUT.exists():
        pytest.skip("target four-method noise candidate has not been generated")
    module = _module()
    summary = module.validate_output(OUTPUT, module.validate_config())
    assert summary["random_identities"] == 57
    assert summary["redraws"] == 0
    assert summary["display_pulse_duration_us"] == 200
    assert summary["full_well_margin_electrons"] > 0.0
    assert summary["parent_phase_wrapping_present"] is True
    assert summary["parent_wrapped_circular_branch_pixel_count"] == 137
