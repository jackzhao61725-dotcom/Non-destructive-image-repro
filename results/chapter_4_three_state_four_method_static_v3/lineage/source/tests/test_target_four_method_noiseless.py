from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from isolated_non_destructive_image import load_isolated_non_destructive_image_module


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/target_three_state_four_method_noiseless_v4.json"
RUNNER = ROOT / "scripts/run_target_four_method_noiseless.py"
OUTPUT = ROOT / ".scratch/target_three_state_four_method_noiseless_v4"
MODULE = load_isolated_non_destructive_image_module(
    "target_four_method_noiseless",
    namespace="_test_target_four_method_noiseless",
)


def test_contract_uses_target_profiles_and_one_incident_intensity_scale() -> None:
    config = MODULE.validate_config(ROOT, CONFIG)
    assert config["physical_contract"]["reference_pulse_duration_us"] == 200
    assert config["readout_contract"]["common_display_denominator"] == "incident_I0"
    assert config["readout_contract"]["dpfi_display_quantity"] == (
        "I_H_over_I0_minus_I_V_over_I0"
    )
    assert config["grid_and_transfer"]["object_shape"] == [161, 641]
    profile = json.loads(
        (ROOT / config["sources"]["profile_contract"]["path"]).read_text()
    )
    assert profile["target_scale_anchor"]["source_config"].endswith(
        "target_geometry_finite_temperature_closure_v4.json"
    )


def test_source_check_is_non_generating() -> None:
    existed = OUTPUT.exists()
    completed = subprocess.run(
        [str(ROOT / ".venv/Scripts/python.exe"), str(RUNNER), "--source-check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"status": "source_check_pass"' in completed.stdout
    assert OUTPUT.exists() is existed


def test_generated_target_comparison_replays_when_present() -> None:
    if not OUTPUT.exists():
        pytest.skip("target noiseless diagnostic has not been generated")
    config = MODULE.validate_config(ROOT, CONFIG)
    summary = MODULE.validate_output(OUTPUT, config)
    replay = MODULE.replay_output(OUTPUT, config)
    assert summary["random_draws"] == 0
    assert summary["phase_wrapping_present"] is True
    assert summary["maximum_absolute_circular_branch_phase_rad"] >= np.pi
    assert summary["wrapped_circular_branch_pixel_count"] > 0
    assert replay["maximum_replay_error"] < 1e-12
    assert summary["dgi_negative_pixel_count"] > 0
    assert summary["dpfi_difference_negative_pixel_count"] > 0


def test_validator_rejects_a_changed_dpfi_port_difference(tmp_path: Path) -> None:
    if not OUTPUT.exists():
        pytest.skip("target noiseless diagnostic has not been generated")
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    payload_path = copied / "data" / MODULE.NPZ_NAME
    with np.load(payload_path) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    key = "separated_droplets__dpfi_difference_camera_intensity_over_i0"
    payload[key] = np.array(payload[key], copy=True)
    payload[key][payload[key].shape[0] // 2, payload[key].shape[1] // 2] += 1e-3
    np.savez_compressed(payload_path, **payload)
    with pytest.raises(ValueError, match="artifact manifest changed"):
        MODULE.validate_output(copied, MODULE.validate_config(ROOT, CONFIG))
