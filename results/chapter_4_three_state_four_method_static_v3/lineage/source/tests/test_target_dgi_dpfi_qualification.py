from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from isolated_non_destructive_image import load_isolated_non_destructive_image_module


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/target_dgi_dpfi_qualification_v5.json"
RUNNER = ROOT / "scripts/run_target_dgi_dpfi_qualification.py"
OUTPUT = ROOT / ".scratch/target_dgi_dpfi_qualification_v5"
MODULE = load_isolated_non_destructive_image_module(
    "target_dgi_dpfi_qualification", namespace="_test_target_dgi_dpfi_qualification_v5"
)


def test_contract_selects_methods_without_feature_extraction() -> None:
    config = MODULE.validate_config(ROOT, CONFIG)
    assert config["selection_contract"]["candidate_methods"] == ["pci", "dgi", "dffi", "dpfi"]
    assert config["selection_contract"]["selected_methods"] == ["dpfi", "dgi"]
    assert config["selection_contract"]["not_a_universal_method_ranking"] is True
    assert config["dgi_acquisition"]["signed_residual_policy"] == "retain_without_clipping"
    assert config["dpfi_acquisition"]["registration"] == "exact"
    assert config["detector"]["full_well_gate"].startswith("maximum_expected_plus_6sigma")
    assert "peak_spacing" not in str(config).lower()


def test_parent_evidence_selects_dpfi_then_dgi_for_every_state() -> None:
    config = MODULE.validate_config(ROOT, CONFIG)
    summary, ranking = MODULE._build_summary(ROOT, config)
    assert summary["selected_methods"] == ["dpfi", "dgi"]
    for state in MODULE.STATES:
        selected = [row for row in ranking if row["state"] == state]
        assert [row["method"] for row in selected[:2]] == ["dpfi", "dgi"]
    assert summary["maximum_expected_processing_error"] <= 1e-12
    assert summary["maximum_observed_processing_error"] <= 1e-12
    assert summary["selected_full_well_margin_electrons"] > 13000.0
    assert summary["phase_wrapping_scope"] == "ID_stronger_circular_branch_only"


def test_source_check_does_not_generate_output() -> None:
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


def test_generated_qualification_replays_when_present() -> None:
    if not OUTPUT.exists():
        pytest.skip("target DGI/DPFI qualification has not been generated")
    config = MODULE.validate_config(ROOT, CONFIG)
    summary = MODULE.validate_output(OUTPUT, config, ROOT)
    assert summary["random_draws"] == 0
    assert summary["new_random_identities"] == 0
    assert summary["new_optical_propagations"] == 0
    assert summary["new_fits"] == 0
    assert summary["selected_raw_seed_identities"] == 33
