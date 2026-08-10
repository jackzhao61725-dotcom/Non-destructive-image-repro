"""Tests for the machine-readable public software-verification reference."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import check_reference, reproduce_forward_model


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REFERENCE = ROOT / "reference" / "expected_results.json"


@pytest.fixture(scope="module")
def forward_payload() -> dict[str, Any]:
    return reproduce_forward_model.reproduce(CONFIGS)


def test_reference_checker_passes_without_writing_output(
    forward_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        check_reference.reproduce_forward_model,
        "reproduce",
        lambda *_args, **_kwargs: forward_payload,
    )

    assert check_reference.main([]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("PASS forward (25 keys)")
    assert captured.err == ""


def test_reference_checker_returns_nonzero_for_mismatch(
    forward_payload: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    mutated = deepcopy(reference)
    mutated["sections"]["forward"]["checks"][1]["expected"] = 99.0
    path = tmp_path / "mismatch.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")
    monkeypatch.setattr(
        check_reference.reproduce_forward_model,
        "reproduce",
        lambda *_args, **_kwargs: forward_payload,
    )

    assert check_reference.main(["--reference", str(path)]) == 1
    assert "FAIL forward (1 mismatches)" in capsys.readouterr().out


def test_reference_checker_rejects_config_drift_before_reproduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reference["config_sha256"]["model.json"] = "0" * 64
    path = tmp_path / "wrong-config-hash.json"
    path.write_text(json.dumps(reference), encoding="utf-8")

    def fail_if_called(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise AssertionError("reproduction must not run after config identity failure")

    monkeypatch.setattr(
        check_reference.reproduce_forward_model,
        "reproduce",
        fail_if_called,
    )
    assert check_reference.main(["--reference", str(path)]) == 2
    assert "FAIL reference identity (1 mismatches)" in capsys.readouterr().out


def test_numeric_check_accepts_inside_and_rejects_outside_tolerance() -> None:
    section = {
        "checks": [
            {
                "key": "value",
                "path": ["value"],
                "expected": 1.0,
                "absolute_tolerance": 0.01,
            }
        ]
    }
    inside, _ = check_reference.check_section({"value": 1.0099}, section)
    outside, _ = check_reference.check_section({"value": 1.0101}, section)

    assert inside == []
    assert len(outside) == 1


def test_inference_reference_uses_bounded_tolerances_without_start_identity() -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert (
        reference["classification"]
        == "software_verification_reference_not_experimental_data"
    )
    checks = reference["sections"]["inference"]["checks"]
    paths = [check["path"] for check in checks]
    assert all("selected_start_id" not in path for path in paths)
    endpoint_checks = [check for check in checks if "observables" in check["path"]]
    contrast_checks = [check for check in checks if "estimate_um" in check["key"]]
    assert len(endpoint_checks) == 4
    assert len(contrast_checks) == 2
    assert all(check["absolute_tolerance"] == 0.01 for check in endpoint_checks)
    assert all(check["relative_tolerance"] == 0.002 for check in endpoint_checks)
    assert all(check["absolute_tolerance"] == 0.02 for check in contrast_checks)
    assert all(check["relative_tolerance"] == 0.005 for check in contrast_checks)
