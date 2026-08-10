"""Tests for the bundled retained synthetic evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import check_evidence


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "retained_v1"


def test_evidence_checker_passes_without_rerunning_analysis(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert check_evidence.main([]) == 0
    captured = capsys.readouterr()
    assert captured.out == "PASS evidence (16 files, 2366025 bytes)\n"
    assert captured.err == ""


def test_evidence_manifest_preserves_the_claim_boundary() -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == (
        "curated_public_derivative_of_admitted_synthetic_evidence"
    )
    assert manifest["scientific_compute_executed"] is False
    assert len(manifest["source_families"]) == 3
    assert len(manifest["derivatives"]) == 1


def test_orientation_refit_derivative_is_complete_and_row_preserving() -> None:
    path = EVIDENCE / "orientation" / "conditional_refits.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 128
    assert {int(row["draw_index"]) for row in rows} == set(range(64))
    assert {row["endpoint_label"] for row in rows} == {
        "B_parallel_y",
        "B_parallel_z",
    }
    assert {row["fit_status"] for row in rows} == {"success"}
