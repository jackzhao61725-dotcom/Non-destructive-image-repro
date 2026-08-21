from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

import non_destructive_image.chapter4_static_retained_staging as staging


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/chapter_4_three_state_four_method_static_retained_v3.json"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("chapter4_static") / "staging"
    return staging.prepare_staging(ROOT, CONFIG, staging_override=target)


def _refresh_manifest(directory: Path) -> None:
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    artifacts = [
        staging._file_record(path, directory)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    manifest["artifacts"] = artifacts
    manifest["artifact_count"] = len(artifacts)
    manifest["total_bytes"] = sum(int(record["bytes"]) for record in artifacts)
    staging._write_json(directory / "artifact_manifest.json", manifest)


def test_source_contract_and_active_parents_are_exact() -> None:
    config = staging.validate_sources(ROOT, CONFIG)
    assert config["family"] == "chapter_4_three_state_four_method_static_v3"
    assert config["lifecycle"]["admitted"] is False
    assert config["recomputation"] == {
        "new_random_draws": 0,
        "new_fits": 0,
        "new_optical_propagations": 0,
        "new_thermodynamic_solves": 0,
        "new_presentations": 3,
    }


def test_prepared_staging_is_self_contained_and_byte_preserves_payloads(prepared: Path) -> None:
    result = staging.validate_staging(prepared)
    assert result["status"] == "PASS"
    assert result["admitted"] is False and result["consumable"] is False
    assert result["snr_rows"] == 144 and result["seed_identities"] == 57
    assert result["selected_methods"] == ["dpfi", "dgi"]
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pairs = {
        ROOT / config["parents"]["noiseless"]["directory"] / "data/target_four_method_camera_and_fields.npz": prepared / "data/noiseless/target_four_method_camera_and_fields.npz",
        ROOT / config["parents"]["noiseless"]["directory"] / "data/state_method_extrema.csv": prepared / "data/noiseless/state_method_extrema.csv",
        ROOT / config["parents"]["noise"]["directory"] / "data/four_method_noise_arrays.npz": prepared / "data/noise/four_method_noise_arrays.npz",
        ROOT / config["parents"]["noise"]["directory"] / "data/four_method_snr.csv": prepared / "data/noise/four_method_snr.csv",
        ROOT / config["parents"]["noise"]["directory"] / "data/seed_ledger.csv": prepared / "data/noise/seed_ledger.csv",
        ROOT / config["parents"]["qualification"]["directory"] / "data/selected_acquisition_contract.json": prepared / "data/selection/selected_acquisition_contract.json",
        ROOT / config["parents"]["qualification"]["directory"] / "data/selection_at_200us.csv": prepared / "data/selection/selection_at_200us.csv",
        ROOT / config["parents"]["qualification"]["directory"] / "selection_statement.txt": prepared / "data/selection/selection_statement.txt",
    }
    assert all(source.read_bytes() == copied.read_bytes() for source, copied in pairs.items())


def test_final_width_font_and_pdf_geometry(prepared: Path) -> None:
    config = json.loads((prepared / "config_snapshot.json").read_text(encoding="utf-8"))
    assert config["presentation"]["width_inches"] == 6.61
    assert min(config["presentation"]["font_sizes_pt"].values()) >= 9.0
    for name, height in (
        ("figure_4_target_four_method_noiseless_common_scale.pdf", config["presentation"]["panel_height_inches"]),
        ("figure_4_target_four_method_noisy_200us.pdf", config["presentation"]["panel_height_inches"]),
        ("figure_4_target_four_method_snr.pdf", config["presentation"]["snr_height_inches"]),
    ):
        assert staging._page_size_points(prepared / "presentation" / name) == pytest.approx((6.61 * 72.0, height * 72.0), abs=0.02)


def test_selection_is_exactly_dpfi_then_dgi_at_200us(prepared: Path) -> None:
    with (prepared / "data/selection/selection_at_200us.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for state in staging.STATES:
        selected = [row for row in rows if row["state"] == state]
        assert [row["method"] for row in selected[:2]] == ["dpfi", "dgi"]


def test_manifest_tamper_is_rejected_before_config_or_payload_load(prepared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    changed = tmp_path / "changed"
    shutil.copytree(prepared, changed)
    (changed / "captions/figure_4_target_four_method_snr.txt").write_text("changed\n", encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("payload loaded before manifest authentication")

    monkeypatch.setattr(staging, "_load_config", forbidden)
    with pytest.raises(ValueError, match="artifact manifest changed"):
        staging.validate_staging(changed)


def test_numeric_tamper_is_rejected_after_manifest_refresh(prepared: Path, tmp_path: Path) -> None:
    changed = tmp_path / "numeric"
    shutil.copytree(prepared, changed)
    path = changed / "data/selection/selection_at_200us.csv"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("79.55975981916528", "79.0", 1), encoding="utf-8")
    _refresh_manifest(changed)
    with pytest.raises(ValueError, match="payload parent identity changed"):
        staging.validate_staging(changed, replay_presentations=False)


def test_refreshed_manifest_cannot_admit_extra_artifact(prepared: Path, tmp_path: Path) -> None:
    changed = tmp_path / "extra"
    shutil.copytree(prepared, changed)
    (changed / "unexpected.txt").write_text("extra\n", encoding="utf-8")
    _refresh_manifest(changed)
    with pytest.raises(ValueError, match="exact artifact inventory changed"):
        staging.validate_staging(changed, replay_presentations=False)


def test_refreshed_manifest_cannot_hide_caption_and_config_drift(prepared: Path, tmp_path: Path) -> None:
    changed = tmp_path / "caption"
    shutil.copytree(prepared, changed)
    config_path = changed / "config_snapshot.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["captions"]["snr"] = "Changed reader claim."
    staging._write_json(config_path, config)
    (changed / staging.CAPTION_FILES["snr"]).write_text(
        "Changed reader claim.\n", encoding="utf-8", newline=""
    )
    _refresh_manifest(changed)
    with pytest.raises(ValueError, match="frozen config semantics changed"):
        staging.validate_staging(changed, replay_presentations=False)


@pytest.mark.parametrize("field", ["supersession", "excluded_scope"])
def test_refreshed_manifest_cannot_remove_frozen_scope(
    prepared: Path, tmp_path: Path, field: str
) -> None:
    changed = tmp_path / field
    shutil.copytree(prepared, changed)
    config_path = changed / "config_snapshot.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if field == "supersession":
        config["superseded_presentations"] = config["superseded_presentations"][1:]
    else:
        config["claim_boundary"]["does_not_support"] = [
            value
            for value in config["claim_boundary"]["does_not_support"]
            if "Oxford" not in value
        ]
    staging._write_json(config_path, config)
    _refresh_manifest(changed)
    with pytest.raises(ValueError, match="(frozen config semantics|claim boundary) changed"):
        staging.validate_staging(changed, replay_presentations=False)


def test_v3_rendered_pixels_match_v2(prepared: Path) -> None:
    predecessor = ROOT / ".scratch/chapter_4_three_state_four_method_static_v2_admission_staging/presentation"
    for name in (
        "figure_4_target_four_method_noiseless_common_scale.png",
        "figure_4_target_four_method_noisy_200us.png",
        "figure_4_target_four_method_snr.png",
    ):
        assert np.array_equal(
            staging.plt.imread(predecessor / name),
            staging.plt.imread(prepared / "presentation" / name),
        )


def test_live_scratch_is_not_used_by_staging_validator(prepared: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.exists

    def guarded(path: Path) -> bool:
        if ".scratch" in path.parts and prepared not in path.parents and path != prepared:
            raise AssertionError(f"live scratch access: {path}")
        return original(path)

    monkeypatch.setattr(Path, "exists", guarded)
    result = staging.validate_staging(prepared, replay_presentations=False)
    assert result["status"] == "PASS"


def test_no_overwrite_rejects_existing_staging(prepared: Path) -> None:
    with pytest.raises(ValueError, match="staging or building already exists"):
        staging.prepare_staging(ROOT, CONFIG, staging_override=prepared)
