"""Select DGI and DPFI from the validated Trap-III four-method comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_LABEL = "target_dgi_dpfi_qualification_v5"
CANDIDATE_METHODS = ("pci", "dgi", "dffi", "dpfi")
SELECTED_METHODS = ("dpfi", "dgi")
STATES = ("smooth_bec", "connected_modulated", "separated_droplets")
ROLES = {
    "dgi": ("atom_stop", "leakage_stop", "stop_dark", "open_reference", "open_dark"),
    "dpfi": ("atom_h", "atom_v", "blank_h", "blank_v", "dark_h", "dark_v"),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _source(root: Path, record: dict[str, Any]) -> Path:
    path = (root / record["path"]).resolve()
    _require(path.is_file(), f"source missing: {record['path']}")
    _require(_sha256(path) == record["sha256"], f"source changed: {record['path']}")
    return path


def _git_record(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=False, capture_output=True, text=True
        ).stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def _manifest(directory: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative == "artifact_manifest.json":
            continue
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return {
        "schema_version": 1,
        "family": EXPECTED_LABEL,
        "status": "diagnostic_not_retained",
        "admitted": False,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(item["bytes"]) for item in artifacts),
        "artifacts": artifacts,
    }


def validate_config(repository_root: Path, config_path: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    config = _read_json(config_path.resolve())
    _require(config.get("schema_version") == 1, "schema changed")
    _require(config.get("label") == EXPECTED_LABEL, "config identity changed")
    _require(
        config["lifecycle"]
        == {
            "status": "diagnostic_not_retained",
            "admitted": False,
            "output_directory": ".scratch/target_dgi_dpfi_qualification_v5",
            "random_draws": 0,
            "overwrite": False,
            "results_writes": 0,
        },
        "lifecycle changed",
    )
    selection = config["selection_contract"]
    _require(tuple(selection["candidate_methods"]) == CANDIDATE_METHODS, "candidate methods changed")
    _require(tuple(selection["selected_methods"]) == SELECTED_METHODS, "selected methods changed")
    _require(
        selection["evidence_rule"]
        == "DPFI_first_and_DGI_second_by_whole_image_SNR_for_each_state_at_200_us",
        "selection rule changed",
    )
    dgi = config["dgi_acquisition"]
    _require(
        float(dgi["stop_optical_depth"]) == 4.0
        and float(dgi["residual_reference_field_amplitude"]) == 0.01
        and float(dgi["open_to_stop_gain"]) == 1.0
        and tuple(dgi["raw_roles"]) == ROLES["dgi"],
        "DGI acquisition changed",
    )
    dpfi = config["dpfi_acquisition"]
    _require(
        dpfi["port_convention"] == "H_plus_V_minus"
        and float(dpfi["h_gain"]) == 1.0
        and float(dpfi["v_gain"]) == 1.0
        and float(dpfi["cross_port_leakage_fraction"]) == 0.0
        and dpfi["registration"] == "exact"
        and dpfi["shared_psf"] is True
        and tuple(dpfi["raw_roles"]) == ROLES["dpfi"],
        "DPFI acquisition changed",
    )
    records = [*config["sources"]["parents"], *config["sources"]["generation"]]
    _require(len(records) == len({item["path"] for item in records}), "source path duplicated")
    for record in records:
        _source(root, record)
    return config


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _process_dgi(raw: dict[str, np.ndarray]) -> np.ndarray:
    denominator = raw["open_reference"] - raw["open_dark"]
    _require(np.all(denominator > 0.0), "DGI denominator failed")
    return (raw["atom_stop"] - raw["leakage_stop"]) / denominator


def _process_dpfi(raw: dict[str, np.ndarray]) -> np.ndarray:
    h = raw["atom_h"] - raw["dark_h"]
    v = raw["atom_v"] - raw["dark_v"]
    denominator = (raw["blank_h"] - raw["dark_h"]) + (
        raw["blank_v"] - raw["dark_v"]
    )
    _require(np.all(denominator > 0.0), "DPFI denominator failed")
    return (h - v) / denominator


def _ranking(rows: list[dict[str, str]], duration_us: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if int(row["pulse_duration_us"]) == duration_us]
    _require(len(selected) == len(CANDIDATE_METHODS) * len(STATES), "SNR slice changed")
    ranking: list[dict[str, Any]] = []
    for state in STATES:
        ordered = sorted(
            (row for row in selected if row["state"] == state),
            key=lambda row: float(row["image_snr"]),
            reverse=True,
        )
        _require(tuple(row["method"] for row in ordered[:2]) == SELECTED_METHODS, f"selection failed for {state}")
        for rank, row in enumerate(ordered, start=1):
            ranking.append(
                {
                    "state": state,
                    "rank": rank,
                    "method": row["method"],
                    "whole_image_snr": float(row["image_snr"]),
                    "pulse_duration_us": duration_us,
                }
            )
    return ranking


def _replay_selected(arrays_path: Path) -> tuple[float, float, float, int]:
    maximum_expected = 0.0
    expected_error = 0.0
    observed_error = 0.0
    role_arrays = 0
    with np.load(arrays_path) as data:
        for method in SELECTED_METHODS:
            process = _process_dpfi if method == "dpfi" else _process_dgi
            for state in STATES:
                expected: dict[str, np.ndarray] = {}
                observed: dict[str, np.ndarray] = {}
                for role in ROLES[method]:
                    expected_key = f"{state}__{method}__{role}__expected_electrons"
                    observed_key = f"{state}__{method}__{role}__observed_electrons"
                    _require(expected_key in data and observed_key in data, f"raw role missing: {method}/{state}/{role}")
                    expected[role] = np.asarray(data[expected_key], dtype=float)
                    observed[role] = np.asarray(data[observed_key], dtype=float)
                    maximum_expected = max(maximum_expected, float(np.max(expected[role])))
                    role_arrays += 2
                expected_error = max(
                    expected_error,
                    float(np.max(np.abs(process(expected) - np.asarray(data[f"{state}__{method}__expected_signal"], dtype=float)))),
                )
                observed_error = max(
                    observed_error,
                    float(np.max(np.abs(process(observed) - np.asarray(data[f"{state}__{method}__observed_signal"], dtype=float)))),
                )
    return expected_error, observed_error, maximum_expected, role_arrays


def _build_summary(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parents = {record["name"]: root / record["path"] for record in config["sources"]["parents"]}
    noise_summary = _read_json(parents["noise_summary"])
    noiseless_summary = _read_json(parents["noiseless_summary"])
    rows = _read_csv(parents["noise_snr_csv"])
    _require(len(rows) == 144, "SNR inventory changed")
    ranking = _ranking(rows, int(config["detector"]["pulse_duration_us"]))
    expected_error, observed_error, maximum_expected, role_arrays = _replay_selected(parents["noise_arrays"])
    _require(expected_error <= 1e-12 and observed_error <= 1e-12, "selected readout replay failed")

    seed_rows = _read_csv(parents["noise_seed_ledger"])
    _require(len(seed_rows) == 57, "seed inventory changed")
    selected_seed_rows = [row for row in seed_rows if row["method"] in SELECTED_METHODS]
    _require(len(selected_seed_rows) == 33, "selected seed inventory changed")
    _require(len({row["seed"] for row in selected_seed_rows}) == 33, "selected seed duplicated")

    read_noise = float(config["detector"]["read_noise_electrons_rms"])
    maximum_plus_6sigma = maximum_expected + 6.0 * np.sqrt(maximum_expected + read_noise**2)
    full_well = float(config["detector"]["full_well_electrons"])
    _require(maximum_plus_6sigma < full_well, "full-well gate failed")
    _require(bool(noise_summary["parent_phase_wrapping_present"]), "noise phase-wrap boundary changed")
    _require(bool(noiseless_summary["phase_wrapping_present"]), "noiseless phase-wrap boundary changed")

    return (
        {
            "schema_version": 1,
            "label": EXPECTED_LABEL,
            "status": "diagnostic_not_retained",
            "admitted": False,
            "candidate_methods": list(CANDIDATE_METHODS),
            "selected_methods": list(SELECTED_METHODS),
            "selection_basis": config["selection_contract"]["evidence_rule"],
            "selected_raw_seed_identities": len(selected_seed_rows),
            "selected_raw_role_arrays_replayed": role_arrays,
            "maximum_expected_processing_error": expected_error,
            "maximum_observed_processing_error": observed_error,
            "maximum_selected_expected_electrons": maximum_expected,
            "maximum_selected_expected_plus_6sigma_electrons": maximum_plus_6sigma,
            "selected_full_well_margin_electrons": full_well - maximum_plus_6sigma,
            "phase_wrapping_present": True,
            "phase_wrapping_scope": "ID_stronger_circular_branch_only",
            "whole_image_snr_role": "aggregate_image_strength_not_feature_precision",
            "calibration_boundary": config["claim_boundary"],
            "random_draws": 0,
            "new_random_identities": 0,
            "new_optical_propagations": 0,
            "new_fits": 0,
            "results_writes": 0,
        },
        ranking,
    )


def run_qualification(repository_root: Path, config_path: Path) -> Path:
    root = repository_root.resolve()
    config_path = config_path.resolve()
    config = validate_config(root, config_path)
    output = root / config["lifecycle"]["output_directory"]
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    staging.mkdir(parents=True)
    (staging / "data").mkdir()
    summary, ranking = _build_summary(root, config)
    with (staging / "data/selection_at_200us.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    _write_json(
        staging / "data/selected_acquisition_contract.json",
        {
            "dgi": config["dgi_acquisition"],
            "dpfi": config["dpfi_acquisition"],
            "detector": config["detector"],
            "claim_boundary": config["claim_boundary"],
        },
    )
    _write_json(staging / "summary.json", summary)
    _write_json(staging / "config_snapshot.json", config)
    _write_json(
        staging / "provenance.json",
        {
            "schema_version": 1,
            "status": "diagnostic_not_retained",
            "admitted": False,
            "config_sha256": _sha256(config_path),
            "source_hashes": {
                record["path"]: record["sha256"]
                for record in [*config["sources"]["parents"], *config["sources"]["generation"]]
            },
            "git": _git_record(root),
            "execution": {
                "random_draws": 0,
                "new_random_identities": 0,
                "new_optical_propagations": 0,
                "new_fits": 0,
                "results_writes": 0,
            },
        },
    )
    (staging / "selection_statement.txt").write_text(
        "Under the declared Trap-III geometry, ideal matched readout and detector model, "
        "DPFI gives the largest whole-image SNR for BEC, SSP and ID, while DGI is second "
        "for all three. DPFI is therefore continued as the signed port-difference readout "
        "with leading-order linear response, and DGI as the dark-background quadratic "
        "comparator. This is a configuration-specific "
        "scope decision, not a universal method ranking or a feature-recovery result.\n",
        encoding="utf-8",
    )
    _write_json(staging / "artifact_manifest.json", _manifest(staging))
    validate_output(staging, config, root)
    os.replace(staging, output)
    return output


def validate_output(directory: Path, config: dict[str, Any], repository_root: Path) -> dict[str, Any]:
    directory = directory.resolve()
    root = repository_root.resolve()
    _require(_read_json(directory / "artifact_manifest.json") == _manifest(directory), "manifest changed")
    _require(_read_json(directory / "config_snapshot.json") == config, "config snapshot changed")
    expected_summary, expected_ranking = _build_summary(root, config)
    summary = _read_json(directory / "summary.json")
    _require(summary == expected_summary, "summary changed")
    ranking = _read_csv(directory / "data/selection_at_200us.csv")
    _require(len(ranking) == 12, "selection table changed")
    for stored, expected in zip(ranking, expected_ranking, strict=True):
        _require(stored["state"] == expected["state"], "selection state changed")
        _require(int(stored["rank"]) == expected["rank"], "selection rank changed")
        _require(stored["method"] == expected["method"], "selection method changed")
        _require(np.isclose(float(stored["whole_image_snr"]), expected["whole_image_snr"], rtol=0.0, atol=1e-12), "selection SNR changed")
        _require(int(stored["pulse_duration_us"]) == 200, "selection duration changed")
    expected_contract = {
        "dgi": config["dgi_acquisition"],
        "dpfi": config["dpfi_acquisition"],
        "detector": config["detector"],
        "claim_boundary": config["claim_boundary"],
    }
    _require(_read_json(directory / "data/selected_acquisition_contract.json") == expected_contract, "selected acquisition contract changed")
    _require("peak" not in (directory / "selection_statement.txt").read_text(encoding="utf-8").lower(), "selection anticipates feature extraction")
    return summary
