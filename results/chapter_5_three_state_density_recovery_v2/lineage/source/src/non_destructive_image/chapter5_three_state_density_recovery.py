"""Section 5.3 model-matched SSP/ID density-recovery evidence.

This module only aggregates completed DGI and DPFI fits. It creates no camera
draw, optical propagation or nonlinear fit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.stat().st_size != source.stat().st_size or _sha256(destination) != _sha256(source):
        raise ValueError(f"copy identity failed: {source}")


def _copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        _copy(path, destination / path.relative_to(source))


def _manifest_inventory(directory: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(
        item for item in directory.rglob("*")
        if item.is_file() and item.name != "artifact_manifest.json"
    ):
        output.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return output


def _write_manifest(
    directory: Path,
    *,
    family: str,
    status: str,
    admitted: bool,
    claim_boundary: Mapping[str, Any],
) -> None:
    inventory = _manifest_inventory(directory)
    _write_json(
        directory / "artifact_manifest.json",
        {
            "schema_version": 1,
            "family": family,
            "status": status,
            "admitted": admitted,
            "artifact_count": len(inventory),
            "total_bytes": sum(int(item["bytes"]) for item in inventory),
            "artifacts": inventory,
            "claim_boundary": claim_boundary,
        },
    )


def _validate_manifest(
    directory: Path,
    *,
    expected_status: str | None = None,
    expected_admitted: bool | None = None,
) -> dict[str, Any]:
    manifest_path = directory / "artifact_manifest.json"
    manifest = _read_json(manifest_path)
    if expected_status is not None and manifest.get("status") != expected_status:
        raise ValueError(f"lifecycle status changed: {directory}")
    if expected_admitted is not None and manifest.get("admitted") is not expected_admitted:
        raise ValueError(f"admission status changed: {directory}")
    inventory = list(manifest["artifacts"])
    expected = {item["path"] for item in inventory}
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if expected != actual:
        raise ValueError(f"manifest inventory mismatch: {directory}")
    for item in inventory:
        path = directory / item["path"]
        if path.stat().st_size != int(item["bytes"]) or _sha256(path) != item["sha256"]:
            raise ValueError(f"artifact identity changed: {path}")
    return manifest


def _authenticate_parent(root: Path, parent: Mapping[str, Any]) -> Path:
    directory = root / str(parent["directory"])
    manifest = directory / "artifact_manifest.json"
    if _sha256(manifest) != parent["manifest_sha256"]:
        raise ValueError(f"parent manifest changed: {manifest}")
    _validate_manifest(directory)
    for key, filename in (
        ("selected_fits_sha256", "data/selected_fits.csv"),
        ("start_outcomes_sha256", "data/start_outcomes.csv"),
        ("seed_ledger_sha256", "data/seed_ledger.csv"),
        ("reclassified_fits_sha256", "data/reclassified_fits.csv"),
        ("support_summary_sha256", "data/support_summary.csv"),
    ):
        if key in parent and _sha256(directory / filename) != parent[key]:
            raise ValueError(f"parent payload changed: {directory / filename}")
    return directory


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _expected_keys(config: Mapping[str, Any]) -> set[tuple[str, str, int, int]]:
    return {
        (method, state, int(duration), int(draw))
        for method in config["methods"]
        for state in config["states"]
        for duration in config["durations_us"]
        for draw in config["draw_ids"]
    }


def _combine_rows(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_rows: list[dict[str, str]] = []
    parent_directories: dict[str, Path] = {}
    for method in config["methods"]:
        directory = _authenticate_parent(root, config["parents"][method])
        parent_directories[method] = directory
        rows = _read_csv(directory / "data/selected_fits.csv")
        if any(row["method"] != method for row in rows):
            raise ValueError(f"method identity changed: {method}")
        selected_rows.extend(rows)
    support_directory = _authenticate_parent(root, config["parents"]["support"])
    support_rows = _read_csv(support_directory / "data/reclassified_fits.csv")
    support_by_key = {
        (row["method"], row["state"], int(row["duration_us"]), int(row["draw_id"])): row
        for row in support_rows
    }
    if len(support_by_key) != len(support_rows):
        raise ValueError("duplicate support key")

    expected = _expected_keys(config)
    selected_by_key = {
        (row["method"], row["state"], int(row["duration_us"]), int(row["draw_id"])): row
        for row in selected_rows
    }
    if len(selected_by_key) != len(selected_rows) or set(selected_by_key) != expected:
        raise ValueError("selected-fit identity inventory changed")
    if set(support_by_key) != expected:
        raise ValueError("support identity inventory changed")

    combined: list[dict[str, Any]] = []
    qualification: list[dict[str, Any]] = []
    for key in sorted(expected):
        row = selected_by_key[key]
        support = support_by_key[key]
        if _bool(row["optimizer_failure"]) or _bool(row["residual_failure"]):
            raise ValueError(f"failed selected fit: {key}")
        if not _bool(row["topology_valid"]):
            raise ValueError(f"invalid selected density topology: {key}")
        numeric = {
            name: float(row[name])
            for name in (
                "eta_hat",
                "d_peak_hat_um",
                "d_peak_truth_um",
                "nu_vp_hat",
                "nu_vp_truth",
            )
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            raise ValueError(f"non-finite fitted value: {key}")
        if not math.isclose(numeric["d_peak_hat_um"], float(support["d_peak_hat_um"]), abs_tol=1e-12):
            raise ValueError(f"peak-spacing join changed: {key}")
        if not math.isclose(numeric["nu_vp_hat"], float(support["nu_vp_hat"]), abs_tol=1e-12):
            raise ValueError(f"valley-ratio join changed: {key}")
        combined.append(
            {
                "method": key[0],
                "state": key[1],
                "display_label": config["display_labels"][key[1]],
                "duration_us": key[2],
                "draw_id": key[3],
                "eta_hat": numeric["eta_hat"],
                "eta_truth": 1.0,
                "d_peak_hat_um": numeric["d_peak_hat_um"],
                "d_peak_truth_um": numeric["d_peak_truth_um"],
                "nu_vp_hat": numeric["nu_vp_hat"],
                "nu_vp_truth": numeric["nu_vp_truth"],
                "nuisance_censored": _bool(row["nuisance_censored"]),
                "d_peak_supported": _bool(support["d_peak_supported"]),
                "nu_vp_supported": _bool(support["nu_vp_supported"]),
            }
        )

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in combined:
        groups[(row["method"], row["state"], int(row["duration_us"]))].append(row)
    for key in sorted(groups):
        rows = groups[key]
        qualification.append(
            {
                "method": key[0],
                "state": key[1],
                "display_label": config["display_labels"][key[1]],
                "duration_us": key[2],
                "draw_count": len(rows),
                "nuisance_censored_count": sum(bool(row["nuisance_censored"]) for row in rows),
                "d_peak_supported_count": sum(bool(row["d_peak_supported"]) for row in rows),
                "nu_vp_supported_count": sum(bool(row["nu_vp_supported"]) for row in rows),
            }
        )
    return combined, qualification


def summarise(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["method"]),
                str(row["state"]),
                str(row["display_label"]),
                int(row["duration_us"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        for observable in config["observables"]:
            values = np.asarray([float(row[observable["column"]]) for row in group], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite recovery distribution: {key}")
            if "truth" in observable:
                truth = float(observable["truth"])
            else:
                truths = {float(row[observable["truth_column"]]) for row in group}
                if len(truths) != 1:
                    raise ValueError(f"truth changed within cell: {key}")
                truth = truths.pop()
            output.append(
                {
                    "method": key[0],
                    "state": key[1],
                    "display_label": key[2],
                    "duration_us": key[3],
                    "observable": observable["name"],
                    "unit": observable["unit"],
                    "truth": truth,
                    "draw_count": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q16": float(np.quantile(values, 0.16)),
                    "q84": float(np.quantile(values, 0.84)),
                    "empirical_sd": float(np.std(values, ddof=1)),
                    "bias": float(np.mean(values) - truth),
                }
            )
    return output


def _render(summary_rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    methods = ("dpfi", "dgi")
    method_style = {
        "dpfi": {"label": "DPFI", "color": "#326C9B", "marker": "s"},
        "dgi": {"label": "DGI", "color": "#D17A22", "marker": "o"},
    }
    states = (("connected_modulated", "SSP"), ("separated_droplets", "ID"))
    observables = (
        ("eta", r"$\eta_s$"),
        ("d_peak", r"$\overline{d}_{\mathrm{pk}}\;(\mu\mathrm{m})$"),
        ("nu_vp", r"$\nu_{\mathrm{vp}}$"),
    )
    durations = (25, 50, 100, 200, 400)
    positions = np.arange(len(durations), dtype=float)
    lookup = {
        (str(row["method"]), str(row["state"]), str(row["observable"]), int(row["duration_us"])): row
        for row in summary_rows
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(3, 2, figsize=(7.15, 7.6), sharex=True)
    for column, (state, title) in enumerate(states):
        axes[0, column].set_title(title, pad=8, fontweight="bold")
        for row_index, (observable, ylabel) in enumerate(observables):
            axis = axes[row_index, column]
            truth = float(lookup[("dpfi", state, observable, durations[0])]["truth"])
            axis.axhline(truth, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
            for method in methods:
                cells = [lookup[(method, state, observable, duration)] for duration in durations]
                medians = np.asarray([float(cell["median"]) for cell in cells])
                q16 = np.asarray([float(cell["q16"]) for cell in cells])
                q84 = np.asarray([float(cell["q84"]) for cell in cells])
                style = method_style[method]
                axis.fill_between(
                    positions,
                    q16,
                    q84,
                    color=style["color"],
                    alpha=0.16 if method == "dpfi" else 0.13,
                    linewidth=0,
                    zorder=2,
                )
                axis.plot(
                    positions,
                    medians,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=4.8,
                    markerfacecolor="white" if method == "dgi" else style["color"],
                    markeredgewidth=1.0,
                    linewidth=1.25,
                    label=style["label"],
                    zorder=3,
                )
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_xlim(-0.25, len(durations) - 0.75)
            if row_index == 2:
                axis.set_xticks(positions, [str(value) for value in durations])
                axis.set_xlabel(r"Pulse duration $\tau\;(\mu\mathrm{s})$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    truth_handle = plt.Line2D([0], [0], color="#333333", linewidth=1.0, linestyle="--")
    figure.legend(
        handles + [truth_handle],
        labels + ["Input"],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.997),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
    )
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.935, hspace=0.26, wspace=0.28)
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output / "figure_5_5_three_state_density_recovery.pdf",
        bbox_inches="tight",
        metadata={"Creator": "Non-destructive-image", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        output / "figure_5_5_three_state_density_recovery.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "Non-destructive-image"},
    )
    plt.close(figure)


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(args, cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    return {
        "branch": run("git", "branch", "--show-current"),
        "head": run("git", "rev-parse", "HEAD"),
        "dirty_tree": bool(run("git", "status", "--porcelain")),
    }


def _candidate_staging(candidate: Path) -> Path:
    return candidate.parent / f".{candidate.name}-staging"


def build_candidate(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    candidate = root / config["candidate"]
    staging = _candidate_staging(candidate)
    if candidate.exists() or staging.exists():
        raise FileExistsError(candidate)
    rows, qualification = _combine_rows(root, config)
    summaries = summarise(rows, config)
    staging.mkdir(parents=True)
    _write_csv(staging / "data/recovery_draws.csv", rows)
    _write_csv(staging / "data/recovery_summary.csv", summaries)
    _write_csv(staging / "data/qualification_summary.csv", qualification)
    for method in config["methods"]:
        source = root / config["parents"][method]["directory"] / "data"
        for name in ("selected_fits.csv", "start_outcomes.csv", "seed_ledger.csv"):
            _copy(source / name, staging / "parents" / method / name)
    support_source = root / config["parents"]["support"]["directory"] / "data"
    for name in ("reclassified_fits.csv", "support_summary.csv"):
        _copy(support_source / name, staging / "parents" / "support" / name)
    _render(summaries, staging / "figures")
    (staging / "caption.txt").write_text(
        "Median and central 68% recovery ranges for condensate population, visible mean peak separation and valley-to-neighbouring-peak density ratio across 64 model-matched camera-noise acquisitions per point.\n",
        encoding="utf-8",
    )
    _copy(config_path, staging / "config_snapshot.json")
    _write_json(
        staging / "summary.json",
        {
            "schema_version": 1,
            "family": config["label"],
            "status": "complete_pending_independent_admission",
            "admitted": False,
            "selected_fit_rows": len(rows),
            "summary_rows": len(summaries),
            "qualification_rows": len(qualification),
            "methods": config["methods"],
            "states": config["states"],
            "durations_us": config["durations_us"],
            "draws_per_cell": len(config["draw_ids"]),
            "new_random_draws": 0,
            "new_optical_propagations": 0,
            "new_nonlinear_fits": 0,
            "claim_boundary": config["claim_boundary"],
        },
    )
    _write_json(
        staging / "provenance.json",
        {
            "schema_version": 1,
            "parents": config["parents"],
            "source_git_state": _git_state(root),
            "aggregation": "all finite selected fits; no support filtering",
            "interval": "16th to 84th percentiles across 64 camera-noise acquisitions",
            "new_random_draws": 0,
            "new_optical_propagations": 0,
            "new_nonlinear_fits": 0,
        },
    )
    _write_manifest(
        staging,
        family=config["label"],
        status="complete_pending_independent_admission",
        admitted=False,
        claim_boundary=config["claim_boundary"],
    )
    check = validate(staging, expected_status="complete_pending_independent_admission", expected_admitted=False)
    os.replace(staging, candidate)
    final = validate(candidate, expected_status="complete_pending_independent_admission", expected_admitted=False)
    if check != final:
        raise ValueError("candidate validation changed after publication")
    return {"candidate": str(candidate), **final}


def _admission_contract(config: Mapping[str, Any], candidate_manifest_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "family": config["label"],
        "status": "admitted_immutable",
        "admitted": True,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "aggregation": {
            "selected_fit_population": "all finite selected fits",
            "camera_noise_draws_per_cell": len(config["draw_ids"]),
            "interval": "central 68 percent, bounded by the 16th and 84th percentiles",
            "support_filter_applied_to_plotted_distribution": False,
        },
        "new_random_draws": 0,
        "new_optical_propagations": 0,
        "new_nonlinear_fits": 0,
        "claim_boundary": config["claim_boundary"],
    }


def admit(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    candidate = root / config["candidate"]
    _validate_manifest(
        candidate,
        expected_status="complete_pending_independent_admission",
        expected_admitted=False,
    )
    validate(candidate, expected_status="complete_pending_independent_admission", expected_admitted=False)
    staging = root / config["staging"]
    target = root / config["target"]
    if staging.exists() or target.exists():
        raise FileExistsError(target)
    staging.mkdir(parents=True)
    for name in ("data", "parents", "figures"):
        _copy_tree(candidate / name, staging / name)
    _copy(candidate / "caption.txt", staging / "caption.txt")
    for source, name in (
        (candidate / "artifact_manifest.json", "candidate_manifest_snapshot.json"),
        (candidate / "summary.json", "candidate_summary_snapshot.json"),
        (candidate / "provenance.json", "candidate_provenance_snapshot.json"),
        (config_path, "admission_config_snapshot.json"),
    ):
        _copy(source, staging / "lineage" / name)
    for path in (
        root / "src/non_destructive_image/chapter5_three_state_density_recovery.py",
        root / "scripts/build_chapter_5_three_state_density_recovery.py",
        root / "tests/test_chapter5_three_state_density_recovery.py",
    ):
        _copy(path, staging / "lineage/source" / path.relative_to(root))
    for name, parent in config["parents"].items():
        source = root / parent["directory"] / "artifact_manifest.json"
        _copy(source, staging / "lineage/parent_manifests" / f"{name}.json")
    contract = _admission_contract(config, _sha256(candidate / "artifact_manifest.json"))
    _write_json(staging / "admission_contract_snapshot.json", contract)
    candidate_summary = _read_json(candidate / "summary.json")
    _write_json(
        staging / "summary.json",
        {
            **candidate_summary,
            "status": "admitted_immutable",
            "admitted": True,
            "candidate_manifest_sha256": _sha256(candidate / "artifact_manifest.json"),
        },
    )
    _write_json(
        staging / "provenance.json",
        {
            "schema_version": 1,
            "candidate": config["candidate"],
            "candidate_manifest_sha256": _sha256(candidate / "artifact_manifest.json"),
            "admission_git_state": _git_state(root),
            "new_random_draws": 0,
            "new_optical_propagations": 0,
            "new_nonlinear_fits": 0,
        },
    )
    _write_manifest(
        staging,
        family=config["label"],
        status="admitted_immutable",
        admitted=True,
        claim_boundary=config["claim_boundary"],
    )
    before = validate(staging, expected_status="admitted_immutable", expected_admitted=True)
    os.replace(staging, target)
    after = validate(target, expected_status="admitted_immutable", expected_admitted=True)
    if before != after:
        raise ValueError("admitted validation changed after publication")
    return {"target": str(target), **after}


def validate(
    directory: Path,
    *,
    expected_status: str | None = None,
    expected_admitted: bool | None = None,
) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = _validate_manifest(
        directory,
        expected_status=expected_status,
        expected_admitted=expected_admitted,
    )
    config_path = (
        directory / "lineage/admission_config_snapshot.json"
        if (directory / "lineage/admission_config_snapshot.json").exists()
        else directory / "config_snapshot.json"
    )
    config = _read_json(config_path)
    rows = _read_csv(directory / "data/recovery_draws.csv")
    if len(rows) != 1280:
        raise ValueError("recovery draw inventory changed")
    keys = {
        (row["method"], row["state"], int(row["duration_us"]), int(row["draw_id"]))
        for row in rows
    }
    if keys != _expected_keys(config):
        raise ValueError("recovery identity inventory changed")
    recomputed = summarise(rows, config)
    saved = _read_csv(directory / "data/recovery_summary.csv")
    if len(saved) != 60 or len(recomputed) != 60:
        raise ValueError("recovery summary inventory changed")
    expected_rows = [{key: str(value) for key, value in row.items()} for row in recomputed]
    if saved != expected_rows:
        raise ValueError("recovery summary does not replay")
    qualification = _read_csv(directory / "data/qualification_summary.csv")
    if len(qualification) != 20 or any(int(row["draw_count"]) != 64 for row in qualification):
        raise ValueError("qualification inventory changed")
    for row in rows:
        for name in ("eta_hat", "d_peak_hat_um", "d_peak_truth_um", "nu_vp_hat", "nu_vp_truth"):
            if not math.isfinite(float(row[name])):
                raise ValueError("non-finite retained recovery value")
    for name in (
        "figure_5_5_three_state_density_recovery.pdf",
        "figure_5_5_three_state_density_recovery.png",
    ):
        path = directory / "figures" / name
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing presentation figure: {path}")
    summary = _read_json(directory / "summary.json")
    if summary["claim_boundary"] != manifest["claim_boundary"]:
        raise ValueError("claim boundary diverged")
    return {
        "status": "PASS",
        "selected_fit_rows": len(rows),
        "summary_rows": len(saved),
        "qualification_rows": len(qualification),
        "manifest_sha256": _sha256(directory / "artifact_manifest.json"),
        "new_random_draws": 0,
        "new_optical_propagations": 0,
        "new_nonlinear_fits": 0,
    }
