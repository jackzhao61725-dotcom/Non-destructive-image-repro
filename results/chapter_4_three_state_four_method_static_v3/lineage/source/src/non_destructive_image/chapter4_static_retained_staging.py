"""Prepare and validate the static Chapter 4 four-method admission staging.

The staging is deliberately non-admitted and scratch-only.  It byte-preserves
the active noiseless, noisy and selection numerical payloads, snapshots the one
admitted count-scale parent that they require, and replaces only the oversized
presentation derivatives with fixed-width successors.  Validation is
self-contained once the staging directory exists: it never reads a live
``.scratch`` parent.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import platform
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402


FAMILY = "chapter_4_three_state_four_method_static_v3"
STAGING_RELATIVE = f".scratch/{FAMILY}_admission_staging"
TARGET_RELATIVE = f"results/{FAMILY}"
METHODS = ("pci", "dgi", "dffi", "dpfi")
STATES = ("smooth_bec", "connected_modulated", "separated_droplets")
STATE_LABELS = dict(zip(STATES, ("BEC", "SSP", "ID"), strict=True))
METHOD_LABELS = {method: method.upper() for method in METHODS}
ROLE_IDS = {
    "pci": {"atom": 1, "bright_reference": 2, "dark": 3},
    "dgi": {
        "atom_stop": 1,
        "leakage_stop": 2,
        "stop_dark": 3,
        "open_reference": 4,
        "open_dark": 5,
    },
    "dffi": {
        "crossed_atom": 1,
        "crossed_blank": 2,
        "crossed_dark": 3,
        "open_reference": 4,
        "open_dark": 5,
    },
    "dpfi": {
        "atom_h": 1,
        "atom_v": 2,
        "blank_h": 3,
        "blank_v": 4,
        "dark_h": 5,
        "dark_v": 6,
    },
}
METHOD_IDS = {method: index for index, method in enumerate(METHODS, start=1)}
STATE_IDS = {state: index for index, state in enumerate(STATES, start=1)}
EXPECTED_NUMERICAL_FILES = {
    "data/noiseless/state_method_extrema.csv",
    "data/noiseless/target_four_method_camera_and_fields.npz",
    "data/noise/four_method_noise_arrays.npz",
    "data/noise/four_method_snr.csv",
    "data/noise/seed_ledger.csv",
    "data/selection/selected_acquisition_contract.json",
    "data/selection/selection_at_200us.csv",
    "data/selection/selection_statement.txt",
}
EXPECTED_PRESENTATION_FILES = {
    "presentation/figure_4_target_four_method_noiseless_common_scale.pdf",
    "presentation/figure_4_target_four_method_noiseless_common_scale.png",
    "presentation/figure_4_target_four_method_noisy_200us.pdf",
    "presentation/figure_4_target_four_method_noisy_200us.png",
    "presentation/figure_4_target_four_method_snr.pdf",
    "presentation/figure_4_target_four_method_snr.png",
}
EXPECTED_ARTIFACT_COUNT = 79
EXPECTED_ARTIFACT_PATHS_SHA256 = (
    "4078d06645fb8e2a7ec3844b372994ec5e9dd1b183813b607a7b8a00dae59d43"
)
EXPECTED_CONFIG_CORE_SHA256 = (
    "7b8c799160d92a41b03d735f8934f491c216087557060501ffb399c2dae4ccfa"
)
EXPECTED_CLAIM_BOUNDARY = {
    "supports": [
        "common-I0 noiseless fractional camera response for PCI, DGI, DFFI and DPFI under the declared final Trap-III geometry",
        "one fixed draw-0 200-us noisy camera acquisition for each of the four methods and three target states under the declared ideal detector model",
        "whole-image template SNR over the 25-300-us duration grid",
        "configuration-specific continuation with DPFI first and DGI second by whole-image SNR for each state at 200 us",
        "forward ID response with explicit stronger-branch phase wrapping",
    ],
    "does_not_support": [
        "universal method superiority or installed-apparatus performance",
        "feature precision, spacing recovery, state classification or experimental probability",
        "a globally single-valued ID density inversion",
        "calibrated extinction, port-gain balance, registration or differential-transfer performance",
        "an optimum pulse duration, detector threshold or experimental validation",
        "Oxford, multiframe, Chapter-5, binary-mixture or dynamical claims",
    ],
}
EXPECTED_PAYLOAD_SHA256 = {
    "data/noiseless/state_method_extrema.csv": "b202daa4b0b4b4817f5e713118d6dd6a7df71b7fc5842eb02600fa07fce58ea1",
    "data/noiseless/target_four_method_camera_and_fields.npz": "dc3bae3101ffda28ee5a8f8a5c32c3b12950c577fabac2df72177206c5e7707e",
    "data/noise/four_method_noise_arrays.npz": "04f969962be2ade2fd43b80b2a0534169eef4ce1739a2b8fdc3e28472628df20",
    "data/noise/four_method_snr.csv": "8ae12590e954d14577a853f809eb6ad9555d6c15a9b82289b4a52a9ef774b460",
    "data/noise/seed_ledger.csv": "7209071bb016d0e9c86238ea5d40b02cd7b283056e03d48bc814376dc0ff2763",
    "data/selection/selected_acquisition_contract.json": "2c5b563cac01aeaa66096d158aa935e12775f1757ebc0472e6ade0cf413cb1f3",
    "data/selection/selection_at_200us.csv": "c9cf3034300e45ca9c4ae2faab1f460eb04acb3e806eabedf4ef94a2468b2e05",
    "data/selection/selection_statement.txt": "a6f9411b565be950ed6e115ed0736b21a4d1b29cec3358c826441c11670eeb03",
}
CAPTION_FILES = {
    "noiseless": "captions/figure_4_target_four_method_noiseless_common_scale.txt",
    "noisy": "captions/figure_4_target_four_method_noisy_200us.txt",
    "snr": "captions/figure_4_target_four_method_snr.txt",
}
PARENT_SNAPSHOT_FILES = {
    "noiseless_manifest": "lineage/parents/noiseless_artifact_manifest.json",
    "noiseless_config": "lineage/parents/noiseless_config_snapshot.json",
    "noiseless_summary": "lineage/parents/noiseless_summary.json",
    "noiseless_provenance": "lineage/parents/noiseless_provenance.json",
    "noiseless_presentation_manifest": "lineage/parents/noiseless_presentation_manifest.json",
    "noiseless_presentation_metadata": "lineage/parents/noiseless_presentation_metadata.json",
    "noise_manifest": "lineage/parents/noise_artifact_manifest.json",
    "noise_config": "lineage/parents/noise_config_snapshot.json",
    "noise_summary": "lineage/parents/noise_summary.json",
    "noise_provenance": "lineage/parents/noise_provenance.json",
    "noise_presentation_manifest": "lineage/parents/noise_presentation_manifest.json",
    "noise_presentation_metadata": "lineage/parents/noise_presentation_metadata.json",
    "qualification_manifest": "lineage/parents/qualification_artifact_manifest.json",
    "qualification_config": "lineage/parents/qualification_config_snapshot.json",
    "qualification_summary": "lineage/parents/qualification_summary.json",
    "qualification_provenance": "lineage/parents/qualification_provenance.json",
    "count_scale_manifest": "lineage/count_scale_parent/artifact_manifest.json",
    "count_scale_csv": "lineage/count_scale_parent/deterministic_validation.csv",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, f"CSV header missing: {path}")
        return list(reader), list(reader.fieldnames)


def _relative(value: Any, context: str) -> str:
    _require(isinstance(value, str) and value and "\\" not in value, f"invalid {context}")
    path = Path(value)
    _require(
        not path.is_absolute() and ".." not in path.parts and path.as_posix() == value,
        f"invalid {context}",
    )
    return value


def _config_core_sha256(config: Mapping[str, Any]) -> str:
    core = dict(config)
    closure = config.get("closure_sources")
    _require(isinstance(closure, list), "closure sources missing")
    core["closure_sources"] = [
        {"path": record.get("path")} if isinstance(record, Mapping) else record
        for record in closure
    ]
    encoded = json.dumps(
        core, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(config.get("schema_version") == 1, "config schema changed")
    _require(config.get("family") == FAMILY, "family changed")
    _require(config.get("staging") == STAGING_RELATIVE, "staging path changed")
    _require(config.get("target") == TARGET_RELATIVE, "target path changed")
    lifecycle = config.get("lifecycle")
    _require(
        lifecycle
        == {
            "status": "admission_staging",
            "admitted": False,
            "overwrite": False,
            "resume": False,
            "results_writes": 0,
        },
        "lifecycle changed",
    )
    recomputation = config.get("recomputation")
    _require(
        recomputation
        == {
            "new_random_draws": 0,
            "new_fits": 0,
            "new_optical_propagations": 0,
            "new_thermodynamic_solves": 0,
            "new_presentations": 3,
        },
        "recomputation boundary changed",
    )
    figure = config.get("presentation")
    _require(isinstance(figure, Mapping), "presentation contract missing")
    _require(float(figure.get("width_inches", 0.0)) == 6.61, "figure width changed")
    _require(float(figure.get("minimum_font_pt", 0.0)) >= 9.0, "font floor changed")
    fonts = figure.get("font_sizes_pt")
    _require(
        isinstance(fonts, Mapping)
        and fonts
        and min(float(value) for value in fonts.values()) >= float(figure["minimum_font_pt"]),
        "font contract changed",
    )
    _require(
        figure.get("noiseless_common_scale") == 2.011999399447263
        and figure.get("noisy_common_scale") == 2.1105806605656316,
        "display scales changed",
    )
    parents = config.get("parents")
    _require(isinstance(parents, Mapping), "parents missing")
    for name in (
        "noiseless",
        "noiseless_presentation",
        "noise",
        "noise_presentation",
        "qualification",
        "count_scale",
    ):
        record = parents.get(name)
        _require(isinstance(record, Mapping), f"parent missing: {name}")
        _relative(record.get("directory"), f"{name} directory")
        _require(
            isinstance(record.get("manifest_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(record["manifest_sha256"])),
            f"parent hash missing: {name}",
        )
    closure = config.get("closure_sources")
    _require(isinstance(closure, list) and closure, "closure sources missing")
    paths: list[str] = []
    for record in closure:
        _require(isinstance(record, Mapping), "closure record invalid")
        paths.append(_relative(record.get("path"), "closure path"))
        _require(
            isinstance(record.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])),
            "closure hash invalid",
        )
    _require(len(paths) == len(set(paths)), "closure paths are not unique")
    _require(config.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY, "claim boundary changed")
    _require(
        _config_core_sha256(config) == EXPECTED_CONFIG_CORE_SHA256,
        "frozen config semantics changed",
    )
    return dict(config)


def _load_config(path: Path) -> dict[str, Any]:
    return _validate_config(_read_json(path))


def _manifest_records(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("artifacts")
    _require(isinstance(rows, list), "manifest artifacts missing")
    records: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "manifest record invalid")
        relative = _relative(row.get("path"), "artifact path")
        _require(relative not in records, "duplicate artifact path")
        records[relative] = row
    return records


def _authenticate_parent(root: Path, record: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    directory = root / str(record["directory"])
    manifest_path = directory / "artifact_manifest.json"
    _require(
        manifest_path.is_file() and _sha256(manifest_path) == record["manifest_sha256"],
        f"parent manifest changed: {directory}",
    )
    manifest = _read_json(manifest_path)
    for relative, digest in record.get("required_files", {}).items():
        path = directory / _relative(relative, "parent artifact")
        _require(path.is_file() and _sha256(path) == digest, f"parent file changed: {path}")
        _require(
            _manifest_records(manifest).get(relative, {}).get("sha256") == digest,
            f"parent membership changed: {path}",
        )
    return directory, manifest


def validate_sources(root: Path, config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    for record in config["closure_sources"]:
        path = root / record["path"]
        _require(path.is_file() and _sha256(path) == record["sha256"], f"source changed: {path}")
    for record in config["parents"].values():
        _authenticate_parent(root, record)
    return config


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _file_record(path: Path, directory: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(directory).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _manifest(directory: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = [
        _file_record(path, directory)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    paths = [record["path"] for record in artifacts]
    _require(
        len(paths) == EXPECTED_ARTIFACT_COUNT
        and hashlib.sha256(("\n".join(paths) + "\n").encode("utf-8")).hexdigest()
        == EXPECTED_ARTIFACT_PATHS_SHA256,
        "exact artifact inventory changed",
    )
    return {
        "schema_version": 1,
        "family": FAMILY,
        "status": "admission_staging",
        "admitted": False,
        "consumable": False,
        "self_hash_policy": "manifest_excluded",
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(row["bytes"]) for row in artifacts),
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
        "artifacts": artifacts,
    }


def _verify_manifest_before_load(directory: Path) -> dict[str, Any]:
    manifest = _read_json(directory / "artifact_manifest.json")
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("family") == FAMILY
        and manifest.get("status") == "admission_staging"
        and manifest.get("admitted") is False
        and manifest.get("consumable") is False
        and manifest.get("self_hash_policy") == "manifest_excluded"
        and manifest.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY,
        "artifact manifest semantics changed",
    )
    records = _manifest_records(manifest)
    paths = sorted(records)
    _require(
        manifest.get("artifact_count") == EXPECTED_ARTIFACT_COUNT
        and len(paths) == EXPECTED_ARTIFACT_COUNT
        and hashlib.sha256(("\n".join(paths) + "\n").encode("utf-8")).hexdigest()
        == EXPECTED_ARTIFACT_PATHS_SHA256,
        "exact artifact inventory changed",
    )
    actual = {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _require(set(actual) == set(records), "exact artifact inventory changed")
    total_bytes = 0
    for relative, path in actual.items():
        record = records[relative]
        size = path.stat().st_size
        digest = _sha256(path)
        _require(
            record.get("bytes") == size and record.get("sha256") == digest,
            f"artifact manifest changed: {relative}",
        )
        total_bytes += size
    _require(manifest.get("total_bytes") == total_bytes, "artifact byte total changed")
    return manifest


def _page_size_points(path: Path) -> tuple[float, float]:
    boxes = re.findall(
        rb"/MediaBox\s*\[\s*([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s+([-+0-9.]+)\s*\]",
        path.read_bytes(),
    )
    _require(len(boxes) == 1, f"one PDF page required: {path}")
    x0, y0, x1, y1 = (float(item) for item in boxes[0])
    return x1 - x0, y1 - y0


def _display_arrays(noiseless_path: Path) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, str], np.ndarray]]:
    with np.load(noiseless_path, allow_pickle=False) as data:
        y = np.asarray(data["camera_y_m"], dtype=float) * 1e6
        z = np.asarray(data["camera_z_m"], dtype=float) * 1e6
        arrays: dict[tuple[str, str], np.ndarray] = {}
        for state in STATES:
            arrays[(state, "pci")] = np.asarray(data[f"{state}__pci_camera_intensity_over_i0"], dtype=float) - 0.9025
            arrays[(state, "dgi")] = np.asarray(data[f"{state}__dgi_camera_intensity_over_i0"], dtype=float) - 1e-4
            arrays[(state, "dffi")] = np.asarray(data[f"{state}__dffi_camera_intensity_over_i0"], dtype=float)
            arrays[(state, "dpfi")] = np.asarray(data[f"{state}__dpfi_difference_camera_intensity_over_i0"], dtype=float)
    return y, z, arrays


def _extent(y: np.ndarray, z: np.ndarray) -> list[float]:
    return [
        float(y[0] - np.diff(y).mean() / 2.0),
        float(y[-1] + np.diff(y).mean() / 2.0),
        float(z[0] - np.diff(z).mean() / 2.0),
        float(z[-1] + np.diff(z).mean() / 2.0),
    ]


def _figure_rc(config: Mapping[str, Any]) -> dict[str, Any]:
    fonts = config["presentation"]["font_sizes_pt"]
    return {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": float(fonts["base"]),
        "axes.titlesize": float(fonts["title"]),
        "axes.labelsize": float(fonts["axes_label"]),
        "xtick.labelsize": float(fonts["tick"]),
        "ytick.labelsize": float(fonts["tick"]),
        "legend.fontsize": float(fonts["legend"]),
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.75,
    }


def _save(figure: Any, pdf: Path, png: Path, title: str, dpi: int) -> None:
    fixed = dt.datetime(2026, 8, 18, tzinfo=dt.timezone.utc)
    figure.savefig(
        pdf,
        metadata={
            "Title": title,
            "Creator": FAMILY,
            "Producer": "Matplotlib",
            "CreationDate": fixed,
            "ModDate": fixed,
        },
    )
    figure.savefig(png, dpi=dpi, metadata={"Software": FAMILY})
    plt.close(figure)


def _render_panel_grid(
    data_path: Path,
    output: Path,
    config: Mapping[str, Any],
    *,
    noisy: bool,
) -> None:
    figure_contract = config["presentation"]
    width = float(figure_contract["width_inches"])
    height = float(figure_contract["panel_height_inches"])
    dpi = int(figure_contract["dpi"])
    limit = float(
        figure_contract["noisy_common_scale"] if noisy else figure_contract["noiseless_common_scale"]
    )
    if noisy:
        with np.load(data_path, allow_pickle=False) as stored:
            y = np.asarray(stored["camera_y_m"], dtype=float) * 1e6
            z = np.asarray(stored["camera_z_m"], dtype=float) * 1e6
            arrays = {
                (state, method): np.asarray(stored[f"{state}__{method}__observed_signal"], dtype=float)
                for method in METHODS
                for state in STATES
            }
    else:
        y, z, arrays = _display_arrays(data_path)
    replay_limit = max(float(np.max(np.abs(values))) for values in arrays.values())
    _require(math.isclose(replay_limit, limit, rel_tol=0.0, abs_tol=1e-12), "display limit changed")
    with plt.rc_context(_figure_rc(config)):
        figure, axes = plt.subplots(4, 3, figsize=(width, height), sharex=True, sharey=True)
        normalisation = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = None
        for row, method in enumerate(METHODS):
            for column, state in enumerate(STATES):
                axis = axes[row, column]
                image = axis.imshow(
                    arrays[(state, method)],
                    origin="lower",
                    extent=_extent(y, z),
                    cmap="RdBu_r",
                    norm=normalisation,
                    interpolation="nearest",
                    aspect="equal",
                )
                if row == 0:
                    axis.set_title(STATE_LABELS[state], pad=2.0)
                if column == 0:
                    axis.set_ylabel(r"$z\ (\mu\mathrm{m})$", labelpad=1.5)
                if row == 3:
                    axis.set_xlabel(r"$y\ (\mu\mathrm{m})$", labelpad=1.5)
                axis.set_yticks([-1, 1])
                axis.set_xticks([-10, -5, 0, 5, 10])
                axis.tick_params(length=2.2, pad=1.5)
            figure.text(0.018, (0.84, 0.63, 0.42, 0.21)[row], METHOD_LABELS[method], ha="left", va="center")
        _require(image is not None, "panel rendering failed")
        figure.subplots_adjust(left=0.135, right=0.875, bottom=0.075, top=0.965, wspace=0.08, hspace=0.12)
        colour_axis = figure.add_axes([0.90, 0.13, 0.018, 0.78])
        colourbar = figure.colorbar(image, cax=colour_axis)
        colourbar.set_ticks([-2.0, -1.0, 0.0, 1.0, 2.0])
        colourbar.set_label(
            "Processed camera response / incident $I_0$" if noisy else "Fractional camera response / incident $I_0$",
            labelpad=3.0,
        )
        stem = "figure_4_target_four_method_noisy_200us" if noisy else "figure_4_target_four_method_noiseless_common_scale"
        _save(figure, output / f"{stem}.pdf", output / f"{stem}.png", stem, dpi)


def _render_snr(csv_path: Path, output: Path, config: Mapping[str, Any]) -> None:
    rows, _ = _read_csv(csv_path)
    figure_contract = config["presentation"]
    colours = {"smooth_bec": "#395C8C", "connected_modulated": "#B26B31", "separated_droplets": "#4F7B52"}
    styles = {"smooth_bec": ("-", "o", "white"), "connected_modulated": ("--", "s", None), "separated_droplets": (":", "D", "white")}
    with plt.rc_context(_figure_rc(config)):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(float(figure_contract["width_inches"]), float(figure_contract["snr_height_inches"])),
            sharex=True,
            sharey=True,
        )
        for axis, method in zip(axes.flat, METHODS, strict=True):
            for state in STATES:
                selected = [row for row in rows if row["method"] == method and row["state"] == state]
                line, marker, face = styles[state]
                markevery = (0, 2) if method == "pci" and state == "smooth_bec" else ((1, 2) if method == "pci" and state == "separated_droplets" else None)
                axis.plot(
                    [float(row["pulse_duration_us"]) for row in selected],
                    [float(row["image_snr"]) for row in selected],
                    linestyle=line,
                    marker=marker,
                    markersize=3.2,
                    color=colours[state],
                    markerfacecolor=colours[state] if face is None else face,
                    markeredgecolor=colours[state],
                    markeredgewidth=0.8,
                    linewidth=1.4,
                    markevery=markevery,
                    label=STATE_LABELS[state],
                )
            axis.axvline(200.0, color="#555555", linestyle="--", linewidth=1.0)
            axis.set_title(METHOD_LABELS[method], pad=2.5)
            axis.grid(alpha=0.18)
            axis.tick_params(length=2.5, pad=2.0)
            if method == "pci":
                axis.text(0.04, 0.08, r"BEC $\simeq$ ID", transform=axis.transAxes, color="#395C8C")
        for axis in axes[:, 0]:
            axis.set_ylabel("Whole-image SNR")
        for axis in axes[1, :]:
            axis.set_xlabel(r"Pulse duration $\tau\ (\mu\mathrm{s})$")
        axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
        figure.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.95, wspace=0.12, hspace=0.20)
        stem = "figure_4_target_four_method_snr"
        _save(
            figure,
            output / f"{stem}.pdf",
            output / f"{stem}.png",
            stem,
            int(figure_contract["dpi"]),
        )


def _process(method: str, raw: Mapping[str, np.ndarray]) -> np.ndarray:
    if method == "pci":
        denominator = raw["bright_reference"] - raw["dark"]
        _require(np.all(denominator > 0.0), "PCI denominator failed")
        return 0.95**2 * ((raw["atom"] - raw["dark"]) / denominator - 1.0)
    if method == "dgi":
        denominator = raw["open_reference"] - raw["open_dark"]
        _require(np.all(denominator > 0.0), "DGI denominator failed")
        return (raw["atom_stop"] - raw["leakage_stop"]) / denominator
    if method == "dffi":
        denominator = raw["open_reference"] - raw["open_dark"]
        _require(np.all(denominator > 0.0), "DFFI denominator failed")
        return (raw["crossed_atom"] - raw["crossed_blank"]) / denominator
    if method == "dpfi":
        denominator = (raw["blank_h"] - raw["dark_h"]) + (raw["blank_v"] - raw["dark_v"])
        _require(np.all(denominator > 0.0), "DPFI denominator failed")
        return ((raw["atom_h"] - raw["dark_h"]) - (raw["atom_v"] - raw["dark_v"])) / denominator
    raise ValueError(method)


def _sample(expected: np.ndarray, read_noise: float, seed: tuple[int, ...]) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    return rng.poisson(expected).astype(float) + rng.normal(0.0, read_noise, expected.shape)


def _validate_noiseless(directory: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = directory / "data/noiseless/target_four_method_camera_and_fields.npz"
    with np.load(path, allow_pickle=False) as data:
        _require(len(data.files) == 66, "noiseless inventory changed")
        _require(np.asarray(data["camera_y_m"]).shape == (71,), "camera y changed")
        _require(np.asarray(data["camera_z_m"]).shape == (17,), "camera z changed")
        wrapped = 0
        maximum_phase = 0.0
        for state in STATES:
            column = np.asarray(data[f"{state}__column_density_m2"], dtype=float)
            phases = np.asarray(data[f"{state}__branch_phase_maps_rad"], dtype=float)
            _require(column.shape == (161, 641) and phases.shape == (2, 161, 641), "object shape changed")
            _require(np.isfinite(column).all() and np.all(column >= 0.0) and np.isfinite(phases).all(), "noiseless domain failed")
            maximum_phase = max(maximum_phase, float(np.max(np.abs(phases))))
            wrapped += int(np.count_nonzero(np.abs(phases) > np.pi))
    _, _, arrays = _display_arrays(path)
    limit = max(float(np.max(np.abs(values))) for values in arrays.values())
    _require(math.isclose(limit, config["presentation"]["noiseless_common_scale"], abs_tol=1e-12), "noiseless scale changed")
    _require(math.isclose(maximum_phase, 3.7073737479043483, abs_tol=1e-12) and wrapped == 137, "phase wrapping changed")
    rows, fields = _read_csv(directory / "data/noiseless/state_method_extrema.csv")
    _require(len(rows) == 12 and fields == ["state", "display_label", "method", "minimum", "maximum", "maximum_absolute", "negative_pixel_count", "sum"], "extrema table changed")
    return {"common_scale": limit, "maximum_phase_rad": maximum_phase, "wrapped_pixels": wrapped}


def _validate_payload_parent_identities(directory: Path) -> None:
    for relative, digest in EXPECTED_PAYLOAD_SHA256.items():
        path = directory / relative
        _require(
            path.is_file() and _sha256(path) == digest,
            f"payload parent identity changed: {relative}",
        )


def _validate_noise(directory: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    data_path = directory / "data/noise/four_method_noise_arrays.npz"
    seed_rows, seed_fields = _read_csv(directory / "data/noise/seed_ledger.csv")
    _require(seed_fields == ["method", "state", "role", "seed"], "seed ledger header changed")
    _require(len(seed_rows) == 57 and len({row["seed"] for row in seed_rows}) == 57, "seed inventory changed")
    expected_keys = {(method, state, role) for method in METHODS for state in STATES for role in ROLE_IDS[method]}
    _require({(row["method"], row["state"], row["role"]) for row in seed_rows} == expected_keys, "seed keys changed")
    read_noise = 0.7
    with np.load(data_path, allow_pickle=False) as data:
        _require(len(data.files) == 152, "noise array inventory changed")
        for row in seed_rows:
            method, state, role = row["method"], row["state"], row["role"]
            seed = tuple(int(item) for item in row["seed"].split(":"))
            expected_seed = (20260818, 4405, METHOD_IDS[method], STATE_IDS[state], ROLE_IDS[method][role], 0)
            _require(seed == expected_seed, "seed schema changed")
            expected = np.asarray(data[f"{state}__{method}__{role}__expected_electrons"], dtype=float)
            observed = np.asarray(data[f"{state}__{method}__{role}__observed_electrons"], dtype=float)
            _require(expected.shape == observed.shape == (17, 71), "raw camera shape changed")
            _require(np.isfinite(expected).all() and np.all(expected >= 0.0), "raw expected domain failed")
            _require(np.array_equal(observed, _sample(expected, read_noise, seed)), "RNG replay failed")
        for method in METHODS:
            for state in STATES:
                raw = {role: np.asarray(data[f"{state}__{method}__{role}__observed_electrons"], dtype=float) for role in ROLE_IDS[method]}
                _require(np.array_equal(_process(method, raw), data[f"{state}__{method}__observed_signal"]), "processed replay failed")
        limit = max(float(np.max(np.abs(data[f"{state}__{method}__observed_signal"]))) for method in METHODS for state in STATES)
        _require(math.isclose(limit, config["presentation"]["noisy_common_scale"], abs_tol=1e-12), "noisy scale changed")
        snr_200 = {
            (method, state): float(np.sqrt(np.sum(np.asarray(data[f"{state}__{method}__expected_signal"]) ** 2 / np.asarray(data[f"{state}__{method}__variance"]))))
            for method in METHODS
            for state in STATES
        }
    rows, fields = _read_csv(directory / "data/noise/four_method_snr.csv")
    _require(fields == ["pulse_duration_us", "method", "state", "image_snr", "count_scale_electrons_per_i0_pixel"] and len(rows) == 144, "SNR table changed")
    keyed = {(int(row["pulse_duration_us"]), row["method"], row["state"]): row for row in rows}
    _require(len(keyed) == 144, "SNR keys changed")
    count_rows, count_fields = _read_csv(
        directory / PARENT_SNAPSHOT_FILES["count_scale_csv"]
    )
    _require(
        count_fields
        == [
            "pulse_duration_us",
            "method",
            "scene",
            "photoelectrons_per_i0_pixel",
            "raw_roles",
            "processed_minimum",
            "processed_maximum",
            "processed_direct_max_abs_error",
            "minimum_reference_denominator_electrons",
            "maximum_expected_raw_electrons",
            "minimum_full_well_margin_electrons_after_6sigma",
            "blank_processed_max_abs",
        ],
        "count-scale parent header changed",
    )
    blank_counts = {
        (int(row["pulse_duration_us"]), row["method"]): float(
            row["photoelectrons_per_i0_pixel"]
        )
        for row in count_rows
        if row["scene"] == "blank"
    }
    _require(len(blank_counts) == 24, "count-scale parent keys changed")
    for duration in range(25, 301, 25):
        pci_count = blank_counts[(duration, "pci")]
        dgi_count = blank_counts[(duration, "dgi")]
        _require(
            math.isclose(pci_count, dgi_count, rel_tol=0.0, abs_tol=1e-12),
            "admitted incident count scales disagree",
        )
        for method in METHODS:
            for state in STATES:
                _require(
                    math.isclose(
                        float(keyed[(duration, method, state)][
                            "count_scale_electrons_per_i0_pixel"
                        ]),
                        pci_count,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    "SNR count-scale lineage changed",
                )
    for key, value in snr_200.items():
        _require(math.isclose(float(keyed[(200, *key)]["image_snr"]), value, abs_tol=1e-12), "200-us SNR replay failed")
    for method in METHODS:
        for state in STATES:
            sequence = [float(keyed[(tau, method, state)]["image_snr"]) for tau in range(25, 301, 25)]
            _require(np.all(np.diff(sequence) > 0.0), "SNR duration sequence changed")
    return {"common_scale": limit, "snr_200us": {f"{state}__{method}": snr_200[(method, state)] for state in STATES for method in METHODS}, "snr_rows": len(rows), "seed_identities": len(seed_rows)}


def _validate_selection(directory: Path, noise: Mapping[str, Any]) -> dict[str, Any]:
    rows, fields = _read_csv(directory / "data/selection/selection_at_200us.csv")
    _require(fields == ["state", "rank", "method", "whole_image_snr", "pulse_duration_us"] and len(rows) == 12, "selection table changed")
    expected: list[tuple[str, int, str, float]] = []
    for state in STATES:
        values = [(method, float(noise["snr_200us"][f"{state}__{method}"])) for method in METHODS]
        values.sort(key=lambda item: item[1], reverse=True)
        expected.extend((state, rank, method, value) for rank, (method, value) in enumerate(values, start=1))
    actual = [(row["state"], int(row["rank"]), row["method"], float(row["whole_image_snr"])) for row in rows]
    _require(actual == expected, "selection ranking changed")
    _require(all([row[2] for row in actual if row[0] == state][:2] == ["dpfi", "dgi"] for state in STATES), "selected methods changed")
    contract = _read_json(directory / "data/selection/selected_acquisition_contract.json")
    _require(contract["dpfi"]["processed_quantity"] == "((atom_h-dark_h)-(atom_v-dark_v))/((blank_h-dark_h)+(blank_v-dark_v))", "DPFI contract changed")
    statement = (directory / "data/selection/selection_statement.txt").read_text(encoding="utf-8")
    _require("configuration-specific" in statement and "not a universal method ranking" in statement, "selection boundary changed")
    return {"selected_methods": ["dpfi", "dgi"], "selection_rows": len(rows)}


def _validate_lineage(directory: Path, config: Mapping[str, Any]) -> None:
    for record in config["closure_sources"]:
        path = directory / "lineage/source" / record["path"]
        _require(path.is_file() and _sha256(path) == record["sha256"], f"closure snapshot changed: {path}")
    parent_map = {
        "noiseless": ("noiseless_manifest", "noiseless_config", "noiseless_summary", "noiseless_provenance"),
        "noiseless_presentation": ("noiseless_presentation_manifest", "noiseless_presentation_metadata"),
        "noise": ("noise_manifest", "noise_config", "noise_summary", "noise_provenance"),
        "noise_presentation": ("noise_presentation_manifest", "noise_presentation_metadata"),
        "qualification": ("qualification_manifest", "qualification_config", "qualification_summary", "qualification_provenance"),
    }
    source_names = {
        "noiseless_manifest": "artifact_manifest.json",
        "noiseless_config": "config_snapshot.json",
        "noiseless_summary": "summary.json",
        "noiseless_provenance": "provenance.json",
        "noiseless_presentation_manifest": "artifact_manifest.json",
        "noiseless_presentation_metadata": "metadata.json",
        "noise_manifest": "artifact_manifest.json",
        "noise_config": "config_snapshot.json",
        "noise_summary": "summary.json",
        "noise_provenance": "provenance.json",
        "noise_presentation_manifest": "artifact_manifest.json",
        "noise_presentation_metadata": "metadata.json",
        "qualification_manifest": "artifact_manifest.json",
        "qualification_config": "config_snapshot.json",
        "qualification_summary": "summary.json",
        "qualification_provenance": "provenance.json",
    }
    for parent, names in parent_map.items():
        record = config["parents"][parent]
        for name in names:
            embedded = directory / PARENT_SNAPSHOT_FILES[name]
            expected_hash = record["manifest_sha256"] if source_names[name] == "artifact_manifest.json" else record["required_files"][source_names[name]]
            _require(embedded.is_file() and _sha256(embedded) == expected_hash, f"parent snapshot changed: {name}")
    count = config["parents"]["count_scale"]
    _require(_sha256(directory / PARENT_SNAPSHOT_FILES["count_scale_manifest"]) == count["manifest_sha256"], "count manifest changed")
    _require(_sha256(directory / PARENT_SNAPSHOT_FILES["count_scale_csv"]) == count["required_files"]["data/deterministic_validation.csv"], "count CSV changed")
    count_manifest = _read_json(directory / PARENT_SNAPSHOT_FILES["count_scale_manifest"])
    _require(count_manifest.get("admitted") is True and count_manifest.get("status") == "admitted_immutable", "count parent lifecycle changed")
    _require(_manifest_records(count_manifest).get("data/deterministic_validation.csv", {}).get("sha256") == count["required_files"]["data/deterministic_validation.csv"], "count parent membership changed")


def _validate_presentations(directory: Path, config: Mapping[str, Any], *, replay: bool) -> None:
    width = float(config["presentation"]["width_inches"])
    heights = {
        "figure_4_target_four_method_noiseless_common_scale.pdf": float(config["presentation"]["panel_height_inches"]),
        "figure_4_target_four_method_noisy_200us.pdf": float(config["presentation"]["panel_height_inches"]),
        "figure_4_target_four_method_snr.pdf": float(config["presentation"]["snr_height_inches"]),
    }
    for name, height in heights.items():
        path = directory / "presentation" / name
        actual = _page_size_points(path)
        _require(math.isclose(actual[0], width * 72.0, abs_tol=0.02) and math.isclose(actual[1], height * 72.0, abs_tol=0.02), f"PDF geometry changed: {name}")
        payload = path.read_bytes()
        _require(b"/FontFile2" in payload and b"/Type3" not in payload, f"PDF fonts changed: {name}")
        png = plt.imread(path.with_suffix(".png"))
        _require(png.shape[1] == round(width * int(config["presentation"]["dpi"])) and png.shape[0] == round(height * int(config["presentation"]["dpi"])), f"PNG geometry changed: {name}")
    if replay:
        with tempfile.TemporaryDirectory(prefix="chapter4-static-replay-") as temp_name:
            temp = Path(temp_name)
            _render_panel_grid(directory / "data/noiseless/target_four_method_camera_and_fields.npz", temp, config, noisy=False)
            _render_panel_grid(directory / "data/noise/four_method_noise_arrays.npz", temp, config, noisy=True)
            _render_snr(directory / "data/noise/four_method_snr.csv", temp, config)
            for relative in EXPECTED_PRESENTATION_FILES:
                _require((directory / relative).read_bytes() == (temp / Path(relative).name).read_bytes(), f"presentation replay changed: {relative}")


def _summary(config: Mapping[str, Any], noiseless: Mapping[str, Any], noise: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "family": FAMILY,
        "status": "admission_staging",
        "admitted": False,
        "consumable": False,
        "noiseless": dict(noiseless),
        "noise": dict(noise),
        "selection": dict(selection),
        "presentation": config["presentation"],
        "recomputation": config["recomputation"],
        "claim_boundary": config["claim_boundary"],
        "superseded_presentations": config["superseded_presentations"],
    }


def _provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "family": FAMILY,
        "status": "admission_staging",
        "admitted": False,
        "consumable": False,
        "calculation": "byte-preserving consolidation of frozen numerical parents plus presentation-only fixed-width successors",
        "parents": config["parents"],
        "closure_sources": config["closure_sources"],
        "recomputation": config["recomputation"],
        "scratch_independent_validation": True,
        "results_writes": 0,
    }


def validate_staging(directory: Path, *, replay_presentations: bool = True) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = _verify_manifest_before_load(directory)
    config = _load_config(directory / "config_snapshot.json")
    actual_files = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    _require(
        EXPECTED_NUMERICAL_FILES <= actual_files
        and EXPECTED_PRESENTATION_FILES <= actual_files,
        "staging inventory incomplete",
    )
    _validate_payload_parent_identities(directory)
    _validate_lineage(directory, config)
    noiseless = _validate_noiseless(directory, config)
    noise = _validate_noise(directory, config)
    selection = _validate_selection(directory, noise)
    _validate_presentations(directory, config, replay=replay_presentations)
    for key, relative in CAPTION_FILES.items():
        expected = config["captions"][key] + "\n"
        _require((directory / relative).read_text(encoding="utf-8") == expected, f"caption changed: {key}")
    _require(_read_json(directory / "summary.json") == _summary(config, noiseless, noise, selection), "summary changed")
    _require(_read_json(directory / "provenance.json") == _provenance(config), "provenance changed")
    environment = _read_json(directory / "environment.json")
    _require(environment.get("python") and environment.get("numpy") and environment.get("matplotlib"), "environment snapshot changed")
    return {
        "status": "PASS",
        "family": FAMILY,
        "admitted": False,
        "consumable": False,
        "artifact_count": manifest["artifact_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_sha256": _sha256(directory / "artifact_manifest.json"),
        "snr_rows": noise["snr_rows"],
        "seed_identities": noise["seed_identities"],
        "selected_methods": selection["selected_methods"],
        "minimum_font_pt": config["presentation"]["minimum_font_pt"],
    }


def prepare_staging(root: Path, config_path: Path, *, staging_override: Path | None = None) -> Path:
    root = root.resolve()
    config_path = config_path.resolve()
    config = validate_sources(root, config_path)
    staging = staging_override.resolve() if staging_override is not None else (root / config["staging"]).resolve()
    building = staging.with_name(staging.name + ".building")
    target = (root / config["target"]).resolve()
    _require(not staging.exists() and not building.exists(), "staging or building already exists")
    _require(not target.exists(), "retained target already exists")
    building.mkdir(parents=True)
    try:
        parent_dirs = {name: _authenticate_parent(root, record)[0] for name, record in config["parents"].items()}
        copy_map = {
            parent_dirs["noiseless"] / "data/state_method_extrema.csv": building / "data/noiseless/state_method_extrema.csv",
            parent_dirs["noiseless"] / "data/target_four_method_camera_and_fields.npz": building / "data/noiseless/target_four_method_camera_and_fields.npz",
            parent_dirs["noise"] / "data/four_method_noise_arrays.npz": building / "data/noise/four_method_noise_arrays.npz",
            parent_dirs["noise"] / "data/four_method_snr.csv": building / "data/noise/four_method_snr.csv",
            parent_dirs["noise"] / "data/seed_ledger.csv": building / "data/noise/seed_ledger.csv",
            parent_dirs["qualification"] / "data/selected_acquisition_contract.json": building / "data/selection/selected_acquisition_contract.json",
            parent_dirs["qualification"] / "data/selection_at_200us.csv": building / "data/selection/selection_at_200us.csv",
            parent_dirs["qualification"] / "selection_statement.txt": building / "data/selection/selection_statement.txt",
        }
        for source, destination in copy_map.items():
            _copy(source, destination)
        snapshot_sources = {
            "noiseless_manifest": parent_dirs["noiseless"] / "artifact_manifest.json",
            "noiseless_config": parent_dirs["noiseless"] / "config_snapshot.json",
            "noiseless_summary": parent_dirs["noiseless"] / "summary.json",
            "noiseless_provenance": parent_dirs["noiseless"] / "provenance.json",
            "noiseless_presentation_manifest": parent_dirs["noiseless_presentation"] / "artifact_manifest.json",
            "noiseless_presentation_metadata": parent_dirs["noiseless_presentation"] / "metadata.json",
            "noise_manifest": parent_dirs["noise"] / "artifact_manifest.json",
            "noise_config": parent_dirs["noise"] / "config_snapshot.json",
            "noise_summary": parent_dirs["noise"] / "summary.json",
            "noise_provenance": parent_dirs["noise"] / "provenance.json",
            "noise_presentation_manifest": parent_dirs["noise_presentation"] / "artifact_manifest.json",
            "noise_presentation_metadata": parent_dirs["noise_presentation"] / "metadata.json",
            "qualification_manifest": parent_dirs["qualification"] / "artifact_manifest.json",
            "qualification_config": parent_dirs["qualification"] / "config_snapshot.json",
            "qualification_summary": parent_dirs["qualification"] / "summary.json",
            "qualification_provenance": parent_dirs["qualification"] / "provenance.json",
            "count_scale_manifest": parent_dirs["count_scale"] / "artifact_manifest.json",
            "count_scale_csv": parent_dirs["count_scale"] / "data/deterministic_validation.csv",
        }
        for name, source in snapshot_sources.items():
            _copy(source, building / PARENT_SNAPSHOT_FILES[name])
        for record in config["closure_sources"]:
            _copy(root / record["path"], building / "lineage/source" / record["path"])
        _copy(config_path, building / "config_snapshot.json")
        for key, relative in CAPTION_FILES.items():
            path = building / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(config["captions"][key] + "\n", encoding="utf-8", newline="")
        presentation = building / "presentation"
        presentation.mkdir(parents=True)
        _render_panel_grid(building / "data/noiseless/target_four_method_camera_and_fields.npz", presentation, config, noisy=False)
        _render_panel_grid(building / "data/noise/four_method_noise_arrays.npz", presentation, config, noisy=True)
        _render_snr(building / "data/noise/four_method_snr.csv", presentation, config)
        noiseless = _validate_noiseless(building, config)
        noise = _validate_noise(building, config)
        selection = _validate_selection(building, noise)
        _write_json(building / "summary.json", _summary(config, noiseless, noise, selection))
        _write_json(building / "provenance.json", _provenance(config))
        _write_json(
            building / "environment.json",
            {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "matplotlib": matplotlib.__version__,
                "executable_name": Path(sys.executable).name,
            },
        )
        _write_json(building / "artifact_manifest.json", _manifest(building, config))
        validate_staging(building)
        building.replace(staging)
        validate_staging(staging)
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise
    return staging


__all__ = ["prepare_staging", "validate_sources", "validate_staging"]
