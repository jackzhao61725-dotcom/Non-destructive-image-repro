"""Generate the bounded 200-us four-method noisy comparison for target profiles."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isolated_non_destructive_image import (  # noqa: E402
    load_isolated_non_destructive_image_module,
)


PCI_DGI = load_isolated_non_destructive_image_module(
    "chapter4_noise_weighted_snr", namespace="_target_noise_pci_dgi"
)
FARADAY = load_isolated_non_destructive_image_module(
    "four_method_camera_noise", namespace="_target_noise_faraday"
)


CONFIG = ROOT / "configs/target_three_state_four_method_noise_v5.json"
STATES = ("smooth_bec", "connected_modulated", "separated_droplets")
METHODS = ("pci", "dgi", "dffi", "dpfi")
STATE_LABELS = dict(zip(STATES, ("BEC", "SSP", "ID"), strict=True))
METHOD_LABELS = {method: method.upper() for method in METHODS}
METHOD_IDS = {method: index for index, method in enumerate(METHODS, start=1)}
STATE_IDS = {state: index for index, state in enumerate(STATES, start=1)}
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sample(expected: np.ndarray, read_noise: float, seed: tuple[int, ...]) -> np.ndarray:
    expected = np.asarray(expected, dtype=float)
    _require(np.isfinite(expected).all() and np.all(expected >= 0.0), "invalid expected counts")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    return rng.poisson(expected).astype(float) + rng.normal(0.0, read_noise, expected.shape)


def _count_scale_100(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["pulse_duration_us"] == "100"
        and row["method"] == "pci"
        and row["scene"] == "blank"
    ]
    _require(len(selected) == 1, "cannot identify the 100-us incident count scale")
    return float(selected[0]["photoelectrons_per_i0_pixel"])


def validate_config(config_path: Path = CONFIG) -> dict[str, Any]:
    config = _json(config_path)
    _require(config.get("schema_version") == 1, "unsupported config schema")
    _require(config.get("label") == "target_three_state_four_method_noise_v5", "label changed")
    lifecycle = config["lifecycle"]
    _require(
        lifecycle
        == {
            "status": "diagnostic_not_retained",
            "admitted": False,
            "output_directory": ".scratch/target_three_state_four_method_noise_v5",
            "overwrite": False,
            "results_writes": 0,
        },
        "lifecycle changed",
    )
    _require(tuple(config["states"]) == STATES and tuple(config["methods"]) == METHODS, "state/method set changed")
    _require(
        config["pulse_durations_us"] == list(range(25, 301, 25))
        and config["display_pulse_duration_us"] == 200,
        "duration contract changed",
    )
    _require(
        float(config["detector"]["read_noise_electrons_rms_per_port_pixel"]) == 0.7
        and float(config["detector"]["dark_mean_electrons_per_pixel"]) == 0.0,
        "detector contract changed",
    )
    _require(
        float(config["detector"]["full_well_electrons"]) == 15000.0
        and config["detector"]["full_well_gate"] == "maximum_expected_plus_6sigma_below_full_well"
        and config["detector"]["clipping"] is False,
        "full-well contract changed",
    )
    randomness = config["randomness"]
    _require(
        randomness["bit_generator"] == "PCG64"
        and randomness["seed_prefix"] == [20260818, 4405]
        and int(randomness["draw_id"]) == 0
        and int(randomness["expected_random_identities"]) == 57
        and randomness["redraws_allowed"] is False,
        "randomness contract changed",
    )
    _require(
        config["acquisition"]["dpfi_port_policy"]
        == "H_and_V_are_camera_sampled_and_noised_independently_before_their_common_I0_difference_is_formed"
        and config["acquisition"]["redraws_allowed"] is False,
        "acquisition contract changed",
    )
    parents = config["parents"]
    for key in (
        "noiseless_manifest",
        "noiseless_payload",
        "noiseless_summary",
        "incident_count_scale",
        "incident_count_manifest",
    ):
        record = parents[key]
        path = ROOT / record["path"]
        _require(path.is_file() and _sha256(path) == record["sha256"], f"parent changed: {key}")
    noiseless_manifest = _json(ROOT / parents["noiseless_manifest"]["path"])
    _require(noiseless_manifest["admitted"] is False, "scratch parent lifecycle changed")
    noiseless_summary = _json(ROOT / parents["noiseless_summary"]["path"])
    _require(
        noiseless_summary["status"] == "diagnostic_not_retained"
        and noiseless_summary["admitted"] is False
        and noiseless_summary["phase_wrapping_present"] is True
        and int(noiseless_summary["wrapped_circular_branch_pixel_count"]) > 0,
        "noiseless phase boundary changed",
    )
    count_manifest = _json(ROOT / parents["incident_count_manifest"]["path"])
    _require(count_manifest["admitted"] is True, "incident count-scale parent is not admitted")
    for name, record in config["implementation"].items():
        path = ROOT / record["path"]
        _require(
            path.is_file() and _sha256(path) == record["sha256"],
            f"implementation source changed: {name}",
        )
    return config


def _noise_model(
    method: str,
    state: str,
    count_scale: float,
    data: np.lib.npyio.NpzFile,
    read_noise: float,
) -> tuple[Any, dict[str, np.ndarray]]:
    shape = data[f"{state}__pci_camera_intensity_over_i0"].shape
    zero = np.zeros(shape, dtype=float)
    one = np.ones(shape, dtype=float)
    if method == "pci":
        means = {
            "atom": count_scale * data[f"{state}__pci_camera_intensity_over_i0"],
            "bright_reference": count_scale * 0.95**2 * one,
            "dark": zero,
        }
        model = PCI_DGI.pci_frame_noise_model(
            means["atom"],
            means["bright_reference"],
            means["dark"],
            zero,
            phase_plate_amplitude_transmittance=0.95,
            read_noise_electrons_rms=read_noise,
        )
        return model, means
    if method == "dgi":
        means = {
            "atom_stop": count_scale * data[f"{state}__dgi_camera_intensity_over_i0"],
            "leakage_stop": count_scale * 1e-4 * one,
            "stop_dark": zero,
            "open_reference": count_scale * one,
            "open_dark": zero,
        }
        model = PCI_DGI.dgi_frame_noise_model(
            means["atom_stop"],
            means["leakage_stop"],
            means["stop_dark"],
            means["open_reference"],
            means["open_dark"],
            zero,
            open_to_stop_scale=1.0,
            read_noise_electrons_rms=read_noise,
        )
        return model, means
    if method == "dffi":
        model = FARADAY.dffi_frame_noise_model(
            count_scale * data[f"{state}__dffi_camera_intensity_over_i0"],
            zero,
            zero,
            count_scale * one,
            zero,
            read_noise_electrons_rms=read_noise,
        )
        return model, dict(model.raw_means)
    if method == "dpfi":
        model = FARADAY.dpfi_frame_noise_model(
            count_scale * data[f"{state}__dpfi_h_camera_intensity_over_i0"],
            count_scale * data[f"{state}__dpfi_v_camera_intensity_over_i0"],
            count_scale * 0.5 * one,
            count_scale * 0.5 * one,
            zero,
            zero,
            read_noise_electrons_rms=read_noise,
        )
        return model, dict(model.raw_means)
    raise ValueError(method)


def _process(method: str, raw: dict[str, np.ndarray]) -> np.ndarray:
    if method == "pci":
        denominator = raw["bright_reference"] - raw["dark"]
        _require(np.all(denominator > 0.0), "sampled PCI denominator is not positive")
        return 0.95**2 * ((raw["atom"] - raw["dark"]) / denominator - 1.0)
    if method == "dgi":
        denominator = raw["open_reference"] - raw["open_dark"]
        _require(np.all(denominator > 0.0), "sampled DGI denominator is not positive")
        return (raw["atom_stop"] - raw["leakage_stop"]) / denominator
    if method == "dffi":
        return FARADAY.process_dffi_counts(
            raw["crossed_atom"], raw["crossed_blank"], raw["open_reference"], raw["open_dark"]
        )
    if method == "dpfi":
        return FARADAY.process_dpfi_counts(
            raw["atom_h"], raw["atom_v"], raw["blank_h"], raw["blank_v"], raw["dark_h"], raw["dark_v"]
        )
    raise ValueError(method)


def _snr(model: Any) -> float:
    valid = np.asarray(model.variance) > 0.0
    return float(np.sqrt(np.sum(np.asarray(model.mu)[valid] ** 2 / np.asarray(model.variance)[valid])))


def _manifest(directory: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.relative_to(directory).as_posix() == "artifact_manifest.json":
            continue
        artifacts.append(
            {"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return {
        "schema_version": 1,
        "family": "target_three_state_four_method_noise_v5",
        "status": "diagnostic_not_retained",
        "admitted": False,
        "artifact_count": len(artifacts),
        "total_bytes": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
    }


def _render_noisy(staging: Path, arrays: dict[str, np.ndarray], limit: float) -> None:
    y = arrays["camera_y_m"] * 1e6
    z = arrays["camera_z_m"] * 1e6
    dy = float(np.mean(np.diff(y)))
    dz = float(np.mean(np.diff(z)))
    extent = [y[0] - dy / 2.0, y[-1] + dy / 2.0, z[0] - dz / 2.0, z[-1] + dz / 2.0]
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    plt.rcParams.update({"font.family": "serif", "font.size": 9.5})
    figure, axes = plt.subplots(4, 3, figsize=(7.2, 5.55), sharex=True, sharey=True)
    image = None
    for row, method in enumerate(METHODS):
        for column, state in enumerate(STATES):
            axis = axes[row, column]
            image = axis.imshow(
                arrays[f"{state}__{method}__observed_signal"],
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            if row == 0:
                axis.set_title(STATE_LABELS[state])
            if column == 0:
                axis.set_ylabel(r"$z\ (\mu\mathrm{m})$")
            if row == 3:
                axis.set_xlabel(r"$y\ (\mu\mathrm{m})$")
            axis.set_yticks([-1, 1])
            axis.set_xticks([-10, -5, 0, 5, 10])
    for position, name in zip((0.83, 0.62, 0.41, 0.20), ("PCI", "DGI", "DFFI", "DPFI"), strict=True):
        figure.text(0.022, position, name, ha="left", va="center", fontsize=10.5)
    figure.subplots_adjust(left=0.14, right=0.88, bottom=0.09, top=0.95, wspace=0.08, hspace=0.13)
    _require(image is not None, "no noisy image was rendered")
    cax = figure.add_axes([0.905, 0.14, 0.018, 0.74])
    colourbar = figure.colorbar(image, cax=cax)
    colourbar.set_label(r"Processed camera response relative to incident $I_0$")
    colourbar.set_ticks([-limit, -limit / 2.0, 0.0, limit / 2.0, limit])
    figure.savefig(staging / "figures/figure_4_target_four_method_noisy_200us.pdf", bbox_inches="tight")
    figure.savefig(staging / "figures/figure_4_target_four_method_noisy_200us.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _render_snr(staging: Path, rows: list[dict[str, Any]], display_tau: int) -> None:
    colours = {"smooth_bec": "#395C8C", "connected_modulated": "#B26B31", "separated_droplets": "#4F7B52"}
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, METHODS, strict=True):
        for state in STATES:
            selected = [row for row in rows if row["method"] == method and row["state"] == state]
            axis.plot(
                [row["pulse_duration_us"] for row in selected],
                [row["image_snr"] for row in selected],
                marker="o",
                markersize=3.2,
                color=colours[state],
                label=STATE_LABELS[state],
            )
        axis.axvline(display_tau, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_title(METHOD_LABELS[method])
        axis.grid(alpha=0.18)
    for axis in axes[:, 0]:
        axis.set_ylabel("Whole-image SNR")
    for axis in axes[1, :]:
        axis.set_xlabel(r"Pulse duration $\tau\ (\mu\mathrm{s})$")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.94, wspace=0.12, hspace=0.20)
    figure.savefig(staging / "figures/figure_4_target_four_method_snr.pdf")
    figure.savefig(staging / "figures/figure_4_target_four_method_snr.png", dpi=300)
    plt.close(figure)


def run(config_path: Path = CONFIG) -> Path:
    config = validate_config(config_path)
    output = ROOT / config["lifecycle"]["output_directory"]
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    staging.mkdir(parents=True)
    (staging / "data").mkdir()
    (staging / "figures").mkdir()
    parents = config["parents"]
    payload_path = ROOT / parents["noiseless_payload"]["path"]
    count_path = ROOT / parents["incident_count_scale"]["path"]
    count_scale_100 = _count_scale_100(count_path)
    read_noise = float(config["detector"]["read_noise_electrons_rms_per_port_pixel"])
    display_tau = int(config["display_pulse_duration_us"])
    prefix = tuple(int(value) for value in config["randomness"]["seed_prefix"])
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    max_expected = 0.0
    with np.load(payload_path) as data:
        arrays["camera_y_m"] = np.asarray(data["camera_y_m"])
        arrays["camera_z_m"] = np.asarray(data["camera_z_m"])
        for tau in config["pulse_durations_us"]:
            count_scale = count_scale_100 * float(tau) / 100.0
            for method in METHODS:
                for state in STATES:
                    model, means = _noise_model(method, state, count_scale, data, read_noise)
                    max_expected = max(
                        max_expected,
                        *(float(np.max(np.asarray(expected))) for expected in means.values()),
                    )
                    rows.append(
                        {
                            "pulse_duration_us": int(tau),
                            "method": method,
                            "state": state,
                            "image_snr": _snr(model),
                            "count_scale_electrons_per_i0_pixel": count_scale,
                        }
                    )
                    if int(tau) != display_tau:
                        continue
                    sampled: dict[str, np.ndarray] = {}
                    for role, expected in means.items():
                        seed = (*prefix, METHOD_IDS[method], STATE_IDS[state], ROLE_IDS[method][role], 0)
                        sampled[role] = _sample(np.asarray(expected), read_noise, seed)
                        arrays[f"{state}__{method}__{role}__expected_electrons"] = np.asarray(expected)
                        arrays[f"{state}__{method}__{role}__observed_electrons"] = sampled[role]
                        seed_rows.append(
                            {"method": method, "state": state, "role": role, "seed": ":".join(str(value) for value in seed)}
                        )
                    arrays[f"{state}__{method}__expected_signal"] = np.asarray(model.expected_signal)
                    arrays[f"{state}__{method}__variance"] = np.asarray(model.variance)
                    arrays[f"{state}__{method}__observed_signal"] = _process(method, sampled)
    limit = max(
        float(np.max(np.abs(arrays[f"{state}__{method}__observed_signal"])))
        for method in METHODS
        for state in STATES
    )
    full_well = float(config["detector"]["full_well_electrons"])
    maximum_expected_plus_6sigma = max_expected + 6.0 * np.sqrt(max_expected + read_noise**2)
    _require(maximum_expected_plus_6sigma < full_well, "full-well gate failed")
    _render_noisy(staging, arrays, limit)
    _render_snr(staging, rows, display_tau)
    np.savez_compressed(staging / "data/four_method_noise_arrays.npz", **arrays)
    with (staging / "data/four_method_snr.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (staging / "data/seed_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "state", "role", "seed"))
        writer.writeheader()
        writer.writerows(seed_rows)
    summary = {
        "schema_version": 1,
        "label": config["label"],
        "status": "diagnostic_not_retained",
        "admitted": False,
        "display_pulse_duration_us": display_tau,
        "common_noisy_zero_centred_display_limit": limit,
        "random_identities": len(seed_rows),
        "redraws": 0,
        "maximum_expected_electrons": max_expected,
        "maximum_expected_plus_6sigma_electrons": maximum_expected_plus_6sigma,
        "full_well_margin_electrons": full_well - maximum_expected_plus_6sigma,
        "parent_phase_wrapping_present": bool(
            _json(ROOT / parents["noiseless_summary"]["path"])["phase_wrapping_present"]
        ),
        "parent_maximum_absolute_circular_branch_phase_rad": float(
            _json(ROOT / parents["noiseless_summary"]["path"])[
                "maximum_absolute_circular_branch_phase_rad"
            ]
        ),
        "parent_wrapped_circular_branch_pixel_count": int(
            _json(ROOT / parents["noiseless_summary"]["path"])[
                "wrapped_circular_branch_pixel_count"
            ]
        ),
        "whole_image_snr_role": config["image_snr"]["role"],
        "claim_boundary": config["claim_boundary"],
    }
    (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "config_snapshot.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "diagnostic_not_retained",
                "admitted": False,
                "config_sha256": _sha256(config_path),
                "runner_sha256": _sha256(Path(__file__)),
                "parent_hashes": {name: record["sha256"] for name, record in parents.items()},
                "random_identities": len(seed_rows),
                "redraws": 0,
                "results_writes": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (staging / "caption_noisy.txt").write_text(
        "One simulated 200 microsecond acquisition for each target-scale state and readout. "
        "Every panel uses the same incident probe dose, camera sampling and Poisson plus 0.7-electron read-noise model. "
        "The PCI, DGI and DFFI raw-frame roles and both DPFI ports are sampled independently; the H and V port intensities are combined only after camera sampling and noise. "
        "All twelve panels share one zero-centred scale relative to incident I0. The balanced DPFI mean background cancels, but the independent shot and read-noise variances of its two bright ports do not. "
        "The ID control inherits the local circular-branch phase wrapping identified in the noiseless forward field, so image strength is not a globally single-valued density coordinate.\n",
        encoding="utf-8",
    )
    (staging / "caption_snr.txt").write_text(
        "Expected whole-image SNR versus pulse duration under the same ideal readout and detector model. "
        "The dashed line marks the 200 microsecond acquisition shown in the noisy-image figure. "
        "This template statistic measures aggregate image strength; it is not a resolution, feature-precision or state-classification probability.\n",
        encoding="utf-8",
    )
    (staging / "artifact_manifest.json").write_text(json.dumps(_manifest(staging), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_output(staging, config)
    os.replace(staging, output)
    return output


def validate_output(directory: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = validate_config(CONFIG) if config is None else config
    _require(_json(directory / "artifact_manifest.json") == _manifest(directory), "manifest changed")
    _require(_json(directory / "config_snapshot.json") == config, "config snapshot changed")
    summary = _json(directory / "summary.json")
    _require(
        summary["label"] == config["label"]
        and summary["admitted"] is False
        and summary["redraws"] == 0
        and summary["random_identities"] == 57,
        "summary lifecycle changed",
    )
    parent_summary = _json(ROOT / config["parents"]["noiseless_summary"]["path"])
    _require(
        summary["parent_phase_wrapping_present"]
        == parent_summary["phase_wrapping_present"]
        and np.isclose(
            float(summary["parent_maximum_absolute_circular_branch_phase_rad"]),
            float(parent_summary["maximum_absolute_circular_branch_phase_rad"]),
            rtol=0.0,
            atol=0.0,
        )
        and int(summary["parent_wrapped_circular_branch_pixel_count"])
        == int(parent_summary["wrapped_circular_branch_pixel_count"]),
        "inherited phase boundary changed",
    )
    seed_rows = list(csv.DictReader((directory / "data/seed_ledger.csv").open(newline="", encoding="utf-8")))
    _require(len(seed_rows) == 57 and len({row["seed"] for row in seed_rows}) == 57, "seed inventory changed")
    expected_seed_keys = {
        (method, state, role)
        for method in METHODS
        for state in STATES
        for role in ROLE_IDS[method]
    }
    _require(
        {(row["method"], row["state"], row["role"]) for row in seed_rows} == expected_seed_keys,
        "seed key inventory changed",
    )
    seed_prefix = tuple(int(value) for value in config["randomness"]["seed_prefix"])
    for row in seed_rows:
        expected_seed = (
            *seed_prefix,
            METHOD_IDS[row["method"]],
            STATE_IDS[row["state"]],
            ROLE_IDS[row["method"]][row["role"]],
            0,
        )
        _require(tuple(int(value) for value in row["seed"].split(":")) == expected_seed, "seed schema changed")

    parents = config["parents"]
    payload_path = ROOT / parents["noiseless_payload"]["path"]
    count_scale_100 = _count_scale_100(ROOT / parents["incident_count_scale"]["path"])
    read_noise = float(config["detector"]["read_noise_electrons_rms_per_port_pixel"])
    max_expected = 0.0
    with np.load(directory / "data/four_method_noise_arrays.npz") as data, np.load(payload_path) as parent:
        for row in seed_rows:
            state, method, role = row["state"], row["method"], row["role"]
            expected = np.asarray(data[f"{state}__{method}__{role}__expected_electrons"])
            observed = np.asarray(data[f"{state}__{method}__{role}__observed_electrons"])
            seed = tuple(int(value) for value in row["seed"].split(":"))
            _, raw_means = _noise_model(method, state, 2.0 * count_scale_100, parent, read_noise)
            _require(
                np.allclose(expected, np.asarray(raw_means[role]), rtol=0.0, atol=1e-12),
                f"raw mean replay failed: {state} {method} {role}",
            )
            replay = _sample(expected, read_noise, seed)
            _require(np.array_equal(observed, replay), f"RNG replay failed: {state} {method} {role}")
            max_expected = max(max_expected, float(np.max(expected)))
        for method in METHODS:
            for state in STATES:
                raw = {
                    role: np.asarray(data[f"{state}__{method}__{role}__observed_electrons"])
                    for role in ROLE_IDS[method]
                }
                _require(
                    np.array_equal(_process(method, raw), data[f"{state}__{method}__observed_signal"]),
                    f"processed replay failed: {state} {method}",
                )
        replay_limit = max(
            float(np.max(np.abs(np.asarray(data[f"{state}__{method}__observed_signal"]))))
            for method in METHODS
            for state in STATES
        )
        _require(
            np.isclose(replay_limit, float(summary["common_noisy_zero_centred_display_limit"]), rtol=0.0, atol=1e-12),
            "common noisy display limit changed",
        )
    rows = list(csv.DictReader((directory / "data/four_method_snr.csv").open(newline="", encoding="utf-8")))
    _require(len(rows) == 144, "SNR inventory changed")
    rows_by_key = {(int(row["pulse_duration_us"]), row["method"], row["state"]): row for row in rows}
    _require(len(rows_by_key) == 144, "SNR key inventory changed")
    max_expected_all_tau = 0.0
    with np.load(payload_path) as parent:
        for tau in config["pulse_durations_us"]:
            count_scale = count_scale_100 * float(tau) / 100.0
            for method in METHODS:
                sequence = []
                for state in STATES:
                    model, means = _noise_model(method, state, count_scale, parent, read_noise)
                    expected_snr = _snr(model)
                    row = rows_by_key[(int(tau), method, state)]
                    _require(
                        np.isclose(float(row["image_snr"]), expected_snr, rtol=0.0, atol=1e-12)
                        and np.isclose(float(row["count_scale_electrons_per_i0_pixel"]), count_scale, rtol=0.0, atol=1e-12),
                        f"SNR replay failed: {tau} {method} {state}",
                    )
                    sequence.append(expected_snr)
                    max_expected_all_tau = max(
                        max_expected_all_tau,
                        *(float(np.max(np.asarray(expected))) for expected in means.values()),
                    )
    for method in METHODS:
        for state in STATES:
            sequence = [float(rows_by_key[(tau, method, state)]["image_snr"]) for tau in config["pulse_durations_us"]]
            _require(np.all(np.diff(sequence) > 0.0), f"SNR duration sequence is not monotonic: {method} {state}")
    full_well = float(config["detector"]["full_well_electrons"])
    maximum_expected_plus_6sigma = max_expected_all_tau + 6.0 * np.sqrt(max_expected_all_tau + read_noise**2)
    _require(maximum_expected_plus_6sigma < full_well, "full-well replay failed")
    _require(
        np.isclose(float(summary["maximum_expected_electrons"]), max_expected_all_tau, rtol=0.0, atol=1e-12)
        and np.isclose(float(summary["maximum_expected_plus_6sigma_electrons"]), maximum_expected_plus_6sigma, rtol=0.0, atol=1e-12)
        and np.isclose(float(summary["full_well_margin_electrons"]), full_well - maximum_expected_plus_6sigma, rtol=0.0, atol=1e-12),
        "full-well summary changed",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps({"status": "generation_pass", "output": str(run())}))
