"""Analytic Chapter 4 noise-weighted image and Oxford change SNR."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_retained_family(root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = _read_json(manifest_path)
    _require(manifest.get("admitted") is True, "three-state family is not admitted")
    records = manifest.get("artifact_inventory_excluding_manifest")
    _require(isinstance(records, list) and len(records) == 9, "retained artifact inventory changed")
    family = manifest_path.parent.resolve()
    verified: list[dict[str, Any]] = []
    for raw in records:
        path = (family / str(raw["path"])).resolve()
        path.relative_to(family)
        _require(path.is_file(), f"missing retained artifact: {path.name}")
        _require(path.stat().st_size == int(raw["size_bytes"]), f"retained byte mismatch: {path.name}")
        _require(_sha256(path) == raw["sha256"], f"retained hash mismatch: {path.name}")
        verified.append(_record(path, root))
    return verified


def verify_scratch_manifest(root: Path, manifest_path: Path, expected_count: int) -> list[dict[str, Any]]:
    manifest = _read_json(manifest_path)
    _require(manifest.get("admitted") is False, "scratch source was unexpectedly admitted")
    records = manifest.get("artifacts")
    _require(isinstance(records, list) and len(records) == expected_count, "scratch artifact inventory changed")
    verified: list[dict[str, Any]] = []
    for raw in records:
        path = (root / str(raw["path"])).resolve()
        path.relative_to((root / ".scratch").resolve())
        _require(path.is_file(), f"missing scratch artifact: {path.name}")
        _require(path.stat().st_size == int(raw["bytes"]), f"scratch byte mismatch: {path.name}")
        _require(_sha256(path) == raw["sha256"], f"scratch hash mismatch: {path.name}")
        verified.append(_record(path, root))
    return verified


def load_expected_npz(path: Path) -> dict[str, np.ndarray]:
    """Load deterministic arrays while structurally excluding fixed-draw observations."""

    with np.load(path) as loaded:
        selected = {
            name: np.asarray(loaded[name])
            for name in loaded.files
            if "observed" not in name
        }
    _require(selected and not any("observed" in name for name in selected), "observed draw leaked into D4c")
    return selected


def diagonal_noise_weighted_snr(mu: ArrayLike, variance: ArrayLike) -> float:
    signal = np.asarray(mu, dtype=float)
    noise = np.asarray(variance, dtype=float)
    if signal.shape != noise.shape or signal.size == 0:
        raise ValueError("mu and variance must share one non-empty shape")
    if not np.isfinite(signal).all() or not np.isfinite(noise).all() or np.any(noise <= 0.0):
        raise ValueError("mu must be finite and variance finite positive")
    return float(np.sqrt(np.sum(signal**2 / noise)))


def legacy_block_snr(mu: ArrayLike, variance: ArrayLike) -> float:
    signal = np.asarray(mu, dtype=float)
    noise = np.asarray(variance, dtype=float)
    if signal.shape != noise.shape or signal.size == 0 or np.any(noise <= 0.0):
        raise ValueError("legacy block inputs are invalid")
    return float(abs(np.sum(signal)) / np.sqrt(np.sum(noise)))


@dataclass(frozen=True)
class FrameNoiseModel:
    method: str
    expected_signal: FloatArray
    blank_signal: FloatArray
    mu: FloatArray
    variance: FloatArray
    jacobians: Mapping[str, FloatArray]
    raw_variances: Mapping[str, FloatArray]
    role_variance_contributions: Mapping[str, FloatArray]


def _raw_variance(expected: ArrayLike, read_noise: float) -> FloatArray:
    values = np.asarray(expected, dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("raw expected electrons must be finite and non-negative")
    return np.asarray(values + float(read_noise) ** 2, dtype=float)


def pci_frame_noise_model(
    p: ArrayLike,
    b: ArrayLike,
    d: ArrayLike,
    blank_signal: ArrayLike,
    *,
    phase_plate_amplitude_transmittance: float,
    read_noise_electrons_rms: float,
) -> FrameNoiseModel:
    p_value, b_value, d_value = (np.asarray(value, dtype=float) for value in (p, b, d))
    if p_value.shape != b_value.shape or p_value.shape != d_value.shape:
        raise ValueError("PCI raw means must share one shape")
    denominator = b_value - d_value
    _require(np.all(denominator > 0.0), "PCI expected denominator is non-positive")
    tp2 = float(phase_plate_amplitude_transmittance) ** 2
    expected = tp2 * ((p_value - d_value) / denominator - 1.0)
    blank = np.asarray(blank_signal, dtype=float)
    _require(blank.shape == expected.shape, "PCI blank shape changed")
    jacobians = {
        "P": tp2 / denominator,
        "B": -tp2 * (p_value - d_value) / denominator**2,
        "D": tp2 * (p_value - b_value) / denominator**2,
    }
    raw = {
        "P": _raw_variance(p_value, read_noise_electrons_rms),
        "B": _raw_variance(b_value, read_noise_electrons_rms),
        "D": _raw_variance(d_value, read_noise_electrons_rms),
    }
    contributions = {name: jacobians[name] ** 2 * raw[name] for name in raw}
    variance = sum(contributions.values())
    return FrameNoiseModel("pci", expected, blank, expected - blank, variance, jacobians, raw, contributions)


def dgi_frame_noise_model(
    p_s: ArrayLike,
    l_s: ArrayLike,
    d_s: ArrayLike,
    b_o: ArrayLike,
    d_o: ArrayLike,
    blank_signal: ArrayLike,
    *,
    open_to_stop_scale: float,
    read_noise_electrons_rms: float,
) -> FrameNoiseModel:
    p, leakage, stop_dark, bright, open_dark = (
        np.asarray(value, dtype=float) for value in (p_s, l_s, d_s, b_o, d_o)
    )
    shape = p.shape
    if any(value.shape != shape for value in (leakage, stop_dark, bright, open_dark)):
        raise ValueError("DGI raw means must share one shape")
    scale = float(open_to_stop_scale)
    open_level = bright - open_dark
    denominator = scale * open_level
    _require(scale > 0.0 and np.all(denominator > 0.0), "DGI expected denominator is non-positive")
    expected = (p - leakage) / denominator
    blank = np.asarray(blank_signal, dtype=float)
    _require(blank.shape == expected.shape, "DGI blank shape changed")
    jacobians = {
        "P_s": 1.0 / denominator,
        "L_s": -1.0 / denominator,
        "D_s": np.zeros(shape, dtype=float),
        "B_o": -expected / open_level,
        "D_o": expected / open_level,
    }
    raw = {
        "P_s": _raw_variance(p, read_noise_electrons_rms),
        "L_s": _raw_variance(leakage, read_noise_electrons_rms),
        "D_s": _raw_variance(stop_dark, read_noise_electrons_rms),
        "B_o": _raw_variance(bright, read_noise_electrons_rms),
        "D_o": _raw_variance(open_dark, read_noise_electrons_rms),
    }
    contributions = {name: jacobians[name] ** 2 * raw[name] for name in raw}
    variance = sum(contributions.values())
    return FrameNoiseModel("dgi", expected, blank, expected - blank, variance, jacobians, raw, contributions)


@dataclass(frozen=True)
class ChangeNoiseModel:
    method: str
    delta_mu: FloatArray
    variance: FloatArray
    independent_frame_variance: FloatArray
    cross_covariance: FloatArray
    contributions: Mapping[str, FloatArray]


def oxford_change_noise_model(later: FrameNoiseModel, first: FrameNoiseModel) -> ChangeNoiseModel:
    if later.method != first.method or later.mu.shape != first.mu.shape:
        raise ValueError("change frames must share method and shape")
    delta = later.mu - first.mu
    if later.method == "pci":
        for role in ("B", "D"):
            _require(np.array_equal(later.raw_variances[role], first.raw_variances[role]), f"shared PCI {role} variance changed")
        contributions = {
            "atom_later": later.role_variance_contributions["P"],
            "atom_q1": first.role_variance_contributions["P"],
            "B_shared": (later.jacobians["B"] - first.jacobians["B"]) ** 2 * first.raw_variances["B"],
            "D_shared": (later.jacobians["D"] - first.jacobians["D"]) ** 2 * first.raw_variances["D"],
        }
        cross = (
            later.jacobians["B"] * first.jacobians["B"] * first.raw_variances["B"]
            + later.jacobians["D"] * first.jacobians["D"] * first.raw_variances["D"]
        )
    else:
        for role in ("L_s", "D_s", "B_o", "D_o"):
            _require(np.array_equal(later.raw_variances[role], first.raw_variances[role]), f"shared DGI {role} variance changed")
        contributions = {
            "atom_later": later.role_variance_contributions["P_s"],
            "atom_q1": first.role_variance_contributions["P_s"],
            "L_s_shared": (later.jacobians["L_s"] - first.jacobians["L_s"]) ** 2 * first.raw_variances["L_s"],
            "D_s_shared": (later.jacobians["D_s"] - first.jacobians["D_s"]) ** 2 * first.raw_variances["D_s"],
            "B_o_shared": (later.jacobians["B_o"] - first.jacobians["B_o"]) ** 2 * first.raw_variances["B_o"],
            "D_o_shared": (later.jacobians["D_o"] - first.jacobians["D_o"]) ** 2 * first.raw_variances["D_o"],
        }
        cross = sum(
            later.jacobians[role] * first.jacobians[role] * first.raw_variances[role]
            for role in ("L_s", "D_s", "B_o", "D_o")
        )
    variance = sum(contributions.values())
    independent = later.variance + first.variance
    _require(np.allclose(variance, independent - 2.0 * cross, rtol=2e-13, atol=2e-18), "shared covariance identity failed")
    return ChangeNoiseModel(later.method, delta, variance, independent, cross, contributions)


def exact_common_grid_indices(
    three_y: ArrayLike,
    three_z: ArrayLike,
    oxford_y: ArrayLike,
    oxford_z: ArrayLike,
) -> tuple[slice, slice]:
    ty, tz, oy, oz = (np.asarray(value, dtype=float) for value in (three_y, three_z, oxford_y, oxford_z))
    _require(ty.shape == (71,) and tz.shape == (35,) and oy.shape == (111,) and oz.shape == (35,), "camera grid inventory changed")
    matches = [np.flatnonzero(oy == value) for value in ty]
    _require(all(item.size == 1 for item in matches), "three-state y coordinate is not an exact Oxford subset")
    indices = np.asarray([item[0] for item in matches])
    _require(np.array_equal(indices, np.arange(20, 91)) and np.array_equal(oy[20:91], ty), "common y subset is not centered and exact")
    _require(np.array_equal(oz, tz), "common z coordinates differ")
    return slice(0, 35), slice(20, 91)


def _crop(values: np.ndarray, row_slice: slice, column_slice: slice) -> np.ndarray:
    return np.asarray(values[row_slice, column_slice])


def _frame_row(
    *,
    dataset: str,
    condition: str,
    label: str,
    method: str,
    fluence: float,
    q: int | None,
    model: FrameNoiseModel,
    row_slice: slice,
    column_slice: slice,
) -> dict[str, Any]:
    mu = _crop(model.mu, row_slice, column_slice)
    variance = _crop(model.variance, row_slice, column_slice)
    snr = diagonal_noise_weighted_snr(mu, variance)
    return {
        "dataset": dataset,
        "condition_id": condition,
        "display_label": label,
        "method": method,
        "fluence_mw_us": fluence,
        "frame_q": "single" if q is None else q,
        "grid_id": "centered_common_camera_grid_35x71",
        "pixel_count": mu.size,
        "mu_source": "deterministic_expected_processed_atom_image_minus_expected_blank",
        "covariance_source": "expected_raw_Poisson_plus_0p7e_read_noise_propagated_by_expected_processing_Jacobian",
        "snr_image": snr,
        "snr_image_squared": snr**2,
        "sum_mu_squared": float(np.sum(mu**2)),
        "variance_min": float(np.min(variance)),
        "variance_max": float(np.max(variance)),
        "variance_sum": float(np.sum(variance)),
        "observed_fixed_draw_consumed": False,
    }


def _crop_row(
    *,
    dataset: str,
    condition: str,
    method: str,
    statistic_kind: str,
    comparison: str,
    mu: np.ndarray,
    variance: np.ndarray,
    row_slice: slice,
    column_slice: slice,
) -> dict[str, Any]:
    common_mu = _crop(mu, row_slice, column_slice)
    common_variance = _crop(variance, row_slice, column_slice)
    common = diagonal_noise_weighted_snr(common_mu, common_variance)
    native = diagonal_noise_weighted_snr(mu, variance)
    fraction = common**2 / native**2 if native > 0.0 else 1.0
    return {
        "dataset": dataset,
        "condition_id": condition,
        "method": method,
        "statistic_kind": statistic_kind,
        "comparison": comparison,
        "common_grid_shape": "35x71",
        "native_grid_shape": f"{mu.shape[0]}x{mu.shape[1]}",
        "coordinate_match": "exact_subset_no_interpolation",
        "oxford_native_y_slice": "20:91" if dataset == "oxford_tfbec" else "native_equals_common",
        "common_grid_snr": common,
        "native_full_grid_snr": native,
        "common_fraction_of_native_snr_squared": fraction,
        "omitted_native_snr_squared_fraction": max(0.0, 1.0 - fraction),
    }


def validate_config(repository_root: Path, config_path: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    config = _read_json(config_path.resolve())
    _require(config.get("schema_version") == 1, "unsupported D4c schema")
    lifecycle = config["lifecycle"]
    _require(
        lifecycle.get("admitted") is False
        and lifecycle.get("overwrite_existing_output") is False
        and all(lifecycle.get(name) == 0 for name in ("optical_propagations", "thermodynamic_solves", "random_draws", "ensemble_draws", "fits")),
        "D4c execution boundary changed",
    )
    output = (root / lifecycle["output_directory"]).resolve()
    output.relative_to((root / ".scratch").resolve())
    if output.exists():
        raise FileExistsError(f"exclusive D4c output already exists: {output}")
    statistic = config["statistic"]
    _require(
        statistic["name"] == "noise-weighted image SNR"
        and statistic["observed_fixed_draw_consumed"] is False
        and statistic["threshold"] is None
        and float(statistic["read_noise_electrons_rms"]) == 0.7,
        "D4c statistic changed",
    )
    _require(config["common_grid"]["shape_z_y"] == [35, 71] and config["common_grid"]["interpolation"] is False, "D4c common grid changed")
    return config


def run_noise_weighted_snr(repository_root: Path, config_path: Path) -> Path:
    started = time.perf_counter()
    root = repository_root.resolve()
    config_path = config_path.resolve()
    config = validate_config(root, config_path)
    sources = config["sources"]
    authenticated = {
        "three_state": verify_retained_family(root, root / sources["three_state_manifest"]),
        "d4a": verify_scratch_manifest(root, root / sources["d4a_manifest"], 6),
        "d4b": verify_scratch_manifest(root, root / sources["d4b_manifest"], 7),
    }
    three = load_expected_npz(root / sources["three_state_arrays"])
    oxford = load_expected_npz(root / sources["d4b_arrays"])
    row_slice, oxford_columns = exact_common_grid_indices(
        three["camera_y_m"], three["camera_z_m"], oxford["camera_y_m"], oxford["camera_z_m"]
    )
    three_columns = slice(0, 71)
    read_noise = float(config["statistic"]["read_noise_electrons_rms"])
    tp = float(config["processing"]["pci_phase_plate_amplitude_transmittance"])
    scale = float(config["processing"]["dgi_open_to_stop_scale"])

    frame_models: dict[tuple[str, str, str], FrameNoiseModel] = {}
    expected_processing_max_error = 0.0
    states = config["conditions"]["three_state"]["states"]
    labels = config["conditions"]["three_state"]["display_labels"]
    for state in states:
        pci = pci_frame_noise_model(
            three[f"raw_pci_{state}_atom_expected_electrons"],
            three[f"raw_pci_{state}_bright_reference_expected_electrons"],
            three[f"raw_pci_{state}_dark_expected_electrons"],
            three["blank_pci_processed_expected_delta_i_over_i0"],
            phase_plate_amplitude_transmittance=tp,
            read_noise_electrons_rms=read_noise,
        )
        dgi = dgi_frame_noise_model(
            three[f"raw_dgi_{state}_atom_stop_expected_electrons"],
            three[f"raw_dgi_{state}_leakage_stop_expected_electrons"],
            three[f"raw_dgi_{state}_stop_dark_expected_electrons"],
            three[f"raw_dgi_{state}_open_reference_expected_electrons"],
            three[f"raw_dgi_{state}_open_dark_expected_electrons"],
            three["blank_dgi_processed_expected_delta_i_over_i0"],
            open_to_stop_scale=scale,
            read_noise_electrons_rms=read_noise,
        )
        for method, model in (("pci", pci), ("dgi", dgi)):
            stored = three[f"processed_{method}_{state}_expected_delta_i_over_i0"]
            expected_processing_max_error = max(expected_processing_max_error, float(np.max(np.abs(model.expected_signal - stored))))
            frame_models[("three_state", state, method)] = model

    for q in (1, 2, 3):
        pci = pci_frame_noise_model(
            oxford[f"raw_pci_P_q{q}_expected_electrons"],
            oxford["raw_pci_B_shared_expected_electrons"],
            oxford["raw_pci_D_shared_expected_electrons"],
            np.zeros((35, 111)),
            phase_plate_amplitude_transmittance=tp,
            read_noise_electrons_rms=read_noise,
        )
        dgi = dgi_frame_noise_model(
            oxford[f"raw_dgi_P_s_q{q}_expected_electrons"],
            oxford["raw_dgi_L_s_shared_expected_electrons"],
            oxford["raw_dgi_D_s_shared_expected_electrons"],
            oxford["raw_dgi_B_o_shared_expected_electrons"],
            oxford["raw_dgi_D_o_shared_expected_electrons"],
            np.zeros((35, 111)),
            open_to_stop_scale=scale,
            read_noise_electrons_rms=read_noise,
        )
        for method, model in (("pci", pci), ("dgi", dgi)):
            stored = oxford[f"processed_{method}_q{q}_expected_delta_i_over_i0"]
            expected_processing_max_error = max(expected_processing_max_error, float(np.max(np.abs(model.expected_signal - stored))))
            frame_models[("oxford", str(q), method)] = model

    snr_rows: list[dict[str, Any]] = []
    crop_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    audit_arrays: dict[str, np.ndarray] = {
        "common_camera_y_m": three["camera_y_m"],
        "common_camera_z_m": three["camera_z_m"],
    }

    def add_frame(dataset: str, condition: str, label: str, method: str, fluence: float, q: int | None, model: FrameNoiseModel, columns: slice) -> None:
        row = _frame_row(
            dataset=dataset,
            condition=condition,
            label=label,
            method=method,
            fluence=fluence,
            q=q,
            model=model,
            row_slice=row_slice,
            column_slice=columns,
        )
        snr_rows.append(row)
        comparison = "single" if q is None else f"q{q}_vs_blank"
        crop_rows.append(
            _crop_row(
                dataset=dataset,
                condition=condition,
                method=method,
                statistic_kind="image",
                comparison=comparison,
                mu=model.mu,
                variance=model.variance,
                row_slice=row_slice,
                column_slice=columns,
            )
        )
        common_mu = _crop(model.mu, row_slice, columns)
        common_variance = _crop(model.variance, row_slice, columns)
        center_rows = slice((35 - 5) // 2, (35 - 5) // 2 + 5)
        center_columns = slice((71 - 5) // 2, (71 - 5) // 2 + 5)
        mu5 = common_mu[center_rows, center_columns]
        variance5 = common_variance[center_rows, center_columns]
        nw5 = diagonal_noise_weighted_snr(mu5, variance5)
        replacement_rows.append(
            {
                "dataset": dataset,
                "condition_id": condition,
                "display_label": label,
                "method": method,
                "frame_q": "single" if q is None else q,
                "legacy_centered_5x5_block_snr": legacy_block_snr(mu5, variance5),
                "noise_weighted_snr_on_centered_5x5": nw5,
                "common_grid_noise_weighted_snr": row["snr_image"],
                "centered_5x5_fraction_of_common_snr_squared": nw5**2 / float(row["snr_image_squared"]),
                "role": "replacement_audit_only_not_main_result_or_threshold",
            }
        )
        key = f"{dataset}_{condition}_{method}".replace("-", "_")
        audit_arrays[f"{key}_common_mu"] = common_mu
        audit_arrays[f"{key}_common_variance_total"] = common_variance
        for role, values in model.role_variance_contributions.items():
            audit_arrays[f"{key}_common_variance_{role}"] = _crop(values, row_slice, columns)

    for state in states:
        for method in ("pci", "dgi"):
            add_frame(
                "three_state_single_exposure",
                state,
                labels[state],
                method,
                100.0,
                None,
                frame_models[("three_state", state, method)],
                three_columns,
            )
    for q in (1, 2, 3):
        for method in ("pci", "dgi"):
            add_frame(
                "oxford_tfbec",
                f"oxford_q{q}",
                f"Oxford q{q}",
                method,
                300.0,
                q,
                frame_models[("oxford", str(q), method)],
                oxford_columns,
            )

    change_rows: list[dict[str, Any]] = []
    for later_q in (2, 3):
        for method in ("pci", "dgi"):
            first = frame_models[("oxford", "1", method)]
            later = frame_models[("oxford", str(later_q), method)]
            change = oxford_change_noise_model(later, first)
            common_mu = _crop(change.delta_mu, row_slice, oxford_columns)
            common_variance = _crop(change.variance, row_slice, oxford_columns)
            common_independent = _crop(change.independent_frame_variance, row_slice, oxford_columns)
            snr = diagonal_noise_weighted_snr(common_mu, common_variance)
            independent_snr = diagonal_noise_weighted_snr(common_mu, common_independent)
            contribution_sums = {
                name: float(np.sum(_crop(values, row_slice, oxford_columns)))
                for name, values in change.contributions.items()
            }
            atom_sum = contribution_sums.pop("atom_later") + contribution_sums.pop("atom_q1")
            shared_sum = sum(contribution_sums.values())
            row = {
                "dataset": "oxford_tfbec",
                "method": method,
                "fluence_mw_us": 300.0,
                "later_frame_q": later_q,
                "reference_frame_q": 1,
                "grid_id": "centered_common_camera_grid_35x71",
                "pixel_count": common_mu.size,
                "snr_change": snr,
                "snr_change_squared": snr**2,
                "snr_if_frames_incorrectly_treated_independent": independent_snr,
                "shared_covariance_snr_ratio_correct_over_independent": snr / independent_snr,
                "atom_frame_variance_sum": atom_sum,
                "shared_reference_variance_sum": shared_sum,
                "total_variance_sum": float(np.sum(common_variance)),
                "cross_frame_covariance_sum": float(np.sum(_crop(change.cross_covariance, row_slice, oxford_columns))),
                "L_s_shared_variance_sum": contribution_sums.get("L_s_shared", 0.0),
                "D_s_shared_variance_sum": contribution_sums.get("D_s_shared", 0.0),
                "B_or_B_o_shared_variance_sum": contribution_sums.get("B_shared", contribution_sums.get("B_o_shared", 0.0)),
                "D_or_D_o_shared_variance_sum": contribution_sums.get("D_shared", contribution_sums.get("D_o_shared", 0.0)),
                "covariance_source": "analytic_expected_raw_Jacobians_with_actual_shared_reference_identity",
                "observed_fixed_draw_consumed": False,
            }
            change_rows.append(row)
            crop_rows.append(
                _crop_row(
                    dataset="oxford_tfbec",
                    condition=f"q{later_q}_minus_q1",
                    method=method,
                    statistic_kind="change",
                    comparison=f"q{later_q}-q1",
                    mu=change.delta_mu,
                    variance=change.variance,
                    row_slice=row_slice,
                    column_slice=oxford_columns,
                )
            )
            key = f"oxford_change_q{later_q}_minus_q1_{method}"
            audit_arrays[f"{key}_common_delta_mu"] = common_mu
            audit_arrays[f"{key}_common_variance_total"] = common_variance
            audit_arrays[f"{key}_common_variance_independent_wrong"] = common_independent
            audit_arrays[f"{key}_common_cross_covariance"] = _crop(change.cross_covariance, row_slice, oxford_columns)
            for role, values in change.contributions.items():
                audit_arrays[f"{key}_common_variance_{role}"] = _crop(values, row_slice, oxford_columns)

    output = (root / config["lifecycle"]["output_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot_path = output / "config_snapshot.json"
    shutil.copyfile(config_path, snapshot_path)
    snr_path = output / "snr_image.csv"
    change_path = output / "snr_change.csv"
    crop_path = output / "crop_convergence.csv"
    replacement_path = output / "5x5_replacement_audit.csv"
    covariance_path = output / "snr_covariance_arrays.npz"
    _write_csv(snr_path, snr_rows)
    _write_csv(change_path, change_rows)
    _write_csv(crop_path, crop_rows)
    _write_csv(replacement_path, replacement_rows)
    np.savez_compressed(covariance_path, **audit_arrays)

    summary = {
        "label": config["label"],
        "status": config["lifecycle"]["status"],
        "admitted": False,
        "runtime_seconds": time.perf_counter() - started,
        "definition": config["statistic"],
        "common_grid": {
            "shape": [35, 71],
            "three_state_native_equals_common": True,
            "Oxford_y_slice": [20, 91],
            "coordinate_max_abs_error_m": 0.0,
            "interpolation": False,
        },
        "inventory": {
            "snr_image_rows": len(snr_rows),
            "snr_change_rows": len(change_rows),
            "crop_convergence_rows": len(crop_rows),
            "replacement_audit_rows": len(replacement_rows),
            "observed_arrays_consumed": 0,
            "optical_propagations": 0,
            "thermodynamic_solves": 0,
            "random_draws": 0,
            "fits": 0,
        },
        "validation": {
            "expected_processing_max_abs_error": expected_processing_max_error,
            "all_covariance_arrays_finite": all(np.isfinite(value).all() for value in audit_arrays.values()),
            "all_variance_arrays_positive": all(
                np.all(value > 0.0)
                for name, value in audit_arrays.items()
                if "variance_total" in name or "variance_independent_wrong" in name
            ),
            "DGI_change_L_s_variance_max": max(row["L_s_shared_variance_sum"] for row in change_rows if row["method"] == "dgi"),
            "DGI_change_D_s_variance_max": max(row["D_s_shared_variance_sum"] for row in change_rows if row["method"] == "dgi"),
        },
        "claim_boundary": config["claim_boundary"],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {
        "config": _record(config_path, root),
        "authenticated_inputs": authenticated,
        "direct_input_files": [
            _record(root / sources["three_state_arrays"], root),
            _record(root / sources["d4b_arrays"], root),
            _record(root / sources["d4b_summary"], root),
        ],
        "maintained_source_files": [
            _record(root / "src/non_destructive_image/chapter4_noise_weighted_snr.py", root),
        ],
        "consumption_policy": {
            "expected_arrays_only": True,
            "observed_fixed_draw_arrays_consumed": False,
            "input_array_names": sorted([*three.keys(), *oxford.keys()]),
        },
        "execution": config["lifecycle"],
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = [snapshot_path, snr_path, change_path, crop_path, replacement_path, covariance_path, summary_path, provenance_path]
    manifest = {
        "label": config["label"],
        "admitted": False,
        "exclusive_output": True,
        "artifacts": [_record(path, root) for path in artifacts],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


__all__ = [
    "ChangeNoiseModel",
    "FrameNoiseModel",
    "diagonal_noise_weighted_snr",
    "dgi_frame_noise_model",
    "exact_common_grid_indices",
    "legacy_block_snr",
    "load_expected_npz",
    "oxford_change_noise_model",
    "pci_frame_noise_model",
    "run_noise_weighted_snr",
    "validate_config",
]
