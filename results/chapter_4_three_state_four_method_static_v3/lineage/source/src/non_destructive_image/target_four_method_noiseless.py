"""Noiseless four-readout comparison for the target-scale three-state objects."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .camera import centered_camera_shape, resample_to_camera_pixels
from .equilibrium_imaging import optical_transfer_from_objective_config
from .four_method_jones_imaging import simulate_matched_four_method_jones_images
from .target_three_state_profiles import (
    build_target_three_state_profiles,
    load_target_three_state_profile_config,
)


EXPECTED_LABEL = "target_three_state_four_method_noiseless_v4"
STATES = ("smooth_bec", "connected_modulated", "separated_droplets")
METHODS = ("pci", "dgi", "dffi", "dpfi")
NPZ_NAME = "target_four_method_camera_and_fields.npz"


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
    path = (root / str(record["path"])).resolve()
    _require(path.is_file(), f"source is missing: {record['path']}")
    _require(_sha256(path) == str(record["sha256"]), f"source hash changed: {record['path']}")
    return path


def _manifest(directory: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.relative_to(directory).as_posix() == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "family": EXPECTED_LABEL,
        "status": "diagnostic_not_retained",
        "admitted": False,
        "self_hash_policy": "root_manifest_excluded",
        "artifact_count": len(artifacts),
        "total_bytes": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
    }


def _git_record(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=False, capture_output=True, text=True
        ).stdout.strip()

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def validate_config(repository_root: Path, config_path: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    config = _read_json(config_path.resolve())
    _require(config.get("schema_version") == 1, "unsupported config schema")
    _require(config.get("label") == EXPECTED_LABEL, "config label changed")
    lifecycle = config["lifecycle"]
    _require(
        lifecycle
        == {
            "status": "diagnostic_not_retained",
            "admitted": False,
            "random_draws": 0,
            "results_writes_allowed": False,
            "overwrite_existing_output": False,
            "output_directory": ".scratch/target_three_state_four_method_noiseless_v4",
        },
        "lifecycle changed",
    )
    physical = config["physical_contract"]
    _require(tuple(physical["states"]) == STATES, "state set changed")
    _require(tuple(physical["methods"]) == METHODS, "method set changed")
    _require(
        physical["geometry"]
        == "B_parallel_k_parallel_plus_x_input_z_faraday_y_camera_y_z"
        and float(physical["detuning_hz"]) == 1.5e9
        and int(physical["reference_pulse_duration_us"]) == 200
        and physical["state_feedback"] == "none"
        and physical["oxford_tfbec_in_scope"] is False,
        "physical contract changed",
    )
    _require(
        physical["phase_wrapping_policy"]
        == "record_by_state_and_circular_branch_without_clipping_or_rejecting_the_forward_field",
        "phase-wrapping policy changed",
    )
    grid = config["grid_and_transfer"]
    _require(
        float(grid["object_pixel_m"]) == 0.05e-6
        and grid["object_shape"] == [161, 641]
        and float(grid["camera_pixel_m"]) == 4.44596443228454e-7
        and grid["camera_shape"] == [17, 71]
        and grid["optical_case"] == "measured_best",
        "grid or transfer contract changed",
    )
    readout = config["readout_contract"]
    _require(
        float(readout["pci_phase_plate_amplitude"]) == 0.95
        and float(readout["pci_phase_plate_phase_rad"]) == np.pi / 2.0
        and float(readout["dgi_stop_optical_depth"]) == 4.0
        and readout["dpfi_display_quantity"] == "I_H_over_I0_minus_I_V_over_I0"
        and readout["common_display_denominator"] == "incident_I0",
        "readout contract changed",
    )
    sources = config["sources"]
    for name in (
        "profile_contract",
        "finite_temperature_contract",
        "orientation_contract",
        "atomic_config",
        "objective_config",
    ):
        _source(root, sources[name])
    records = [*sources["implementation_sources"], *sources["generation_sources"]]
    _require(len({item["path"] for item in records}) == len(records), "duplicate source path")
    for record in records:
        _source(root, record)
    profiles = _read_json(root / sources["profile_contract"]["path"])
    _require(
        profiles["target_scale_anchor"]["source_config"]
        == sources["finite_temperature_contract"]["path"],
        "profile contract is not bound to the active finite-temperature contract",
    )
    orientation = _read_json(root / sources["orientation_contract"]["path"])
    geometry = orientation["geometries"]["three_state_equilibrium"]["jones_atomic_response"]["geometry"]
    _require(
        geometry["probe_wavevector_unit_vector"] == [1.0, 0.0, 0.0]
        and geometry["quantisation_axis_unit_vector"] == [1.0, 0.0, 0.0]
        and geometry["input_linear_polarisation_unit_vector"] == [0.0, 0.0, 1.0]
        and geometry["faraday_orthogonal_axis_unit_vector"] == [0.0, 1.0, 0.0],
        "Jones orientation changed",
    )
    return config


def _payload_key(state: str, name: str) -> str:
    return f"{state}__{name}"


def _processed(result: Any) -> dict[str, np.ndarray]:
    return {
        "pci": np.asarray(result.pci.pci_signal_over_i0, dtype=float),
        "dgi": np.asarray(result.dgi.dgi_signal_over_i0, dtype=float),
        "dffi": np.asarray(result.dffi_camera_intensity_over_i0, dtype=float),
        "dpfi": np.asarray(result.dpfi_difference_camera_intensity_over_i0, dtype=float),
    }


def _build_inputs(root: Path, config: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
    sources = config["sources"]
    profile_config = load_target_three_state_profile_config(
        root / sources["profile_contract"]["path"]
    )
    profile_set = build_target_three_state_profiles(profile_config)
    model = _read_json(root / sources["atomic_config"]["path"])
    orientation = _read_json(root / sources["orientation_contract"]["path"])
    objective = _read_json(root / sources["objective_config"]["path"])
    transfer = optical_transfer_from_objective_config(
        objective,
        "measured_best",
        profile_set.profiles[0].column_density_m2.shape,
        float(config["grid_and_transfer"]["object_pixel_m"]),
    )
    return profile_set, model, orientation, objective, transfer


def run_diagnostic(repository_root: Path, config_path: Path) -> Path:
    root = repository_root.resolve()
    config_path = config_path.resolve()
    config = validate_config(root, config_path)
    output = (root / config["lifecycle"]["output_directory"]).resolve()
    scratch = (root / ".scratch").resolve()
    _require(output.parent == scratch, "output must be one direct .scratch child")
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(output)
    staging.mkdir(parents=True)
    (staging / "data").mkdir()

    profile_set, model, orientation, _, transfer = _build_inputs(root, config)
    grid = config["grid_and_transfer"]
    readout = config["readout_contract"]
    object_pixel = float(grid["object_pixel_m"])
    camera_pixel = float(grid["camera_pixel_m"])
    camera_shape = tuple(int(value) for value in grid["camera_shape"])
    _require(
        centered_camera_shape(
            profile_set.profiles[0].column_density_m2.shape,
            object_pixel,
            camera_pixel,
        )
        == camera_shape,
        "declared camera shape is not the largest centred physical array",
    )
    pci_reference = float(readout["pci_phase_plate_amplitude"]) * np.exp(
        1j * float(readout["pci_phase_plate_phase_rad"])
    )
    dgi_reference = 10.0 ** (-float(readout["dgi_stop_optical_depth"]) / 2.0)
    payload: dict[str, np.ndarray] = {
        "object_y_m": np.asarray(profile_set.y_axis_m),
        "object_z_m": np.asarray(profile_set.z_axis_m),
        "object_pixel_size_m": np.asarray(object_pixel),
        "camera_pixel_size_m": np.asarray(camera_pixel),
        "optical_transfer": np.asarray(transfer.transfer),
        "pci_reference_field": np.asarray(pci_reference),
        "dgi_reference_field": np.asarray(dgi_reference),
    }
    rows: list[dict[str, Any]] = []
    camera_y: np.ndarray | None = None
    camera_z: np.ndarray | None = None
    common_limit = 0.0
    max_phase = 0.0
    wrapped_phase_pixels = 0
    phase_diagnostics: dict[str, dict[str, dict[str, float | int]]] = {}
    dgi_negative = 0
    dpfi_negative = 0

    for profile in profile_set.profiles:
        state = profile.definition.state_id
        result = simulate_matched_four_method_jones_images(
            profile.column_density_m2,
            profile_set.y_axis_m,
            profile_set.z_axis_m,
            model_config=model,
            orientation_config=orientation,
            optical_transfer=transfer,
            detuning_hz=float(config["physical_contract"]["detuning_hz"]),
            camera_pixel_size_m=camera_pixel,
            phase_plate_transmittance=float(readout["pci_phase_plate_amplitude"]),
            phase_plate_phase_rad=float(readout["pci_phase_plate_phase_rad"]),
            dgi_stop_optical_depth=float(readout["dgi_stop_optical_depth"]),
        )
        _require(result.pci.camera_intensity_over_i0.shape == camera_shape, "camera shape changed")
        if camera_y is None:
            camera_y = np.asarray(result.camera_y_m)
            camera_z = np.asarray(result.camera_z_m)
            payload["camera_y_m"] = camera_y
            payload["camera_z_m"] = camera_z
        else:
            _require(
                np.array_equal(camera_y, result.camera_y_m)
                and np.array_equal(camera_z, result.camera_z_m),
                "camera axes changed between states",
            )
        arrays = {
            "column_density_m2": profile.column_density_m2,
            "branch_phase_maps_rad": result.pci.branch_phase_maps_rad,
            "branch_optical_depth_maps": result.pci.branch_optical_depth_maps,
            "circular_transmission_fields": result.circular_transmission_fields,
            "co_polarised_object_field": result.pci.co_polarised_object_field,
            "faraday_orthogonal_object_field": result.pci.faraday_orthogonal_object_field,
            "co_polarised_propagated_field": result.co_polarised_propagated_field,
            "faraday_propagated_field": result.faraday_propagated_field,
            "pci_camera_intensity_over_i0": result.pci.camera_intensity_over_i0,
            "dgi_camera_intensity_over_i0": result.dgi.camera_intensity_over_i0,
            "dffi_camera_intensity_over_i0": result.dffi_camera_intensity_over_i0,
            "dpfi_h_camera_intensity_over_i0": result.dpfi_h_camera_intensity_over_i0,
            "dpfi_v_camera_intensity_over_i0": result.dpfi_v_camera_intensity_over_i0,
            "dpfi_sum_camera_intensity_over_i0": result.dpfi_sum_camera_intensity_over_i0,
            "dpfi_difference_camera_intensity_over_i0": result.dpfi_difference_camera_intensity_over_i0,
            "dpfi_normalised_difference": result.dpfi_normalised_difference,
            "pci_co_camera_intensity_over_i0": result.pci_co_camera_intensity_over_i0,
            "dgi_co_camera_intensity_over_i0": result.dgi_co_camera_intensity_over_i0,
            "open_co_camera_intensity_over_i0": result.open_co_camera_intensity_over_i0,
        }
        for name, values in arrays.items():
            payload[_payload_key(state, name)] = np.asarray(values)
        processed = _processed(result)
        branch_phase = np.asarray(result.pci.branch_phase_maps_rad, dtype=float)
        branch_records: dict[str, dict[str, float | int]] = {}
        for branch_index, branch_label in enumerate(("sigma_minus", "sigma_plus")):
            branch_abs = np.abs(branch_phase[branch_index])
            branch_max = float(np.max(branch_abs))
            branch_wrapped = int(np.count_nonzero(branch_abs >= np.pi))
            branch_records[branch_label] = {
                "maximum_absolute_phase_rad": branch_max,
                "wrapped_pixel_count": branch_wrapped,
            }
            max_phase = max(max_phase, branch_max)
            wrapped_phase_pixels += branch_wrapped
        phase_diagnostics[state] = branch_records
        dgi_negative += int(np.count_nonzero(processed["dgi"] < 0.0))
        dpfi_negative += int(np.count_nonzero(processed["dpfi"] < 0.0))
        for method, values in processed.items():
            common_limit = max(common_limit, float(np.max(np.abs(values))))
            rows.append(
                {
                    "state": state,
                    "display_label": profile.definition.label,
                    "method": method,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "maximum_absolute": float(np.max(np.abs(values))),
                    "negative_pixel_count": int(np.count_nonzero(values < 0.0)),
                    "sum": float(np.sum(values)),
                }
            )

    _require(camera_y is not None and camera_z is not None, "camera axes were not created")
    np.savez_compressed(staging / "data" / NPZ_NAME, **payload)
    with (staging / "data" / "state_method_extrema.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "label": EXPECTED_LABEL,
        "status": "diagnostic_not_retained",
        "admitted": False,
        "random_draws": 0,
        "states": list(STATES),
        "methods": list(METHODS),
        "reference_pulse_duration_us": 200,
        "common_I0_zero_centred_display_limit": common_limit,
        "maximum_absolute_circular_branch_phase_rad": max_phase,
        "phase_wrapping_present": bool(max_phase >= np.pi),
        "wrapped_circular_branch_pixel_count": wrapped_phase_pixels,
        "phase_diagnostics_by_state_and_branch": phase_diagnostics,
        "dgi_negative_pixel_count": dgi_negative,
        "dpfi_difference_negative_pixel_count": dpfi_negative,
        "claim_boundary": config["claim_boundary"],
    }
    _write_json(staging / "summary.json", summary)
    _write_json(
        staging / "provenance.json",
        {
            "schema_version": 1,
            "label": EXPECTED_LABEL,
            "status": "diagnostic_not_retained",
            "admitted": False,
            "config_path": config_path.relative_to(root).as_posix(),
            "config_sha256": _sha256(config_path),
            "source_hashes": {
                record["path"]: record["sha256"]
                for record in [
                    config["sources"][name]
                    for name in (
                        "profile_contract",
                        "finite_temperature_contract",
                        "orientation_contract",
                        "atomic_config",
                        "objective_config",
                    )
                ]
                + config["sources"]["implementation_sources"]
                + config["sources"]["generation_sources"]
            },
            "git": _git_record(root),
            "execution": {
                "object_evaluations": 3,
                "random_draws": 0,
                "new_fits": 0,
                "results_writes": 0,
            },
        },
    )
    _write_json(staging / "config_snapshot.json", config)
    _write_json(staging / "artifact_manifest.json", _manifest(staging))
    validate_output(staging, config)
    os.replace(staging, output)
    return output


def _max_error(left: np.ndarray, right: np.ndarray | float | complex) -> float:
    return float(np.max(np.abs(np.asarray(left) - right)))


def replay_output(directory: Path, config: dict[str, Any]) -> dict[str, Any]:
    grid = config["grid_and_transfer"]
    camera_shape = tuple(int(value) for value in grid["camera_shape"])
    with np.load(directory / "data" / NPZ_NAME) as data:
        transfer = np.asarray(data["optical_transfer"], dtype=float)
        object_pixel = float(data["object_pixel_size_m"])
        camera_pixel = float(data["camera_pixel_size_m"])
        pci_reference = complex(data["pci_reference_field"])
        dgi_reference = complex(data["dgi_reference_field"])
        maximum = 0.0
        common_limit = 0.0
        maximum_phase = 0.0
        wrapped_phase_pixels = 0
        phase_diagnostics: dict[str, dict[str, dict[str, float | int]]] = {}
        for state in STATES:
            prefix = f"{state}__"
            phase = np.asarray(data[prefix + "branch_phase_maps_rad"], dtype=float)
            branch_records: dict[str, dict[str, float | int]] = {}
            for branch_index, branch_label in enumerate(("sigma_minus", "sigma_plus")):
                branch_abs = np.abs(phase[branch_index])
                branch_max = float(np.max(branch_abs))
                branch_wrapped = int(np.count_nonzero(branch_abs >= np.pi))
                branch_records[branch_label] = {
                    "maximum_absolute_phase_rad": branch_max,
                    "wrapped_pixel_count": branch_wrapped,
                }
                maximum_phase = max(maximum_phase, branch_max)
                wrapped_phase_pixels += branch_wrapped
            phase_diagnostics[state] = branch_records
            optical_depth = np.asarray(data[prefix + "branch_optical_depth_maps"], dtype=float)
            circular = np.exp(-optical_depth / 2.0 + 1j * phase)
            co = (circular[0] + circular[1]) / 2.0
            faraday = 0.5j * (circular[0] - circular[1])
            co_prop = np.fft.ifft2(np.fft.fft2(co - 1.0) * transfer)
            f_prop = np.fft.ifft2(np.fft.fft2(faraday) * transfer)
            open_co = 1.0 + co_prop
            optical = {
                "pci_camera_intensity_over_i0": np.abs(pci_reference + co_prop) ** 2 + np.abs(f_prop) ** 2,
                "dgi_camera_intensity_over_i0": np.abs(dgi_reference + co_prop) ** 2 + np.abs(f_prop) ** 2,
                "dffi_camera_intensity_over_i0": np.abs(f_prop) ** 2,
                "dpfi_h_camera_intensity_over_i0": np.abs(open_co + f_prop) ** 2 / 2.0,
                "dpfi_v_camera_intensity_over_i0": np.abs(open_co - f_prop) ** 2 / 2.0,
            }
            camera = {
                name: resample_to_camera_pixels(values, object_pixel, camera_pixel, camera_shape)
                for name, values in optical.items()
            }
            for name, expected in camera.items():
                maximum = max(maximum, _max_error(data[prefix + name], expected))
            maximum = max(
                maximum,
                _max_error(data[prefix + "circular_transmission_fields"], circular),
                _max_error(data[prefix + "co_polarised_object_field"], co),
                _max_error(data[prefix + "faraday_orthogonal_object_field"], faraday),
                _max_error(data[prefix + "co_polarised_propagated_field"], co_prop),
                _max_error(data[prefix + "faraday_propagated_field"], f_prop),
            )
            h = camera["dpfi_h_camera_intensity_over_i0"]
            v = camera["dpfi_v_camera_intensity_over_i0"]
            difference = h - v
            total = h + v
            maximum = max(
                maximum,
                _max_error(data[prefix + "dpfi_sum_camera_intensity_over_i0"], total),
                _max_error(data[prefix + "dpfi_difference_camera_intensity_over_i0"], difference),
                _max_error(data[prefix + "dpfi_normalised_difference"], difference / total),
            )
            processed = (
                camera["pci_camera_intensity_over_i0"] - abs(pci_reference) ** 2,
                camera["dgi_camera_intensity_over_i0"] - abs(dgi_reference) ** 2,
                camera["dffi_camera_intensity_over_i0"],
                difference,
            )
            common_limit = max(common_limit, *(float(np.max(np.abs(value))) for value in processed))
        return {
            "maximum_replay_error": maximum,
            "common_I0_zero_centred_display_limit": common_limit,
            "maximum_absolute_circular_branch_phase_rad": maximum_phase,
            "wrapped_circular_branch_pixel_count": wrapped_phase_pixels,
            "phase_diagnostics_by_state_and_branch": phase_diagnostics,
        }


def validate_output(directory: Path, config: dict[str, Any]) -> dict[str, Any]:
    directory = directory.resolve()
    manifest = _read_json(directory / "artifact_manifest.json")
    _require(manifest == _manifest(directory), "artifact manifest changed")
    summary = _read_json(directory / "summary.json")
    _require(summary["label"] == EXPECTED_LABEL and summary["admitted"] is False, "summary identity changed")
    _require(_read_json(directory / "config_snapshot.json") == config, "config snapshot changed")
    provenance = _read_json(directory / "provenance.json")
    _require(provenance["status"] == "diagnostic_not_retained" and provenance["admitted"] is False, "provenance lifecycle changed")
    replay = replay_output(directory, config)
    tolerance = float(config["numerical_tolerances"]["maximum_replay_error"])
    _require(replay["maximum_replay_error"] <= tolerance, "independent field/camera replay failed")
    _require(
        abs(replay["common_I0_zero_centred_display_limit"] - summary["common_I0_zero_centred_display_limit"])
        <= tolerance,
        "common display limit changed",
    )
    _require(
        abs(
            float(summary["maximum_absolute_circular_branch_phase_rad"])
            - float(replay["maximum_absolute_circular_branch_phase_rad"])
        )
        <= tolerance,
        "maximum circular phase changed",
    )
    _require(
        int(summary["wrapped_circular_branch_pixel_count"])
        == int(replay["wrapped_circular_branch_pixel_count"]),
        "phase-wrapped pixel inventory changed",
    )
    _require(
        summary["phase_diagnostics_by_state_and_branch"]
        == replay["phase_diagnostics_by_state_and_branch"],
        "state/branch phase diagnostics changed",
    )
    _require(
        bool(summary["phase_wrapping_present"])
        == (float(summary["maximum_absolute_circular_branch_phase_rad"]) >= np.pi),
        "phase-wrapping status changed",
    )
    return summary


__all__ = [
    "EXPECTED_LABEL",
    "METHODS",
    "NPZ_NAME",
    "STATES",
    "replay_output",
    "run_diagnostic",
    "validate_config",
    "validate_output",
]
