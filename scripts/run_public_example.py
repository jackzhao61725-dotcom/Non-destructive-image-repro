"""Run the bounded public example for the current dissertation model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from non_destructive_image.camera import centered_camera_shape  # noqa: E402
from non_destructive_image.equilibrium_imaging import (  # noqa: E402
    optical_transfer_from_objective_config,
)
from non_destructive_image.four_method_jones_imaging import (  # noqa: E402
    simulate_matched_four_method_jones_images,
)
from non_destructive_image.target_multiframe_noise_acquisition import (  # noqa: E402
    sample_raw_frame,
)
from non_destructive_image.public_inference import (  # noqa: E402
    PublicBECFitContext,
    fit_public_dpfi,
    three_peak_observables,
)
from non_destructive_image.target_three_state_profiles import (  # noqa: E402
    build_target_three_state_profiles,
    load_target_three_state_profile_config,
)


DEFAULT_CONFIG = ROOT / "configs" / "public_example_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != 1 or config.get("label") != "public_example_v1":
        raise ValueError("unexpected public example config")
    for name, identity in config["sources"].items():
        source = ROOT / identity["path"]
        if not source.is_file() or _sha256(source) != identity["sha256"]:
            raise ValueError(f"public example source identity changed: {name}")
    return config


def _camera_axes(shape: tuple[int, int], pixel_m: float) -> tuple[np.ndarray, np.ndarray]:
    z = (np.arange(shape[0], dtype=float) - (shape[0] - 1) / 2.0) * pixel_m
    y = (np.arange(shape[1], dtype=float) - (shape[1] - 1) / 2.0) * pixel_m
    return y, z


def _fit_context(
    config: Mapping[str, Any],
    profile_set: Any,
    atomic: Mapping[str, Any],
    orientation: Mapping[str, Any],
    transfer: Any,
) -> PublicBECFitContext:
    """Build the bounded Section 5.1 example from explicit public inputs."""

    optical = config["optical_model"]
    fit = config["one_fit"]
    bec = profile_set.profiles[0]
    camera_shape = centered_camera_shape(
        bec.column_density_m2.shape,
        float(optical["object_pixel_m"]),
        float(optical["camera_pixel_m"]),
    )
    camera_y_m, camera_z_m = _camera_axes(
        camera_shape, float(optical["camera_pixel_m"])
    )
    return PublicBECFitContext(
        physical_model=fit["physical_model"],
        solver=fit["solver"],
        n0=float(bec.definition.atom_number),
        ry0_m=float(bec.definition.radius_y_m),
        rz0_m=float(bec.definition.radius_z_m),
        y_m=np.asarray(profile_set.y_axis_m, dtype=float),
        z_m=np.asarray(profile_set.z_axis_m, dtype=float),
        object_pixel_m=float(optical["object_pixel_m"]),
        atomic_config=atomic,
        orientation_config=orientation,
        optical_transfer=transfer,
        camera_y_m=camera_y_m,
        camera_z_m=camera_z_m,
        camera_pixel_m=float(optical["camera_pixel_m"]),
        detuning_hz=float(optical["detuning_hz"]),
        dgi_stop_optical_depth=float(optical["dgi_stop_optical_depth"]),
        count_scale=int(fit["pulse_duration_us"])
        * float(fit["photoelectrons_per_i0_pixel_per_us"]),
        read_noise_electrons_rms=float(fit["read_noise_electrons_rms"]),
    )


def _state_summary(profile: Any, image: Any) -> dict[str, Any]:
    definition = profile.definition
    summary: dict[str, Any] = {
        "state": definition.state_id,
        "atom_number": float(definition.atom_number),
        "peak_column_density_m2": float(np.max(profile.column_density_m2)),
        "camera_response_range": {
            "pci": [
                float(np.min(image.pci.pci_signal_over_i0)),
                float(np.max(image.pci.pci_signal_over_i0)),
            ],
            "dgi": [
                float(np.min(image.dgi.dgi_signal_over_i0)),
                float(np.max(image.dgi.dgi_signal_over_i0)),
            ],
            "dffi": [
                float(np.min(image.dffi_camera_intensity_over_i0)),
                float(np.max(image.dffi_camera_intensity_over_i0)),
            ],
            "dpfi": [
                float(np.min(image.dpfi_difference_camera_intensity_over_i0)),
                float(np.max(image.dpfi_difference_camera_intensity_over_i0)),
            ],
        },
    }
    if definition.morphology != "smooth_bec":
        observables = three_peak_observables(
            float(definition.component_centres_y_m[2]) * 1e6,
            float(definition.component_sigma_y_m) * 1e6,
        )
        summary["visible_peak_spacing_um"] = observables["d_peak_um"]
        summary["valley_to_peak_ratio"] = observables["nu_vp"]
    return summary


def _one_dpfi_fit(context: PublicBECFitContext, config: Mapping[str, Any]) -> dict[str, Any]:
    fit = config["one_fit"]
    truth = np.asarray(fit["truth_eta_rho_y_y0_um"], dtype=float)
    truth_physical = np.asarray(
        [math.log(truth[0]), math.log(truth[1]), truth[2]], dtype=float
    )
    expected = context.expected_roles(context.truth_vector(truth_physical))
    seed_prefix = tuple(int(value) for value in fit["seed"])
    observed = {
        role: sample_raw_frame(
            values,
            read_noise_electrons_rms=context.read_noise_electrons_rms,
            seed=(*seed_prefix, role_index),
        )
        for role_index, (role, values) in enumerate(expected.items())
    }
    start = np.asarray(fit["start_eta_rho_y_y0_um"], dtype=float)
    physical_start = np.asarray(
        [math.log(start[0]), math.log(start[1]), start[2]], dtype=float
    )
    outcome = fit_public_dpfi(context, observed, physical_start)
    if not outcome["converged"]:
        raise RuntimeError(f"public DPFI fit did not converge: {outcome['termination_message']}")
    return {
        "method": "DPFI",
        "truth": {"eta": truth[0], "rho_y": truth[1], "y0_um": truth[2]},
        "start": {"eta": start[0], "rho_y": start[1], "y0_um": start[2]},
        "estimate": {
            "eta": outcome["eta_hat"],
            "rho_y": outcome["rho_y_hat"],
            "y0_um": outcome["y0_hat_um"],
        },
        "reduced_raw_objective": outcome["reduced_raw_objective"],
        "function_evaluations": outcome["nfev_total"],
        "optical_evaluations_including_truth": context.optical_evaluations,
        "interpretation": "one fixed-seed workflow check; not a recovery distribution",
    }


def _repeated_bec_summary(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = ROOT / config["sources"]["repeated_bec_summary"]["path"]
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    view = config["repeated_bec_view"]
    durations = {int(view["pulse_duration_us"])}
    q_values = {int(value) for value in view["image_q"]}
    observables = set(view["observables"])
    selected = [
        row
        for row in rows
        if int(row["duration_us"]) in durations
        and int(row["image_q"]) in q_values
        and row["observable"] in observables
    ]
    expected = len(durations) * len(q_values) * len(observables)
    if len(selected) != expected:
        raise ValueError("authenticated repeated-BEC view is incomplete")
    return [
        {
            "pulse_duration_us": int(row["duration_us"]),
            "image_q": int(row["image_q"]),
            "observable": row["observable"],
            "equilibrium_input": float(row["truth"]),
            "recovered_median": float(row["median"]),
            "central_68_percent": [float(row["q16"]), float(row["q84"])],
        }
        for row in selected
    ]


def run_example(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_config(config_path)
    sources = config["sources"]
    profile_config = load_target_three_state_profile_config(
        ROOT / sources["profile_contract"]["path"]
    )
    profile_set = build_target_three_state_profiles(profile_config)
    atomic = _read_json(ROOT / sources["atomic_config"]["path"])
    orientation = _read_json(ROOT / sources["orientation_contract"]["path"])
    objective = _read_json(ROOT / sources["objective_config"]["path"])
    optical = config["optical_model"]
    transfer = optical_transfer_from_objective_config(
        objective,
        str(optical["objective_case"]),
        profile_set.profiles[0].column_density_m2.shape,
        float(optical["object_pixel_m"]),
    )

    state_summaries = []
    for profile in profile_set.profiles:
        image = simulate_matched_four_method_jones_images(
            profile.column_density_m2,
            profile_set.y_axis_m,
            profile_set.z_axis_m,
            model_config=atomic,
            orientation_config=orientation,
            optical_transfer=transfer,
            detuning_hz=float(optical["detuning_hz"]),
            camera_pixel_size_m=float(optical["camera_pixel_m"]),
            phase_plate_transmittance=float(optical["phase_plate_amplitude"]),
            phase_plate_phase_rad=float(optical["phase_plate_phase_rad"]),
            dgi_stop_optical_depth=float(optical["dgi_stop_optical_depth"]),
        )
        state_summaries.append(_state_summary(profile, image))

    context = _fit_context(config, profile_set, atomic, orientation, transfer)
    return {
        "schema_version": 1,
        "status": "public_example_complete",
        "static_three_state_route": state_summaries,
        "one_bec_fit": _one_dpfi_fit(context, config),
        "repeated_bec_route": _repeated_bec_summary(config),
        "claim_boundary": config["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    config = _load_config(arguments.config.resolve())
    if arguments.validate_only:
        result: dict[str, Any] = {
            "status": "pass",
            "label": config["label"],
            "source_count": len(config["sources"]),
        }
    else:
        result = run_example(arguments.config.resolve())
    if arguments.output is not None:
        output = arguments.output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
