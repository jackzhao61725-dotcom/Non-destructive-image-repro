"""Shared configuration and endpoint construction for public drivers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Any, Mapping

import numpy as np
import scipy

from non_destructive_image.orientation_endpoints import (
    OrientationEndpointBuildContract,
    OrientationEndpointSpec,
    build_orientation_endpoint_pair,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILENAMES = ("model.json", "reference_state.json", "reproduction.json")


def read_json(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_configs(config_dir: Path) -> tuple[dict[str, Any], ...]:
    """Load and minimally validate the three public configuration authorities."""

    resolved = config_dir.resolve()
    paths = tuple(resolved / name for name in CONFIG_FILENAMES)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing public config files: {missing}")
    model, reference, reproduction = tuple(read_json(path) for path in paths)
    for label, config in zip(CONFIG_FILENAMES, (model, reference, reproduction), strict=True):
        if config.get("schema_version") != 1:
            raise ValueError(f"{label} has an unsupported schema_version")
    source = model.get("source")
    if not isinstance(source, Mapping) or not re.fullmatch(
        r"[0-9a-f]{40}", str(source.get("commit", ""))
    ):
        raise ValueError("model source.commit must be a full Git SHA-1")
    fluences = reproduction.get("probe", {}).get("fluence_scan_mw_us")
    if not isinstance(fluences, list) or len(fluences) != 17:
        raise ValueError("reproduction config must list the 17 fluence values")
    if any(not np.isfinite(float(value)) or float(value) <= 0.0 for value in fluences):
        raise ValueError("fluence values must be positive and finite")
    if len(set(float(value) for value in fluences)) != len(fluences):
        raise ValueError("fluence values must be unique")
    return model, reference, reproduction


def input_identity(config_dir: Path) -> dict[str, Any]:
    """Return the hashes and software versions attached to regenerated output."""

    resolved = config_dir.resolve()
    return {
        "configs": {
            name: sha256_file(resolved / name) for name in CONFIG_FILENAMES
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }


def endpoint_products(
    model: Mapping[str, Any],
    reference: Mapping[str, Any],
    reproduction: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Build the two independently instantiated orientation endpoints."""

    endpoint_records = reproduction["endpoints"]
    if not isinstance(endpoint_records, list) or len(endpoint_records) != 2:
        raise ValueError("exactly two orientation endpoints are required")
    specs = tuple(
        OrientationEndpointSpec(
            label=str(item["label"]),
            dipole_axis=str(item["dipole_axis"]),
            dipole_axis_index=int(item["dipole_axis_index"]),
            theta_d_deg=float(item["theta_d_deg"]),
            probe_axis=tuple(float(value) for value in item["probe_axis"]),
            quantisation_axis=tuple(
                float(value) for value in item["quantisation_axis"]
            ),
            polarisation_axis=tuple(
                float(value) for value in item["polarisation_axis"]
            ),
        )
        for item in endpoint_records
    )
    acquisition = reproduction["acquisition"]
    probe = reproduction["probe"]
    contract = OrientationEndpointBuildContract(
        source_condition_id=str(reproduction["reference"]["condition_id"]),
        source_repetition_id=str(reproduction["reference"]["repetition_id"]),
        detuning_hz=float(probe["detuning_hz"]),
        field_of_view_m=float(acquisition["field_of_view_m"]),
        canonical_ngrid=int(acquisition["canonical_ngrid"]),
        inverse_ngrid=int(acquisition["inverse_ngrid"]),
        camera_pixel_size_m=float(acquisition["camera_pixel_size_m"]),
        camera_output_shape=tuple(int(value) for value in acquisition["camera_output_shape"]),
        numerical_aperture=float(acquisition["numerical_aperture"]),
        wavelength_m=float(model["atom"]["transition_wavelength_m"]),
        photoelectrons_per_i0_pixel=float(
            acquisition["photoelectrons_per_i0_pixel_at_300_mw_us"]
        ),
        read_noise_electrons=float(acquisition["read_noise_electrons"]),
        phase_plate_transmittance=float(model["pci"]["phase_plate_transmittance"]),
        phase_plate_phase_rad=float(model["pci"]["phase_plate_phase_rad"]),
        independent_exposures_by_role=acquisition["independent_exposures_by_role"],
    )
    products = build_orientation_endpoint_pair(
        specs=(specs[0], specs[1]),
        contract=contract,
        model_config=model,
        initial_condition_config=reference,
    )
    return products[0], products[1]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one stable UTF-8 JSON object."""

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(target)
