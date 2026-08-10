"""Shared configuration and endpoint construction for public drivers."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import tomllib
from typing import Any, Mapping

import numpy as np
import non_destructive_image
import scipy

from non_destructive_image.orientation_endpoints import (
    OrientationEndpointBuildContract,
    OrientationEndpointSpec,
    build_orientation_endpoint_pair,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILENAMES = ("model.json", "reference_state.json", "reproduction.json")
PACKAGE_DISTRIBUTION = "equilibrium-dispersive-imaging-reproduction"


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


def _portable_path(value: Path) -> str:
    """Return a useful path identity without recording a user-specific root."""

    resolved = value.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def _json_value(value: Any) -> Any:
    """Return a JSON-safe representation of one parsed CLI value."""

    if isinstance(value, Path):
        return _portable_path(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported invocation argument type: {type(value).__name__}")


def parsed_invocation_arguments(
    arguments: argparse.Namespace | Mapping[str, Any],
) -> dict[str, Any]:
    """Return all parsed invocation arguments in stable JSON-safe form."""

    values = vars(arguments) if isinstance(arguments, argparse.Namespace) else arguments
    return {str(key): _json_value(value) for key, value in values.items()}


def input_identity(
    config_dir: Path,
    *,
    invocation_arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the hashes and software versions attached to regenerated output."""

    resolved = config_dir.resolve()
    software: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    importable_version = getattr(non_destructive_image, "__version__", None)
    try:
        distribution_version: str | None = version(PACKAGE_DISTRIBUTION)
    except PackageNotFoundError:
        distribution_version = None
    if importable_version is not None:
        package_version = str(importable_version)
        version_source = "importable_package"
    elif distribution_version is not None:
        package_version = distribution_version
        version_source = "installed_distribution_metadata"
    else:
        with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        package_version = str(project.get("version", "unknown"))
        version_source = "repository_pyproject"
    software["package"] = {
        "distribution": PACKAGE_DISTRIBUTION,
        "version": package_version,
        "version_source": version_source,
        "importable_version": (
            None if importable_version is None else str(importable_version)
        ),
        "distribution_metadata_version": distribution_version,
        "versions_match": (
            None
            if importable_version is None or distribution_version is None
            else str(importable_version) == distribution_version
        ),
    }
    return {
        "configs": {
            name: sha256_file(resolved / name) for name in CONFIG_FILENAMES
        },
        "software": software,
        "invocation_arguments": parsed_invocation_arguments(
            {} if invocation_arguments is None else invocation_arguments
        ),
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


def require_output_available(path: Path, *, overwrite: bool) -> None:
    """Refuse an existing output before starting an expensive reproduction."""

    if not overwrite and path.resolve().exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path.resolve()}")


def write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Publish one stable JSON object atomically, without overwrite by default."""

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    require_output_available(target, overwrite=overwrite)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise FileExistsError(
                    f"refusing to overwrite existing output: {target}"
                ) from error
            except OSError as error:
                raise OSError(
                    "could not publish the output exclusively; use a local "
                    "filesystem with hard-link support or pass --overwrite "
                    "after checking the target"
                ) from error
    finally:
        temporary.unlink(missing_ok=True)
