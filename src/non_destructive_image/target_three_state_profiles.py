"""Build the target-scale BEC, SSP and ID profiles from one contract."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .equilibrium_profiles import (
    EquilibriumProfile,
    EquilibriumProfileDefinition,
    build_equilibrium_profile,
)


EXPECTED_LABEL = "three_state_target_trap_profiles_v4"
EXPECTED_STATE_IDS = (
    "smooth_bec",
    "connected_modulated",
    "separated_droplets",
)


@dataclass(frozen=True)
class TargetThreeStateProfileSet:
    """Sampled target-scale profiles and their shared coordinate axes."""

    config: Mapping[str, Any]
    y_axis_m: NDArray[np.floating]
    z_axis_m: NDArray[np.floating]
    profiles: tuple[EquilibriumProfile, ...]


def load_target_three_state_profile_config(path: str | Path) -> dict[str, Any]:
    """Load and validate identity and cross-field contract invariants."""

    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("label") != EXPECTED_LABEL:
        raise ValueError("unexpected target three-state profile contract")
    profiles = config.get("profiles")
    if not isinstance(profiles, list) or tuple(
        profile.get("state_id") for profile in profiles
    ) != EXPECTED_STATE_IDS:
        raise ValueError("target profile order or state identity changed")

    source = config["source_morphology"]
    target = config["target_scale_anchor"]
    construction = config["construction_boundary"]
    source_centres = tuple(float(value) for value in source["axial_peak_centres_um"])
    spacing_um = float(source["representative_neighbour_spacing_um"])
    if not np.allclose(np.diff(source_centres), spacing_um, rtol=0.0, atol=1e-12):
        raise ValueError("source peak centres do not implement the declared spacing")
    if not np.isclose(
        float(construction["axial_peak_spacing_um"]),
        spacing_um,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("construction and source spacing disagree")
    for profile in profiles[1:]:
        if not np.allclose(
            np.asarray(profile["component_centres_y_um"], dtype=float),
            source_centres,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("modulated profile centres disagree with the source scale")
        if not np.allclose(
            np.asarray(profile["component_weights"], dtype=float),
            np.asarray(source["axial_peak_weights"], dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("modulated profile weights disagree with the contract")
    common_atoms = float(target["condensate_atoms"])
    if any(
        not np.isclose(
            float(profile["atom_number"]), common_atoms, rtol=0.0, atol=1e-12
        )
        for profile in profiles
    ):
        raise ValueError("target and profile condensate atom numbers disagree")
    target_radii = np.asarray(target["dipolar_tf_radii_um_xyz"], dtype=float)
    if not np.allclose(
        [profiles[0]["radius_y_um"], profiles[0]["radius_z_um"]],
        target_radii[[1, 2]],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("smooth BEC radii disagree with the target TF core")
    target_rms_z = float(target["projected_tf_rms_um_yz"][1])
    if any(
        not np.isclose(
            float(profile["component_sigma_z_um"]),
            target_rms_z,
            rtol=0.0,
            atol=1e-12,
        )
        for profile in profiles[1:]
    ):
        raise ValueError("modulated transverse widths disagree with the target TF core")
    return config


def _axis(minimum_um: float, maximum_um: float, spacing_um: float) -> NDArray[np.floating]:
    if not np.isfinite([minimum_um, maximum_um, spacing_um]).all():
        raise ValueError("sampling coordinates must be finite")
    if spacing_um <= 0.0 or maximum_um <= minimum_um:
        raise ValueError("sampling interval must be positive and ordered")
    intervals = int(round((maximum_um - minimum_um) / spacing_um))
    if not np.isclose(
        minimum_um + intervals * spacing_um,
        maximum_um,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("sampling endpoints must contain an integer number of steps")
    axis = np.linspace(minimum_um, maximum_um, intervals + 1, dtype=float) * 1e-6
    axis.setflags(write=False)
    return axis


def _definition(profile: Mapping[str, Any]) -> EquilibriumProfileDefinition:
    common = {
        "state_id": str(profile["state_id"]),
        "label": str(profile["label"]),
        "morphology": str(profile["morphology"]),
        "atom_number": float(profile["atom_number"]),
    }
    if profile["morphology"] == "smooth_bec":
        return EquilibriumProfileDefinition(
            **common,
            radius_y_m=float(profile["radius_y_um"]) * 1e-6,
            radius_z_m=float(profile["radius_z_um"]) * 1e-6,
        )
    return EquilibriumProfileDefinition(
        **common,
        component_centres_y_m=tuple(
            float(value) * 1e-6 for value in profile["component_centres_y_um"]
        ),
        component_weights=tuple(float(value) for value in profile["component_weights"]),
        component_sigma_y_m=float(profile["component_sigma_y_um"]) * 1e-6,
        component_sigma_z_m=float(profile["component_sigma_z_um"]) * 1e-6,
    )


def build_target_three_state_profiles(
    config: Mapping[str, Any],
    *,
    grid_spacing_um: float | None = None,
) -> TargetThreeStateProfileSet:
    """Build all three profiles on the contract grid or a convergence grid."""

    sampling = config["sampling_and_validation"]
    spacing_um = (
        float(sampling["grid_spacing_um"])
        if grid_spacing_um is None
        else float(grid_spacing_um)
    )
    y_axis = _axis(float(sampling["y_min_um"]), float(sampling["y_max_um"]), spacing_um)
    z_axis = _axis(float(sampling["z_min_um"]), float(sampling["z_max_um"]), spacing_um)
    y_grid, z_grid = np.meshgrid(y_axis, z_axis)
    profiles = tuple(
        build_equilibrium_profile(
            _definition(profile),
            y_grid,
            z_grid,
            minimum_peak_distance_m=float(sampling["minimum_peak_distance_um"]) * 1e-6,
            peak_prominence_fraction=float(sampling["peak_prominence_fraction"]),
        )
        for profile in config["profiles"]
    )
    return TargetThreeStateProfileSet(
        config=config,
        y_axis_m=y_axis,
        z_axis_m=z_axis,
        profiles=profiles,
    )


def boundary_to_peak_ratio(profile: EquilibriumProfile) -> float:
    """Return the largest sampled boundary value relative to the map peak."""

    values = profile.column_density_m2
    boundary = max(
        float(np.max(values[0, :])),
        float(np.max(values[-1, :])),
        float(np.max(values[:, 0])),
        float(np.max(values[:, -1])),
    )
    return boundary / float(np.max(values))


__all__ = [
    "EXPECTED_LABEL",
    "TargetThreeStateProfileSet",
    "boundary_to_peak_ratio",
    "build_target_three_state_profiles",
    "load_target_three_state_profile_config",
]
