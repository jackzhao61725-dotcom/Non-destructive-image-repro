from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from isolated_non_destructive_image import (  # noqa: E402
    load_isolated_non_destructive_image_module,
)


_PROFILES = load_isolated_non_destructive_image_module(
    "equilibrium_profiles",
    namespace="_ndi_corrected_acquisition_scientific_tests_v1",
)
EquilibriumProfileDefinition = _PROFILES.EquilibriumProfileDefinition
build_equilibrium_profile = _PROFILES.build_equilibrium_profile
measure_morphology = _PROFILES.measure_morphology


def _grid() -> tuple[np.ndarray, np.ndarray]:
    axis_m = (np.arange(512, dtype=float) - 256) * (40e-6 / 512)
    return np.meshgrid(axis_m, axis_m)


def _definitions() -> tuple[EquilibriumProfileDefinition, ...]:
    return (
        EquilibriumProfileDefinition(
            state_id="smooth_bec",
            label="Smooth BEC",
            morphology="smooth_bec",
            atom_number=5.0e4,
            radius_y_m=9.0e-6,
            radius_z_m=3.2e-6,
        ),
        EquilibriumProfileDefinition(
            state_id="connected_modulated",
            label="Connected density-modulated profile",
            morphology="connected_modulated",
            atom_number=5.0e4,
            component_centres_y_m=(-4e-6, 0.0, 4e-6),
            component_weights=(0.82, 1.0, 0.82),
            component_sigma_y_m=1.35e-6,
            component_sigma_z_m=1.0e-6,
        ),
        EquilibriumProfileDefinition(
            state_id="separated_droplets",
            label="Separated droplet array",
            morphology="separated_droplets",
            atom_number=5.0e4,
            component_centres_y_m=(-4e-6, 0.0, 4e-6),
            component_weights=(0.82, 1.0, 0.82),
            component_sigma_y_m=0.8e-6,
            component_sigma_z_m=0.9e-6,
        ),
    )


def test_three_profiles_share_atom_number_grid_and_axial_scale() -> None:
    y_grid_m, z_grid_m = _grid()
    profiles = [
        build_equilibrium_profile(
            definition,
            y_grid_m,
            z_grid_m,
            minimum_peak_distance_m=2.4e-6,
            peak_prominence_fraction=0.05,
        )
        for definition in _definitions()
    ]

    for profile in profiles:
        assert profile.column_density_m2.shape == y_grid_m.shape
        assert np.isfinite(profile.column_density_m2).all()
        assert np.all(profile.column_density_m2 >= 0.0)
        assert profile.observables.integrated_weight == pytest.approx(5.0e4, rel=2e-14)
        assert not profile.column_density_m2.flags.writeable

    rms_y = np.asarray([profile.observables.rms_y_m for profile in profiles])
    assert np.ptp(rms_y) / np.mean(rms_y) < 0.1


def test_truth_observables_resolve_smooth_connected_and_separated_profiles() -> None:
    y_grid_m, z_grid_m = _grid()
    profiles = [
        build_equilibrium_profile(
            definition,
            y_grid_m,
            z_grid_m,
            minimum_peak_distance_m=2.4e-6,
            peak_prominence_fraction=0.05,
        )
        for definition in _definitions()
    ]
    smooth, connected, separated = (profile.observables for profile in profiles)

    assert smooth.peak_count == 1
    assert smooth.mean_peak_spacing_m is None
    assert smooth.modulation_contrast == 0.0
    assert connected.peak_count == 3
    assert separated.peak_count == 3
    assert connected.mean_peak_spacing_m == pytest.approx(4.0e-6, abs=0.1e-6)
    assert separated.mean_peak_spacing_m == pytest.approx(4.0e-6, abs=0.1e-6)
    assert 0.1 < connected.modulation_contrast < 0.7
    assert separated.modulation_contrast > connected.modulation_contrast + 0.25


def test_profile_contract_rejects_inconsistent_or_nonphysical_inputs() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        EquilibriumProfileDefinition(
            state_id="bad",
            label="Bad",
            morphology="connected_modulated",
            atom_number=1.0,
            component_centres_y_m=(0.0, 0.0),
            component_weights=(1.0, 1.0),
            component_sigma_y_m=1.0,
            component_sigma_z_m=1.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        measure_morphology(
            np.asarray([[0.0, -1.0, 0.0]] * 3),
            np.arange(3, dtype=float),
            np.arange(3, dtype=float),
            minimum_peak_distance_m=1.0,
            peak_prominence_fraction=0.05,
        )
