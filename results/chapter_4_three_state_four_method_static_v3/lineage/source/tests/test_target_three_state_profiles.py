from __future__ import annotations

import json
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


_TARGET = load_isolated_non_destructive_image_module(
    "target_three_state_profiles",
    namespace="_ndi_target_three_state_profile_tests_v1",
)
CONFIG = ROOT / "configs" / "three_state_target_trap_profiles_v4.json"


def _profile_set(spacing_um: float | None = None):
    config = _TARGET.load_target_three_state_profile_config(CONFIG)
    return _TARGET.build_target_three_state_profiles(
        config,
        grid_spacing_um=spacing_um,
    )


def test_target_profile_contract_implements_frozen_parameters() -> None:
    profile_set = _profile_set()
    config = profile_set.config
    smooth, connected, separated = config["profiles"]

    assert config["source_morphology"]["representative_neighbour_spacing_um"] == 2.8
    assert config["source_morphology"]["source_example_atom_number"] == 50000.0
    assert smooth["atom_number"] == pytest.approx(50701.094679638154)
    assert smooth["radius_y_um"] == pytest.approx(14.652216669933539)
    assert smooth["radius_z_um"] == pytest.approx(1.470434840433167)
    assert connected["component_centres_y_um"] == [-2.8, 0.0, 2.8]
    assert separated["component_centres_y_um"] == [-2.8, 0.0, 2.8]
    assert connected["component_sigma_y_um"] == pytest.approx(0.945)
    assert separated["component_sigma_y_um"] == pytest.approx(0.56)
    assert connected["component_sigma_z_um"] == pytest.approx(0.5557721295587291)
    assert separated["component_sigma_z_um"] == pytest.approx(0.5557721295587291)


def test_generated_profiles_are_normalised_supported_and_morphologically_distinct() -> None:
    profile_set = _profile_set()
    config = profile_set.config
    validation = config["sampling_and_validation"]
    smooth, connected, separated = profile_set.profiles

    assert profile_set.y_axis_m.size == 641
    assert profile_set.z_axis_m.size == 161
    for profile in profile_set.profiles:
        assert profile.observables.integrated_weight == pytest.approx(
            50701.094679638154,
            rel=validation["atom_number_relative_tolerance"],
        )
        assert np.isfinite(profile.column_density_m2).all()
        assert np.all(profile.column_density_m2 >= 0.0)
        assert _TARGET.boundary_to_peak_ratio(profile) < validation[
            "maximum_boundary_to_peak_ratio"
        ]

    assert smooth.observables.peak_count == 1
    assert smooth.observables.mean_peak_spacing_m is None
    assert connected.observables.peak_count == 3
    assert separated.observables.peak_count == 3
    target_spacing = config["construction_boundary"]["axial_peak_spacing_um"]
    tolerance = validation["peak_spacing_absolute_tolerance_um"]
    assert connected.observables.mean_peak_spacing_m * 1e6 == pytest.approx(
        target_spacing,
        abs=tolerance,
    )
    assert separated.observables.mean_peak_spacing_m * 1e6 == pytest.approx(
        target_spacing,
        abs=tolerance,
    )
    low, high = validation["ssp_contrast_interval"]
    assert low < connected.observables.modulation_contrast < high
    assert separated.observables.modulation_contrast > validation[
        "id_minimum_contrast"
    ]
    assert (
        separated.observables.modulation_contrast
        - connected.observables.modulation_contrast
        > validation["minimum_id_minus_ssp_contrast"]
    )


def test_recorded_generated_truth_replays_from_the_contract() -> None:
    profile_set = _profile_set()
    truth = profile_set.config["generated_truth_on_contract_grid"]
    assert [row["state_id"] for row in truth] == [
        profile.definition.state_id for profile in profile_set.profiles
    ]
    for row, profile in zip(truth, profile_set.profiles, strict=True):
        observables = profile.observables
        assert row["integrated_atoms"] == pytest.approx(observables.integrated_weight)
        assert row["rms_y_um"] == pytest.approx(observables.rms_y_m * 1e6)
        assert row["rms_z_um"] == pytest.approx(observables.rms_z_m * 1e6)
        assert row["peak_count"] == observables.peak_count
        assert row["peak_positions_y_um"] == pytest.approx(
            np.asarray(observables.peak_positions_y_m) * 1e6
        )
        if observables.mean_peak_spacing_m is None:
            assert row["mean_peak_spacing_um"] is None
        else:
            assert row["mean_peak_spacing_um"] == pytest.approx(
                observables.mean_peak_spacing_m * 1e6
            )
        assert row["modulation_contrast"] == pytest.approx(
            observables.modulation_contrast
        )
        assert row["peak_column_density_um2"] == pytest.approx(
            float(np.max(profile.column_density_m2)) / 1e12
        )
        assert row["boundary_to_peak_ratio"] == pytest.approx(
            _TARGET.boundary_to_peak_ratio(profile)
        )


def test_profile_truth_is_converged_across_declared_grids() -> None:
    config = _TARGET.load_target_three_state_profile_config(CONFIG)
    spacings = config["sampling_and_validation"]["grid_convergence_spacings_um"]
    profile_sets = [
        _TARGET.build_target_three_state_profiles(config, grid_spacing_um=spacing)
        for spacing in spacings
    ]
    reference = profile_sets[-1]
    rms_tolerance = config["sampling_and_validation"][
        "grid_convergence_rms_absolute_tolerance_um"
    ]
    contrast_tolerance = config["sampling_and_validation"][
        "grid_convergence_contrast_absolute_tolerance"
    ]
    for candidate in profile_sets[:-1]:
        for candidate_profile, reference_profile in zip(
            candidate.profiles,
            reference.profiles,
            strict=True,
        ):
            assert candidate_profile.observables.rms_y_m * 1e6 == pytest.approx(
                reference_profile.observables.rms_y_m * 1e6,
                abs=rms_tolerance,
            )
            assert candidate_profile.observables.rms_z_m * 1e6 == pytest.approx(
                reference_profile.observables.rms_z_m * 1e6,
                abs=rms_tolerance,
            )
            assert candidate_profile.observables.modulation_contrast == pytest.approx(
                reference_profile.observables.modulation_contrast,
                abs=contrast_tolerance,
            )


def test_contract_rejects_spacing_mismatch(tmp_path: Path) -> None:
    config = _TARGET.load_target_three_state_profile_config(CONFIG)
    config["profiles"][1]["component_centres_y_um"][-1] = 4.0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="centres disagree"):
        _TARGET.load_target_three_state_profile_config(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config["profiles"][0].__setitem__("atom_number", 49999.0),
            "atom numbers disagree",
        ),
        (
            lambda config: config["profiles"][0].__setitem__("radius_z_um", 1.2),
            "radii disagree",
        ),
        (
            lambda config: config["profiles"][2].__setitem__(
                "component_sigma_z_um", 0.5
            ),
            "transverse widths disagree",
        ),
    ],
)
def test_contract_rejects_broken_common_scale(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    config = _TARGET.load_target_three_state_profile_config(CONFIG)
    mutation(config)
    path = tmp_path / "bad-scale.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _TARGET.load_target_three_state_profile_config(path)
