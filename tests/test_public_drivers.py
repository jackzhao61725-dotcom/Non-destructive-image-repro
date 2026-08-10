"""Contract tests for the code-only reproduction entry points."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from non_destructive_image.multiframe_thermodynamics import (
    closure_validation_residuals,
    measured_scale_non_saturation,
    nominal_critical_atoms,
    oxford_multiframe_contract_from_configs,
)
from scripts import _common, reproduce_forward_model, reproduce_inference
from scripts._common import endpoint_products, load_configs, write_json


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def test_public_configs_build_independent_endpoints() -> None:
    model, reference, reproduction = load_configs(CONFIGS)
    first, second = endpoint_products(model, reference, reproduction)

    assert first.spec.label == "B_parallel_y"
    assert second.spec.label == "B_parallel_z"
    assert first.state is not second.state
    assert first.canonical_operator is not second.canonical_operator
    assert np.all(np.asarray(first.state.radii_m) > 0.0)
    assert np.all(np.asarray(second.state.radii_m) > 0.0)


def test_forward_payload_is_finite_and_tracks_orientation_contrast() -> None:
    invocation = {
        "config_dir": CONFIGS,
        "output": ROOT / "outputs" / "forward_model.json",
        "validate_only": False,
        "overwrite": False,
    }
    payload = reproduce_forward_model.reproduce(
        CONFIGS,
        invocation_arguments=invocation,
    )

    contrast = payload["orientation_contrast_um"]
    assert contrast["delta_sigma_y"] > 0.0
    assert contrast["delta_sigma_z"] > 0.0
    assert payload["reference_probe"]["total_scattered_photons_per_atom"] > 0.0
    assert len(payload["conditional_thermodynamic_sequence"]["states"]) >= 2
    assert payload["upstream_source_commit"]
    assert "source_commit" not in payload
    identity = payload["identity"]
    assert identity["invocation_arguments"] == {
        "config_dir": "configs",
        "output": "outputs/forward_model.json",
        "validate_only": False,
        "overwrite": False,
    }
    assert identity["software"]["python"]
    assert identity["software"]["platform"]
    assert "machine" in identity["software"]
    assert identity["software"]["package"]["version"]
    assert identity["software"]["package"]["version_source"] in {
        "installed_distribution_metadata",
        "importable_package",
        "repository_pyproject",
    }
    package_identity = identity["software"]["package"]
    assert package_identity["importable_version"] == "1.1.0"
    if package_identity["distribution_metadata_version"] is not None:
        assert isinstance(package_identity["versions_match"], bool)


def test_oxford_fig2c_fit_is_transformed_once_for_runtime() -> None:
    model, reference, _reproduction = load_configs(CONFIGS)
    contract = oxford_multiframe_contract_from_configs(reference, model)
    thermodynamics = next(
        condition
        for condition in reference["initial_conditions"]
        if condition["id"] == "oxford_bimodal_300ms"
    )["multiframe_thermodynamics"]
    closure = thermodynamics["closure"]
    fit = closure["fig2c_calibrated_fit"]
    beta = float(closure["combined_calibration"]["beta"])

    assert "measured_scale" not in closure
    assert contract.fig2c_fit_intercept == pytest.approx(fit["intercept"])
    assert contract.fig2c_fit_slope == pytest.approx(fit["slope"])
    assert contract.measured_fit_intercept == pytest.approx(
        beta * float(fit["intercept"]),
        abs=1e-12,
    )
    assert contract.measured_fit_slope == pytest.approx(
        beta ** (3.0 / 5.0) * float(fit["slope"]),
        abs=1e-12,
    )

    fig2c_points = np.asarray(contract.closure_validation_xy, dtype=float)
    check = thermodynamics["closure_check_337_5_ms"]
    measured_x = float(check["x_two_fifths"])
    measured_y = float(check["y_thermal_over_critical"])
    assert measured_x == pytest.approx(
        beta ** (2.0 / 5.0) * fig2c_points[2, 0],
        rel=2e-15,
    )
    assert measured_y == pytest.approx(beta * fig2c_points[2, 1], rel=2e-15)

    initial = contract.initial_state("oxford_300ms_rep_1_central")
    critical = nominal_critical_atoms(initial.temperature_nk, contract)
    runtime_x = (initial.condensate_atoms / critical) ** (2.0 / 5.0)
    assert measured_scale_non_saturation(
        initial.temperature_nk,
        initial.condensate_atoms,
        contract,
    ) == pytest.approx(
        critical
        * (
            contract.measured_fit_intercept
            + contract.measured_fit_slope * runtime_x
        )
    )
    rms, mean_absolute, maximum = closure_validation_residuals(contract)
    assert rms == pytest.approx(0.031, abs=0.001)
    assert mean_absolute == pytest.approx(0.022, abs=0.001)
    assert maximum <= 0.075


def test_public_config_rejects_a_second_stored_beta_transform() -> None:
    model, reference, _reproduction = load_configs(CONFIGS)
    mutated = deepcopy(reference)
    closure = next(
        condition
        for condition in mutated["initial_conditions"]
        if condition["id"] == "oxford_bimodal_300ms"
    )["multiframe_thermodynamics"]["closure"]
    closure["measured_scale"] = {"intercept": 1.1, "slope": 0.31}
    with pytest.raises(ValueError, match="derived exactly once"):
        oxford_multiframe_contract_from_configs(mutated, model)


def test_public_config_rejects_the_legacy_source_fit_key() -> None:
    model, reference, _reproduction = load_configs(CONFIGS)
    mutated = deepcopy(reference)
    closure = next(
        condition
        for condition in mutated["initial_conditions"]
        if condition["id"] == "oxford_bimodal_300ms"
    )["multiframe_thermodynamics"]["closure"]
    closure["source_fit"] = closure.pop("fig2c_calibrated_fit")
    with pytest.raises(ValueError, match="fig2c_calibrated_fit"):
        oxford_multiframe_contract_from_configs(mutated, model)


def test_public_config_rejects_the_wrong_fig2c_coordinates() -> None:
    model, reference, _reproduction = load_configs(CONFIGS)
    mutated = deepcopy(reference)
    closure = next(
        condition
        for condition in mutated["initial_conditions"]
        if condition["id"] == "oxford_bimodal_300ms"
    )["multiframe_thermodynamics"]["closure"]
    closure["fig2c_calibrated_fit"]["coordinate_system"] = (
        "unidentified_coordinates"
    )
    with pytest.raises(ValueError, match="beta-corrected Fig. 2c"):
        oxford_multiframe_contract_from_configs(mutated, model)


def test_inference_seed_tree_repeats_one_raw_block() -> None:
    _model, _reference, reproduction = load_configs(CONFIGS)
    mean = tuple(np.full((3, 4), 10.0 + index) for index in range(3))
    first = reproduce_inference._raw_block(
        reproduction,
        expected=mean,
        read_noise_electrons=0.7,
        fluence_index=16,
        draw_id=0,
        endpoint_index=0,
    )
    second = reproduce_inference._raw_block(
        reproduction,
        expected=mean,
        read_noise_electrons=0.7,
        fluence_index=16,
        draw_id=0,
        endpoint_index=0,
    )
    third = reproduce_inference._raw_block(
        reproduction,
        expected=mean,
        read_noise_electrons=0.7,
        fluence_index=16,
        draw_id=1,
        endpoint_index=0,
    )

    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            first.observed_electrons, second.observed_electrons, strict=True
        )
    )
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(
            first.observed_electrons, third.observed_electrons, strict=True
        )
    )


def test_estimator_contract_has_four_bounded_starts() -> None:
    _model, _reference, reproduction = load_configs(CONFIGS)
    ids, starts, lower, upper = reproduce_inference._parameter_contract(reproduction)

    assert ids == ("neutral", "low_peak", "high_peak", "shifted")
    assert len(starts) == 4
    assert all(np.all(vector >= lower) and np.all(vector <= upper) for vector in starts)


def test_json_publication_is_exclusive_by_default(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    write_json(output, {"generation": 1})

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_json(output, {"generation": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {"generation": 1}
    assert list(tmp_path.iterdir()) == [output]


def test_json_publication_explains_missing_hard_link_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("unsupported")

    monkeypatch.setattr(_common.os, "link", fail_link)
    with pytest.raises(OSError, match="hard-link support"):
        write_json(output, {"generation": 1})

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_json_publication_reports_a_racing_target_as_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result.json"

    def racing_link(*_args: object, **_kwargs: object) -> None:
        raise FileExistsError("target appeared")

    monkeypatch.setattr(_common.os, "link", racing_link)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_json(output, {"generation": 1})

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("module", "extra_arguments"),
    [
        (reproduce_forward_model, []),
        (reproduce_inference, ["--draws", "1"]),
    ],
)
def test_public_driver_refuses_existing_output_before_reproduction(
    module: object,
    extra_arguments: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text('{"kept": true}\n', encoding="utf-8")
    called = False

    def fail_if_called(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(module, "reproduce", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "driver",
            "--config-dir",
            str(CONFIGS),
            "--output",
            str(output),
            *extra_arguments,
        ],
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        module.main()

    assert called is False
    assert json.loads(output.read_text(encoding="utf-8")) == {"kept": True}


@pytest.mark.parametrize(
    ("module", "extra_arguments", "expected_draws"),
    [
        (reproduce_forward_model, [], None),
        (reproduce_inference, ["--draws", "1"], 1),
    ],
)
def test_public_driver_explicit_overwrite_writes_valid_json_and_full_arguments(
    module: object,
    extra_arguments: list[str],
    expected_draws: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("not the replacement", encoding="utf-8")

    def payload(*_args: object, **kwargs: object) -> dict[str, object]:
        return {"identity": {"invocation_arguments": kwargs["invocation_arguments"]}}

    monkeypatch.setattr(module, "reproduce", payload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "driver",
            "--config-dir",
            str(CONFIGS),
            "--output",
            str(output),
            *extra_arguments,
            "--overwrite",
        ],
    )
    module.main()

    written = json.loads(output.read_text(encoding="utf-8"))
    arguments = written["identity"]["invocation_arguments"]
    assert arguments["config_dir"] == "configs"
    assert arguments["output"] == f"<external>/{output.name}"
    assert arguments["validate_only"] is False
    assert arguments["overwrite"] is True
    if expected_draws is None:
        assert "draws" not in arguments
    else:
        assert arguments["draws"] == expected_draws
