from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import non_destructive_image.reconstruction.linked_observable_information as information
from non_destructive_image.reconstruction.contracts import (
    DetectorContract,
    ReconstructionGrid,
)
from non_destructive_image.reconstruction.linked_observable_information import (
    ConditionalObservableEstimate,
    ConfidenceDecomposition,
    LinkedSyntheticBlankReference,
    LinkedObservableInformationBootstrap,
    OneSidedObservableBound,
    ObservableIdentifiabilityRecord,
    ReferenceLightInferenceProvenance,
    analyse_linked_raw_residuals,
    analyse_linked_zero_density_evidence,
    analyse_two_frame_observable_identifiability,
    bilinear_effective_support_mask,
    bootstrap_linked_observable_information,
    classify_information_level,
    fit_linked_zero_density_null,
    refit_linked_observable_bootstrap_draw,
    select_q1_observation_for_reference_sensitivity,
    summarise_two_frame_information,
)
from non_destructive_image.reconstruction.linked_scalar_fit import (
    LinkedRawObservation,
    LinkedScalarFitOptions,
    LinkedScalarFitResult,
    fit_linked_scalar_sequence,
)
from non_destructive_image.reconstruction.object_models import (
    NonnegativeBilinearDensityModel,
)
from non_destructive_image.reconstruction.observable_calibration import OBSERVABLE_NAMES
from non_destructive_image.reconstruction.observables import (
    ObservableIntegrationSupport,
)
from non_destructive_image.reconstruction.regularisation import (
    build_curvature_regularisation,
)
from non_destructive_image.reconstruction.scalar_measurements import (
    DGILinkedRawOperator,
    DGINuisanceValues,
    DGITransferContract,
    PCILinkedRawOperator,
    PCINuisanceValues,
    PCITransferContract,
    ScalarOpticalResponseContract,
)


PRIMARY_PROVENANCE = ReferenceLightInferenceProvenance(
    contract_label="linked_two_exposure_information_v1",
    method="PCI",
    fluence_mw_us=300.0,
    detuning_ghz=1.5,
    selected_eigenmode="perpendicular",
    exposure_indices=(1, 2),
    imaged_pre_pulse_state_indices=(0, 1),
    observation_source="synthetic linked raw counts used only by focused tests",
    initialisation_source="fixed predeclared focused-test coefficients",
    support_source="fixed focused-test object grid",
    regularisation_source="none",
    regularisation_applied=False,
    thermodynamic_prediction_used=False,
    truth_template_used=False,
    reference_template_used=False,
    temporal_coupling_used=False,
    truth_derived_initialisation_used=False,
    target_derived_support_used=False,
    truth_derived_affine_calibration_used=False,
)
DGI_PROVENANCE = replace(PRIMARY_PROVENANCE, method="DGI")


def _small_pci_information_case() -> tuple[
    NonnegativeBilinearDensityModel,
    PCILinkedRawOperator,
    LinkedRawObservation,
    LinkedScalarFitResult,
    ObservableIntegrationSupport,
    LinkedScalarFitOptions,
]:
    axis_m = (np.arange(8, dtype=float) - 4.0) * 0.5e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    grid = ReconstructionGrid.from_arrays(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        pupil=np.ones((8, 8), dtype=complex),
        bin_size=2,
        roi_mask=np.ones((4, 4), dtype=bool),
    )
    model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-1.5, 0.0, 1.5],
        knot_z_um=[-1.5, 0.0, 1.5],
        coefficient_scale_m2=1e14,
    )
    operator = PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(1e5, 0.7),
        response=ScalarOpticalResponseContract(1.9e-16, 3.8e-18),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 1,
            "dark": 1,
        },
        jacobian_batch_size=3,
    )
    q1 = np.asarray([0.05, 0.1, 0.05, 0.15, 0.9, 0.2, 0.04, 0.12, 0.06])
    q2 = 0.9 * q1
    nuisance = PCINuisanceValues(1e5, 0.2)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        [q1, q2],
        nuisance,
    )
    options = LinkedScalarFitOptions(max_nfev=60)
    point = fit_linked_scalar_sequence(
        operator,
        model,
        LinkedRawObservation(prediction.role_names, prediction.expected_electrons),
        initial_density_coefficients=np.stack([q1, q2]),
        density_coefficient_upper=2.0,
        initial_nuisance=nuisance,
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        options=options,
    )
    assert point.diagnostics.success
    observation = LinkedRawObservation(
        point.prediction.role_names,
        point.prediction.expected_electrons,
    )
    support = ObservableIntegrationSupport(
        y_grid_m,
        z_grid_m,
        support_mask=bilinear_effective_support_mask(model),
    )
    return model, operator, observation, point, support, options


def test_primary_provenance_rejects_truth_or_target_derived_inputs() -> None:
    with pytest.raises(ValueError, match="F=300"):
        replace(PRIMARY_PROVENANCE, fluence_mw_us=210.0)
    with pytest.raises(ValueError, match=r"\+1.5 GHz"):
        replace(PRIMARY_PROVENANCE, detuning_ghz=-1.5)
    with pytest.raises(ValueError, match="target-derived"):
        replace(PRIMARY_PROVENANCE, truth_derived_initialisation_used=True)
    with pytest.raises(ValueError, match="target-derived"):
        replace(PRIMARY_PROVENANCE, target_derived_support_used=True)
    with pytest.raises(ValueError, match="target-derived"):
        replace(PRIMARY_PROVENANCE, truth_derived_affine_calibration_used=True)


def _complete_ensemble() -> LinkedObservableInformationBootstrap:
    point = np.asarray(
        [
            [10.0, 0.0, 0.0, 2.0, 1.0],
            [8.0, 0.5, -0.2, 2.5, 1.2],
        ]
    )
    samples = np.asarray(
        [
            [[9.5, -0.1, 0.0, 1.9, 0.95], [7.7, 0.4, -0.3, 2.4, 1.15]],
            [[10.0, 0.0, 0.1, 2.0, 1.00], [8.0, 0.5, -0.2, 2.5, 1.20]],
            [[10.5, 0.1, -0.1, 2.1, 1.05], [8.3, 0.6, -0.1, 2.6, 1.25]],
        ]
    )
    return LinkedObservableInformationBootstrap(
        parameter_names=OBSERVABLE_NAMES,
        requested_draws=3,
        fit_success_mask=np.ones(3, dtype=bool),
        point_estimates=point,
        point_supported_mask=np.ones_like(point, dtype=bool),
        samples=samples,
        supported_mask=np.ones_like(samples, dtype=bool),
        route_provenance=PRIMARY_PROVENANCE,
        assumptions=("test conditional bootstrap",),
    )


def test_two_frame_summary_uses_aligned_draws_for_changes_and_ratios() -> None:
    ensemble = _complete_ensemble()
    summary = summarise_two_frame_information(ensemble, confidence_level=0.8)

    integrated = summary.observables["A"]
    assert integrated.q1.status == "complete"
    assert integrated.q2.status == "complete"
    assert integrated.delta_21.estimate == pytest.approx(-2.0)
    assert integrated.ratio_21 is not None
    assert integrated.ratio_21.estimate == pytest.approx(0.8)
    expected_delta = ensemble.samples[:, 1, 0] - ensemble.samples[:, 0, 0]
    expected_ratio = ensemble.samples[:, 1, 0] / ensemble.samples[:, 0, 0]
    np.testing.assert_allclose(
        [integrated.delta_21.lower, integrated.delta_21.upper],
        np.quantile(expected_delta, [0.1, 0.9]),
    )
    np.testing.assert_allclose(
        [integrated.ratio_21.lower, integrated.ratio_21.upper],
        np.quantile(expected_ratio, [0.1, 0.9]),
    )
    assert summary.observables["y_c_um"].ratio_21 is None
    assert summary.derived_aspect_ratio is not None
    assert summary.derived_aspect_ratio.q1.estimate == pytest.approx(2.0)
    assert summary.derived_aspect_ratio.q2.estimate == pytest.approx(2.5 / 1.2)
    mutable_assumptions = list(summary.assumptions)
    frozen_summary = replace(summary, assumptions=mutable_assumptions)
    mutable_assumptions.clear()
    assert frozen_summary.assumptions == summary.assumptions


def test_missing_width_does_not_delete_integrated_or_centroid_information() -> None:
    point = np.asarray(
        [[10.0, 0.0, 0.0, 2.0, 1.0], [8.0, 0.5, -0.2, 2.5, 1.2]]
    )
    samples = np.full((3, 2, len(OBSERVABLE_NAMES)), np.nan)
    samples[0] = point
    samples[2, :, :3] = point[:, :3] * np.asarray([0.98, 1.0, 1.0])
    fit_success = np.asarray([True, False, True])
    ensemble = LinkedObservableInformationBootstrap(
        parameter_names=OBSERVABLE_NAMES,
        requested_draws=3,
        fit_success_mask=fit_success,
        point_estimates=point,
        point_supported_mask=np.ones_like(point, dtype=bool),
        samples=samples,
        supported_mask=np.isfinite(samples),
        route_provenance=PRIMARY_PROVENANCE,
        assumptions=("test aligned missingness",),
    )

    summary = summarise_two_frame_information(ensemble, confidence_level=0.8)
    assert summary.observables["A"].delta_21.status == "partial"
    assert summary.observables["A"].delta_21.supported_draws == 2
    assert summary.observables["A"].delta_21.estimate == pytest.approx(-2.0)
    assert summary.observables["A"].delta_21.lower is None
    assert summary.observables["A"].delta_21.upper is None
    with pytest.raises(ValueError, match="partial estimates cannot report"):
        replace(
            summary.observables["A"].delta_21,
            lower=-2.1,
            upper=-1.9,
        )
    assert summary.observables["sigma_z_um"].delta_21.status == "partial"
    assert summary.observables["sigma_z_um"].delta_21.supported_draws == 1
    assert summary.observables["sigma_z_um"].delta_21.lower is None
    assert summary.derived_aspect_ratio is not None
    assert summary.derived_aspect_ratio.delta_21.status == "partial"


def test_paired_change_preserves_shared_draw_cancellation() -> None:
    point = np.asarray(
        [[10.0, 0.0, 0.0, 2.0, 1.0], [8.0, 0.5, -0.2, 2.5, 1.2]]
    )
    common_offsets = np.asarray([-1.0, -0.5, 0.5, 1.0])
    samples = np.repeat(point[None, :, :], common_offsets.size, axis=0)
    samples[:, 0, 0] += common_offsets
    samples[:, 1, 0] += common_offsets
    ensemble = LinkedObservableInformationBootstrap(
        parameter_names=OBSERVABLE_NAMES,
        requested_draws=common_offsets.size,
        fit_success_mask=np.ones(common_offsets.size, dtype=bool),
        point_estimates=point,
        point_supported_mask=np.ones_like(point, dtype=bool),
        samples=samples,
        supported_mask=np.ones_like(samples, dtype=bool),
        route_provenance=PRIMARY_PROVENANCE,
        assumptions=("shared additive response perturbation",),
    )

    integrated = summarise_two_frame_information(
        ensemble,
        confidence_level=0.8,
    ).observables["A"]
    assert integrated.q1.lower < integrated.q1.upper
    assert integrated.q2.lower < integrated.q2.upper
    assert integrated.delta_21.lower == pytest.approx(-2.0)
    assert integrated.delta_21.upper == pytest.approx(-2.0)


def test_nonpositive_q1_denominator_withholds_ratio_without_hiding_integral() -> None:
    point = np.asarray(
        [[0.0, np.nan, np.nan, np.nan, np.nan], [1.0, 0.0, 0.0, 1.0, 1.0]]
    )
    samples = np.repeat(point[None, :, :], 2, axis=0)
    ensemble = LinkedObservableInformationBootstrap(
        parameter_names=OBSERVABLE_NAMES,
        requested_draws=2,
        fit_success_mask=np.ones(2, dtype=bool),
        point_estimates=point,
        point_supported_mask=np.isfinite(point),
        samples=samples,
        supported_mask=np.isfinite(samples),
        route_provenance=PRIMARY_PROVENANCE,
        assumptions=("blank q1 denominator",),
    )

    integrated = summarise_two_frame_information(
        ensemble,
        confidence_level=0.8,
    ).observables["A"]
    assert integrated.q1.status == "complete"
    assert integrated.q1.estimate == pytest.approx(0.0)
    assert integrated.ratio_21 is not None
    assert integrated.ratio_21.status == "unresolved"
    assert integrated.ratio_21.estimate is None


def test_failed_refit_rows_cannot_contain_supported_values() -> None:
    point = np.ones((2, len(OBSERVABLE_NAMES)))
    samples = np.ones((2, 2, len(OBSERVABLE_NAMES)))
    with pytest.raises(ValueError, match="failed refits"):
        LinkedObservableInformationBootstrap(
            parameter_names=OBSERVABLE_NAMES,
            requested_draws=2,
            fit_success_mask=np.asarray([True, False]),
            point_estimates=point,
            point_supported_mask=np.ones_like(point, dtype=bool),
            samples=samples,
            supported_mask=np.ones_like(samples, dtype=bool),
            route_provenance=PRIMARY_PROVENANCE,
            assumptions=("invalid failed row",),
        )


def test_public_evidence_records_freeze_text_and_reject_fractional_draw_counts() -> None:
    bootstrap_assumptions = ["mutable bootstrap assumption"]
    frozen_bootstrap = replace(
        _complete_ensemble(),
        assumptions=bootstrap_assumptions,
    )
    bootstrap_assumptions.clear()
    assert frozen_bootstrap.assumptions == ("mutable bootstrap assumption",)

    reasons = ["point is unsupported"]
    estimate = ConditionalObservableEstimate(
        observable_name="A",
        quantity="delta_21",
        unit="response_integral",
        estimate=None,
        status="unresolved",
        confidence_level=0.8,
        requested_draws=2,
        successful_fit_draws=2,
        supported_draws=2,
        lower=None,
        upper=None,
        null_value=0.0,
        excludes_null=None,
        reasons=reasons,
    )
    reasons.clear()
    assert estimate.reasons == ("point is unsupported",)

    bound_assumptions = ["predeclared one-sided construction"]
    bound = OneSidedObservableBound(
        observable_name="A",
        quantity="q1",
        unit="response_integral",
        bound_value=1.0,
        direction="lower",
        confidence_level=0.8,
        construction="profile likelihood",
        predeclared_rule_id="immutability_test",
        assumptions=bound_assumptions,
    )
    bound_assumptions.clear()
    assert bound.assumptions == ("predeclared one-sided construction",)

    confidence_reasons = ["detector rule pending"]
    confidence = ConfidenceDecomposition(
        fit_and_data="adequate",
        detector_statistical="not_assessed",
        identifiability="adequate",
        calibration="adequate",
        forward_model="adequate",
        basis_model="stable",
        support="stable",
        reference="not_applicable",
        regularisation="stable",
        repeatability="adequate",
        relative_change="not_assessed",
        reasons=confidence_reasons,
    )
    confidence_reasons.clear()
    assert confidence.reasons == ("detector rule pending",)

    identifiability_reasons = ["gradient unavailable"]
    identifiability_record = ObservableIdentifiabilityRecord(
        observable_name="A",
        quantity="q1",
        estimate=None,
        scaled_gradient_norm=None,
        data_null_space_fraction=None,
        active_bound_gradient_fraction=None,
        identified_subspace_standard_uncertainty=None,
        supported=False,
        reasons=identifiability_reasons,
    )
    identifiability_reasons.clear()
    assert identifiability_record.reasons == ("gradient unavailable",)

    with pytest.raises(TypeError, match="requested_draws must be an integer"):
        replace(_complete_ensemble(), requested_draws=1.9)
    with pytest.raises(TypeError, match="requested_draws must be an integer"):
        replace(_complete_ensemble(), requested_draws=True)
    with pytest.raises(TypeError, match="successful_fit_draws must be an integer"):
        replace(estimate, successful_fit_draws=1.5)


def test_confidence_classification_keeps_components_separate() -> None:
    summary = summarise_two_frame_information(
        _complete_ensemble(),
        confidence_level=0.8,
    )
    estimate = summary.observables["A"].delta_21
    ratio_estimate = summary.observables["A"].ratio_21
    assert ratio_estimate is not None
    assert ratio_estimate.unit == "1"
    width_ratio = summary.observables["sigma_y_um"].ratio_21
    assert width_ratio is not None
    assert width_ratio.unit == "1"
    with pytest.raises(ValueError, match="positive observables"):
        replace(ratio_estimate, observable_name="y_c_um")
    adequate = ConfidenceDecomposition(
        fit_and_data="adequate",
        detector_statistical="adequate",
        identifiability="adequate",
        calibration="adequate",
        forward_model="adequate",
        basis_model="stable",
        support="stable",
        reference="not_applicable",
        regularisation="stable",
        repeatability="adequate",
        relative_change="adequate",
        reasons=(),
    )
    assert classify_information_level(estimate, adequate) == "quantitatively_resolved"
    explicit_bound = OneSidedObservableBound(
        observable_name="A",
        quantity="ratio_21",
        unit="1",
        bound_value=0.7,
        direction="lower",
        confidence_level=0.8,
        construction="predeclared one-sided profile-likelihood construction",
        predeclared_rule_id="focused_test_bound_rule",
        assumptions=("focused-test construction only",),
    )
    assert classify_information_level(explicit_bound, adequate) == "bounded"
    with pytest.raises(ValueError, match="always applicable"):
        replace(adequate, fit_and_data="not_applicable")
    with pytest.raises(ValueError, match="always applicable"):
        replace(adequate, support="not_applicable")

    q1_estimate = summarise_two_frame_information(
        _complete_ensemble(),
        confidence_level=0.8,
    ).observables["A"].q1
    with pytest.raises(ValueError, match="single-frame"):
        classify_information_level(q1_estimate, adequate)
    q1_adequate = replace(adequate, relative_change="not_applicable")
    assert (
        classify_information_level(q1_estimate, q1_adequate)
        == "quantitatively_resolved"
    )
    with pytest.raises(ValueError, match="delta_21 and ratio_21"):
        classify_information_level(
            estimate,
            replace(
                adequate,
                relative_change="not_applicable",
            ),
        )

    limited = ConfidenceDecomposition(
        fit_and_data="adequate",
        detector_statistical="limited",
        identifiability="adequate",
        calibration="not_assessed",
        forward_model="not_assessed",
        basis_model="not_assessed",
        support="not_assessed",
        reference="not_applicable",
        regularisation="not_assessed",
        repeatability="limited",
        relative_change="limited",
        reasons=("formal detector and calibration rules are not frozen",),
    )
    assert (
        classify_information_level(estimate, limited)
        == "informative_but_inconclusive"
    )
    sensitive = ConfidenceDecomposition(
        fit_and_data="adequate",
        detector_statistical="adequate",
        identifiability="adequate",
        calibration="adequate",
        forward_model="adequate",
        basis_model="stable",
        support="stable",
        reference="sensitive",
        regularisation="stable",
        repeatability="adequate",
        relative_change="adequate",
        reasons=("reference-informed sensitivity moves the estimate",),
    )
    assert classify_information_level(estimate, sensitive) == "prior_sensitive"
    invalid_support = replace(
        adequate,
        support="failed",
        reasons=("observable support is outside the effective density basis",),
    )
    assert classify_information_level(estimate, invalid_support) == "unresolved"
    unidentified = ConfidenceDecomposition(
        fit_and_data="adequate",
        detector_statistical="adequate",
        identifiability="failed",
        calibration="adequate",
        forward_model="adequate",
        basis_model="stable",
        support="stable",
        reference="not_applicable",
        regularisation="stable",
        repeatability="not_assessed",
        relative_change="not_assessed",
        reasons=("observable gradient lies in the local data null space",),
    )
    assert classify_information_level(estimate, unidentified) == "unresolved"
    failed = ConfidenceDecomposition(
        fit_and_data="failed",
        detector_statistical="not_assessed",
        identifiability="not_assessed",
        calibration="not_assessed",
        forward_model="not_assessed",
        basis_model="not_assessed",
        support="not_assessed",
        reference="not_applicable",
        regularisation="not_assessed",
        repeatability="not_assessed",
        relative_change="not_assessed",
        reasons=("linked raw fit failed",),
    )
    assert classify_information_level(estimate, failed) == "fit_or_data_failure"

    with pytest.raises(ValueError, match="rule id"):
        OneSidedObservableBound(
            observable_name="A",
            quantity="ratio_21",
            unit="1",
            bound_value=0.7,
            direction="lower",
            confidence_level=0.8,
            construction="profile likelihood",
            predeclared_rule_id="",
            assumptions=("focused-test construction only",),
        )
    with pytest.raises(ValueError, match="non-negative domain"):
        OneSidedObservableBound(
            observable_name="A",
            quantity="ratio_21",
            unit="1",
            bound_value=-1.0,
            direction="upper",
            confidence_level=0.8,
            construction="profile likelihood",
            predeclared_rule_id="invalid_negative_ratio_bound",
            assumptions=("focused-test construction only",),
        )


def test_unresolved_estimate_cannot_be_upgraded_by_adequate_component_labels() -> None:
    unresolved = ConditionalObservableEstimate(
        observable_name="A",
        quantity="q1",
        unit="response_integral",
        estimate=1.0,
        status="unresolved",
        confidence_level=0.8,
        requested_draws=3,
        successful_fit_draws=3,
        supported_draws=0,
        lower=None,
        upper=None,
        null_value=None,
        excludes_null=None,
        reasons=("no_supported_bootstrap_draws",),
    )
    adequate = ConfidenceDecomposition(
        fit_and_data="adequate",
        detector_statistical="adequate",
        identifiability="adequate",
        calibration="adequate",
        forward_model="adequate",
        basis_model="stable",
        support="stable",
        reference="not_applicable",
        regularisation="stable",
        repeatability="adequate",
        relative_change="not_applicable",
        reasons=(),
    )

    assert classify_information_level(unresolved, adequate) == "unresolved"
    with pytest.raises(ValueError, match="status"):
        ConditionalObservableEstimate(
            observable_name="A",
            quantity="q1",
            unit="response_integral",
            estimate=1.0,
            status="garbage",  # type: ignore[arg-type]
            confidence_level=0.8,
            requested_draws=3,
            successful_fit_draws=3,
            supported_draws=0,
            lower=None,
            upper=None,
            null_value=None,
            excludes_null=None,
            reasons=("invalid status",),
        )


def test_point_unsupported_with_supported_refits_is_retained_without_interval() -> None:
    point = np.asarray(
        [[10.0, 0.0, 0.0, np.nan, np.nan], [8.0, 0.5, -0.2, 2.5, 1.2]]
    )
    samples = np.repeat(
        np.asarray(
            [[[10.0, 0.0, 0.0, 2.0, 1.0], [8.0, 0.5, -0.2, 2.5, 1.2]]]
        ),
        3,
        axis=0,
    )
    ensemble = LinkedObservableInformationBootstrap(
        parameter_names=OBSERVABLE_NAMES,
        requested_draws=3,
        fit_success_mask=np.ones(3, dtype=bool),
        point_estimates=point,
        point_supported_mask=np.isfinite(point),
        samples=samples,
        supported_mask=np.isfinite(samples),
        route_provenance=PRIMARY_PROVENANCE,
        assumptions=("point width unsupported but refits finite",),
    )

    summary = summarise_two_frame_information(ensemble, confidence_level=0.8)
    q1_width = summary.observables["sigma_y_um"].q1
    assert q1_width.status == "unresolved"
    assert q1_width.estimate is None
    assert q1_width.supported_draws == 3
    assert q1_width.lower is None
    assert "point_estimate_not_numerically_supported" in q1_width.reasons


def test_reference_light_bootstrap_is_seeded_and_two_frame_aligned() -> None:
    axis_m = (np.arange(12, dtype=float) - 6.0) * 0.5e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    grid = ReconstructionGrid.from_arrays(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        pupil=np.ones((12, 12), dtype=complex),
        bin_size=2,
        roi_mask=np.ones((6, 6), dtype=bool),
    )
    model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-2.0, 0.0, 2.0],
        knot_z_um=[-2.0, 0.0, 2.0],
        coefficient_scale_m2=1e14,
    )
    operator = PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(1e5, 0.7),
        response=ScalarOpticalResponseContract(1.9e-16, 3.8e-18),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 1,
            "dark": 1,
        },
        jacobian_batch_size=3,
    )
    first = np.asarray([0.05, 0.1, 0.05, 0.15, 0.9, 0.2, 0.04, 0.12, 0.06])
    second = 0.9 * first
    nuisance = PCINuisanceValues(1e5, 0.2)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        [first, second],
        nuisance,
    )
    linked_observation = LinkedRawObservation(
        prediction.role_names,
        prediction.expected_electrons,
    )
    point = fit_linked_scalar_sequence(
        operator,
        model,
        linked_observation,
        initial_density_coefficients=0.95 * np.stack([first, second]),
        density_coefficient_upper=2.0,
        initial_nuisance=PCINuisanceValues(0.98e5, 0.1),
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        options=LinkedScalarFitOptions(max_nfev=80),
    )
    q1_observation = select_q1_observation_for_reference_sensitivity(
        operator,
        model,
        linked_observation,
        point,
        reference_light_provenance=PRIMARY_PROVENANCE,
        regularisation=None,
    )
    assert q1_observation.role_names == (
        "atom_000",
        "bright_reference",
        "dark",
    )
    assert "atom_001" not in q1_observation.role_names
    for name, values in zip(
        q1_observation.role_names,
        q1_observation.observed_electrons,
        strict=True,
    ):
        source_index = prediction.role_names.index(name)
        np.testing.assert_array_equal(
            values,
            prediction.expected_electrons[source_index],
        )
    altered_arrays = [np.array(values, copy=True) for values in prediction.expected_electrons]
    altered_arrays[0][0, 0] += 1.0
    with pytest.raises(ValueError, match="raw sequence that produced the fit"):
        select_q1_observation_for_reference_sensitivity(
            operator,
            model,
            LinkedRawObservation(prediction.role_names, tuple(altered_arrays)),
            point,
            reference_light_provenance=PRIMARY_PROVENANCE,
            regularisation=None,
        )
    support = ObservableIntegrationSupport(
        y_grid_m,
        z_grid_m,
        support_mask=bilinear_effective_support_mask(model),
    )
    kwargs = dict(
        reference_light_provenance=PRIMARY_PROVENANCE,
        integration_support=support,
        density_coefficient_upper=2.0,
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        options=LinkedScalarFitOptions(max_nfev=80),
        draws=2,
    )
    fractional_draw_kwargs = dict(kwargs)
    fractional_draw_kwargs["draws"] = 1.9
    with pytest.raises(TypeError, match="draws must be an integer"):
        bootstrap_linked_observable_information(
            operator,
            model,
            point,
            rng=np.random.default_rng(41),
            **fractional_draw_kwargs,
        )
    left = bootstrap_linked_observable_information(
        operator,
        model,
        point,
        rng=np.random.default_rng(41),
        **kwargs,
    )
    right = bootstrap_linked_observable_information(
        operator,
        model,
        point,
        rng=np.random.default_rng(41),
        **kwargs,
    )

    assert left.samples.shape == (2, 2, len(OBSERVABLE_NAMES))
    np.testing.assert_array_equal(left.fit_success_mask, right.fit_success_mask)
    np.testing.assert_allclose(left.samples, right.samples, equal_nan=True)
    summary = summarise_two_frame_information(left, confidence_level=0.8)
    assert summary.observables["A"].ratio_21 is not None


def test_bootstrap_rejects_coordinate_drift_in_observable_support() -> None:
    axis_m = (np.arange(8, dtype=float) - 4.0) * 0.5e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    grid = ReconstructionGrid.from_arrays(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        pupil=np.ones((8, 8), dtype=complex),
        bin_size=2,
        roi_mask=np.ones((4, 4), dtype=bool),
    )
    model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-1.5, 0.0, 1.5],
        knot_z_um=[-1.5, 0.0, 1.5],
        coefficient_scale_m2=1e14,
    )
    operator = PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(1e5, 0.7),
        response=ScalarOpticalResponseContract(1.9e-16, 3.8e-18),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={"atom": 1, "bright_reference": 1, "dark": 1},
    )
    coefficients = np.full((2, model.parameter_count), 0.2)
    nuisance = PCINuisanceValues(1e5, 0.2)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        list(coefficients),
        nuisance,
    )
    point = fit_linked_scalar_sequence(
        operator,
        model,
        LinkedRawObservation(prediction.role_names, prediction.expected_electrons),
        initial_density_coefficients=coefficients,
        density_coefficient_upper=2.0,
        initial_nuisance=nuisance,
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        options=LinkedScalarFitOptions(max_nfev=40),
    )
    shifted_support = ObservableIntegrationSupport(y_grid_m + 1e-12, z_grid_m)
    with pytest.raises(ValueError, match="coordinates"):
        bootstrap_linked_observable_information(
            operator,
            model,
            point,
            reference_light_provenance=PRIMARY_PROVENANCE,
            integration_support=shifted_support,
            density_coefficient_upper=2.0,
            nuisance_lower=[0.8e5, 0.0],
            nuisance_upper=[1.2e5, 2.0],
            regularisation=None,
            options=LinkedScalarFitOptions(max_nfev=40),
            draws=1,
            rng=np.random.default_rng(4),
        )

    narrow_model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-0.5, 0.5],
        knot_z_um=[-0.5, 0.5],
        coefficient_scale_m2=1e14,
        support_mask=np.ones(y_grid_m.shape, dtype=bool),
    )
    effective_support = bilinear_effective_support_mask(narrow_model)
    assert not effective_support.flags.writeable
    with pytest.raises(ValueError, match="fixed to zero"):
        information._validate_observable_support_contract(
            operator,
            narrow_model,
            ObservableIntegrationSupport(y_grid_m, z_grid_m),
        )
    information._validate_observable_support_contract(
        operator,
        narrow_model,
        ObservableIntegrationSupport(
            y_grid_m,
            z_grid_m,
            support_mask=effective_support,
        ),
    )

    with pytest.raises(TypeError, match="NonnegativeBilinearDensityModel"):
        bootstrap_linked_observable_information(
            operator,
            object(),
            point,
            reference_light_provenance=PRIMARY_PROVENANCE,
            integration_support=ObservableIntegrationSupport(y_grid_m, z_grid_m),
            density_coefficient_upper=2.0,
            nuisance_lower=[0.8e5, 0.0],
            nuisance_upper=[1.2e5, 2.0],
            regularisation=None,
            options=LinkedScalarFitOptions(max_nfev=40),
            draws=1,
            rng=np.random.default_rng(4),
        )


def test_scaled_jacobian_rank_uses_direct_svd_for_weak_supported_direction() -> None:
    jacobian = np.asarray([[1.0, 1.0], [0.0, 1e-8]])

    singular, vectors, rank, condition = information._scaled_jacobian_subspaces(
        jacobian,
        parameter_count=2,
        relative_rank_tolerance=1e-10,
    )

    assert singular.shape == (2,)
    assert vectors.shape == (2, 2)
    assert rank == 2
    assert np.isfinite(condition)


def test_bilinear_effective_support_matches_actual_representable_density() -> None:
    axis_m = np.asarray([-1.0, 0.0, 1.0], dtype=float) * 1e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-0.5, 0.5],
        knot_z_um=[-0.5, 0.5],
        coefficient_scale_m2=1e14,
    )

    expected = model.column_density(np.ones(model.parameter_count)) > 0.0
    effective_support = bilinear_effective_support_mask(model)
    np.testing.assert_array_equal(effective_support, expected)
    np.testing.assert_array_equal(
        effective_support,
        np.asarray(
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ]
        ),
    )
    assert not effective_support.flags.writeable


@pytest.mark.parametrize(
    ("represented_density", "message"),
    (
        (np.ones((2, 2)), "wrong grid shape"),
        (np.full((3, 3), np.nan), "must be finite"),
        (-np.ones((3, 3)), "must be non-negative"),
        (np.zeros((3, 3)), "no representable support"),
    ),
)
def test_bilinear_effective_support_rejects_invalid_actual_density(
    monkeypatch: pytest.MonkeyPatch,
    represented_density: np.ndarray,
    message: str,
) -> None:
    axis_m = np.asarray([-1.0, 0.0, 1.0], dtype=float) * 1e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-1.0, 1.0],
        knot_z_um=[-1.0, 1.0],
        coefficient_scale_m2=1e14,
    )
    monkeypatch.setattr(
        NonnegativeBilinearDensityModel,
        "column_density",
        lambda _model, _parameters: represented_density,
    )

    with pytest.raises(RuntimeError, match=message):
        bilinear_effective_support_mask(model)


def test_bilinear_effective_support_rejects_density_outside_declared_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axis_m = np.asarray([-1.0, 0.0, 1.0], dtype=float) * 1e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    declared_support = np.ones((3, 3), dtype=bool)
    declared_support[0, 0] = False
    model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-1.0, 1.0],
        knot_z_um=[-1.0, 1.0],
        coefficient_scale_m2=1e14,
        support_mask=declared_support,
    )
    monkeypatch.setattr(
        NonnegativeBilinearDensityModel,
        "column_density",
        lambda _model, _parameters: np.ones((3, 3)),
    )

    with pytest.raises(RuntimeError, match="exceeds declared support"):
        bilinear_effective_support_mask(model)


def test_tall_scaled_jacobian_factorises_only_small_r_matrix(monkeypatch) -> None:
    rng = np.random.default_rng(7)
    jacobian = rng.normal(size=(5000, 4))
    original_svd = np.linalg.svd
    svd_input_shapes: list[tuple[int, ...]] = []

    def recording_svd(values, *args, **kwargs):
        svd_input_shapes.append(np.asarray(values).shape)
        return original_svd(values, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", recording_svd)
    singular, vectors, rank, _ = information._scaled_jacobian_subspaces(
        jacobian,
        parameter_count=4,
        relative_rank_tolerance=1e-10,
    )

    assert svd_input_shapes == [(4, 4)]
    assert singular.shape == (4,)
    assert vectors.shape == (4, 4)
    assert rank == 4


def test_wide_scaled_jacobian_retains_complete_parameter_null_basis() -> None:
    jacobian = np.asarray([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

    singular, vectors, rank, condition = information._scaled_jacobian_subspaces(
        jacobian,
        parameter_count=4,
        relative_rank_tolerance=1e-10,
    )

    assert singular.shape == (4,)
    assert vectors.shape == (4, 4)
    assert rank == 2
    assert condition == float("inf")
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(4), atol=1e-12)


def test_axis_observable_parameter_gradients_match_finite_differences() -> None:
    axis_m = (np.arange(10, dtype=float) - 5.0) * 0.5e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-2.0, 0.0, 2.0],
        knot_z_um=[-2.0, 0.0, 2.0],
        coefficient_scale_m2=1e14,
    )
    support = ObservableIntegrationSupport(
        y_grid_m,
        z_grid_m,
        support_mask=bilinear_effective_support_mask(model),
    )
    parameters = np.asarray([0.2, 0.3, 0.25, 0.4, 0.9, 0.35, 0.15, 0.25, 0.2])
    values, supported, gradients = information._axis_observable_parameter_gradients(
        model,
        parameters,
        support,
        jacobian_batch_size=3,
    )
    assert np.all(supported)
    assert np.all(np.isfinite(values))
    step = 1e-6
    numerical = np.empty_like(gradients)
    for index in range(model.parameter_count):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        plus_values, _ = information._observable_vector_with_support(
            model.column_density(plus),
            support,
        )
        minus_values, _ = information._observable_vector_with_support(
            model.column_density(minus),
            support,
        )
        numerical[:, index] = (plus_values - minus_values) / (2.0 * step)
    np.testing.assert_allclose(gradients, numerical, rtol=2e-5, atol=1e-8)


def test_linked_raw_residual_and_local_identifiability_keep_role_boundaries() -> None:
    axis_m = (np.arange(10, dtype=float) - 5.0) * 0.5e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    grid = ReconstructionGrid.from_arrays(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        pupil=np.ones((10, 10), dtype=complex),
        bin_size=2,
        roi_mask=np.ones((5, 5), dtype=bool),
    )
    model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-2.0, 0.0, 2.0],
        knot_z_um=[-2.0, 0.0, 2.0],
        coefficient_scale_m2=1e14,
    )
    operator = PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(1e5, 0.7),
        response=ScalarOpticalResponseContract(1.9e-16, 3.8e-18),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={"atom": 1, "bright_reference": 1, "dark": 1},
        jacobian_batch_size=3,
    )
    q1 = np.asarray([0.1, 0.2, 0.1, 0.3, 0.8, 0.25, 0.08, 0.18, 0.09])
    q2 = np.asarray([0.09, 0.18, 0.09, 0.27, 0.72, 0.23, 0.07, 0.16, 0.08])
    nuisance = PCINuisanceValues(1e5, 0.2)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        [q1, q2],
        nuisance,
    )
    fit = fit_linked_scalar_sequence(
        operator,
        model,
        LinkedRawObservation(prediction.role_names, prediction.expected_electrons),
        initial_density_coefficients=np.stack([q1, q2]),
        density_coefficient_upper=2.0,
        initial_nuisance=nuisance,
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        options=LinkedScalarFitOptions(max_nfev=60),
    )
    exact_fitted_observation = LinkedRawObservation(
        fit.prediction.role_names,
        fit.prediction.expected_electrons,
    )
    residual_summary = analyse_linked_raw_residuals(
        operator,
        model,
        exact_fitted_observation,
        fit,
        reference_light_provenance=PRIMARY_PROVENANCE,
        regularisation=None,
    )
    residuals = residual_summary.roles
    assert residual_summary.route_provenance is PRIMARY_PROVENANCE
    mutable_roles = list(residual_summary.roles)
    mutable_raw_assumptions = list(residual_summary.assumptions)
    frozen_residual_summary = replace(
        residual_summary,
        roles=mutable_roles,
        assumptions=mutable_raw_assumptions,
    )
    mutable_roles.clear()
    mutable_raw_assumptions.clear()
    assert len(frozen_residual_summary.roles) == 4
    assert frozen_residual_summary.assumptions == residual_summary.assumptions
    assert [item.frame_index for item in residuals] == [0, 1, None, None]
    assert [item.shared_role for item in residuals] == [False, False, True, True]
    assert all(item.roi_rms == pytest.approx(0.0) for item in residuals)
    altered_arrays = [
        np.array(values, copy=True)
        for values in exact_fitted_observation.observed_electrons
    ]
    altered_arrays[0][0, 0] += 1.0
    with pytest.raises(ValueError, match="raw sequence that produced the fit"):
        analyse_linked_raw_residuals(
            operator,
            model,
            LinkedRawObservation(fit.prediction.role_names, tuple(altered_arrays)),
            fit,
            reference_light_provenance=PRIMARY_PROVENANCE,
            regularisation=None,
        )

    support = ObservableIntegrationSupport(
        y_grid_m,
        z_grid_m,
        support_mask=bilinear_effective_support_mask(model),
    )
    identifiability = analyse_two_frame_observable_identifiability(
        operator,
        model,
        fit,
        reference_light_provenance=PRIMARY_PROVENANCE,
        integration_support=support,
        density_parameter_lower=0.0,
        density_coefficient_upper=2.0,
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        relative_rank_tolerance=1e-10,
        relative_active_bound_tolerance=1e-6,
    )
    assert identifiability.parameter_count == 2 * model.parameter_count + 2
    assert 0 <= identifiability.data_rank <= identifiability.parameter_count
    assert len(identifiability.records) == 22
    assert {item.quantity for item in identifiability.records} == {
        "q1",
        "q2",
        "delta_21",
        "ratio_21",
    }
    for item in identifiability.records:
        if item.supported:
            assert item.data_null_space_fraction is not None
            assert 0.0 <= item.data_null_space_fraction <= 1.0
            assert item.active_bound_gradient_fraction is not None
            assert 0.0 <= item.active_bound_gradient_fraction <= 1.0
    aspect_records = [
        item
        for item in identifiability.records
        if item.observable_name == "aspect_ratio_y_over_z"
    ]
    assert [item.quantity for item in aspect_records] == [
        "q1",
        "q2",
        "delta_21",
        "ratio_21",
    ]
    assert all(item.supported for item in aspect_records)
    mutable_records = list(identifiability.records)
    mutable_ident_assumptions = list(identifiability.assumptions)
    frozen_identifiability = replace(
        identifiability,
        records=mutable_records,
        assumptions=mutable_ident_assumptions,
    )
    mutable_records.clear()
    mutable_ident_assumptions.clear()
    assert len(frozen_identifiability.records) == 22
    assert frozen_identifiability.assumptions == identifiability.assumptions

    raw_observation = LinkedRawObservation(
        prediction.role_names,
        prediction.expected_electrons,
    )
    regularisation = build_curvature_regularisation(
        model.knot_y_um,
        model.knot_z_um,
        density_scale_m2=1e14,
        weight_um2=10.0,
    )
    regularised_fit = fit_linked_scalar_sequence(
        operator,
        model,
        raw_observation,
        initial_density_coefficients=np.stack([q1, q2]),
        density_coefficient_upper=2.0,
        initial_nuisance=nuisance,
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=regularisation,
        options=LinkedScalarFitOptions(max_nfev=80),
    )
    regularised_provenance = replace(
        PRIMARY_PROVENANCE,
        regularisation_source="focused-test curvature weight 10 um2",
        regularisation_applied=True,
    )
    with pytest.raises(ValueError, match="regularisation objective"):
        analyse_linked_raw_residuals(
            operator,
            model,
            raw_observation,
            regularised_fit,
            reference_light_provenance=PRIMARY_PROVENANCE,
            regularisation=None,
        )
    wrong_regularisation = build_curvature_regularisation(
        model.knot_y_um,
        model.knot_z_um,
        density_scale_m2=1e14,
        weight_um2=20.0,
    )
    with pytest.raises(ValueError, match="regularisation objective"):
        analyse_linked_raw_residuals(
            operator,
            model,
            raw_observation,
            regularised_fit,
            reference_light_provenance=regularised_provenance,
            regularisation=wrong_regularisation,
        )
    regularised_summary = analyse_linked_raw_residuals(
        operator,
        model,
        raw_observation,
        regularised_fit,
        reference_light_provenance=regularised_provenance,
        regularisation=regularisation,
    )
    assert regularised_summary.route_provenance is regularised_provenance


def test_dgi_raw_residuals_keep_two_atom_roles_and_four_shared_roles() -> None:
    axis_m = (np.arange(8, dtype=float) - 4.0) * 0.5e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    grid = ReconstructionGrid.from_arrays(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        pupil=np.ones((8, 8), dtype=complex),
        bin_size=2,
        roi_mask=np.ones((4, 4), dtype=bool),
    )
    model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-1.5, 0.0, 1.5],
        knot_z_um=[-1.5, 0.0, 1.5],
        coefficient_scale_m2=1e14,
    )
    operator = DGILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(140.0, 0.7),
        response=ScalarOpticalResponseContract(1.9e-16, 3.8e-18),
        transfer=DGITransferContract(2.0),
        independent_exposures_by_role={
            "atom_stop": 1,
            "leakage_stop": 1,
            "stop_dark": 1,
            "open_reference": 1,
            "open_dark": 1,
        },
        jacobian_batch_size=3,
    )
    q1 = np.asarray([0.08, 0.15, 0.05, 0.25, 0.9, 0.2, 0.04, 0.12, 0.06])
    q2 = 0.9 * q1
    nuisance = DGINuisanceValues(140.0, 0.4, 0.2, 0.9)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        [q1, q2],
        nuisance,
    )
    observation = LinkedRawObservation(
        prediction.role_names,
        prediction.expected_electrons,
    )
    fit = fit_linked_scalar_sequence(
        operator,
        model,
        observation,
        initial_density_coefficients=np.stack([q1, q2]),
        density_coefficient_upper=2.0,
        initial_nuisance=nuisance,
        nuisance_lower=[80.0, 0.0, 0.0, 0.5],
        nuisance_upper=[200.0, 5.0, 5.0, 1.5],
        regularisation=None,
        options=LinkedScalarFitOptions(max_nfev=20),
    )

    fitted_observation = LinkedRawObservation(
        fit.prediction.role_names,
        fit.prediction.expected_electrons,
    )
    diagnostic_summary = analyse_linked_raw_residuals(
        operator,
        model,
        fitted_observation,
        fit,
        reference_light_provenance=DGI_PROVENANCE,
        regularisation=None,
    )
    diagnostics = diagnostic_summary.roles
    assert diagnostic_summary.route_provenance is DGI_PROVENANCE
    assert [item.frame_index for item in diagnostics] == [
        0,
        1,
        None,
        None,
        None,
        None,
    ]
    assert [item.role_name for item in diagnostics] == [
        "atom_stop_000",
        "atom_stop_001",
        "leakage_stop",
        "stop_dark",
        "open_reference",
        "open_dark",
    ]
    assert all(item.roi_rms == pytest.approx(0.0) for item in diagnostics)
    q1_sensitivity_observation = select_q1_observation_for_reference_sensitivity(
        operator,
        model,
        fitted_observation,
        fit,
        reference_light_provenance=DGI_PROVENANCE,
        regularisation=None,
    )
    assert q1_sensitivity_observation.role_names == (
        "atom_stop_000",
        "leakage_stop",
        "stop_dark",
        "open_reference",
        "open_dark",
    )
    assert "atom_stop_001" not in q1_sensitivity_observation.role_names
    dgi_null = fit_linked_zero_density_null(
        operator,
        model,
        fitted_observation,
        fit,
        reference_light_provenance=DGI_PROVENANCE,
        initial_nuisance=DGINuisanceValues(*fit.nuisance_values),
        nuisance_lower=[80.0, 0.0, 0.0, 0.5],
        nuisance_upper=[200.0, 5.0, 5.0, 1.5],
        regularisation=None,
        options=LinkedScalarFitOptions(max_nfev=20),
    )
    assert dgi_null.success
    assert dgi_null.prediction.role_names == fit.prediction.role_names
    assert dgi_null.prediction.density_parameter_slices == ()

    pci_operator = PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(140.0, 0.7),
        response=ScalarOpticalResponseContract(1.9e-16, 3.8e-18),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 1,
            "dark": 1,
        },
    )
    with pytest.raises(TypeError, match="DGI provenance requires"):
        analyse_linked_raw_residuals(
            pci_operator,
            model,
            fitted_observation,
            fit,
            reference_light_provenance=DGI_PROVENANCE,
            regularisation=None,
        )


def test_single_bootstrap_refit_supplies_multiple_named_supports_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, operator, observation, point, support, options = _small_pci_information_case()
    central_mask = np.asarray(support.support_mask, dtype=bool) & (
        np.abs(support.y_grid_m) <= 1.0e-6
    )
    central = ObservableIntegrationSupport(
        support.y_grid_m,
        support.z_grid_m,
        support_mask=central_mask,
    )
    actual_fit = information.fit_linked_scalar_sequence
    calls = 0

    def counted_fit(*args: object, **kwargs: object) -> LinkedScalarFitResult:
        nonlocal calls
        calls += 1
        return actual_fit(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        information,
        "draw_linked_raw_observation",
        lambda *_args, **_kwargs: observation,
    )
    monkeypatch.setattr(information, "fit_linked_scalar_sequence", counted_fit)
    draw = refit_linked_observable_bootstrap_draw(
        operator,
        model,
        point,
        reference_light_provenance=PRIMARY_PROVENANCE,
        integration_supports={"full": support, "central": central},
        density_coefficient_upper=2.0,
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        options=options,
        rng=np.random.default_rng(17),
    )

    assert calls == 1
    assert draw.fit_success
    assert draw.support_names == ("full", "central")
    assert draw.values.shape == (2, 2, len(OBSERVABLE_NAMES))
    assert np.all(draw.supported_mask[:, :, 0])
    assert np.all(draw.values[1, :, 0] < draw.values[0, :, 0])


def test_single_bootstrap_refit_retains_explicit_all_nan_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, operator, observation, point, support, options = _small_pci_information_case()
    failed = replace(
        point,
        diagnostics=replace(
            point.diagnostics,
            success=False,
            message="forced focused-test failure",
        ),
    )
    monkeypatch.setattr(
        information,
        "draw_linked_raw_observation",
        lambda *_args, **_kwargs: observation,
    )
    monkeypatch.setattr(
        information,
        "fit_linked_scalar_sequence",
        lambda *_args, **_kwargs: failed,
    )

    draw = refit_linked_observable_bootstrap_draw(
        operator,
        model,
        point,
        reference_light_provenance=PRIMARY_PROVENANCE,
        integration_supports={"full": support},
        density_coefficient_upper=2.0,
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        options=options,
        rng=np.random.default_rng(19),
    )

    assert draw.status == "fit_failure"
    assert not draw.fit_success
    assert "forced focused-test failure" in draw.fit_message
    assert np.all(np.isnan(draw.values))
    assert not np.any(draw.supported_mask)

    def raised_fit(*_args: object, **_kwargs: object) -> LinkedScalarFitResult:
        raise ValueError("forced numerical-domain failure")

    monkeypatch.setattr(information, "fit_linked_scalar_sequence", raised_fit)
    raised = refit_linked_observable_bootstrap_draw(
        operator,
        model,
        point,
        reference_light_provenance=PRIMARY_PROVENANCE,
        integration_supports={"full": support},
        density_coefficient_upper=2.0,
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        options=options,
        rng=np.random.default_rng(23),
    )
    assert raised.status == "fit_failure"
    assert "forced numerical-domain failure" in raised.fit_message
    assert np.all(np.isnan(raised.values))


def test_linked_zero_density_refits_nuisance_and_only_reports_development_rank() -> None:
    model, operator, observation, point, _support, options = _small_pci_information_case()
    null_fit = fit_linked_zero_density_null(
        operator,
        model,
        observation,
        point,
        reference_light_provenance=PRIMARY_PROVENANCE,
        initial_nuisance=PCINuisanceValues(*point.nuisance_values),
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        options=options,
    )

    assert null_fit.success
    assert null_fit.irls_iterations == point.diagnostics.irls_iterations
    assert null_fit.prediction.density_parameter_slices == ()
    assert null_fit.prediction.jacobian.shape[1] == len(operator.nuisance_names)
    fitted_null_nuisance = PCINuisanceValues(*null_fit.nuisance_values)
    zero = np.zeros_like(operator.grid.y_grid_m)
    expected_names, expected_roles = operator.expected_linked_sequence_from_density_maps(
        [zero, zero],
        fitted_null_nuisance,
    )
    assert null_fit.prediction.role_names == expected_names
    for stored, expected in zip(
        null_fit.prediction.expected_electrons,
        expected_roles,
        strict=True,
    ):
        np.testing.assert_allclose(stored, expected, rtol=1e-12, atol=0.0)

    model_only = analyse_linked_zero_density_evidence(
        operator,
        model,
        observation,
        point,
        null_fit,
        reference_light_provenance=PRIMARY_PROVENANCE,
        regularisation=None,
        pipeline_fingerprint="focused-pipeline-v1",
        condition_fingerprint="focused-condition-v1",
    )
    assert model_only.evidence_level == "model_only"
    assert model_only.delta_data_gaussian_quasi_deviance > 0.0
    reference = LinkedSyntheticBlankReference(
        delta_data_gaussian_quasi_deviance=np.asarray(
            [
                model_only.delta_data_gaussian_quasi_deviance - 1.0,
                model_only.delta_data_gaussian_quasi_deviance + 1.0,
            ]
        ),
        case_ids=("blank-success-1", "blank-success-2"),
        attempted_count=3,
        failed_case_ids=("blank-failed-1",),
        pipeline_fingerprint="focused-pipeline-v1",
        condition_fingerprint="focused-condition-v1",
    )
    ranked = analyse_linked_zero_density_evidence(
        operator,
        model,
        observation,
        point,
        null_fit,
        reference_light_provenance=PRIMARY_PROVENANCE,
        regularisation=None,
        pipeline_fingerprint="focused-pipeline-v1",
        condition_fingerprint="focused-condition-v1",
        synthetic_blank_reference=reference,
    )
    assert ranked.evidence_level == "synthetic_blank_development_rank"
    assert ranked.development_rank_from_largest == 2
    assert ranked.failed_reference_count == 1
    assert not hasattr(ranked, "p_value")
    assert not hasattr(ranked, "threshold")


def test_linked_zero_density_null_uses_frozen_lsmr_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, operator, observation, point, _support, _options = _small_pci_information_case()
    actual_least_squares = information.least_squares
    calls: list[dict[str, object]] = []

    def recording_least_squares(*args: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        return actual_least_squares(*args, **kwargs)

    monkeypatch.setattr(information, "least_squares", recording_least_squares)
    options = LinkedScalarFitOptions(
        max_nfev=20,
        trust_region_solver="lsmr",
        lsmr_atol=2e-6,
        lsmr_btol=3e-6,
        lsmr_conlim=4e7,
        lsmr_maxiter=31,
        lsmr_regularize=False,
    )
    null_fit = fit_linked_zero_density_null(
        operator,
        model,
        observation,
        point,
        reference_light_provenance=PRIMARY_PROVENANCE,
        initial_nuisance=PCINuisanceValues(*point.nuisance_values),
        nuisance_lower=[0.8e5, 0.0],
        nuisance_upper=[1.2e5, 2.0],
        regularisation=None,
        options=options,
    )

    assert null_fit.success
    assert len(calls) == options.irls_iterations
    for call in calls:
        assert call["method"] == "trf"
        assert call["loss"] == "linear"
        assert call["x_scale"] == "jac"
        assert call["tr_solver"] == "lsmr"
        assert call["tr_options"] == {
            "atol": pytest.approx(2e-6),
            "btol": pytest.approx(3e-6),
            "conlim": pytest.approx(4e7),
            "maxiter": 31,
            "regularize": False,
        }
def test_dgi_zero_density_prediction_preserves_raw_roles_and_exposure_variances() -> None:
    axis_m = (np.arange(8, dtype=float) - 4.0) * 0.5e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    grid = ReconstructionGrid.from_arrays(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        pupil=np.ones((8, 8), dtype=complex),
        bin_size=2,
        roi_mask=np.ones((4, 4), dtype=bool),
    )
    operator = DGILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(140.0, 0.7),
        response=ScalarOpticalResponseContract(1.9e-16, 3.8e-18),
        transfer=DGITransferContract(2.0),
        independent_exposures_by_role={
            "atom_stop": 2,
            "leakage_stop": 3,
            "stop_dark": 4,
            "open_reference": 5,
            "open_dark": 6,
        },
    )
    nuisance = DGINuisanceValues(140.0, 0.4, 0.2, 0.9)
    prediction = information._zero_density_linked_prediction(operator, 2, nuisance)
    zero = np.zeros_like(y_grid_m)
    names, roles = operator.expected_linked_sequence_from_density_maps(
        [zero, zero],
        nuisance,
    )

    assert prediction.role_names == names
    assert prediction.density_parameter_slices == ()
    assert prediction.jacobian.shape[1] == 4
    for index, (name, frame_index, stored, expected, variance) in enumerate(
        zip(
            prediction.role_names,
            prediction.role_frame_indices,
            prediction.expected_electrons,
            roles,
            prediction.conditional_variance_electrons2,
            strict=True,
        )
    ):
        np.testing.assert_allclose(stored, expected, rtol=1e-12, atol=0.0)
        base_role = "atom_stop" if frame_index is not None else name
        exposure_count = operator.independent_exposures_by_role[base_role]
        np.testing.assert_allclose(
            variance,
            (expected + operator.read_noise_electrons**2) / exposure_count,
            rtol=1e-12,
            atol=0.0,
            err_msg=f"variance mismatch for role index {index}",
        )
