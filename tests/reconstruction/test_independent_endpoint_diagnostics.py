from __future__ import annotations

from dataclasses import replace
from typing import get_args

import numpy as np
import pytest

from non_destructive_image.reconstruction.contracts import (
    DetectorContract,
    ReconstructionGrid,
)
from non_destructive_image.reconstruction.independent_endpoint_diagnostics import (
    IndependentConfidenceComponents,
    IndependentQuantityStatus,
    InformationLevel,
    analyse_independent_endpoint_identifiability,
    analyse_independent_endpoint_residuals,
    analyse_independent_zero_density_null,
    classify_independent_information_level,
)
from non_destructive_image.reconstruction.independent_endpoint_information import (
    ENDPOINT_LABELS,
    RAW_ROLE_NAMES,
    IndependentEndpointFit,
    IndependentEndpointFitInput,
    IndependentEndpointPairFit,
    IndependentEndpointPairProvenance,
    IndependentEndpointRawBlock,
    fit_independent_pci_endpoints,
    summarise_independent_endpoint_information,
    IndependentEndpointBootstrap,
)
from non_destructive_image.reconstruction.linked_scalar_fit import (
    LinkedScalarFitOptions,
)
from non_destructive_image.reconstruction.object_models import (
    NonnegativeBilinearDensityModel,
)
from non_destructive_image.reconstruction.observables import (
    ObservableIntegrationSupport,
)
from non_destructive_image.reconstruction.scalar_measurements import (
    PCILinkedRawOperator,
    PCINuisanceValues,
    PCITransferContract,
    ScalarOpticalResponseContract,
)


def _case() -> tuple[
    tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    IndependentEndpointPairProvenance,
]:
    axis_m = (np.arange(8, dtype=float) - 3.5) * 0.5e-6
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
        knot_y_um=(-1.5, 1.5),
        knot_z_um=(-1.5, 1.5),
        coefficient_scale_m2=1e14,
    )
    support = ObservableIntegrationSupport(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        support_mask=model.support_mask,
    )
    coefficients = (
        np.asarray([0.18, 0.42, 0.26, 0.70]),
        np.asarray([0.12, 0.35, 0.38, 0.82]),
    )
    truth_nuisances = (PCINuisanceValues(120.0, 0.25), PCINuisanceValues(133.0, 0.55))
    inputs: list[IndependentEndpointFitInput] = []
    for index in range(2):
        operator = PCILinkedRawOperator(
            grid=grid,
            detector=DetectorContract(120.0, 0.4),
            response=ScalarOpticalResponseContract(
                1.8e-16,
                3.0e-18,
            ),
            transfer=PCITransferContract(0.95, np.pi / 2.0),
            independent_exposures_by_role={
                "atom": 1,
                "bright_reference": 1,
                "dark": 1,
            },
            jacobian_batch_size=2,
        )
        prediction = operator.expected_linked_sequence_and_jacobian_model(
            model,
            [coefficients[index]],
            truth_nuisances[index],
        )
        prefix = ("by", "bz")[index]
        raw = IndependentEndpointRawBlock(
            endpoint_label=ENDPOINT_LABELS[index],
            field_orientation=("y", "z")[index],
            role_names=RAW_ROLE_NAMES,
            role_owner_ids=(
                f"{prefix}_atom",
                f"{prefix}_bright",
                f"{prefix}_dark",
            ),
            observed_electrons=prediction.expected_electrons,
        )
        inputs.append(
            IndependentEndpointFitInput(
                operator=operator,
                model=model,
                raw_block=raw,
                observable_support=support,
                initial_density_coefficients=0.8 * coefficients[index],
                density_parameter_lower=0.0,
                density_coefficient_upper=2.0,
                initial_nuisance=PCINuisanceValues(
                    0.95 * truth_nuisances[index].i0_photoelectrons_per_pixel,
                    0.5 * truth_nuisances[index].dark_electrons_per_pixel,
                ),
                nuisance_lower=[70.0, 0.0],
                nuisance_upper=[180.0, 3.0],
                regularisation=None,
                options=LinkedScalarFitOptions(
                    irls_iterations=1,
                    max_nfev=80,
                    xtol=1e-10,
                    ftol=1e-10,
                    gtol=1e-10,
                ),
            )
        )
    provenance = IndependentEndpointPairProvenance(
        contract_label="chapter_5_orientation_information_contract_v1",
        endpoint_labels=ENDPOINT_LABELS,
        field_orientations=("y", "z"),
        imaging_axis="x",
        raw_count_unit="electrons",
        density_unit="m^-2",
        independent_preparations=True,
        independent_raw_blocks=True,
        temporal_coupling_used=False,
        cross_orientation_amplitude_calibration=False,
    )
    return (inputs[0], inputs[1]), provenance


def _fit_case() -> tuple[
    tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    IndependentEndpointPairFit,
]:
    inputs, provenance = _case()
    return inputs, fit_independent_pci_endpoints(inputs, provenance=provenance)


def _clone_operator(
    item: IndependentEndpointFitInput,
    *,
    grid: ReconstructionGrid | None = None,
    phase_scale: float = 1.0,
    transfer_phase_shift: float = 0.0,
) -> PCILinkedRawOperator:
    operator = item.operator
    return PCILinkedRawOperator(
        grid=operator.grid if grid is None else grid,
        detector=operator.detector,
        response=ScalarOpticalResponseContract(
            phase_scale * operator.response.phase_per_column_density_rad_m2,
            operator.response.optical_depth_per_column_density_m2,
        ),
        transfer=PCITransferContract(
            operator.transfer.phase_plate_transmittance,
            operator.transfer.phase_plate_phase_rad + transfer_phase_shift,
        ),
        independent_exposures_by_role=dict(operator.independent_exposures_by_role),
        jacobian_batch_size=operator.jacobian_batch_size,
    )


def test_raw_residuals_keep_three_endpoint_owned_roles_and_fit_identity() -> None:
    inputs, fit = _fit_case()
    residuals = analyse_independent_endpoint_residuals(inputs, fit)

    for endpoint, item in zip(residuals.endpoints, inputs, strict=True):
        assert endpoint.status == "success"
        assert tuple(role.role_name for role in endpoint.roles) == RAW_ROLE_NAMES
        assert tuple(role.role_owner_id for role in endpoint.roles) == (
            item.raw_block.role_owner_ids
        )
        assert max(role.roi_rms for role in endpoint.roles) < 1e-5


def test_residual_diagnostics_reject_raw_owner_identity_drift() -> None:
    inputs, fit = _fit_case()
    changed_raw = replace(
        inputs[0].raw_block,
        role_owner_ids=("changed_atom", "changed_bright", "changed_dark"),
    )
    with pytest.raises(ValueError, match="ownership"):
        analyse_independent_endpoint_residuals(
            (replace(inputs[0], raw_block=changed_raw), inputs[1]),
            fit,
        )


def test_diagnostics_reject_every_post_fit_input_substitution() -> None:
    inputs, fit = _fit_case()
    grid = inputs[0].operator.grid
    changed_grid = ReconstructionGrid.from_arrays(
        y_grid_m=grid.y_grid_m,
        z_grid_m=grid.z_grid_m,
        pupil=0.9 * grid.pupil,
        bin_size=grid.bin_size,
        roi_mask=grid.roi_mask,
    )
    changed_model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=grid.y_grid_m,
        z_grid_m=grid.z_grid_m,
        knot_y_um=inputs[0].model.knot_y_um,
        knot_z_um=inputs[0].model.knot_z_um,
        coefficient_scale_m2=1.1 * inputs[0].model.coefficient_scale_m2,
        support_mask=inputs[0].model.support_mask,
    )
    changed_support_mask = np.array(
        inputs[0].observable_support.support_mask,
        copy=True,
    )
    changed_support_mask[0, 0] = False
    changed_support = ObservableIntegrationSupport(
        y_grid_m=grid.y_grid_m,
        z_grid_m=grid.z_grid_m,
        support_mask=changed_support_mask,
    )
    changed_raw_arrays = [
        np.array(array, copy=True) for array in inputs[0].raw_block.observed_electrons
    ]
    changed_raw_arrays[0][0, 0] += 1.0
    changed_raw = replace(
        inputs[0].raw_block,
        observed_electrons=tuple(changed_raw_arrays),
    )
    variants = {
        "raw": (replace(inputs[0], raw_block=changed_raw), inputs[1]),
        "bounds": (
            replace(inputs[0], density_coefficient_upper=2.1),
            inputs[1],
        ),
        "options": (
            replace(
                inputs[0],
                options=replace(
                    inputs[0].options,
                    max_nfev=inputs[0].options.max_nfev + 1,
                ),
            ),
            inputs[1],
        ),
        "support": (
            replace(inputs[0], observable_support=changed_support),
            replace(inputs[1], observable_support=changed_support),
        ),
        "model": (
            replace(inputs[0], model=changed_model),
            replace(inputs[1], model=changed_model),
        ),
        "response": (
            replace(inputs[0], operator=_clone_operator(inputs[0], phase_scale=0.9)),
            replace(inputs[1], operator=_clone_operator(inputs[1], phase_scale=0.9)),
        ),
        "grid": (
            replace(inputs[0], operator=_clone_operator(inputs[0], grid=changed_grid)),
            replace(inputs[1], operator=_clone_operator(inputs[1], grid=changed_grid)),
        ),
        "operator": (
            replace(
                inputs[0],
                operator=_clone_operator(inputs[0], transfer_phase_shift=0.1),
            ),
            replace(
                inputs[1],
                operator=_clone_operator(inputs[1], transfer_phase_shift=0.1),
            ),
        ),
    }
    for _label, changed_inputs in variants.items():
        with pytest.raises(ValueError, match="fit input identity"):
            analyse_independent_endpoint_residuals(changed_inputs, fit)


def test_residual_failure_record_preserves_successful_peer() -> None:
    inputs, fit = _fit_case()
    failed = IndependentEndpointFit(
        endpoint_label=fit.endpoints[1].endpoint_label,
        field_orientation=fit.endpoints[1].field_orientation,
        role_owner_ids=fit.endpoints[1].role_owner_ids,
        fit_input_sha256=fit.endpoints[1].fit_input_sha256,
        status="fit_failure",
        message="synthetic numerical failure",
        fit_result=None,
        observables=None,
    )
    partial_fit = IndependentEndpointPairFit(
        endpoints=(fit.endpoints[0], failed),
        provenance=fit.provenance,
    )
    residuals = analyse_independent_endpoint_residuals(inputs, partial_fit)

    assert residuals.endpoints[0].status == "success"
    assert residuals.endpoints[1].status == "fit_failure"
    assert residuals.endpoints[1].roles == ()
    identifiability = analyse_independent_endpoint_identifiability(inputs, partial_fit)
    assert identifiability.endpoints[0].status == "success"
    assert any(record.supported for record in identifiability.endpoints[0].records)
    assert identifiability.endpoints[1].status == "fit_failure"
    assert not any(record.supported for record in identifiability.endpoints[1].records)
    assert not any(record.supported for record in identifiability.paired_records)
    null = analyse_independent_zero_density_null(inputs, partial_fit)
    assert null.endpoints[0].status == "success"
    assert null.endpoints[1].status == "fit_failure"


def test_zero_density_null_refits_nuisance_per_endpoint() -> None:
    inputs, fit = _fit_case()
    null = analyse_independent_zero_density_null(inputs, fit)

    for endpoint in null.endpoints:
        assert endpoint.status == "success"
        assert endpoint.nuisance_values is not None
        assert endpoint.null_weighted_chi_square is not None
        assert endpoint.fitted_weighted_chi_square is not None
        assert endpoint.improvement_over_null is not None
        assert endpoint.null_weighted_chi_square >= 0.0
        assert endpoint.fitted_weighted_chi_square >= 0.0
        assert endpoint.improvement_over_null > 0.0
        assert endpoint.evidence_level == "model_only"


def test_identifiability_uses_two_local_blocks_and_withholds_amplitude_ratio() -> None:
    inputs, fit = _fit_case()
    result = analyse_independent_endpoint_identifiability(inputs, fit)

    assert not hasattr(result, "joint_jacobian")
    for endpoint, item, fit_endpoint in zip(
        result.endpoints,
        inputs,
        fit.endpoints,
        strict=True,
    ):
        assert endpoint.status == "success"
        assert endpoint.parameter_count == item.model.parameter_count + 2
        assert fit_endpoint.fit_result is not None
        assert fit_endpoint.fit_result.prediction.jacobian.shape[1] == (
            endpoint.parameter_count
        )
        assert endpoint.data_rank <= endpoint.parameter_count
        assert endpoint.records
        assert all(record.quantity == endpoint.endpoint_label for record in endpoint.records)
        assert endpoint.relative_rank_tolerance == 1e-10

    amplitude_ratio = next(
        record
        for record in result.paired_records
        if record.observable_name == "A" and record.quantity == "ratio_Bz_over_By"
    )
    assert amplitude_ratio.supported is False
    assert amplitude_ratio.estimate is None
    assert amplitude_ratio.reasons == (
        "cross_orientation_amplitude_calibration_not_supplied",
    )
    with pytest.raises(ValueError, match="no cross-orientation amplitude calibration"):
        replace(fit.provenance, cross_orientation_amplitude_calibration=True)
    sigma_delta = next(
        record
        for record in result.paired_records
        if record.observable_name == "sigma_y_um"
        and record.quantity == "delta_Bz_minus_By"
    )
    assert sigma_delta.supported is True


def test_changing_bz_likelihood_does_not_change_by_identifiability_block() -> None:
    inputs, fit = _fit_case()
    baseline = analyse_independent_endpoint_identifiability(inputs, fit)
    changed_coefficients = np.asarray([0.72, 0.18, 0.64, 0.24])
    changed_prediction = inputs[1].operator.expected_linked_sequence_and_jacobian_model(
        inputs[1].model,
        [changed_coefficients],
        PCINuisanceValues(133.0, 0.55),
    )
    changed_raw = replace(
        inputs[1].raw_block,
        observed_electrons=changed_prediction.expected_electrons,
    )
    changed_input = replace(
        inputs[1],
        raw_block=changed_raw,
        initial_density_coefficients=0.8 * changed_coefficients,
    )
    changed_fit = fit_independent_pci_endpoints(
        (inputs[0], changed_input),
        provenance=fit.provenance,
    )
    changed = analyse_independent_endpoint_identifiability(
        (inputs[0], changed_input),
        changed_fit,
    )

    np.testing.assert_allclose(
        changed.endpoints[0].singular_values,
        baseline.endpoints[0].singular_values,
        rtol=0.0,
        atol=0.0,
    )
    assert not np.allclose(
        changed.endpoints[1].singular_values,
        baseline.endpoints[1].singular_values,
        rtol=1e-6,
        atol=0.0,
    )


def test_confidence_components_do_not_collapse_incomplete_evidence() -> None:
    inputs, fit = _fit_case()
    by_observables = fit.endpoints[0].observables
    bz_observables = fit.endpoints[1].observables
    assert by_observables is not None and bz_observables is not None
    point = np.stack([by_observables.values, bz_observables.values])
    samples = np.stack((0.99 * point, 1.01 * point))
    bootstrap = IndependentEndpointBootstrap(
        point_fit=fit,
        fit_success_mask=np.ones((2, 2), dtype=bool),
        samples=samples,
        supported_mask=np.isfinite(samples),
    )
    information = summarise_independent_endpoint_information(
        bootstrap,
        confidence_level=0.68,
    )
    sigma_y = next(
        item for item in information.observables if item.observable_name == "sigma_y_um"
    )
    complete = IndependentConfidenceComponents(
        fit_and_data="adequate",
        detector_statistical="adequate",
        identifiability="adequate",
        calibration="adequate",
        forward_model="adequate",
        basis_model="stable",
        support="stable",
        reference="not_applicable",
        regularisation="not_applicable",
        repeatability="adequate",
        relative_change="adequate",
        reasons=(),
    )
    assert classify_independent_information_level(sigma_y.delta_b, complete) == (
        "quantitatively_resolved"
    )
    status = IndependentQuantityStatus(
        estimate=sigma_y.delta_b,
        information_level="quantitatively_resolved",
        confidence=complete,
    )
    assert status.estimate is sigma_y.delta_b
    limited = replace(
        complete,
        repeatability="not_assessed",
        reasons=("experimental repeatability has not been assessed",),
    )
    assert classify_independent_information_level(sigma_y.delta_b, limited) == (
        "informative_but_inconclusive"
    )


def test_orientation_v1_does_not_expose_an_unimplemented_bounded_level() -> None:
    assert "bounded" not in get_args(InformationLevel)
