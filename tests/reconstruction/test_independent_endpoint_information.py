from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from non_destructive_image.reconstruction.contracts import (
    DetectorContract,
    ReconstructionGrid,
)
from non_destructive_image.reconstruction.independent_endpoint_information import (
    ENDPOINT_LABELS,
    OBSERVABLE_NAMES,
    RAW_ROLE_NAMES,
    EndpointObservableVector,
    IndependentEndpointBootstrap,
    IndependentEndpointFitInput,
    IndependentEndpointPairFit,
    IndependentEndpointPairProvenance,
    IndependentEndpointPointFitSnapshot,
    IndependentEndpointRawBlock,
    SUPPORT_NAMES,
    assemble_independent_endpoint_bootstrap,
    draw_and_refit_independent_endpoint_bootstrap,
    fit_independent_pci_endpoints,
    postprocess_independent_endpoint_supports,
    restore_successful_independent_endpoint_pair_fit,
    snapshot_successful_independent_endpoint_pair_fit,
    summarise_independent_endpoint_information,
)
from non_destructive_image.reconstruction.linked_scalar_fit import (
    LinkedScalarFitOptions,
    fit_linked_scalar_sequence,
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


def _grid(*, pupil_scale: float = 1.0) -> ReconstructionGrid:
    axis_m = (np.arange(8, dtype=float) - 3.5) * 0.5e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    return ReconstructionGrid.from_arrays(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        pupil=np.full((8, 8), pupil_scale, dtype=complex),
        bin_size=2,
        roi_mask=np.ones((4, 4), dtype=bool),
    )


def _model(
    grid: ReconstructionGrid,
    *,
    knot_y_um: tuple[float, ...] = (-1.5, 1.5),
) -> NonnegativeBilinearDensityModel:
    return NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=grid.y_grid_m,
        z_grid_m=grid.z_grid_m,
        knot_y_um=knot_y_um,
        knot_z_um=(-1.5, 1.5),
        coefficient_scale_m2=1e14,
    )


def _operator(
    grid: ReconstructionGrid,
    *,
    phase_response: float,
) -> PCILinkedRawOperator:
    return PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(120.0, 0.4),
        response=ScalarOpticalResponseContract(phase_response, 3.0e-18),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 1,
            "dark": 1,
        },
        jacobian_batch_size=2,
    )


def _provenance(
    *,
    cross_orientation_amplitude_calibration: bool = False,
) -> IndependentEndpointPairProvenance:
    return IndependentEndpointPairProvenance(
        contract_label="chapter_5_orientation_information_contract_v1",
        endpoint_labels=("B_parallel_y", "B_parallel_z"),
        field_orientations=("y", "z"),
        imaging_axis="x",
        raw_count_unit="electrons",
        density_unit="m^-2",
        independent_preparations=True,
        independent_raw_blocks=True,
        temporal_coupling_used=False,
        cross_orientation_amplitude_calibration=(
            cross_orientation_amplitude_calibration
        ),
    )


def _fit_inputs() -> tuple[
    tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    tuple[np.ndarray, np.ndarray],
]:
    grid = _grid()
    model = _model(grid)
    support = ObservableIntegrationSupport(
        y_grid_m=grid.y_grid_m,
        z_grid_m=grid.z_grid_m,
        support_mask=model.support_mask,
    )
    coefficients = (
        np.asarray([0.18, 0.42, 0.26, 0.70]),
        np.asarray([0.12, 0.35, 0.38, 0.82]),
    )
    nuisances = (PCINuisanceValues(120.0, 0.25), PCINuisanceValues(133.0, 0.55))
    operators = (
        _operator(grid, phase_response=1.8e-16),
        _operator(grid, phase_response=1.8e-16),
    )
    inputs: list[IndependentEndpointFitInput] = []
    for index, (operator, coefficient, nuisance) in enumerate(
        zip(operators, coefficients, nuisances, strict=True)
    ):
        prediction = operator.expected_linked_sequence_and_jacobian_model(
            model,
            [coefficient],
            nuisance,
        )
        prefix = "by" if index == 0 else "bz"
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
                initial_density_coefficients=0.8 * coefficient,
                density_parameter_lower=0.0,
                density_coefficient_upper=2.0,
                initial_nuisance=PCINuisanceValues(
                    0.95 * nuisance.i0_photoelectrons_per_pixel,
                    0.5 * nuisance.dark_electrons_per_pixel,
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
    return (inputs[0], inputs[1]), coefficients


def test_factorised_pair_fit_equals_two_direct_single_frame_fits() -> None:
    inputs, _ = _fit_inputs()
    pair = fit_independent_pci_endpoints(inputs, provenance=_provenance())

    direct = tuple(
        fit_linked_scalar_sequence(
            item.operator,
            item.model,
            item.raw_block.as_linked_observation(),
            initial_density_coefficients=item.initial_density_coefficients[None, :],
            density_parameter_lower=item.density_parameter_lower,
            density_coefficient_upper=item.density_coefficient_upper,
            initial_nuisance=item.initial_nuisance,
            nuisance_lower=item.nuisance_lower,
            nuisance_upper=item.nuisance_upper,
            regularisation=item.regularisation,
            options=item.options,
        )
        for item in inputs
    )

    assert pair.fit_success_mask == (True, True)
    for endpoint, expected in zip(pair.endpoints, direct, strict=True):
        assert endpoint.fit_result is not None
        np.testing.assert_allclose(
            endpoint.fit_result.density_coefficients,
            expected.density_coefficients,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            endpoint.fit_result.nuisance_values,
            expected.nuisance_values,
            rtol=0.0,
            atol=0.0,
        )


def test_pair_has_two_local_nuisance_and_jacobian_blocks_only() -> None:
    inputs, _ = _fit_inputs()
    pair = fit_independent_pci_endpoints(inputs, provenance=_provenance())

    assert set(pair.endpoints[0].nuisance_owner_ids).isdisjoint(
        pair.endpoints[1].nuisance_owner_ids
    )
    assert not hasattr(pair, "joint_jacobian")
    for endpoint, item in zip(pair.endpoints, inputs, strict=True):
        assert endpoint.fit_result is not None
        prediction = endpoint.fit_result.prediction
        assert prediction.nuisance_names == (
            "i0_photoelectrons_per_pixel",
            "dark_electrons_per_pixel",
        )
        assert prediction.jacobian.shape[1] == item.model.parameter_count + 2
        assert prediction.density_parameter_slices == (
            slice(0, item.model.parameter_count),
        )


def test_successful_point_fit_snapshot_roundtrip_recomputes_derived_arrays() -> None:
    inputs, _ = _fit_inputs()
    original = fit_independent_pci_endpoints(inputs, provenance=_provenance())
    snapshot = snapshot_successful_independent_endpoint_pair_fit(original)
    restored = restore_successful_independent_endpoint_pair_fit(inputs, snapshot)

    assert snapshot.provenance == original.provenance
    assert restored.fit_success_mask == (True, True)
    for original_endpoint, restored_endpoint, record in zip(
        original.endpoints,
        restored.endpoints,
        snapshot.endpoints,
        strict=True,
    ):
        assert not hasattr(record, "prediction")
        assert not hasattr(record, "column_density_m2")
        assert not hasattr(record, "observables")
        assert original_endpoint.fit_result is not None
        assert restored_endpoint.fit_result is not None
        assert original_endpoint.observables is not None
        assert restored_endpoint.observables is not None
        np.testing.assert_array_equal(
            restored_endpoint.fit_result.density_coefficients,
            original_endpoint.fit_result.density_coefficients,
        )
        np.testing.assert_array_equal(
            restored_endpoint.fit_result.nuisance_values,
            original_endpoint.fit_result.nuisance_values,
        )
        for actual, expected in zip(
            restored_endpoint.fit_result.prediction.expected_electrons,
            original_endpoint.fit_result.prediction.expected_electrons,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            restored_endpoint.observables.values,
            original_endpoint.observables.values,
        )


def test_point_fit_snapshot_rejects_substitution_swap_and_numerical_tampering() -> None:
    inputs, _ = _fit_inputs()
    original = fit_independent_pci_endpoints(inputs, provenance=_provenance())
    snapshot = snapshot_successful_independent_endpoint_pair_fit(original)

    changed_atom = np.array(inputs[0].raw_block.observed_electrons[0], copy=True)
    changed_atom[0, 0] += 1.0
    changed_raw = replace(
        inputs[0].raw_block,
        observed_electrons=(
            changed_atom,
            inputs[0].raw_block.observed_electrons[1],
            inputs[0].raw_block.observed_electrons[2],
        ),
    )
    with pytest.raises(ValueError, match="input identity"):
        restore_successful_independent_endpoint_pair_fit(
            (replace(inputs[0], raw_block=changed_raw), inputs[1]),
            snapshot,
        )

    with pytest.raises(ValueError, match="canonical By/Bz order"):
        IndependentEndpointPointFitSnapshot(
            endpoints=tuple(reversed(snapshot.endpoints)),
            provenance=snapshot.provenance,
        )

    changed_coefficients = np.array(
        snapshot.endpoints[0].density_coefficients,
        copy=True,
    )
    changed_coefficients[0] += 0.01
    tampered_record = replace(
        snapshot.endpoints[0],
        density_coefficients=changed_coefficients,
    )
    tampered_snapshot = replace(
        snapshot,
        endpoints=(tampered_record, snapshot.endpoints[1]),
    )
    with pytest.raises(ValueError, match="residual identity"):
        restore_successful_independent_endpoint_pair_fit(inputs, tampered_snapshot)

    tampered_diagnostics = replace(
        snapshot.endpoints[0].diagnostics,
        weighted_chi_square=(
            snapshot.endpoints[0].diagnostics.weighted_chi_square + 1.0
        ),
    )
    diagnostic_snapshot = replace(
        snapshot,
        endpoints=(
            replace(snapshot.endpoints[0], diagnostics=tampered_diagnostics),
            snapshot.endpoints[1],
        ),
    )
    with pytest.raises(ValueError, match="chi-square identity"):
        restore_successful_independent_endpoint_pair_fit(inputs, diagnostic_snapshot)

    diagnostics = snapshot.endpoints[0].diagnostics
    changed_rank = diagnostics.data_jacobian_rank - 1
    assert changed_rank >= 0
    changed_dof = diagnostics.whitened_residual_vector.size - changed_rank
    rank_diagnostics = replace(
        diagnostics,
        data_jacobian_rank=changed_rank,
        degrees_of_freedom=changed_dof,
        reduced_chi_square=diagnostics.weighted_chi_square / changed_dof,
    )
    rank_snapshot = replace(
        snapshot,
        endpoints=(
            replace(snapshot.endpoints[0], diagnostics=rank_diagnostics),
            snapshot.endpoints[1],
        ),
    )
    with pytest.raises(ValueError, match="Jacobian rank identity"):
        restore_successful_independent_endpoint_pair_fit(inputs, rank_snapshot)

    changed_condition = (
        1.0
        if np.isinf(diagnostics.data_jacobian_condition)
        else diagnostics.data_jacobian_condition * 1.01
    )
    condition_snapshot = replace(
        snapshot,
        endpoints=(
            replace(
                snapshot.endpoints[0],
                diagnostics=replace(
                    diagnostics,
                    data_jacobian_condition=changed_condition,
                ),
            ),
            snapshot.endpoints[1],
        ),
    )
    with pytest.raises(ValueError, match="Jacobian condition identity"):
        restore_successful_independent_endpoint_pair_fit(inputs, condition_snapshot)


def test_point_fit_snapshot_rejects_a_failed_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _ = _fit_inputs()
    import non_destructive_image.reconstruction.independent_endpoint_information as module

    original_fit = module.fit_linked_scalar_sequence

    def fail_by(operator: PCILinkedRawOperator, *args: object, **kwargs: object):
        if operator is inputs[0].operator:
            raise RuntimeError("synthetic primary endpoint failure")
        return original_fit(operator, *args, **kwargs)

    monkeypatch.setattr(module, "fit_linked_scalar_sequence", fail_by)
    failed = fit_independent_pci_endpoints(inputs, provenance=_provenance())
    assert failed.fit_success_mask == (False, True)
    with pytest.raises(ValueError, match="two successful endpoints"):
        snapshot_successful_independent_endpoint_pair_fit(failed)


def test_endpoint_specific_operator_instances_are_required() -> None:
    inputs, _ = _fit_inputs()
    shared_operator = replace(inputs[1], operator=inputs[0].operator)

    with pytest.raises(ValueError, match="distinct operator instances"):
        fit_independent_pci_endpoints(
            (inputs[0], shared_operator),
            provenance=_provenance(),
        )


def test_v1_requires_distinct_equal_responses_and_one_exposure_per_role() -> None:
    inputs, _ = _fit_inputs()
    first = inputs[0].operator
    second = inputs[1].operator
    shared_response_operator = PCILinkedRawOperator(
        grid=second.grid,
        detector=second.detector,
        response=first.response,
        transfer=second.transfer,
        independent_exposures_by_role=dict(second.independent_exposures_by_role),
        jacobian_batch_size=second.jacobian_batch_size,
    )
    with pytest.raises(ValueError, match="distinct response instances"):
        fit_independent_pci_endpoints(
            (inputs[0], replace(inputs[1], operator=shared_response_operator)),
            provenance=_provenance(),
        )

    changed_response_operator = _operator(
        second.grid,
        phase_response=0.9 * second.response.phase_per_column_density_rad_m2,
    )
    with pytest.raises(ValueError, match="rotation-covariant scalar response"):
        fit_independent_pci_endpoints(
            (inputs[0], replace(inputs[1], operator=changed_response_operator)),
            provenance=_provenance(),
        )

    repeated_role_operator = PCILinkedRawOperator(
        grid=second.grid,
        detector=second.detector,
        response=ScalarOpticalResponseContract(
            second.response.phase_per_column_density_rad_m2,
            second.response.optical_depth_per_column_density_m2,
        ),
        transfer=second.transfer,
        independent_exposures_by_role={
            "atom": 2,
            "bright_reference": 2,
            "dark": 2,
        },
        jacobian_batch_size=second.jacobian_batch_size,
    )
    with pytest.raises(ValueError, match="exactly one independent exposure"):
        replace(inputs[1], operator=repeated_role_operator)


def test_expected_numerical_fit_exception_preserves_peer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _ = _fit_inputs()
    import non_destructive_image.reconstruction.independent_endpoint_information as module

    original = module.fit_linked_scalar_sequence

    def fail_first(operator: PCILinkedRawOperator, *args: object, **kwargs: object):
        if operator is inputs[0].operator:
            raise np.linalg.LinAlgError("synthetic endpoint factorisation failure")
        return original(operator, *args, **kwargs)

    monkeypatch.setattr(module, "fit_linked_scalar_sequence", fail_first)
    pair = fit_independent_pci_endpoints(inputs, provenance=_provenance())

    assert pair.fit_success_mask == (False, True)
    assert pair.endpoints[0].fit_result is None
    assert pair.endpoints[0].observables is None
    assert "LinAlgError" in pair.endpoints[0].message
    assert pair.endpoints[1].fit_result is not None
    assert pair.endpoints[1].observables is not None


def test_raw_identity_rejects_shared_owners_and_wrong_semantics() -> None:
    inputs, _ = _fit_inputs()
    valid = inputs[0].raw_block
    with pytest.raises(ValueError, match="canonical By/Bz order"):
        fit_independent_pci_endpoints(
            (inputs[1], inputs[0]),
            provenance=_provenance(),
        )
    with pytest.raises(ValueError, match="orientation"):
        IndependentEndpointRawBlock(
            endpoint_label="B_parallel_y",
            field_orientation="z",
            role_names=RAW_ROLE_NAMES,
            role_owner_ids=("a", "b", "c"),
            observed_electrons=valid.observed_electrons,
        )
    with pytest.raises(ValueError, match="canonical PCI order"):
        IndependentEndpointRawBlock(
            endpoint_label="B_parallel_y",
            field_orientation="y",
            role_names=("dark", "bright_reference", "atom_000"),
            role_owner_ids=("a", "b", "c"),
            observed_electrons=valid.observed_electrons,
        )
    with pytest.raises(ValueError, match="unit"):
        replace(valid, unit="adu")

    shared_bright = replace(
        inputs[1].raw_block,
        role_owner_ids=(
            "bz_atom",
            inputs[0].raw_block.role_owner_ids[1],
            "bz_dark",
        ),
    )
    with pytest.raises(ValueError, match="six.*unique"):
        fit_independent_pci_endpoints(
            (inputs[0], replace(inputs[1], raw_block=shared_bright)),
            provenance=_provenance(),
        )


def test_provenance_rejects_order_orientation_axes_and_units() -> None:
    base = _provenance()
    with pytest.raises(ValueError, match="canonical By/Bz order"):
        replace(base, endpoint_labels=tuple(reversed(base.endpoint_labels)))
    with pytest.raises(ValueError, match="canonical y/z order"):
        replace(base, field_orientations=("z", "y"))
    with pytest.raises(ValueError, match="axis"):
        replace(base, imaging_axis="z")
    with pytest.raises(ValueError, match="raw-count unit"):
        replace(base, raw_count_unit="adu")
    with pytest.raises(ValueError, match="density unit"):
        replace(base, density_unit="cm^-2")
    with pytest.raises(ValueError, match="no cross-orientation amplitude calibration"):
        replace(base, cross_orientation_amplitude_calibration=True)


def test_grid_basis_and_observable_support_mismatch_are_rejected() -> None:
    inputs, _ = _fit_inputs()
    other_grid = _grid(pupil_scale=0.9)
    grid_mismatch = replace(
        inputs[1],
        operator=_operator(other_grid, phase_response=1.8e-16),
        model=_model(other_grid),
        observable_support=ObservableIntegrationSupport(
            y_grid_m=other_grid.y_grid_m,
            z_grid_m=other_grid.z_grid_m,
        ),
    )
    with pytest.raises(ValueError, match="reconstruction grids"):
        fit_independent_pci_endpoints(
            (inputs[0], grid_mismatch),
            provenance=_provenance(),
        )

    basis_mismatch = replace(
        inputs[1],
        model=_model(inputs[1].operator.grid, knot_y_um=(-1.5, 0.0, 1.5)),
        initial_density_coefficients=np.full(6, 0.2),
        density_parameter_lower=0.0,
        density_coefficient_upper=2.0,
    )
    with pytest.raises(ValueError, match="density bases"):
        fit_independent_pci_endpoints(
            (inputs[0], basis_mismatch),
            provenance=_provenance(),
        )

    changed_mask = np.array(
        inputs[1].observable_support.support_mask,
        copy=True,
    )
    changed_mask[0, 0] = False
    shifted_support = ObservableIntegrationSupport(
        y_grid_m=inputs[1].observable_support.y_grid_m,
        z_grid_m=inputs[1].observable_support.z_grid_m,
        support_mask=changed_mask,
    )
    with pytest.raises(ValueError, match="observable supports"):
        fit_independent_pci_endpoints(
            (inputs[0], replace(inputs[1], observable_support=shifted_support)),
            provenance=_provenance(),
        )


def _bootstrap_with_one_bz_failure() -> IndependentEndpointBootstrap:
    inputs, _ = _fit_inputs()
    pair = fit_independent_pci_endpoints(inputs, provenance=_provenance())
    by_observables = pair.endpoints[0].observables
    bz_observables = pair.endpoints[1].observables
    assert by_observables is not None and bz_observables is not None
    point = np.stack([by_observables.values, bz_observables.values])
    samples = np.stack([point * 0.98, point * 1.02, point * 1.01])
    success = np.ones((3, 2), dtype=bool)
    success[1, 1] = False
    samples[1, 1] = np.nan
    return IndependentEndpointBootstrap(
        point_fit=pair,
        fit_success_mask=success,
        samples=samples,
        supported_mask=np.isfinite(samples),
    )


def test_one_endpoint_failure_keeps_peer_and_suppresses_delta_interval() -> None:
    summary = summarise_independent_endpoint_information(
        _bootstrap_with_one_bz_failure(),
        confidence_level=0.68,
    )
    sigma_y = next(
        item for item in summary.observables if item.observable_name == "sigma_y_um"
    )
    assert sigma_y.by.status == "complete"
    assert sigma_y.by.lower is not None
    assert sigma_y.bz.status == "point_only"
    assert sigma_y.delta_b.status == "point_only"
    assert sigma_y.delta_b.lower is None
    assert sigma_y.delta_b.upper is None


def test_aspect_ratio_requires_both_widths_and_ratios_require_positive_values() -> None:
    bootstrap = _bootstrap_with_one_bz_failure()
    by = bootstrap.point_fit.endpoints[0]
    assert by.observables is not None
    values = np.array(by.observables.values, copy=True)
    values[OBSERVABLE_NAMES.index("sigma_z_um")] = np.nan
    no_z_width = replace(
        by,
        observables=EndpointObservableVector(
            names=OBSERVABLE_NAMES,
            units=by.observables.units,
            values=values,
            supported_mask=np.isfinite(values),
        ),
    )
    point_fit = IndependentEndpointPairFit(
        endpoints=(no_z_width, bootstrap.point_fit.endpoints[1]),
        provenance=bootstrap.point_fit.provenance,
    )
    samples = np.array(bootstrap.samples, copy=True)
    samples[:, 0, OBSERVABLE_NAMES.index("sigma_z_um")] = np.nan
    no_width_bootstrap = IndependentEndpointBootstrap(
        point_fit=point_fit,
        fit_success_mask=bootstrap.fit_success_mask,
        samples=samples,
        supported_mask=np.isfinite(samples),
    )
    summary = summarise_independent_endpoint_information(
        no_width_bootstrap,
        confidence_level=0.68,
    )
    aspect = next(
        item
        for item in summary.observables
        if item.observable_name == "aspect_ratio_y_over_z"
    )
    assert aspect.by.status == "unresolved"
    assert aspect.delta_b.status == "unresolved"

    positive_bootstrap = _bootstrap_with_one_bz_failure()
    negative_samples = np.array(positive_bootstrap.samples, copy=True)
    sigma_y_index = OBSERVABLE_NAMES.index("sigma_y_um")
    negative_samples[0, 0, sigma_y_index] = -1.0
    negative = IndependentEndpointBootstrap(
        point_fit=positive_bootstrap.point_fit,
        fit_success_mask=positive_bootstrap.fit_success_mask,
        samples=negative_samples,
        supported_mask=np.isfinite(negative_samples),
    )
    negative_summary = summarise_independent_endpoint_information(
        negative,
        confidence_level=0.68,
    )
    sigma_y = next(
        item
        for item in negative_summary.observables
        if item.observable_name == "sigma_y_um"
    )
    assert sigma_y.ratio_bz_over_by is not None
    assert sigma_y.ratio_bz_over_by.status == "point_only"
    assert sigma_y.ratio_bz_over_by.lower is None


def test_amplitude_ratio_requires_cross_orientation_calibration() -> None:
    summary = summarise_independent_endpoint_information(
        _bootstrap_with_one_bz_failure(),
        confidence_level=0.68,
    )
    amplitude = next(
        item for item in summary.observables if item.observable_name == "A"
    )
    assert amplitude.ratio_bz_over_by is not None
    assert amplitude.ratio_bz_over_by.status == "unresolved"
    assert "cross_orientation_amplitude_calibration_not_supplied" in (
        amplitude.ratio_bz_over_by.reasons
    )


def _three_supports(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
) -> dict[str, ObservableIntegrationSupport]:
    model = inputs[0].model
    inner_mask = np.zeros(model.support_mask.shape, dtype=bool)
    inner_mask[2:6, 2:6] = True
    return {
        "inner": ObservableIntegrationSupport(
            y_grid_m=model.y_grid_m,
            z_grid_m=model.z_grid_m,
            support_mask=inner_mask,
        ),
        "primary": inputs[0].observable_support,
        "outer": ObservableIntegrationSupport(
            y_grid_m=model.y_grid_m,
            z_grid_m=model.z_grid_m,
            support_mask=model.support_mask,
        ),
    }


def test_one_fit_is_postprocessed_on_three_named_supports() -> None:
    inputs, _ = _fit_inputs()
    fit = fit_independent_pci_endpoints(inputs, provenance=_provenance())
    postprocessed = postprocess_independent_endpoint_supports(
        inputs,
        fit,
        _three_supports(inputs),
    )

    assert postprocessed.support_names == SUPPORT_NAMES
    assert postprocessed.values.shape == (3, 2, len(OBSERVABLE_NAMES))
    assert tuple(postprocessed.endpoint_fit_success_mask) == (True, True)
    for endpoint_index, endpoint in enumerate(fit.endpoints):
        assert endpoint.observables is not None
        np.testing.assert_allclose(
            postprocessed.values[postprocessed.support_index("primary"), endpoint_index],
            endpoint.observables.values,
        )


def test_support_postprocessing_rejects_post_fit_input_substitution() -> None:
    inputs, _ = _fit_inputs()
    fit = fit_independent_pci_endpoints(inputs, provenance=_provenance())
    changed_inputs = (
        replace(inputs[0], density_coefficient_upper=2.1),
        inputs[1],
    )

    with pytest.raises(ValueError, match="fit input identity"):
        postprocess_independent_endpoint_supports(
            changed_inputs,
            fit,
            _three_supports(inputs),
        )


def test_conditional_draw_uses_independent_raw_blocks_and_is_seed_reproducible() -> None:
    inputs, _ = _fit_inputs()
    point_fit = fit_independent_pci_endpoints(inputs, provenance=_provenance())
    supports = _three_supports(inputs)
    left = draw_and_refit_independent_endpoint_bootstrap(
        inputs,
        point_fit,
        supports,
        draw_id=7,
        endpoint_rngs=(np.random.default_rng(701), np.random.default_rng(702)),
    )
    right = draw_and_refit_independent_endpoint_bootstrap(
        inputs,
        point_fit,
        supports,
        draw_id=7,
        endpoint_rngs=(np.random.default_rng(701), np.random.default_rng(702)),
    )

    owners = tuple(owner for block in left.raw_blocks for owner in block.role_owner_ids)
    assert len(set(owners)) == 6
    for left_block, right_block in zip(left.raw_blocks, right.raw_blocks, strict=True):
        for left_role, right_role in zip(
            left_block.observed_electrons,
            right_block.observed_electrons,
            strict=True,
        ):
            np.testing.assert_array_equal(left_role, right_role)
    assert not np.array_equal(
        left.raw_blocks[0].observed_electrons[0],
        left.raw_blocks[1].observed_electrons[0],
    )


def test_conditional_refit_exception_preserves_peer_without_redraw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _ = _fit_inputs()
    point_fit = fit_independent_pci_endpoints(inputs, provenance=_provenance())
    import non_destructive_image.reconstruction.independent_endpoint_information as module

    original = module.fit_linked_scalar_sequence
    calls: list[PCILinkedRawOperator] = []

    def fail_second(operator: PCILinkedRawOperator, *args: object, **kwargs: object):
        calls.append(operator)
        if operator is inputs[1].operator:
            raise RuntimeError("synthetic endpoint failure")
        return original(operator, *args, **kwargs)

    monkeypatch.setattr(module, "fit_linked_scalar_sequence", fail_second)
    draw = draw_and_refit_independent_endpoint_bootstrap(
        inputs,
        point_fit,
        _three_supports(inputs),
        draw_id=0,
        endpoint_rngs=(np.random.default_rng(801), np.random.default_rng(802)),
    )

    assert calls == [inputs[0].operator, inputs[1].operator]
    assert draw.fit.fit_success_mask == (True, False)
    assert np.all(np.isfinite(draw.postprocessed.values[:, 0]))
    assert np.all(np.isnan(draw.postprocessed.values[:, 1]))
    bootstrap = assemble_independent_endpoint_bootstrap(
        point_fit,
        (draw,),
    )
    assert bootstrap.fit_success_mask.tolist() == [[True, False]]
    assert np.all(bootstrap.supported_mask[0, 0])
    assert not np.any(bootstrap.supported_mask[0, 1])


def test_summary_reports_requested_and_supported_draw_counts() -> None:
    summary = summarise_independent_endpoint_information(
        _bootstrap_with_one_bz_failure(),
        confidence_level=0.68,
    )
    sigma_z = next(
        item for item in summary.observables if item.observable_name == "sigma_z_um"
    )
    assert (sigma_z.by.requested_draws, sigma_z.by.supported_draws) == (3, 3)
    assert (sigma_z.bz.requested_draws, sigma_z.bz.supported_draws) == (3, 2)
    assert (sigma_z.delta_b.requested_draws, sigma_z.delta_b.supported_draws) == (3, 2)
