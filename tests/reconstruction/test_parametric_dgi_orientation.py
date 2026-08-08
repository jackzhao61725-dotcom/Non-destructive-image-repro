from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import non_destructive_image.reconstruction.parametric_dgi_orientation as dgi_adapter

from non_destructive_image.reconstruction.contracts import (
    DetectorContract,
    ReconstructionGrid,
)
from non_destructive_image.reconstruction.free_radius_model import (
    FreeRadiusCompactDensityModel,
)
from non_destructive_image.reconstruction.independent_endpoint_information import (
    IndependentEndpointRawBlock,
)
from non_destructive_image.reconstruction.linked_scalar_fit import (
    LinkedScalarFitOptions,
)
from non_destructive_image.reconstruction.parameters import (
    SmoothTFParameters,
    from_internal,
    to_internal,
)
from non_destructive_image.reconstruction.parametric_dgi_orientation import (
    DGIParametricEndpointFitInput,
    DGIParametricEndpointRawBlock,
    DGIParametricOrientationProvenance,
    analyse_dgi_parametric_endpoint_diagnostics,
    analyse_dgi_parametric_identifiability,
    analyse_dgi_parametric_residuals,
    analyse_dgi_parametric_zero_density_null,
    fit_independent_parametric_dgi_endpoints,
)
from non_destructive_image.reconstruction.parametric_orientation_diagnostics import (
    ALL_OBSERVABLES,
    PRIMARY_OBSERVABLES,
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


RAW_ROLE_NAMES = (
    "atom_stop_000",
    "leakage_stop",
    "stop_dark",
    "open_reference",
    "open_dark",
)


def _grid() -> ReconstructionGrid:
    axis_m = (np.arange(24, dtype=float) - 12.0) * 0.5e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    return ReconstructionGrid.from_arrays(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        pupil=np.ones((24, 24), dtype=complex),
        bin_size=2,
        roi_mask=np.ones((12, 12), dtype=bool),
    )


def _response() -> ScalarOpticalResponseContract:
    return ScalarOpticalResponseContract(1.9e-16, 3.8e-18)


def _operator(
    grid: ReconstructionGrid,
    *,
    leakage_exposures: int = 1,
    response: ScalarOpticalResponseContract | None = None,
) -> DGILinkedRawOperator:
    return DGILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(500.0, 0.6),
        response=_response() if response is None else response,
        transfer=DGITransferContract(3.0),
        independent_exposures_by_role={
            "atom_stop": 1,
            "leakage_stop": leakage_exposures,
            "stop_dark": 1,
            "open_reference": 1,
            "open_dark": 1,
        },
        jacobian_batch_size=5,
    )


def _provenance(
    *,
    temporal_coupling_used: bool = False,
    generator_reference_used: bool = False,
) -> DGIParametricOrientationProvenance:
    return DGIParametricOrientationProvenance(
        contract_label="chapter_5_orientation_dgi_information_contract_v1",
        endpoint_labels=("B_parallel_y", "B_parallel_z"),
        field_orientations=("y", "z"),
        imaging_axis="x",
        independent_preparations=True,
        independent_raw_blocks=True,
        temporal_coupling_used=temporal_coupling_used,
        generator_reference_used=generator_reference_used,
    )


def _fit_inputs(
    *,
    max_nfev: int = 300,
) -> tuple[
    tuple[DGIParametricEndpointFitInput, DGIParametricEndpointFitInput],
    tuple[SmoothTFParameters, SmoothTFParameters],
]:
    grid = _grid()
    operators = (_operator(grid), _operator(grid))
    models = tuple(
        FreeRadiusCompactDensityModel.from_grid(
            y_grid_m=grid.y_grid_m,
            z_grid_m=grid.z_grid_m,
            profile_exponent=1.5,
        )
        for _ in range(2)
    )
    truths = (
        SmoothTFParameters(4.0e14, -0.25, 0.10, 3.7, 1.9),
        SmoothTFParameters(4.5e14, 0.20, -0.15, 3.0, 2.5),
    )
    nuisance = DGINuisanceValues(220.0, 0.4, 0.2, 0.9)
    base_starts = (
        SmoothTFParameters(3.7e14, -0.15, 0.05, 3.5, 2.0),
        SmoothTFParameters(4.3e14, 0.10, -0.10, 3.2, 2.3),
        SmoothTFParameters(3.2e14, -0.45, 0.25, 4.0, 1.6),
        SmoothTFParameters(5.0e14, 0.35, -0.25, 2.7, 2.8),
    )
    inputs: list[DGIParametricEndpointFitInput] = []
    for index, (operator, model, truth) in enumerate(
        zip(operators, models, truths, strict=True)
    ):
        prediction = operator.expected_linked_sequence_and_jacobian_model(
            model,
            [to_internal(truth)],
            nuisance,
        )
        label = ("B_parallel_y", "B_parallel_z")[index]
        orientation = ("y", "z")[index]
        raw = DGIParametricEndpointRawBlock(
            endpoint_label=label,
            field_orientation=orientation,
            role_names=prediction.role_names,
            role_owner_ids=tuple(f"{label}:{role}" for role in RAW_ROLE_NAMES),
            observed_electrons=prediction.expected_electrons,
        )
        inputs.append(
            DGIParametricEndpointFitInput(
                operator=operator,
                model=model,
                raw_block=raw,
                start_ids=("neutral", "wide", "narrow", "offset"),
                initial_parameter_vectors=tuple(
                    to_internal(value) for value in base_starts
                ),
                parameter_lower=to_internal(
                    SmoothTFParameters(1.0e13, -2.0, -2.0, 0.8, 0.8)
                ),
                parameter_upper=to_internal(
                    SmoothTFParameters(1.0e15, 2.0, 2.0, 5.0, 4.0)
                ),
                initial_nuisance=nuisance,
                nuisance_lower=[50.0, 0.0, 0.0, 0.2],
                nuisance_upper=[500.0, 10.0, 10.0, 2.0],
                options=LinkedScalarFitOptions(
                    irls_iterations=1,
                    max_nfev=max_nfev,
                    xtol=1e-10,
                    ftol=1e-10,
                    gtol=1e-10,
                ),
            )
        )
    return (inputs[0], inputs[1]), truths


def test_dgi_parametric_fit_and_all_endpoint_diagnostics() -> None:
    inputs, truths = _fit_inputs()
    fit = fit_independent_parametric_dgi_endpoints(
        inputs,
        provenance=_provenance(),
    )

    assert tuple(endpoint.status for endpoint in fit.endpoints) == (
        "success",
        "success",
    )
    for endpoint, truth in zip(fit.endpoints, truths, strict=True):
        assert len(endpoint.start_results) == 4
        assert endpoint.selected_fit is not None
        assert endpoint.physical_parameters is not None
        np.testing.assert_allclose(
            to_internal(endpoint.physical_parameters),
            to_internal(truth),
            rtol=2e-6,
            atol=2e-6,
        )

    residuals = analyse_dgi_parametric_residuals(inputs, fit)
    nulls = analyse_dgi_parametric_zero_density_null(inputs, fit)
    geometries = analyse_dgi_parametric_identifiability(inputs, fit)
    bundles = analyse_dgi_parametric_endpoint_diagnostics(inputs, fit)

    for residual in residuals:
        assert residual.status == "success"
        assert tuple(role.role_name for role in residual.roles) == RAW_ROLE_NAMES
        assert all(role.roi_rms < 2e-6 for role in residual.roles)
    for null in nulls:
        assert null.status == "success"
        assert null.nuisance_values is not None
        assert len(null.nuisance_values) == 4
        assert null.improvement_over_null is not None
        assert null.improvement_over_null > 0.0
    for geometry in geometries:
        assert geometry.status == "success"
        assert geometry.parameter_count == 9
        assert geometry.singular_values.shape == (9,)
        assert tuple(record.observable_name for record in geometry.records) == ALL_OBSERVABLES
        amplitude_supported = next(
            record.supported
            for record in geometry.records
            if record.observable_name == "A"
        )
        assert all(
            record.amplitude_gate_supported == amplitude_supported
            for record in geometry.records
            if record.observable_name in PRIMARY_OBSERVABLES
        )
        sigma_z = next(
            record
            for record in geometry.records
            if record.observable_name == "sigma_z_um"
        )
        assert sigma_z.primary_supported is True
        assert sigma_z.supported is False
        assert "amplitude_control_is_not_supported" in sigma_z.reasons
    for bundle in bundles:
        assert len(bundle.multistart) == len(ALL_OBSERVABLES)
        assert all(item.successful_start_count == 4 for item in bundle.multistart)


def test_aspect_geometry_uses_log_ratio_gradient_and_reporting_scale() -> None:
    inputs, _truths = _fit_inputs()
    fit = fit_independent_parametric_dgi_endpoints(inputs, provenance=_provenance())
    endpoint = fit.endpoints[0]
    selected = endpoint.selected_fit
    assert selected is not None
    _values, gradients = dgi_adapter._observable_gradients(
        endpoint,
        profile_exponent=inputs[0].model.profile_exponent,
    )
    internal = np.asarray(selected.density_coefficients[0], dtype=float)
    finite_difference = np.zeros(5, dtype=float)
    step = 1e-6
    for index in range(5):
        plus = internal.copy()
        minus = internal.copy()
        plus[index] += step
        minus[index] -= step
        plus_aspect = dgi_adapter.parametric_observables(
            from_internal(plus),
            profile_exponent=inputs[0].model.profile_exponent,
        ).aspect_ratio_y_over_z
        minus_aspect = dgi_adapter.parametric_observables(
            from_internal(minus),
            profile_exponent=inputs[0].model.profile_exponent,
        ).aspect_ratio_y_over_z
        finite_difference[index] = (
            np.log(plus_aspect) - np.log(minus_aspect)
        ) / (2.0 * step)

    np.testing.assert_allclose(
        gradients["aspect_ratio_y_over_z"],
        finite_difference,
        rtol=1e-9,
        atol=1e-9,
    )
    assert dgi_adapter._reporting_scale("aspect_ratio_y_over_z", 3.0) == pytest.approx(
        np.log(1.2)
    )


def test_zero_density_dgi_atom_role_is_the_blank_leakage_role() -> None:
    grid = _grid()
    operator = _operator(grid)
    nuisance = DGINuisanceValues(220.0, 0.4, 0.2, 0.9)

    role_names, expected = operator.expected_linked_sequence_from_density_maps(
        [np.zeros_like(grid.y_grid_m)],
        nuisance,
    )

    assert role_names == RAW_ROLE_NAMES
    np.testing.assert_array_equal(expected[0], expected[1])


def test_dgi_raw_block_rejects_wrong_roles_owners_and_nonfinite_data() -> None:
    arrays = tuple(np.ones((3, 3)) for _ in range(5))
    owners = tuple(f"owner-{index}" for index in range(5))
    with pytest.raises(ValueError, match="canonical DGI order"):
        DGIParametricEndpointRawBlock(
            "B_parallel_y",
            "y",
            (*RAW_ROLE_NAMES[:-1], "wrong"),
            owners,
            arrays,
        )
    with pytest.raises(ValueError, match="five unique"):
        DGIParametricEndpointRawBlock(
            "B_parallel_y",
            "y",
            RAW_ROLE_NAMES,
            ("same",) * 5,
            arrays,
        )
    nonfinite = list(arrays)
    nonfinite[0] = np.full((3, 3), np.nan)
    with pytest.raises(ValueError, match="finite"):
        DGIParametricEndpointRawBlock(
            "B_parallel_y",
            "y",
            RAW_ROLE_NAMES,
            owners,
            tuple(nonfinite),
        )


def test_dgi_fit_input_rejects_pci_operator_nuisance_and_raw_block() -> None:
    inputs, _truths = _fit_inputs()
    item = inputs[0]
    grid = item.operator.grid
    pci_operator = PCILinkedRawOperator(
        grid=grid,
        detector=item.operator.detector,
        response=_response(),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 1,
            "dark": 1,
        },
    )
    with pytest.raises(TypeError, match="DGILinkedRawOperator"):
        replace(item, operator=pci_operator)
    with pytest.raises(TypeError, match="DGINuisanceValues"):
        replace(item, initial_nuisance=PCINuisanceValues(220.0, 0.2))

    pci_prediction = pci_operator.expected_linked_sequence_and_jacobian_model(
        item.model,
        [item.initial_parameter_vectors[0]],
        PCINuisanceValues(220.0, 0.2),
    )
    pci_raw = IndependentEndpointRawBlock(
        endpoint_label="B_parallel_y",
        field_orientation="y",
        role_names=pci_prediction.role_names,
        role_owner_ids=("pci:atom", "pci:bright", "pci:dark"),
        observed_electrons=pci_prediction.expected_electrons,
    )
    with pytest.raises(TypeError, match="DGIParametricEndpointRawBlock"):
        replace(item, raw_block=pci_raw)


def test_dgi_fit_input_rejects_noncanonical_exposures_and_nuisance_domain() -> None:
    inputs, _truths = _fit_inputs()
    with pytest.raises(ValueError, match="one exposure"):
        replace(inputs[0], operator=_operator(inputs[0].operator.grid, leakage_exposures=2))
    with pytest.raises(ValueError, match="nuisance bounds"):
        replace(inputs[0], nuisance_lower=[0.0, 0.0, 0.0, 0.0])


def test_pair_requires_ten_unique_owner_ids() -> None:
    inputs, _truths = _fit_inputs()
    second = inputs[1]
    duplicate_owners = (
        inputs[0].raw_block.role_owner_ids[0],
        *second.raw_block.role_owner_ids[1:],
    )
    duplicate_raw = replace(second.raw_block, role_owner_ids=duplicate_owners)

    with pytest.raises(ValueError, match="ten DGI orientation"):
        fit_independent_parametric_dgi_endpoints(
            (inputs[0], replace(second, raw_block=duplicate_raw)),
            provenance=_provenance(),
        )


def test_pair_allows_predeclared_orientation_dependent_optical_responses() -> None:
    inputs, _truths = _fit_inputs()
    second = inputs[1]
    response = ScalarOpticalResponseContract(2.1e-16, 4.0e-18)
    operator = _operator(second.operator.grid, response=response)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        second.model,
        [second.initial_parameter_vectors[0]],
        second.initial_nuisance,
    )
    raw = replace(second.raw_block, observed_electrons=prediction.expected_electrons)

    fit = fit_independent_parametric_dgi_endpoints(
        (inputs[0], replace(second, operator=operator, raw_block=raw)),
        provenance=_provenance(),
    )

    assert len(fit.endpoints) == 2


def test_pair_rejects_cross_endpoint_inference_design_drift() -> None:
    inputs, _truths = _fit_inputs()
    first, second = inputs

    with pytest.raises(ValueError, match="start-id policies"):
        fit_independent_parametric_dgi_endpoints(
            (
                first,
                replace(
                    second,
                    start_ids=("other", *second.start_ids[1:]),
                ),
            ),
            provenance=_provenance(),
        )

    wider_upper = np.asarray(second.parameter_upper).copy()
    wider_upper[0] += 0.1
    with pytest.raises(ValueError, match="parameter upper bounds"):
        fit_independent_parametric_dgi_endpoints(
            (first, replace(second, parameter_upper=wider_upper)),
            provenance=_provenance(),
        )

    with pytest.raises(ValueError, match="solver options"):
        fit_independent_parametric_dgi_endpoints(
            (
                first,
                replace(
                    second,
                    options=replace(
                        second.options,
                        max_nfev=second.options.max_nfev + 1,
                    ),
                ),
            ),
            provenance=_provenance(),
        )


def test_residual_identity_rejects_a_different_raw_block() -> None:
    inputs, _truths = _fit_inputs()
    fit = fit_independent_parametric_dgi_endpoints(inputs, provenance=_provenance())
    changed_arrays = list(inputs[0].raw_block.observed_electrons)
    changed_arrays[0] = changed_arrays[0] + 1.0
    changed_raw = replace(
        inputs[0].raw_block,
        observed_electrons=tuple(changed_arrays),
    )

    with pytest.raises(ValueError, match="does not belong"):
        analyse_dgi_parametric_residuals(
            (replace(inputs[0], raw_block=changed_raw), inputs[1]),
            fit,
        )


def test_failed_dgi_fit_retains_all_four_terminals() -> None:
    inputs, _truths = _fit_inputs(max_nfev=1)
    fit = fit_independent_parametric_dgi_endpoints(inputs, provenance=_provenance())

    for endpoint in fit.endpoints:
        assert endpoint.status == "fit_failure"
        assert len(endpoint.start_results) == 4
        assert all(result.status == "fit_failure" for result in endpoint.start_results)
        assert endpoint.selected_fit is None
        assert endpoint.observables is None


def test_nonfinite_objective_is_preserved_as_json_safe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _truths = _fit_inputs()
    baseline = fit_independent_parametric_dgi_endpoints(
        inputs,
        provenance=_provenance(),
    )
    selected = baseline.endpoints[0].selected_fit
    assert selected is not None
    nonfinite = replace(
        selected,
        diagnostics=replace(
            selected.diagnostics,
            weighted_chi_square=float("nan"),
        ),
    )
    monkeypatch.setattr(
        dgi_adapter,
        "fit_linked_scalar_sequence",
        lambda *_args, **_kwargs: nonfinite,
    )

    fit = fit_independent_parametric_dgi_endpoints(
        inputs,
        provenance=_provenance(),
    )

    for endpoint in fit.endpoints:
        assert endpoint.status == "fit_failure"
        assert all(start.weighted_chi_square is None for start in endpoint.start_results)
        assert all(start.fit_result is None for start in endpoint.start_results)
        assert all("non-finite" in start.message for start in endpoint.start_results)


def test_single_frozen_primary_start_is_allowed_for_later_bootstrap() -> None:
    inputs, _truths = _fit_inputs()
    single_start_inputs = tuple(
        replace(
            item,
            start_ids=("frozen_primary",),
            initial_parameter_vectors=(item.initial_parameter_vectors[index],),
            initial_nuisance=DGINuisanceValues(
                220.0 + 10.0 * index,
                0.4,
                0.2,
                0.9,
            ),
        )
        for index, item in enumerate(inputs)
    )

    assert not np.array_equal(
        single_start_inputs[0].initial_parameter_vectors[0],
        single_start_inputs[1].initial_parameter_vectors[0],
    )
    assert single_start_inputs[0].initial_nuisance != (
        single_start_inputs[1].initial_nuisance
    )

    fit = fit_independent_parametric_dgi_endpoints(
        single_start_inputs,
        provenance=_provenance(),
    )

    assert all(endpoint.status == "success" for endpoint in fit.endpoints)
    assert all(len(endpoint.start_results) == 1 for endpoint in fit.endpoints)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("temporal_coupling_used", "temporal coupling"),
        ("generator_reference_used", "generator references"),
    ],
)
def test_dgi_provenance_forbids_coupled_or_generator_assisted_fits(
    field: str,
    match: str,
) -> None:
    values = {
        "temporal_coupling_used": False,
        "generator_reference_used": False,
    }
    values[field] = True
    with pytest.raises(ValueError, match=match):
        _provenance(**values)
