from __future__ import annotations

import numpy as np
import pytest

from non_destructive_image.reconstruction.contracts import (
    DetectorContract,
    ReconstructionGrid,
)
from non_destructive_image.reconstruction.independent_endpoint_information import (
    IndependentEndpointRawBlock,
)
from non_destructive_image.reconstruction.linked_scalar_fit import (
    LinkedScalarFitOptions,
)
from non_destructive_image.reconstruction.free_radius_model import (
    FreeRadiusCompactDensityModel,
)
from non_destructive_image.reconstruction.parameters import (
    SmoothTFParameters,
    to_internal,
)
from non_destructive_image.reconstruction.parametric_orientation import (
    ParametricEndpointFitInput,
    ParametricOrientationProvenance,
    fit_independent_parametric_pci_endpoints,
    parametric_observables,
)
from non_destructive_image.reconstruction.parametric_orientation_diagnostics import (
    ALL_OBSERVABLES,
    PRIMARY_OBSERVABLES,
    analyse_parametric_endpoint_diagnostics,
    analyse_parametric_identifiability,
    analyse_parametric_multistart,
    analyse_parametric_residuals,
    analyse_parametric_zero_density_null,
)
from non_destructive_image.reconstruction.scalar_measurements import (
    PCILinkedRawOperator,
    PCINuisanceValues,
    PCITransferContract,
    ScalarOpticalResponseContract,
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


def _operator(grid: ReconstructionGrid) -> PCILinkedRawOperator:
    return PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(140.0, 0.7),
        response=ScalarOpticalResponseContract(1.9e-16, 3.8e-18),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 1,
            "dark": 1,
        },
        jacobian_batch_size=5,
    )


def _provenance() -> ParametricOrientationProvenance:
    return ParametricOrientationProvenance(
        contract_label="chapter_5_orientation_information_contract_v2",
        endpoint_labels=("B_parallel_y", "B_parallel_z"),
        field_orientations=("y", "z"),
        imaging_axis="x",
        independent_preparations=True,
        independent_raw_blocks=True,
        temporal_coupling_used=False,
        generator_reference_used=False,
    )


def _fit_inputs(
    *,
    max_nfev: int = 100,
    second_exponent: float = 1.5,
) -> tuple[
    tuple[ParametricEndpointFitInput, ParametricEndpointFitInput],
    tuple[SmoothTFParameters, SmoothTFParameters],
]:
    grid = _grid()
    operators = (_operator(grid), _operator(grid))
    models = (
        FreeRadiusCompactDensityModel.from_grid(
            y_grid_m=grid.y_grid_m,
            z_grid_m=grid.z_grid_m,
            profile_exponent=1.5,
        ),
        FreeRadiusCompactDensityModel.from_grid(
            y_grid_m=grid.y_grid_m,
            z_grid_m=grid.z_grid_m,
            profile_exponent=second_exponent,
        ),
    )
    truths = (
        SmoothTFParameters(4.0e14, -0.3, 0.1, 3.6, 1.7),
        SmoothTFParameters(4.6e14, 0.2, -0.15, 3.0, 2.2),
    )
    nuisance = PCINuisanceValues(140.0, 0.4)
    inputs: list[ParametricEndpointFitInput] = []
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
        raw = IndependentEndpointRawBlock(
            endpoint_label=label,
            field_orientation=orientation,
            role_names=prediction.role_names,
            role_owner_ids=(
                f"{label}:atom",
                f"{label}:bright_reference",
                f"{label}:dark",
            ),
            observed_electrons=prediction.expected_electrons,
        )
        starts = (
            SmoothTFParameters(3.5e14, 0.0, 0.0, 3.2, 2.0),
            SmoothTFParameters(2.5e14, -0.5, 0.2, 4.0, 1.4),
        )
        inputs.append(
            ParametricEndpointFitInput(
                operator=operator,
                model=model,
                raw_block=raw,
                start_ids=("neutral", "alternate"),
                initial_parameter_vectors=tuple(to_internal(value) for value in starts),
                parameter_lower=to_internal(
                    SmoothTFParameters(1.0e13, -2.0, -2.0, 1.0, 0.5)
                ),
                parameter_upper=to_internal(
                    SmoothTFParameters(1.0e15, 2.0, 2.0, 5.0, 3.0)
                ),
                initial_nuisance=PCINuisanceValues(130.0, 0.2),
                nuisance_lower=[80.0, 0.0],
                nuisance_upper=[200.0, 5.0],
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


def test_independent_parametric_fit_recovers_noiseless_endpoints() -> None:
    inputs, truths = _fit_inputs()
    result = fit_independent_parametric_pci_endpoints(inputs, provenance=_provenance())

    assert tuple(value.status for value in result.endpoints) == ("success", "success")
    for endpoint, truth in zip(result.endpoints, truths, strict=True):
        assert len(endpoint.start_results) == 2
        assert all(value.status == "success" for value in endpoint.start_results)
        assert endpoint.selected_start_id in ("neutral", "alternate")
        assert endpoint.physical_parameters is not None
        assert endpoint.observables is not None
        np.testing.assert_allclose(
            to_internal(endpoint.physical_parameters),
            to_internal(truth),
            rtol=1e-6,
            atol=2e-6,
        )
        expected = parametric_observables(truth, profile_exponent=1.5)
        assert endpoint.observables.A == pytest.approx(expected.A, rel=2e-6)
        assert endpoint.observables.sigma_y_um == pytest.approx(
            expected.sigma_y_um,
            rel=2e-6,
        )
        assert endpoint.observables.sigma_z_um == pytest.approx(
            expected.sigma_z_um,
            rel=2e-6,
        )


def test_pair_validation_rejects_different_profile_families() -> None:
    inputs, _truths = _fit_inputs(second_exponent=2.0)

    with pytest.raises(ValueError, match="same profile exponent"):
        fit_independent_parametric_pci_endpoints(inputs, provenance=_provenance())


def test_fit_failure_retains_all_starts_without_publishing_observables() -> None:
    inputs, _truths = _fit_inputs(max_nfev=1)
    result = fit_independent_parametric_pci_endpoints(inputs, provenance=_provenance())

    assert tuple(value.status for value in result.endpoints) == (
        "fit_failure",
        "fit_failure",
    )
    for endpoint in result.endpoints:
        assert len(endpoint.start_results) == 2
        assert all(value.status == "fit_failure" for value in endpoint.start_results)
        assert endpoint.selected_fit is None
        assert endpoint.physical_parameters is None
        assert endpoint.observables is None


def test_provenance_forbids_generator_reference_access() -> None:
    with pytest.raises(ValueError, match="generator references"):
        ParametricOrientationProvenance(
            contract_label="chapter_5_orientation_information_contract_v2",
            endpoint_labels=("B_parallel_y", "B_parallel_z"),
            field_orientations=("y", "z"),
            imaging_axis="x",
            independent_preparations=True,
            independent_raw_blocks=True,
            temporal_coupling_used=False,
            generator_reference_used=True,
        )


@pytest.mark.parametrize("profile_exponent", [1.25, 1.5, 2.0])
def test_observable_formula_matches_compact_profile_moments(
    profile_exponent: float,
) -> None:
    parameters = SmoothTFParameters(4.0e14, 0.0, 0.0, 9.0, 3.0)
    observables = parametric_observables(
        parameters,
        profile_exponent=profile_exponent,
    )

    assert observables.A == pytest.approx(
        np.pi
        / (profile_exponent + 1.0)
        * parameters.column_density_peak_m2
        * parameters.radius_y_um
        * parameters.radius_z_um
        * 1e-12
    )
    assert observables.sigma_y_um == pytest.approx(
        parameters.radius_y_um / np.sqrt(2.0 * (profile_exponent + 2.0))
    )
    assert observables.sigma_z_um == pytest.approx(
        parameters.radius_z_um / np.sqrt(2.0 * (profile_exponent + 2.0))
    )
    assert observables.aspect_ratio_y_over_z == pytest.approx(3.0)


def test_parametric_diagnostics_verify_raw_roles_and_local_geometry() -> None:
    inputs, _truths = _fit_inputs()
    fit = fit_independent_parametric_pci_endpoints(inputs, provenance=_provenance())

    residuals = analyse_parametric_residuals(inputs, fit)
    nulls = analyse_parametric_zero_density_null(inputs, fit)
    geometries = analyse_parametric_identifiability(inputs, fit)

    for residual in residuals:
        assert residual.status == "success"
        assert tuple(role.role_name for role in residual.roles) == (
            "atom_000",
            "bright_reference",
            "dark",
        )
        assert all(role.roi_rms < 2e-6 for role in residual.roles)
    for null in nulls:
        assert null.status == "success"
        assert null.improvement_over_null is not None
        assert null.improvement_over_null > 0.0
    for geometry in geometries:
        assert geometry.status == "success"
        assert geometry.parameter_count == 7
        assert geometry.primary_data_rank == 7
        assert tuple(item.observable_name for item in geometry.records) == ALL_OBSERVABLES
        amplitude_supported = next(
            item.supported
            for item in geometry.records
            if item.observable_name == "A"
        )
        assert all(
            item.amplitude_gate_supported == amplitude_supported
            for item in geometry.records
            if item.observable_name in PRIMARY_OBSERVABLES
        )


def test_complete_diagnostic_bundle_retains_multistart_stability() -> None:
    inputs, _truths = _fit_inputs()
    fit = fit_independent_parametric_pci_endpoints(inputs, provenance=_provenance())

    bundles = analyse_parametric_endpoint_diagnostics(inputs, fit)

    for bundle in bundles:
        assert bundle.endpoint_label in ("B_parallel_y", "B_parallel_z")
        assert tuple(item.observable_name for item in bundle.multistart) == ALL_OBSERVABLES
        assert all(item.successful_start_count == 2 for item in bundle.multistart)
        assert all(item.grade == "stable" for item in bundle.multistart)


def test_failed_fit_publishes_no_geometry_or_false_support() -> None:
    inputs, _truths = _fit_inputs(max_nfev=1)
    fit = fit_independent_parametric_pci_endpoints(inputs, provenance=_provenance())

    bundles = analyse_parametric_endpoint_diagnostics(inputs, fit)

    for bundle in bundles:
        assert bundle.residuals.status == "fit_failure"
        assert bundle.zero_density_null.status == "fit_failure"
        assert bundle.geometry.status == "fit_failure"
        assert bundle.geometry.records == ()
        assert all(item.grade == "unresolved" for item in bundle.multistart)


def test_multistart_requires_two_successful_terminals() -> None:
    inputs, _truths = _fit_inputs(max_nfev=1)
    fit = fit_independent_parametric_pci_endpoints(inputs, provenance=_provenance())

    records = analyse_parametric_multistart(
        fit.endpoints[0],
        profile_exponent=1.5,
    )

    assert all(item.maximum_normalised_shift is None for item in records)
    assert all(item.grade == "unresolved" for item in records)
