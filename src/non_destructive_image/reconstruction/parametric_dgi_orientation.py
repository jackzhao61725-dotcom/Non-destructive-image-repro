"""Independent-endpoint parametric DGI inference for orientation studies.

The adapter keeps the existing linked raw-count likelihood and compact density
model.  It adds only the DGI-specific five-role ownership contract, four optical
nuisance parameters and endpoint-local diagnostics needed for a target-free
``B_parallel_y``/``B_parallel_z`` comparison.  The two endpoints are fitted
independently and are paired only after both terminal records exist.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .free_radius_model import FreeRadiusCompactDensityModel
from .linked_scalar_fit import (
    LinkedRawObservation,
    LinkedScalarFitOptions,
    fit_linked_scalar_sequence,
    nuisance_vector,
)
from .parameters import from_internal
from .parametric_orientation import (
    ParametricEndpointFit,
    ParametricStartResult,
    parametric_observables,
)
from .parametric_orientation_diagnostics import (
    ALL_OBSERVABLES,
    PRIMARY_OBSERVABLES,
    ParametricEndpointGeometry,
    ParametricMultistartStability,
    ParametricObservableGeometry,
    ParametricRawRoleResidual,
    analyse_parametric_multistart,
)
from .scalar_measurements import (
    DGILinkedRawOperator,
    DGINuisanceValues,
)


FloatArray = NDArray[np.floating]
_ENDPOINT_LABELS = ("B_parallel_y", "B_parallel_z")
_FIELD_ORIENTATIONS = ("y", "z")
_RAW_ROLE_NAMES = (
    "atom_stop_000",
    "leakage_stop",
    "stop_dark",
    "open_reference",
    "open_dark",
)
_EXPOSURES = {
    "atom_stop": 1,
    "leakage_stop": 1,
    "stop_dark": 1,
    "open_reference": 1,
    "open_dark": 1,
}
_CONTRACT_LABEL = "independent_orientation_dgi_v1"


def _immutable(values: ArrayLike) -> FloatArray:
    source = np.asarray(values, dtype=float)
    result = np.frombuffer(source.tobytes(order="C"), dtype=float)
    return result.reshape(source.shape)


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


@dataclass(frozen=True, eq=False)
class DGIParametricEndpointRawBlock:
    """One endpoint's five independently owned DGI raw-count roles."""

    endpoint_label: str
    field_orientation: str
    role_names: tuple[str, ...]
    role_owner_ids: tuple[str, ...]
    observed_electrons: tuple[ArrayLike, ...]
    unit: str = "electrons"

    def __post_init__(self) -> None:
        label = _nonempty_text(self.endpoint_label, name="endpoint label")
        orientation = _nonempty_text(
            self.field_orientation,
            name="field orientation",
        )
        if label not in _ENDPOINT_LABELS:
            raise ValueError("unknown DGI orientation endpoint label")
        expected_orientation = _FIELD_ORIENTATIONS[_ENDPOINT_LABELS.index(label)]
        if orientation != expected_orientation:
            raise ValueError("endpoint label and field orientation disagree")
        if tuple(self.role_names) != _RAW_ROLE_NAMES:
            raise ValueError("endpoint raw roles must use canonical DGI order")
        owners = tuple(
            _nonempty_text(value, name="raw-role owner id")
            for value in self.role_owner_ids
        )
        if len(owners) != 5 or len(set(owners)) != 5:
            raise ValueError("five unique raw-role owner ids are required per endpoint")
        arrays = tuple(_immutable(value) for value in self.observed_electrons)
        if len(arrays) != 5 or any(array.ndim != 2 for array in arrays):
            raise ValueError("endpoint DGI raw roles must contain five 2D arrays")
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("endpoint raw-role arrays must share one camera shape")
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("endpoint raw electron values must be finite")
        if self.unit != "electrons":
            raise ValueError("endpoint raw-count unit must be electrons")
        object.__setattr__(self, "endpoint_label", label)
        object.__setattr__(self, "field_orientation", orientation)
        object.__setattr__(self, "role_names", tuple(self.role_names))
        object.__setattr__(self, "role_owner_ids", owners)
        object.__setattr__(self, "observed_electrons", arrays)

    def as_linked_observation(self) -> LinkedRawObservation:
        """Return the five raw roles without relabelling or combining them."""

        return LinkedRawObservation(self.role_names, self.observed_electrons)


@dataclass(frozen=True)
class DGIParametricOrientationProvenance:
    """Identity and factorisation boundary for the DGI orientation adapter."""

    contract_label: str
    endpoint_labels: tuple[str, str]
    field_orientations: tuple[str, str]
    imaging_axis: str
    independent_preparations: bool
    independent_raw_blocks: bool
    temporal_coupling_used: bool
    generator_reference_used: bool

    def __post_init__(self) -> None:
        if self.contract_label != _CONTRACT_LABEL:
            raise ValueError("DGI parametric orientation contract label changed")
        if tuple(self.endpoint_labels) != _ENDPOINT_LABELS:
            raise ValueError("endpoint labels must follow canonical By/Bz order")
        if tuple(self.field_orientations) != _FIELD_ORIENTATIONS:
            raise ValueError("field orientations must follow canonical y/z order")
        if self.imaging_axis != "x":
            raise ValueError("orientation DGI imaging axis must be x")
        if self.independent_preparations is not True:
            raise ValueError("orientation endpoints must be independently prepared")
        if self.independent_raw_blocks is not True:
            raise ValueError("orientation endpoints must own independent raw blocks")
        if self.temporal_coupling_used is not False:
            raise ValueError("temporal coupling is forbidden")
        if self.generator_reference_used is not False:
            raise ValueError("generator references are forbidden during endpoint fitting")


@dataclass(frozen=True, eq=False)
class DGIParametricEndpointFitInput:
    """One endpoint's DGI data, compact model, declared starts and fit bounds."""

    operator: DGILinkedRawOperator
    model: FreeRadiusCompactDensityModel
    raw_block: DGIParametricEndpointRawBlock
    start_ids: tuple[str, ...]
    initial_parameter_vectors: tuple[ArrayLike, ...]
    parameter_lower: ArrayLike
    parameter_upper: ArrayLike
    initial_nuisance: DGINuisanceValues
    nuisance_lower: ArrayLike
    nuisance_upper: ArrayLike
    options: LinkedScalarFitOptions

    def __post_init__(self) -> None:
        if not isinstance(self.operator, DGILinkedRawOperator):
            raise TypeError("DGI orientation fits require DGILinkedRawOperator")
        if not isinstance(self.model, FreeRadiusCompactDensityModel):
            raise TypeError("model must be FreeRadiusCompactDensityModel")
        if not isinstance(self.raw_block, DGIParametricEndpointRawBlock):
            raise TypeError("raw_block must be DGIParametricEndpointRawBlock")
        if not isinstance(self.initial_nuisance, DGINuisanceValues):
            raise TypeError("initial_nuisance must be DGINuisanceValues")
        if not isinstance(self.options, LinkedScalarFitOptions):
            raise TypeError("options must be LinkedScalarFitOptions")
        if dict(self.operator.independent_exposures_by_role) != _EXPOSURES:
            raise ValueError("DGI orientation fits require one exposure per raw role")
        if self.raw_block.observed_electrons[0].shape != self.operator.grid.camera_shape:
            raise ValueError("endpoint raw camera shape differs from its operator")
        if not np.array_equal(
            self.model.y_grid_m,
            self.operator.grid.y_grid_m,
        ) or not np.array_equal(
            self.model.z_grid_m,
            self.operator.grid.z_grid_m,
        ):
            raise ValueError("endpoint density model and operator grids differ")

        start_ids = tuple(
            _nonempty_text(value, name="start id") for value in self.start_ids
        )
        if not start_ids or len(set(start_ids)) != len(start_ids):
            raise ValueError("start ids must be non-empty and unique")
        starts = tuple(_immutable(value) for value in self.initial_parameter_vectors)
        if len(starts) != len(start_ids) or any(
            value.shape != (self.model.parameter_count,)
            or np.any(~np.isfinite(value))
            for value in starts
        ):
            raise ValueError("one finite five-parameter vector is required per start")
        lower = _immutable(self.parameter_lower)
        upper = _immutable(self.parameter_upper)
        if (
            lower.shape != (self.model.parameter_count,)
            or upper.shape != lower.shape
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(upper <= lower)
            or any(np.any(value < lower) or np.any(value > upper) for value in starts)
        ):
            raise ValueError("parameter bounds or starts are invalid")
        nuisance_lower = _immutable(self.nuisance_lower)
        nuisance_upper = _immutable(self.nuisance_upper)
        nuisance_initial = nuisance_vector(self.initial_nuisance)
        if (
            nuisance_lower.shape != (4,)
            or nuisance_upper.shape != (4,)
            or np.any(~np.isfinite(nuisance_lower))
            or np.any(~np.isfinite(nuisance_upper))
            or np.any(nuisance_lower < 0.0)
            or nuisance_lower[0] <= 0.0
            or nuisance_lower[3] <= 0.0
            or np.any(nuisance_upper <= nuisance_lower)
            or np.any(nuisance_initial < nuisance_lower)
            or np.any(nuisance_initial > nuisance_upper)
        ):
            raise ValueError("DGI nuisance bounds or initial values are invalid")
        object.__setattr__(self, "start_ids", start_ids)
        object.__setattr__(self, "initial_parameter_vectors", starts)
        object.__setattr__(self, "parameter_lower", lower)
        object.__setattr__(self, "parameter_upper", upper)
        object.__setattr__(self, "nuisance_lower", nuisance_lower)
        object.__setattr__(self, "nuisance_upper", nuisance_upper)


@dataclass(frozen=True)
class DGIParametricOrientationPairFit:
    """Two independent DGI endpoint fits combined as an ordered record."""

    endpoints: tuple[ParametricEndpointFit, ParametricEndpointFit]
    provenance: DGIParametricOrientationProvenance

    def __post_init__(self) -> None:
        if tuple(value.endpoint_label for value in self.endpoints) != _ENDPOINT_LABELS:
            raise ValueError("DGI pair fit must follow canonical By/Bz order")


@dataclass(frozen=True)
class DGIParametricEndpointResiduals:
    """Whitened raw-role residual diagnostics for one DGI endpoint."""

    endpoint_label: str
    status: Literal["success", "fit_failure"]
    roles: tuple[ParametricRawRoleResidual, ...]
    message: str

    def __post_init__(self) -> None:
        if self.status not in ("success", "fit_failure"):
            raise ValueError("unknown DGI endpoint residual status")
        if self.status == "success" and tuple(
            role.role_name for role in self.roles
        ) != _RAW_ROLE_NAMES:
            raise ValueError("successful DGI residual diagnostics require five roles")
        if self.status == "fit_failure" and self.roles:
            raise ValueError("failed residual diagnostics cannot publish role statistics")


@dataclass(frozen=True)
class DGIParametricZeroDensityNull:
    """Endpoint-local zero-density refit of all four DGI nuisances."""

    endpoint_label: str
    status: Literal["success", "fit_failure"]
    nuisance_values: tuple[float, float, float, float] | None
    null_weighted_chi_square: float | None
    fitted_weighted_chi_square: float | None
    improvement_over_null: float | None
    message: str

    def __post_init__(self) -> None:
        values = (
            self.null_weighted_chi_square,
            self.fitted_weighted_chi_square,
            self.improvement_over_null,
        )
        if self.status not in ("success", "fit_failure"):
            raise ValueError("unknown DGI zero-density diagnostic status")
        if self.status == "success":
            if self.nuisance_values is None or any(value is None for value in values):
                raise ValueError("successful DGI zero-density diagnostic is incomplete")
            combined = np.asarray((*self.nuisance_values, *values), dtype=float)
            if np.any(~np.isfinite(combined)):
                raise ValueError("DGI zero-density diagnostic values must be finite")
        elif self.nuisance_values is not None or any(value is not None for value in values):
            raise ValueError("failed zero-density diagnostic cannot publish values")


@dataclass(frozen=True)
class DGIParametricEndpointDiagnostics:
    """Residual, null, geometry and multistart records for one DGI endpoint."""

    endpoint_label: str
    residuals: DGIParametricEndpointResiduals
    zero_density_null: DGIParametricZeroDensityNull
    geometry: ParametricEndpointGeometry
    multistart: tuple[ParametricMultistartStability, ...]

    def __post_init__(self) -> None:
        labels = (
            self.residuals.endpoint_label,
            self.zero_density_null.endpoint_label,
            self.geometry.endpoint_label,
        )
        if any(value != self.endpoint_label for value in labels):
            raise ValueError("DGI endpoint diagnostic labels disagree")
        if tuple(item.observable_name for item in self.multistart) != ALL_OBSERVABLES:
            raise ValueError("DGI endpoint multistart diagnostics are incomplete")


def _same_acquisition_design(
    left: DGILinkedRawOperator,
    right: DGILinkedRawOperator,
) -> bool:
    return (
        left.detector == right.detector
        and left.transfer == right.transfer
        and dict(left.independent_exposures_by_role)
        == dict(right.independent_exposures_by_role)
        and np.array_equal(left.grid.y_grid_m, right.grid.y_grid_m)
        and np.array_equal(left.grid.z_grid_m, right.grid.z_grid_m)
        and np.array_equal(left.grid.pupil, right.grid.pupil)
        and np.array_equal(left.grid.roi_mask, right.grid.roi_mask)
    )


def _validate_pair_inputs(
    inputs: tuple[DGIParametricEndpointFitInput, DGIParametricEndpointFitInput],
    provenance: DGIParametricOrientationProvenance,
) -> None:
    if not isinstance(provenance, DGIParametricOrientationProvenance):
        raise TypeError("DGI orientation provenance has the wrong type")
    if tuple(value.raw_block.endpoint_label for value in inputs) != _ENDPOINT_LABELS:
        raise ValueError("DGI fit inputs must follow canonical By/Bz order")
    if inputs[0].operator is inputs[1].operator:
        raise ValueError("DGI orientation endpoints require distinct operators")
    if not _same_acquisition_design(inputs[0].operator, inputs[1].operator):
        raise ValueError("DGI endpoint acquisition designs differ")
    if inputs[0].model.profile_exponent != inputs[1].model.profile_exponent:
        raise ValueError("DGI endpoints must use the same profile exponent")
    if tuple(inputs[0].model.parameter_names) != tuple(
        inputs[1].model.parameter_names
    ):
        raise ValueError("DGI endpoint parameterisations differ")
    if inputs[0].start_ids != inputs[1].start_ids:
        raise ValueError("DGI endpoint start-id policies differ")
    for label, left, right in (
        (
            "parameter lower bounds",
            inputs[0].parameter_lower,
            inputs[1].parameter_lower,
        ),
        (
            "parameter upper bounds",
            inputs[0].parameter_upper,
            inputs[1].parameter_upper,
        ),
        (
            "nuisance lower bounds",
            inputs[0].nuisance_lower,
            inputs[1].nuisance_lower,
        ),
        (
            "nuisance upper bounds",
            inputs[0].nuisance_upper,
            inputs[1].nuisance_upper,
        ),
    ):
        if not np.array_equal(left, right):
            raise ValueError(f"DGI endpoint {label} differ")
    if inputs[0].options != inputs[1].options:
        raise ValueError("DGI endpoint solver options differ")
    owners = tuple(
        owner for item in inputs for owner in item.raw_block.role_owner_ids
    )
    if len(owners) != 10 or len(set(owners)) != 10:
        raise ValueError("all ten DGI orientation raw-role owner ids must be unique")


def _fit_one_endpoint(item: DGIParametricEndpointFitInput) -> ParametricEndpointFit:
    attempts: list[ParametricStartResult] = []
    for start_id, initial in zip(
        item.start_ids,
        item.initial_parameter_vectors,
        strict=True,
    ):
        try:
            result = fit_linked_scalar_sequence(
                item.operator,
                item.model,
                item.raw_block.as_linked_observation(),
                initial_density_coefficients=initial[None, :],
                density_parameter_lower=item.parameter_lower,
                density_coefficient_upper=item.parameter_upper,
                initial_nuisance=item.initial_nuisance,
                nuisance_lower=item.nuisance_lower,
                nuisance_upper=item.nuisance_upper,
                regularisation=None,
                options=item.options,
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            attempts.append(
                ParametricStartResult(
                    start_id=start_id,
                    status="fit_failure",
                    message=f"{type(exc).__name__}: {exc}",
                    weighted_chi_square=None,
                    fit_result=None,
                )
            )
            continue
        finite_objective = bool(np.isfinite(result.diagnostics.weighted_chi_square))
        if not finite_objective:
            attempts.append(
                ParametricStartResult(
                    start_id=start_id,
                    status="fit_failure",
                    message=(
                        f"{result.diagnostics.message}; non-finite "
                        "weighted_chi_square"
                    ),
                    weighted_chi_square=None,
                    fit_result=None,
                )
            )
            continue
        success = bool(result.diagnostics.success)
        attempts.append(
            ParametricStartResult(
                start_id=start_id,
                status="success" if success else "fit_failure",
                message=result.diagnostics.message,
                weighted_chi_square=float(result.diagnostics.weighted_chi_square),
                fit_result=result,
            )
        )

    successful = [attempt for attempt in attempts if attempt.status == "success"]
    if not successful:
        return ParametricEndpointFit(
            endpoint_label=item.raw_block.endpoint_label,
            field_orientation=item.raw_block.field_orientation,
            status="fit_failure",
            message="no predeclared DGI start produced a successful finite fit",
            start_results=tuple(attempts),
            selected_start_id=None,
            selected_fit=None,
            physical_parameters=None,
            observables=None,
        )
    selected = min(successful, key=lambda attempt: float(attempt.weighted_chi_square))
    if selected.fit_result is None:
        raise RuntimeError("successful selected DGI start lost its fit result")
    parameters = from_internal(selected.fit_result.density_coefficients[0])
    observables = parametric_observables(
        parameters,
        profile_exponent=item.model.profile_exponent,
    )
    return ParametricEndpointFit(
        endpoint_label=item.raw_block.endpoint_label,
        field_orientation=item.raw_block.field_orientation,
        status="success",
        message=f"selected start {selected.start_id}",
        start_results=tuple(attempts),
        selected_start_id=selected.start_id,
        selected_fit=selected.fit_result,
        physical_parameters=parameters,
        observables=observables,
    )


def fit_independent_parametric_dgi_endpoints(
    inputs: tuple[DGIParametricEndpointFitInput, DGIParametricEndpointFitInput],
    *,
    provenance: DGIParametricOrientationProvenance,
) -> DGIParametricOrientationPairFit:
    """Fit By and Bz DGI blocks separately and retain every declared start."""

    if len(inputs) != 2:
        raise ValueError("DGI orientation fitting requires exactly two endpoints")
    _validate_pair_inputs(inputs, provenance)
    return DGIParametricOrientationPairFit(
        endpoints=(_fit_one_endpoint(inputs[0]), _fit_one_endpoint(inputs[1])),
        provenance=provenance,
    )


def _validate_pair_diagnostics(
    inputs: tuple[DGIParametricEndpointFitInput, DGIParametricEndpointFitInput],
    fit: DGIParametricOrientationPairFit,
) -> None:
    if not isinstance(fit, DGIParametricOrientationPairFit):
        raise TypeError("DGI parametric fit has the wrong type")
    if len(inputs) != 2 or len(fit.endpoints) != 2:
        raise ValueError("DGI parametric diagnostics require two endpoints")
    if fit.provenance.contract_label != _CONTRACT_LABEL:
        raise ValueError("DGI diagnostic fit provenance changed")
    for item, endpoint in zip(inputs, fit.endpoints, strict=True):
        if item.raw_block.endpoint_label != endpoint.endpoint_label:
            raise ValueError("DGI diagnostic endpoint order changed")


def _flatten_roles(
    arrays: tuple[ArrayLike, ...],
    roi_mask: NDArray[np.bool_],
) -> FloatArray:
    return np.concatenate(
        [np.asarray(array, dtype=float)[roi_mask] for array in arrays]
    )


def analyse_dgi_parametric_residuals(
    inputs: tuple[DGIParametricEndpointFitInput, DGIParametricEndpointFitInput],
    fit: DGIParametricOrientationPairFit,
) -> tuple[DGIParametricEndpointResiduals, DGIParametricEndpointResiduals]:
    """Verify the selected fit against all five endpoint-owned DGI roles."""

    _validate_pair_diagnostics(inputs, fit)
    endpoints: list[DGIParametricEndpointResiduals] = []
    for item, endpoint in zip(inputs, fit.endpoints, strict=True):
        result = endpoint.selected_fit
        if endpoint.status != "success" or result is None:
            endpoints.append(
                DGIParametricEndpointResiduals(
                    endpoint_label=endpoint.endpoint_label,
                    status="fit_failure",
                    roles=(),
                    message=endpoint.message,
                )
            )
            continue
        prediction = result.prediction
        if prediction.role_names != item.raw_block.role_names:
            raise ValueError("selected DGI prediction raw-role order changed")
        roles: list[ParametricRawRoleResidual] = []
        whitened: list[FloatArray] = []
        roi = item.operator.grid.roi_mask
        for role_name, owner, observed, expected, variance in zip(
            item.raw_block.role_names,
            item.raw_block.role_owner_ids,
            item.raw_block.observed_electrons,
            prediction.expected_electrons,
            prediction.conditional_variance_electrons2,
            strict=True,
        ):
            residual = (np.asarray(observed) - expected) / np.sqrt(variance)
            roi_values = np.asarray(residual[roi], dtype=float)
            if np.any(~np.isfinite(roi_values)):
                raise ValueError("selected DGI fit residuals are non-finite")
            whitened.append(roi_values)
            roles.append(
                ParametricRawRoleResidual(
                    role_name=role_name,
                    role_owner_id=owner,
                    roi_pixel_count=int(roi_values.size),
                    roi_mean=float(np.mean(roi_values)),
                    roi_rms=float(np.sqrt(np.mean(roi_values**2))),
                    roi_standard_deviation=float(np.std(roi_values)),
                    roi_p99_absolute=float(np.quantile(np.abs(roi_values), 0.99)),
                )
            )
        recomputed = np.concatenate(whitened)
        stored = np.asarray(result.diagnostics.whitened_residual_vector, dtype=float)
        tolerance = 32.0 * np.finfo(float).eps * max(
            1.0,
            float(np.max(np.abs(stored))) if stored.size else 1.0,
        )
        if stored.shape != recomputed.shape or not np.allclose(
            stored,
            recomputed,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("selected DGI fit does not belong to the declared raw block")
        endpoints.append(
            DGIParametricEndpointResiduals(
                endpoint_label=endpoint.endpoint_label,
                status="success",
                roles=tuple(roles),
                message="endpoint-local DGI raw residuals verified",
            )
        )
    return endpoints[0], endpoints[1]


def _zero_density_prediction(
    item: DGIParametricEndpointFitInput,
    values: FloatArray,
) -> tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]]:
    nuisance = DGINuisanceValues(*values)
    zero_density = np.zeros_like(item.operator.grid.y_grid_m, dtype=float)
    role_names, expected = item.operator.expected_linked_sequence_from_density_maps(
        [zero_density],
        nuisance,
    )
    if role_names != item.raw_block.role_names:
        raise ValueError("zero-density DGI raw-role order changed")
    variances: list[FloatArray] = []
    for role_name, role in zip(role_names, expected, strict=True):
        base_role = "atom_stop" if role_name == "atom_stop_000" else role_name
        exposures = item.operator.independent_exposures_by_role[base_role]
        variances.append(
            np.asarray(
                (role + item.operator.read_noise_electrons**2) / exposures,
                dtype=float,
            )
        )
    return expected, tuple(variances)


def _fit_zero_density_endpoint(
    item: DGIParametricEndpointFitInput,
    endpoint: ParametricEndpointFit,
) -> DGIParametricZeroDensityNull:
    result = endpoint.selected_fit
    if endpoint.status != "success" or result is None:
        return DGIParametricZeroDensityNull(
            endpoint_label=endpoint.endpoint_label,
            status="fit_failure",
            nuisance_values=None,
            null_weighted_chi_square=None,
            fitted_weighted_chi_square=None,
            improvement_over_null=None,
            message=endpoint.message,
        )
    lower = np.asarray(item.nuisance_lower, dtype=float)
    upper = np.asarray(item.nuisance_upper, dtype=float)
    current = np.asarray(result.nuisance_values, dtype=float)
    observed = _flatten_roles(
        item.raw_block.observed_electrons,
        item.operator.grid.roi_mask,
    )
    try:
        final = None
        for _ in range(item.options.irls_iterations):
            standard_deviation = np.sqrt(
                _flatten_roles(
                    _zero_density_prediction(item, current)[1],
                    item.operator.grid.roi_mask,
                )
            )

            def residual(values: FloatArray) -> FloatArray:
                expected, _variance = _zero_density_prediction(item, values)
                return (
                    observed
                    - _flatten_roles(expected, item.operator.grid.roi_mask)
                ) / standard_deviation

            final = least_squares(
                residual,
                current,
                bounds=(lower, upper),
                method="trf",
                loss="linear",
                x_scale="jac",
                max_nfev=item.options.max_nfev,
                xtol=item.options.xtol,
                ftol=item.options.ftol,
                gtol=item.options.gtol,
            )
            current = np.asarray(final.x, dtype=float)
        if final is None or not final.success or np.any(~np.isfinite(current)):
            return DGIParametricZeroDensityNull(
                endpoint_label=endpoint.endpoint_label,
                status="fit_failure",
                nuisance_values=None,
                null_weighted_chi_square=None,
                fitted_weighted_chi_square=None,
                improvement_over_null=None,
                message=(
                    "zero-density DGI nuisance refit did not execute"
                    if final is None
                    else str(final.message)
                ),
            )
        null_expected, null_variance = _zero_density_prediction(item, current)
        null_residual = (
            observed
            - _flatten_roles(null_expected, item.operator.grid.roi_mask)
        ) / np.sqrt(
            _flatten_roles(null_variance, item.operator.grid.roi_mask)
        )
        null_chi = float(null_residual @ null_residual)
        fitted_chi = float(result.diagnostics.weighted_chi_square)
        return DGIParametricZeroDensityNull(
            endpoint_label=endpoint.endpoint_label,
            status="success",
            nuisance_values=tuple(float(value) for value in current),
            null_weighted_chi_square=null_chi,
            fitted_weighted_chi_square=fitted_chi,
            improvement_over_null=float(null_chi - fitted_chi),
            message="endpoint-local zero-density DGI nuisance refit converged",
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return DGIParametricZeroDensityNull(
            endpoint_label=endpoint.endpoint_label,
            status="fit_failure",
            nuisance_values=None,
            null_weighted_chi_square=None,
            fitted_weighted_chi_square=None,
            improvement_over_null=None,
            message=f"{type(exc).__name__}: {exc}",
        )


def analyse_dgi_parametric_zero_density_null(
    inputs: tuple[DGIParametricEndpointFitInput, DGIParametricEndpointFitInput],
    fit: DGIParametricOrientationPairFit,
) -> tuple[DGIParametricZeroDensityNull, DGIParametricZeroDensityNull]:
    """Refit the four zero-density DGI nuisances for both endpoints."""

    _validate_pair_diagnostics(inputs, fit)
    records = tuple(
        _fit_zero_density_endpoint(item, endpoint)
        for item, endpoint in zip(inputs, fit.endpoints, strict=True)
    )
    return records[0], records[1]


def _observable_gradients(
    endpoint: ParametricEndpointFit,
    *,
    profile_exponent: float,
) -> tuple[dict[str, float], dict[str, FloatArray]]:
    if endpoint.physical_parameters is None:
        raise ValueError("observable gradients require a successful selected fit")
    observables = parametric_observables(
        endpoint.physical_parameters,
        profile_exponent=profile_exponent,
    )
    values = {
        "A": observables.A,
        "centre_y_um": observables.centre_y_um,
        "centre_z_um": observables.centre_z_um,
        "sigma_y_um": observables.sigma_y_um,
        "sigma_z_um": observables.sigma_z_um,
        "aspect_ratio_y_over_z": observables.aspect_ratio_y_over_z,
    }
    gradients = {
        "A": np.asarray([observables.A, 0.0, 0.0, observables.A, observables.A]),
        "centre_y_um": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0]),
        "centre_z_um": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0]),
        "sigma_y_um": np.asarray([0.0, 0.0, 0.0, observables.sigma_y_um, 0.0]),
        "sigma_z_um": np.asarray([0.0, 0.0, 0.0, 0.0, observables.sigma_z_um]),
        # The reporting scale is specified in log-aspect units, so this is
        # d log(sigma_y / sigma_z), not the gradient of the linear ratio.
        "aspect_ratio_y_over_z": np.asarray([0.0, 0.0, 0.0, 1.0, -1.0]),
    }
    return values, gradients


def _reporting_scale(name: str, estimate: float) -> float:
    if name == "A":
        return 0.10 * abs(estimate)
    if name in ("centre_y_um", "centre_z_um"):
        return 0.65
    if name in ("sigma_y_um", "sigma_z_um"):
        return max(0.65, 0.15 * abs(estimate))
    if name == "aspect_ratio_y_over_z":
        return 0.1823215567939546
    raise ValueError("unknown DGI observable reporting scale")


def _geometry_record(
    *,
    name: str,
    estimate: float,
    gradient: FloatArray,
    spans: FloatArray,
    singular_values: FloatArray,
    right_vectors: FloatArray,
    parameters: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    rank_tolerances: tuple[float, ...],
    active_tolerances: tuple[float, ...],
    maximum_null_fraction: float,
    maximum_active_fraction: float,
) -> ParametricObservableGeometry:
    full_gradient = np.zeros(spans.size, dtype=float)
    full_gradient[: gradient.size] = gradient
    scaled_gradient = full_gradient * spans
    gradient_norm = float(np.linalg.norm(scaled_gradient))
    scale = _reporting_scale(name, estimate)
    if gradient_norm == 0.0 or scale <= 0.0:
        return ParametricObservableGeometry(
            observable_name=name,
            estimate=estimate,
            reporting_scale=scale,
            data_null_space_fraction=None,
            active_bound_gradient_fraction=None,
            identified_subspace_standard_uncertainty=None,
            primary_supported=False,
            tolerance_stable=False,
            amplitude_gate_supported=name not in PRIMARY_OBSERVABLES,
            supported=False,
            reasons=("observable_has_no_finite_reporting_geometry",),
        )
    largest = float(singular_values[0]) if singular_values.size else 0.0
    variants: list[tuple[float, float, float]] = []
    for rank_tolerance in rank_tolerances:
        rank = (
            int(np.count_nonzero(singular_values > rank_tolerance * largest))
            if largest > 0.0
            else 0
        )
        identified = right_vectors[:, :rank].T @ scaled_gradient
        null = right_vectors[:, rank:].T @ scaled_gradient
        null_fraction = float(np.linalg.norm(null) / gradient_norm)
        uncertainty = (
            float(np.sqrt(np.sum((identified / singular_values[:rank]) ** 2)))
            if rank
            else float("inf")
        )
        for active_tolerance in active_tolerances:
            active = (
                (parameters - lower <= active_tolerance * spans)
                | (upper - parameters <= active_tolerance * spans)
            )
            active_fraction = float(
                np.linalg.norm(scaled_gradient[active]) / gradient_norm
            )
            variants.append((null_fraction, active_fraction, uncertainty))
    primary_index = (
        rank_tolerances.index(1e-10) * len(active_tolerances)
        + active_tolerances.index(1e-6)
    )
    primary = variants[primary_index]
    decisions = tuple(
        null_fraction <= maximum_null_fraction
        and active_fraction <= maximum_active_fraction
        and uncertainty <= scale
        for null_fraction, active_fraction, uncertainty in variants
    )
    primary_supported = decisions[primary_index]
    tolerance_stable = all(decision == primary_supported for decision in decisions)
    reasons: list[str] = []
    if primary[0] > maximum_null_fraction:
        reasons.append("observable_gradient_has_excess_data_null_component")
    if primary[1] > maximum_active_fraction:
        reasons.append("observable_gradient_depends_on_active_bound")
    if primary[2] > scale:
        reasons.append("identified_uncertainty_exceeds_reporting_scale")
    if not tolerance_stable:
        reasons.append("support_changes_across_rank_or_active_tolerance")
    if not reasons and not primary_supported:
        reasons.append("observable_support_failed")
    return ParametricObservableGeometry(
        observable_name=name,
        estimate=estimate,
        reporting_scale=scale,
        data_null_space_fraction=primary[0],
        active_bound_gradient_fraction=primary[1],
        identified_subspace_standard_uncertainty=primary[2],
        primary_supported=primary_supported,
        tolerance_stable=tolerance_stable,
        amplitude_gate_supported=name not in PRIMARY_OBSERVABLES,
        supported=(
            primary_supported
            and tolerance_stable
            and name not in PRIMARY_OBSERVABLES
        ),
        reasons=(
            tuple(reasons)
            if reasons
            else (
                ()
                if name not in PRIMARY_OBSERVABLES
                else ("amplitude_gate_not_evaluated",)
            )
        ),
    )


def _endpoint_geometry(
    item: DGIParametricEndpointFitInput,
    endpoint: ParametricEndpointFit,
    *,
    rank_tolerances: tuple[float, ...],
    active_tolerances: tuple[float, ...],
    maximum_null_fraction: float,
    maximum_active_fraction: float,
) -> ParametricEndpointGeometry:
    parameter_count = item.model.parameter_count + len(item.operator.nuisance_names)
    result = endpoint.selected_fit
    if endpoint.status != "success" or result is None:
        return ParametricEndpointGeometry(
            endpoint_label=endpoint.endpoint_label,
            status="fit_failure",
            parameter_count=parameter_count,
            primary_data_rank=0,
            primary_condition_number=float("inf"),
            primary_active_bound_count=0,
            singular_values=np.zeros(parameter_count),
            records=(),
            message=endpoint.message,
        )
    lower = np.concatenate(
        [np.asarray(item.parameter_lower), np.asarray(item.nuisance_lower)]
    )
    upper = np.concatenate(
        [np.asarray(item.parameter_upper), np.asarray(item.nuisance_upper)]
    )
    spans = upper - lower
    parameters = np.concatenate(
        [
            result.density_coefficients[0],
            nuisance_vector(DGINuisanceValues(*result.nuisance_values)),
        ]
    )
    variance = _flatten_roles(
        result.prediction.conditional_variance_electrons2,
        item.operator.grid.roi_mask,
    )
    jacobian = np.asarray(result.prediction.jacobian, dtype=float)
    if (
        jacobian.shape != (variance.size, spans.size)
        or np.any(~np.isfinite(jacobian))
        or np.any(~np.isfinite(variance))
        or np.any(variance <= 0.0)
    ):
        raise ValueError("DGI endpoint Jacobian or variance is invalid")
    scaled = jacobian / np.sqrt(variance)[:, None] * spans[None, :]
    triangular = np.linalg.qr(scaled, mode="r")
    _, singular, right_transpose = np.linalg.svd(triangular, full_matrices=True)
    spectrum = np.zeros(spans.size, dtype=float)
    spectrum[: singular.size] = singular
    largest = float(spectrum[0]) if spectrum.size else 0.0
    primary_rank = (
        int(np.count_nonzero(spectrum > 1e-10 * largest)) if largest > 0.0 else 0
    )
    condition = (
        float(spectrum[0] / spectrum[-1])
        if primary_rank == spans.size and spectrum[-1] > 0.0
        else float("inf")
    )
    primary_active = (
        (parameters - lower <= 1e-6 * spans)
        | (upper - parameters <= 1e-6 * spans)
    )
    values, gradients = _observable_gradients(
        endpoint,
        profile_exponent=item.model.profile_exponent,
    )
    records = [
        _geometry_record(
            name=name,
            estimate=values[name],
            gradient=gradients[name],
            spans=spans,
            singular_values=spectrum,
            right_vectors=np.asarray(right_transpose.T),
            parameters=parameters,
            lower=lower,
            upper=upper,
            rank_tolerances=rank_tolerances,
            active_tolerances=active_tolerances,
            maximum_null_fraction=maximum_null_fraction,
            maximum_active_fraction=maximum_active_fraction,
        )
        for name in ALL_OBSERVABLES
    ]
    amplitude = next(record for record in records if record.observable_name == "A")
    gated: list[ParametricObservableGeometry] = []
    for record in records:
        if record.observable_name not in PRIMARY_OBSERVABLES:
            gated.append(record)
            continue
        gate = amplitude.supported
        reasons = tuple(
            reason
            for reason in record.reasons
            if reason != "amplitude_gate_not_evaluated"
        )
        if not gate:
            reasons = (*reasons, "amplitude_control_is_not_supported")
        supported = record.primary_supported and record.tolerance_stable and gate
        gated.append(
            replace(
                record,
                amplitude_gate_supported=gate,
                supported=supported,
                reasons=() if supported else reasons,
            )
        )
    return ParametricEndpointGeometry(
        endpoint_label=endpoint.endpoint_label,
        status="success",
        parameter_count=spans.size,
        primary_data_rank=primary_rank,
        primary_condition_number=condition,
        primary_active_bound_count=int(np.count_nonzero(primary_active)),
        singular_values=spectrum,
        records=tuple(gated),
        message="scaled endpoint-local DGI likelihood geometry evaluated",
    )


def analyse_dgi_parametric_identifiability(
    inputs: tuple[DGIParametricEndpointFitInput, DGIParametricEndpointFitInput],
    fit: DGIParametricOrientationPairFit,
    *,
    rank_tolerances: tuple[float, ...] = (1e-12, 1e-10, 1e-8),
    active_tolerances: tuple[float, ...] = (1e-7, 1e-6, 1e-5),
    maximum_null_fraction: float = 0.10,
    maximum_active_fraction: float = 0.25,
) -> tuple[ParametricEndpointGeometry, ParametricEndpointGeometry]:
    """Evaluate the nine-parameter DGI geometry for every observable."""

    _validate_pair_diagnostics(inputs, fit)
    if 1e-10 not in rank_tolerances or 1e-6 not in active_tolerances:
        raise ValueError("primary DGI identifiability tolerances must be present")
    if any(
        not 0.0 < value < 1.0
        for value in (*rank_tolerances, *active_tolerances)
    ):
        raise ValueError("DGI identifiability tolerances must lie in (0, 1)")
    if (
        not 0.0 <= maximum_null_fraction <= 1.0
        or not 0.0 <= maximum_active_fraction <= 1.0
    ):
        raise ValueError("DGI identifiability fraction limits must lie in [0, 1]")
    records = tuple(
        _endpoint_geometry(
            item,
            endpoint,
            rank_tolerances=rank_tolerances,
            active_tolerances=active_tolerances,
            maximum_null_fraction=maximum_null_fraction,
            maximum_active_fraction=maximum_active_fraction,
        )
        for item, endpoint in zip(inputs, fit.endpoints, strict=True)
    )
    return records[0], records[1]


def analyse_dgi_parametric_endpoint_diagnostics(
    inputs: tuple[DGIParametricEndpointFitInput, DGIParametricEndpointFitInput],
    fit: DGIParametricOrientationPairFit,
) -> tuple[DGIParametricEndpointDiagnostics, DGIParametricEndpointDiagnostics]:
    """Return the complete target-free DGI diagnostic bundle."""

    residuals = analyse_dgi_parametric_residuals(inputs, fit)
    nulls = analyse_dgi_parametric_zero_density_null(inputs, fit)
    geometry = analyse_dgi_parametric_identifiability(inputs, fit)
    records = tuple(
        DGIParametricEndpointDiagnostics(
            endpoint_label=endpoint.endpoint_label,
            residuals=residual,
            zero_density_null=null,
            geometry=local_geometry,
            multistart=analyse_parametric_multistart(
                endpoint,
                profile_exponent=item.model.profile_exponent,
            ),
        )
        for item, endpoint, residual, null, local_geometry in zip(
            inputs,
            fit.endpoints,
            residuals,
            nulls,
            geometry,
            strict=True,
        )
    )
    return records[0], records[1]


__all__ = [
    "DGIParametricEndpointDiagnostics",
    "DGIParametricEndpointFitInput",
    "DGIParametricEndpointRawBlock",
    "DGIParametricEndpointResiduals",
    "DGIParametricOrientationPairFit",
    "DGIParametricOrientationProvenance",
    "DGIParametricZeroDensityNull",
    "analyse_dgi_parametric_endpoint_diagnostics",
    "analyse_dgi_parametric_identifiability",
    "analyse_dgi_parametric_residuals",
    "analyse_dgi_parametric_zero_density_null",
    "fit_independent_parametric_dgi_endpoints",
]
