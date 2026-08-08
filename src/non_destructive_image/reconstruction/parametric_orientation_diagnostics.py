"""Observable-specific diagnostics for low-dimensional orientation inference."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .linked_scalar_fit import nuisance_vector
from .parameters import from_internal
from .parametric_orientation import (
    ParametricEndpointFit,
    ParametricEndpointFitInput,
    ParametricOrientationPairFit,
    parametric_observables,
)
from .scalar_measurements import PCINuisanceValues


FloatArray = NDArray[np.floating]
PRIMARY_OBSERVABLES = (
    "sigma_y_um",
    "sigma_z_um",
    "aspect_ratio_y_over_z",
)
CONTROL_OBSERVABLES = ("A", "centre_y_um", "centre_z_um")
ALL_OBSERVABLES = (*CONTROL_OBSERVABLES, *PRIMARY_OBSERVABLES)


def _immutable(values: ArrayLike) -> FloatArray:
    source = np.asarray(values, dtype=float)
    result = np.frombuffer(source.tobytes(order="C"), dtype=float)
    return result.reshape(source.shape)


@dataclass(frozen=True)
class ParametricRawRoleResidual:
    """Whitened residual statistics for one endpoint-owned raw role."""

    role_name: str
    role_owner_id: str
    roi_pixel_count: int
    roi_mean: float
    roi_rms: float
    roi_standard_deviation: float
    roi_p99_absolute: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.roi_mean,
                self.roi_rms,
                self.roi_standard_deviation,
                self.roi_p99_absolute,
            ),
            dtype=float,
        )
        if not self.role_name or not self.role_owner_id:
            raise ValueError("raw residual role identity cannot be empty")
        if self.roi_pixel_count <= 0 or np.any(~np.isfinite(values)):
            raise ValueError("raw residual statistics must be finite")
        if min(self.roi_rms, self.roi_standard_deviation, self.roi_p99_absolute) < 0.0:
            raise ValueError("raw residual scales cannot be negative")


@dataclass(frozen=True)
class ParametricEndpointResiduals:
    """Raw-role residual diagnostics for one selected endpoint fit."""

    endpoint_label: str
    status: Literal["success", "fit_failure"]
    roles: tuple[ParametricRawRoleResidual, ...]
    message: str

    def __post_init__(self) -> None:
        if self.status not in ("success", "fit_failure"):
            raise ValueError("unknown endpoint residual status")
        if self.status == "success" and len(self.roles) != 3:
            raise ValueError("successful PCI residual diagnostics require three roles")
        if self.status == "fit_failure" and self.roles:
            raise ValueError("failed residual diagnostics cannot publish role statistics")


@dataclass(frozen=True)
class ParametricZeroDensityNull:
    """Endpoint-local zero-density nuisance refit diagnostic."""

    endpoint_label: str
    status: Literal["success", "fit_failure"]
    nuisance_values: tuple[float, float] | None
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
            raise ValueError("unknown zero-density diagnostic status")
        if self.status == "success":
            if self.nuisance_values is None or any(value is None for value in values):
                raise ValueError("successful zero-density diagnostic is incomplete")
            if np.any(~np.isfinite(np.asarray((*self.nuisance_values, *values), dtype=float))):
                raise ValueError("zero-density diagnostic values must be finite")
        elif self.nuisance_values is not None or any(value is not None for value in values):
            raise ValueError("failed zero-density diagnostic cannot publish values")


@dataclass(frozen=True)
class ParametricObservableGeometry:
    """Local information geometry for one endpoint observable."""

    observable_name: str
    estimate: float | None
    reporting_scale: float | None
    data_null_space_fraction: float | None
    active_bound_gradient_fraction: float | None
    identified_subspace_standard_uncertainty: float | None
    primary_supported: bool
    tolerance_stable: bool
    amplitude_gate_supported: bool
    supported: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.observable_name not in ALL_OBSERVABLES:
            raise ValueError("unknown parametric observable")
        optional = (
            self.estimate,
            self.reporting_scale,
            self.data_null_space_fraction,
            self.active_bound_gradient_fraction,
            self.identified_subspace_standard_uncertainty,
        )
        if any(value is not None and not np.isfinite(value) for value in optional):
            raise ValueError("observable geometry metrics must be finite when present")
        if self.supported and (
            not self.primary_supported
            or not self.tolerance_stable
            or not self.amplitude_gate_supported
            or self.reasons
        ):
            raise ValueError("supported observable geometry is internally inconsistent")
        if not self.supported and not self.reasons:
            raise ValueError("unsupported observable geometry requires a reason")


@dataclass(frozen=True, eq=False)
class ParametricEndpointGeometry:
    """Scaled data geometry and observable records for one endpoint."""

    endpoint_label: str
    status: Literal["success", "fit_failure"]
    parameter_count: int
    primary_data_rank: int
    primary_condition_number: float
    primary_active_bound_count: int
    singular_values: FloatArray
    records: tuple[ParametricObservableGeometry, ...]
    message: str

    def __post_init__(self) -> None:
        if self.status not in ("success", "fit_failure"):
            raise ValueError("unknown endpoint geometry status")
        singular = _immutable(self.singular_values)
        if singular.shape != (self.parameter_count,) or np.any(singular < 0.0):
            raise ValueError("endpoint singular-value spectrum is invalid")
        if self.status == "success" and tuple(
            record.observable_name for record in self.records
        ) != ALL_OBSERVABLES:
            raise ValueError("successful endpoint geometry has incomplete observables")
        if self.status == "fit_failure" and self.records:
            raise ValueError("failed endpoint geometry cannot publish observables")
        object.__setattr__(self, "singular_values", singular)


@dataclass(frozen=True)
class ParametricMultistartStability:
    """Selected-to-alternate successful-start spread for one observable."""

    observable_name: str
    successful_start_count: int
    maximum_normalised_shift: float | None
    grade: Literal["stable", "model_sensitive", "unresolved"]

    def __post_init__(self) -> None:
        if self.observable_name not in ALL_OBSERVABLES:
            raise ValueError("unknown multistart observable")
        if self.successful_start_count < 0:
            raise ValueError("successful-start count cannot be negative")
        if self.maximum_normalised_shift is not None and (
            not np.isfinite(self.maximum_normalised_shift)
            or self.maximum_normalised_shift < 0.0
        ):
            raise ValueError("multistart shift must be finite and non-negative")
        if self.grade not in ("stable", "model_sensitive", "unresolved"):
            raise ValueError("unknown multistart grade")


@dataclass(frozen=True)
class ParametricEndpointDiagnostics:
    """Complete fit/data, null, geometry and multistart endpoint diagnostics."""

    endpoint_label: str
    residuals: ParametricEndpointResiduals
    zero_density_null: ParametricZeroDensityNull
    geometry: ParametricEndpointGeometry
    multistart: tuple[ParametricMultistartStability, ...]

    def __post_init__(self) -> None:
        labels = (
            self.residuals.endpoint_label,
            self.zero_density_null.endpoint_label,
            self.geometry.endpoint_label,
        )
        if any(value != self.endpoint_label for value in labels):
            raise ValueError("endpoint diagnostic labels disagree")
        if tuple(item.observable_name for item in self.multistart) != ALL_OBSERVABLES:
            raise ValueError("endpoint multistart diagnostics are incomplete")


def _flatten_roles(arrays: tuple[ArrayLike, ...], roi_mask: NDArray[np.bool_]) -> FloatArray:
    return np.concatenate(
        [np.asarray(array, dtype=float)[roi_mask] for array in arrays]
    )


def _validate_pair(
    inputs: tuple[ParametricEndpointFitInput, ParametricEndpointFitInput],
    fit: ParametricOrientationPairFit,
) -> None:
    if len(inputs) != 2 or len(fit.endpoints) != 2:
        raise ValueError("parametric diagnostics require two endpoints")
    for item, endpoint in zip(inputs, fit.endpoints, strict=True):
        if item.raw_block.endpoint_label != endpoint.endpoint_label:
            raise ValueError("parametric diagnostic endpoint order changed")


def analyse_parametric_residuals(
    inputs: tuple[ParametricEndpointFitInput, ParametricEndpointFitInput],
    fit: ParametricOrientationPairFit,
) -> tuple[ParametricEndpointResiduals, ParametricEndpointResiduals]:
    """Verify and summarise the selected fits against their three raw roles."""

    _validate_pair(inputs, fit)
    endpoints: list[ParametricEndpointResiduals] = []
    for item, endpoint in zip(inputs, fit.endpoints, strict=True):
        result = endpoint.selected_fit
        if endpoint.status != "success" or result is None:
            endpoints.append(
                ParametricEndpointResiduals(
                    endpoint_label=endpoint.endpoint_label,
                    status="fit_failure",
                    roles=(),
                    message=endpoint.message,
                )
            )
            continue
        prediction = result.prediction
        if prediction.role_names != item.raw_block.role_names:
            raise ValueError("selected prediction raw-role order changed")
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
                raise ValueError("selected fit residuals are non-finite")
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
            raise ValueError("selected fit does not belong to the declared raw block")
        endpoints.append(
            ParametricEndpointResiduals(
                endpoint_label=endpoint.endpoint_label,
                status="success",
                roles=tuple(roles),
                message="endpoint-local raw residuals verified",
            )
        )
    return endpoints[0], endpoints[1]


def _fit_zero_density_endpoint(
    item: ParametricEndpointFitInput,
    endpoint: ParametricEndpointFit,
) -> ParametricZeroDensityNull:
    result = endpoint.selected_fit
    if endpoint.status != "success" or result is None:
        return ParametricZeroDensityNull(
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
    zero_density = np.zeros_like(item.operator.grid.y_grid_m, dtype=float)
    role_names, unit_roles = item.operator.expected_linked_sequence_from_density_maps(
        [zero_density],
        PCINuisanceValues(1.0, 0.0),
    )
    if role_names != item.raw_block.role_names:
        raise ValueError("zero-density PCI raw-role order changed")
    observed = _flatten_roles(
        item.raw_block.observed_electrons,
        item.operator.grid.roi_mask,
    )

    def prediction(values: FloatArray) -> tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]]:
        nuisance = PCINuisanceValues(float(values[0]), float(values[1]))
        expected = tuple(
            np.asarray(
                nuisance.i0_photoelectrons_per_pixel * unit
                + nuisance.dark_electrons_per_pixel,
                dtype=float,
            )
            for unit in unit_roles
        )
        variance = tuple(
            np.asarray(role + item.operator.read_noise_electrons**2, dtype=float)
            for role in expected
        )
        return expected, variance

    try:
        final = None
        for _ in range(item.options.irls_iterations):
            standard_deviation = np.sqrt(
                _flatten_roles(
                    prediction(current)[1],
                    item.operator.grid.roi_mask,
                )
            )

            def residual(values: FloatArray) -> FloatArray:
                return (
                    observed
                    - _flatten_roles(
                        prediction(values)[0],
                        item.operator.grid.roi_mask,
                    )
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
            return ParametricZeroDensityNull(
                endpoint_label=endpoint.endpoint_label,
                status="fit_failure",
                nuisance_values=None,
                null_weighted_chi_square=None,
                fitted_weighted_chi_square=None,
                improvement_over_null=None,
                message=(
                    "zero-density nuisance refit did not execute"
                    if final is None
                    else str(final.message)
                ),
            )
        null_expected, null_variance = prediction(current)
        null_residual = (
            observed
            - _flatten_roles(null_expected, item.operator.grid.roi_mask)
        ) / np.sqrt(_flatten_roles(null_variance, item.operator.grid.roi_mask))
        null_chi = float(null_residual @ null_residual)
        fitted_chi = float(result.diagnostics.weighted_chi_square)
        return ParametricZeroDensityNull(
            endpoint_label=endpoint.endpoint_label,
            status="success",
            nuisance_values=(float(current[0]), float(current[1])),
            null_weighted_chi_square=null_chi,
            fitted_weighted_chi_square=fitted_chi,
            improvement_over_null=float(null_chi - fitted_chi),
            message="endpoint-local zero-density nuisance refit converged",
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return ParametricZeroDensityNull(
            endpoint_label=endpoint.endpoint_label,
            status="fit_failure",
            nuisance_values=None,
            null_weighted_chi_square=None,
            fitted_weighted_chi_square=None,
            improvement_over_null=None,
            message=f"{type(exc).__name__}: {exc}",
        )


def analyse_parametric_zero_density_null(
    inputs: tuple[ParametricEndpointFitInput, ParametricEndpointFitInput],
    fit: ParametricOrientationPairFit,
) -> tuple[ParametricZeroDensityNull, ParametricZeroDensityNull]:
    """Refit zero-density nuisances independently for both endpoints."""

    _validate_pair(inputs, fit)
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
    if endpoint.selected_fit is None or endpoint.physical_parameters is None:
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
        # The uncertainty is evaluated in log-aspect units because the
        # reporting contract defines a positive-ratio log scale.
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
    raise ValueError("unknown observable reporting scale")


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
        uncertainty = float(
            np.sqrt(
                np.sum(
                    (identified / singular_values[:rank]) ** 2
                )
            )
        ) if rank else float("inf")
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
        supported=primary_supported and tolerance_stable and name not in PRIMARY_OBSERVABLES,
        reasons=tuple(reasons) if reasons else (() if name not in PRIMARY_OBSERVABLES else ("amplitude_gate_not_evaluated",)),
    )


def _endpoint_geometry(
    item: ParametricEndpointFitInput,
    endpoint: ParametricEndpointFit,
    *,
    rank_tolerances: tuple[float, ...],
    active_tolerances: tuple[float, ...],
    maximum_null_fraction: float,
    maximum_active_fraction: float,
) -> ParametricEndpointGeometry:
    result = endpoint.selected_fit
    if endpoint.status != "success" or result is None:
        return ParametricEndpointGeometry(
            endpoint_label=endpoint.endpoint_label,
            status="fit_failure",
            parameter_count=7,
            primary_data_rank=0,
            primary_condition_number=float("inf"),
            primary_active_bound_count=0,
            singular_values=np.zeros(7),
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
        [result.density_coefficients[0], nuisance_vector(PCINuisanceValues(*result.nuisance_values))]
    )
    variance = _flatten_roles(
        result.prediction.conditional_variance_electrons2,
        item.operator.grid.roi_mask,
    )
    jacobian = np.asarray(result.prediction.jacobian, dtype=float)
    if jacobian.shape != (variance.size, spans.size) or np.any(variance <= 0.0):
        raise ValueError("parametric endpoint Jacobian or variance is invalid")
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
        message="scaled endpoint-local likelihood geometry evaluated",
    )


def analyse_parametric_identifiability(
    inputs: tuple[ParametricEndpointFitInput, ParametricEndpointFitInput],
    fit: ParametricOrientationPairFit,
    *,
    rank_tolerances: tuple[float, ...] = (1e-12, 1e-10, 1e-8),
    active_tolerances: tuple[float, ...] = (1e-7, 1e-6, 1e-5),
    maximum_null_fraction: float = 0.10,
    maximum_active_fraction: float = 0.25,
) -> tuple[ParametricEndpointGeometry, ParametricEndpointGeometry]:
    """Evaluate bound-scaled endpoint-local support for every observable."""

    _validate_pair(inputs, fit)
    if 1e-10 not in rank_tolerances or 1e-6 not in active_tolerances:
        raise ValueError("primary identifiability tolerances must be present")
    if any(not 0.0 < value < 1.0 for value in (*rank_tolerances, *active_tolerances)):
        raise ValueError("identifiability tolerances must lie in (0, 1)")
    if not 0.0 <= maximum_null_fraction <= 1.0 or not 0.0 <= maximum_active_fraction <= 1.0:
        raise ValueError("identifiability fraction limits must lie in [0, 1]")
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


def _normalised_shift(name: str, selected: float, alternate: float) -> float:
    if name == "aspect_ratio_y_over_z":
        if selected <= 0.0 or alternate <= 0.0:
            return float("inf")
        return abs(np.log(alternate / selected)) / _reporting_scale(name, selected)
    return abs(alternate - selected) / _reporting_scale(name, selected)


def analyse_parametric_multistart(
    endpoint: ParametricEndpointFit,
    *,
    profile_exponent: float,
) -> tuple[ParametricMultistartStability, ...]:
    """Grade successful-start observable spread without hiding failed starts."""

    selected = endpoint.observables
    successful = [
        item
        for item in endpoint.start_results
        if item.status == "success" and item.fit_result is not None
    ]
    records: list[ParametricMultistartStability] = []
    for name in ALL_OBSERVABLES:
        if selected is None or len(successful) < 2:
            records.append(
                ParametricMultistartStability(
                    observable_name=name,
                    successful_start_count=len(successful),
                    maximum_normalised_shift=None,
                    grade="unresolved",
                )
            )
            continue
        selected_value = float(getattr(selected, name))
        shifts = []
        for item in successful:
            assert item.fit_result is not None
            alternate = parametric_observables(
                from_internal(item.fit_result.density_coefficients[0]),
                profile_exponent=profile_exponent,
            )
            shifts.append(
                _normalised_shift(name, selected_value, float(getattr(alternate, name)))
            )
        maximum = float(max(shifts))
        grade: Literal["stable", "model_sensitive", "unresolved"]
        if maximum <= 0.5:
            grade = "stable"
        elif maximum <= 1.0:
            grade = "model_sensitive"
        else:
            grade = "unresolved"
        records.append(
            ParametricMultistartStability(
                observable_name=name,
                successful_start_count=len(successful),
                maximum_normalised_shift=maximum,
                grade=grade,
            )
        )
    return tuple(records)


def analyse_parametric_endpoint_diagnostics(
    inputs: tuple[ParametricEndpointFitInput, ParametricEndpointFitInput],
    fit: ParametricOrientationPairFit,
) -> tuple[ParametricEndpointDiagnostics, ParametricEndpointDiagnostics]:
    """Return the complete target-free diagnostic bundle for both endpoints."""

    residuals = analyse_parametric_residuals(inputs, fit)
    nulls = analyse_parametric_zero_density_null(inputs, fit)
    geometry = analyse_parametric_identifiability(inputs, fit)
    records = tuple(
        ParametricEndpointDiagnostics(
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
    "ALL_OBSERVABLES",
    "CONTROL_OBSERVABLES",
    "PRIMARY_OBSERVABLES",
    "ParametricEndpointDiagnostics",
    "ParametricEndpointGeometry",
    "ParametricEndpointResiduals",
    "ParametricMultistartStability",
    "ParametricObservableGeometry",
    "ParametricRawRoleResidual",
    "ParametricZeroDensityNull",
    "analyse_parametric_endpoint_diagnostics",
    "analyse_parametric_identifiability",
    "analyse_parametric_multistart",
    "analyse_parametric_residuals",
    "analyse_parametric_zero_density_null",
]
