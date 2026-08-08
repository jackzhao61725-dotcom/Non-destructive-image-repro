"""Endpoint-local diagnostics for independently prepared PCI measurements.

The two orientation endpoints never enter one likelihood.  Residuals, null
fits, singular spectra and observable gradients are computed per endpoint.
Paired differences and ratios combine only the two already-independent local
linear results, which is algebraically equivalent to a block-diagonal data
likelihood without constructing a joint fit or a joint Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .independent_endpoint_information import (
    DERIVED_OBSERVABLE_NAME,
    ENDPOINT_LABELS,
    OBSERVABLE_NAMES,
    POSITIVE_OBSERVABLE_NAMES,
    RAW_ROLE_NAMES,
    ConditionalEndpointEstimate,
    IndependentEndpointFit,
    IndependentEndpointFitInput,
    IndependentEndpointPairFit,
    IndependentEndpointPairProvenance,
    _validate_fit_input_binding,
    _validate_input_pair,
)
from .linked_scalar_fit import LinkedScalarFitOptions
from .observables import ObservableIntegrationSupport, extract_density_observables
from .scalar_measurements import PCINuisanceValues


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
EndpointDiagnosticStatus = Literal["success", "fit_failure"]
EvidenceGrade = Literal[
    "adequate",
    "limited",
    "not_assessed",
    "not_applicable",
    "failed",
]
ModelDependenceGrade = Literal[
    "stable",
    "sensitive",
    "not_assessed",
    "not_applicable",
    "failed",
]
InformationLevel = Literal[
    "quantitatively_resolved",
    "informative_but_inconclusive",
    "prior_sensitive",
    "unresolved",
    "fit_or_data_failure",
]


def _immutable(value: ArrayLike, *, dtype: type = float) -> NDArray:
    source = np.asarray(value, dtype=dtype)
    result = np.frombuffer(source.tobytes(order="C"), dtype=np.dtype(dtype))
    return result.reshape(source.shape)


def _finite_or_infinite(value: float, *, name: str) -> float:
    result = float(value)
    if np.isnan(result) or result == float("-inf"):
        raise ValueError(f"{name} must be finite or positive infinity")
    return result


def _bound_vector(value: float | ArrayLike, *, size: int, name: str) -> FloatArray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        result = np.full(size, float(result), dtype=float)
    if result.shape != (size,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be a finite scalar or length-{size} vector")
    return result


@dataclass(frozen=True, eq=False)
class EndpointRawRoleResidual:
    """Noise-scaled residual diagnostics for one endpoint-owned raw role."""

    role_name: str
    role_owner_id: str
    standardised_residual_map: FloatArray
    roi_pixel_count: int
    roi_mean: float
    roi_rms: float
    roi_standard_deviation: float

    def __post_init__(self) -> None:
        if not self.role_name or not self.role_owner_id:
            raise ValueError("raw residual role identity cannot be empty")
        residual = np.asarray(self.standardised_residual_map, dtype=float)
        if residual.ndim != 2 or np.any(~np.isfinite(residual)):
            raise ValueError("standardised raw residual map must be finite and 2D")
        if type(self.roi_pixel_count) is not int or self.roi_pixel_count < 1:
            raise ValueError("raw residual ROI pixel count must be positive")
        scalars = np.asarray(
            (self.roi_mean, self.roi_rms, self.roi_standard_deviation),
            dtype=float,
        )
        if np.any(~np.isfinite(scalars)) or np.any(scalars[1:] < 0.0):
            raise ValueError("raw residual statistics are invalid")
        object.__setattr__(self, "standardised_residual_map", _immutable(residual))


@dataclass(frozen=True)
class EndpointResidualDiagnostics:
    """Residual result for one endpoint, including explicit fit failure."""

    endpoint_label: str
    status: EndpointDiagnosticStatus
    roles: tuple[EndpointRawRoleResidual, ...]
    message: str

    def __post_init__(self) -> None:
        if self.endpoint_label not in ENDPOINT_LABELS:
            raise ValueError("unknown residual endpoint label")
        if not self.message:
            raise ValueError("residual diagnostic message cannot be empty")
        if self.status == "success":
            if len(self.roles) != 3:
                raise ValueError("successful PCI residual diagnostics require three roles")
            if tuple(role.role_name for role in self.roles) != RAW_ROLE_NAMES:
                raise ValueError("raw residual roles changed canonical PCI order")
            if len({role.role_owner_id for role in self.roles}) != 3:
                raise ValueError("raw residual owner ids must be endpoint-local")
        elif self.status == "fit_failure":
            if self.roles:
                raise ValueError("failed endpoint residual diagnostics cannot report roles")
        else:
            raise ValueError("unknown endpoint residual status")


@dataclass(frozen=True)
class IndependentEndpointResidualSummary:
    """Two endpoint-local residual records with no pooled statistic."""

    endpoints: tuple[EndpointResidualDiagnostics, EndpointResidualDiagnostics]
    provenance: IndependentEndpointPairProvenance

    def __post_init__(self) -> None:
        if tuple(endpoint.endpoint_label for endpoint in self.endpoints) != ENDPOINT_LABELS:
            raise ValueError("residual summary changed endpoint order")
        if not isinstance(self.provenance, IndependentEndpointPairProvenance):
            raise TypeError("residual provenance has the wrong type")


@dataclass(frozen=True)
class EndpointZeroDensityNullDiagnostic:
    """Endpoint-local zero-density nuisance refit and data-objective comparison."""

    endpoint_label: str
    status: EndpointDiagnosticStatus
    nuisance_values: tuple[float, float] | None
    null_weighted_chi_square: float | None
    fitted_weighted_chi_square: float | None
    improvement_over_null: float | None
    message: str
    evidence_level: Literal["model_only"] = "model_only"

    def __post_init__(self) -> None:
        if self.endpoint_label not in ENDPOINT_LABELS:
            raise ValueError("unknown zero-density endpoint label")
        if not self.message or self.evidence_level != "model_only":
            raise ValueError("zero-density diagnostic identity changed")
        values = (
            self.null_weighted_chi_square,
            self.fitted_weighted_chi_square,
            self.improvement_over_null,
        )
        if self.status == "success":
            if self.nuisance_values is None or any(value is None for value in values):
                raise ValueError("successful zero-density diagnostic is incomplete")
            nuisance = np.asarray(self.nuisance_values, dtype=float)
            scalars = np.asarray(values, dtype=float)
            if (
                nuisance.shape != (2,)
                or np.any(~np.isfinite(nuisance))
                or np.any(nuisance < 0.0)
                or np.any(~np.isfinite(scalars))
                or scalars[0] < 0.0
                or scalars[1] < 0.0
            ):
                raise ValueError("zero-density diagnostic values are invalid")
        elif self.status == "fit_failure":
            if self.nuisance_values is not None or any(value is not None for value in values):
                raise ValueError("failed zero-density diagnostic cannot report values")
        else:
            raise ValueError("unknown zero-density diagnostic status")


@dataclass(frozen=True)
class IndependentEndpointZeroDensitySummary:
    """Independent zero-density null comparisons for By and Bz."""

    endpoints: tuple[
        EndpointZeroDensityNullDiagnostic,
        EndpointZeroDensityNullDiagnostic,
    ]
    provenance: IndependentEndpointPairProvenance

    def __post_init__(self) -> None:
        if tuple(endpoint.endpoint_label for endpoint in self.endpoints) != ENDPOINT_LABELS:
            raise ValueError("zero-density summary changed endpoint order")
        if not isinstance(self.provenance, IndependentEndpointPairProvenance):
            raise TypeError("zero-density provenance has the wrong type")


@dataclass(frozen=True)
class ObservableIdentifiabilityRecord:
    """Continuous local data-support metrics for one endpoint or paired quantity."""

    observable_name: str
    quantity: str
    estimate: float | None
    scaled_gradient_norm: float | None
    data_null_space_fraction: float | None
    active_bound_gradient_fraction: float | None
    identified_subspace_standard_uncertainty: float | None
    supported: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.observable_name not in (*OBSERVABLE_NAMES, DERIVED_OBSERVABLE_NAME):
            raise ValueError("unknown identifiability observable")
        allowed_quantities = (*ENDPOINT_LABELS, "delta_Bz_minus_By", "ratio_Bz_over_By")
        if self.quantity not in allowed_quantities:
            raise ValueError("unknown independent-endpoint quantity")
        metrics = (
            self.scaled_gradient_norm,
            self.data_null_space_fraction,
            self.active_bound_gradient_fraction,
            self.identified_subspace_standard_uncertainty,
        )
        if self.supported:
            if self.estimate is None or not np.isfinite(self.estimate):
                raise ValueError("supported identifiability requires a finite estimate")
            if any(value is None or not np.isfinite(value) for value in metrics):
                raise ValueError("supported identifiability metrics must be finite")
            assert self.scaled_gradient_norm is not None
            assert self.data_null_space_fraction is not None
            assert self.active_bound_gradient_fraction is not None
            assert self.identified_subspace_standard_uncertainty is not None
            if self.scaled_gradient_norm <= 0.0:
                raise ValueError("supported identifiability gradient must be non-zero")
            if not 0.0 <= self.data_null_space_fraction <= 1.0:
                raise ValueError("data-null fraction must lie in [0, 1]")
            if not 0.0 <= self.active_bound_gradient_fraction <= 1.0:
                raise ValueError("active-bound fraction must lie in [0, 1]")
            if self.identified_subspace_standard_uncertainty < 0.0 or self.reasons:
                raise ValueError("supported identifiability record is inconsistent")
        else:
            if any(value is not None for value in metrics):
                raise ValueError("unsupported identifiability cannot report metrics")
            if self.estimate is not None and not np.isfinite(self.estimate):
                raise ValueError("unsupported estimate must be finite or None")
            if not self.reasons:
                raise ValueError("unsupported identifiability requires reasons")


@dataclass(frozen=True, eq=False)
class EndpointIdentifiabilityComponent:
    """Data-only singular geometry for one endpoint likelihood."""

    endpoint_label: str
    status: EndpointDiagnosticStatus
    message: str
    parameter_count: int
    data_rank: int
    relative_rank_tolerance: float
    singular_values: FloatArray
    data_condition_number: float
    active_bound_parameter_count: int
    records: tuple[ObservableIdentifiabilityRecord, ...]

    def __post_init__(self) -> None:
        if self.endpoint_label not in ENDPOINT_LABELS:
            raise ValueError("unknown identifiability endpoint label")
        if self.status not in ("success", "fit_failure") or not self.message:
            raise ValueError("endpoint identifiability status is invalid")
        if (
            type(self.parameter_count) is not int
            or type(self.data_rank) is not int
            or type(self.active_bound_parameter_count) is not int
            or self.parameter_count < 1
            or not 0 <= self.data_rank <= self.parameter_count
            or not 0 <= self.active_bound_parameter_count <= self.parameter_count
        ):
            raise ValueError("endpoint identifiability counts are invalid")
        if not 0.0 < self.relative_rank_tolerance < 1.0:
            raise ValueError("relative rank tolerance must lie in (0, 1)")
        singular = np.asarray(self.singular_values, dtype=float)
        if (
            singular.shape != (self.parameter_count,)
            or np.any(~np.isfinite(singular))
            or np.any(singular < 0.0)
            or np.any(np.diff(singular) > 0.0)
        ):
            raise ValueError("endpoint singular spectrum is invalid")
        condition = _finite_or_infinite(
            self.data_condition_number,
            name="data condition number",
        )
        if condition < 1.0:
            raise ValueError("data condition number cannot be smaller than one")
        if len(self.records) != len(OBSERVABLE_NAMES) + 1:
            raise ValueError("endpoint identifiability record membership changed")
        expected_names = (*OBSERVABLE_NAMES, DERIVED_OBSERVABLE_NAME)
        if tuple(record.observable_name for record in self.records) != expected_names:
            raise ValueError("endpoint identifiability observable order changed")
        if any(record.quantity != self.endpoint_label for record in self.records):
            raise ValueError("endpoint identifiability quantity changed")
        if self.status == "fit_failure" and (
            self.data_rank != 0
            or self.active_bound_parameter_count != 0
            or np.any(singular != 0.0)
            or any(record.supported for record in self.records)
        ):
            raise ValueError("failed endpoint identifiability must remain unsupported")
        object.__setattr__(self, "singular_values", _immutable(singular))


@dataclass(frozen=True)
class IndependentEndpointIdentifiabilitySummary:
    """Endpoint-local spectra and paired records from independent data blocks."""

    endpoints: tuple[EndpointIdentifiabilityComponent, EndpointIdentifiabilityComponent]
    paired_records: tuple[ObservableIdentifiabilityRecord, ...]
    provenance: IndependentEndpointPairProvenance
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if tuple(endpoint.endpoint_label for endpoint in self.endpoints) != ENDPOINT_LABELS:
            raise ValueError("identifiability summary changed endpoint order")
        expected: list[tuple[str, str]] = []
        for name in (*OBSERVABLE_NAMES, DERIVED_OBSERVABLE_NAME):
            expected.append((name, "delta_Bz_minus_By"))
            if name in POSITIVE_OBSERVABLE_NAMES:
                expected.append((name, "ratio_Bz_over_By"))
        actual = [
            (record.observable_name, record.quantity)
            for record in self.paired_records
        ]
        if actual != expected:
            raise ValueError("paired identifiability record membership changed")
        if not isinstance(self.provenance, IndependentEndpointPairProvenance):
            raise TypeError("identifiability provenance has the wrong type")
        if not self.assumptions or any(
            not isinstance(value, str) or not value.strip()
            for value in self.assumptions
        ):
            raise ValueError("identifiability assumptions cannot be empty")


@dataclass(frozen=True)
class IndependentConfidenceComponents:
    """Separate evidence axes for an independent endpoint quantity."""

    fit_and_data: EvidenceGrade
    detector_statistical: EvidenceGrade
    identifiability: EvidenceGrade
    calibration: EvidenceGrade
    forward_model: EvidenceGrade
    basis_model: ModelDependenceGrade
    support: ModelDependenceGrade
    reference: ModelDependenceGrade
    regularisation: ModelDependenceGrade
    repeatability: EvidenceGrade
    relative_change: EvidenceGrade
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        evidence_allowed = {
            "adequate",
            "limited",
            "not_assessed",
            "not_applicable",
            "failed",
        }
        dependence_allowed = {
            "stable",
            "sensitive",
            "not_assessed",
            "not_applicable",
            "failed",
        }
        evidence = (
            self.fit_and_data,
            self.detector_statistical,
            self.identifiability,
            self.calibration,
            self.forward_model,
            self.repeatability,
            self.relative_change,
        )
        dependence = (
            self.basis_model,
            self.support,
            self.reference,
            self.regularisation,
        )
        if any(value not in evidence_allowed for value in evidence):
            raise ValueError("unknown confidence evidence grade")
        if any(value not in dependence_allowed for value in dependence):
            raise ValueError("unknown model-dependence grade")
        mandatory_evidence = (
            self.fit_and_data,
            self.detector_statistical,
            self.identifiability,
            self.calibration,
            self.forward_model,
            self.repeatability,
        )
        if "not_applicable" in mandatory_evidence:
            raise ValueError("mandatory confidence evidence cannot be not_applicable")
        if self.basis_model == "not_applicable" or self.support == "not_applicable":
            raise ValueError("basis and support dependence are always applicable")
        complete = all(value in ("adequate", "not_applicable") for value in evidence)
        complete &= all(value in ("stable", "not_applicable") for value in dependence)
        if not complete and not self.reasons:
            raise ValueError("non-adequate confidence requires explicit reasons")


@dataclass(frozen=True)
class IndependentQuantityStatus:
    """One estimate kept alongside, not collapsed into, its confidence axes."""

    estimate: ConditionalEndpointEstimate
    information_level: InformationLevel
    confidence: IndependentConfidenceComponents

    def __post_init__(self) -> None:
        if not isinstance(self.estimate, ConditionalEndpointEstimate):
            raise TypeError("quantity-status estimate has the wrong type")
        if not isinstance(self.confidence, IndependentConfidenceComponents):
            raise TypeError("quantity-status confidence has the wrong type")
        expected = classify_independent_information_level(
            self.estimate,
            self.confidence,
        )
        if self.information_level != expected:
            raise ValueError("quantity status disagrees with its confidence components")


def classify_independent_information_level(
    estimate: ConditionalEndpointEstimate,
    confidence: IndependentConfidenceComponents,
) -> InformationLevel:
    """Classify orientation-v1 information without fabricating one-sided bounds.

    This packet carries point estimates and optional two-sided conditional
    intervals only.  It cannot return ``bounded`` without a separately typed,
    authority-approved one-sided bound payload.
    """

    if not isinstance(estimate, ConditionalEndpointEstimate):
        raise TypeError("estimate has the wrong type")
    if not isinstance(confidence, IndependentConfidenceComponents):
        raise TypeError("confidence has the wrong type")
    if estimate.quantity in ENDPOINT_LABELS:
        if confidence.relative_change != "not_applicable":
            raise ValueError(
                "relative_change must be not_applicable for an endpoint-local quantity"
            )
    elif confidence.relative_change == "not_applicable":
        raise ValueError(
            "relative_change must be assessed for an orientation difference or ratio"
        )
    if confidence.fit_and_data == "failed":
        return "fit_or_data_failure"
    quantitative_evidence = (
        confidence.detector_statistical,
        confidence.identifiability,
        confidence.calibration,
        confidence.forward_model,
    )
    dependence = (
        confidence.basis_model,
        confidence.support,
        confidence.reference,
        confidence.regularisation,
    )
    if "failed" in quantitative_evidence or "failed" in dependence:
        return "unresolved"
    if estimate.status == "unresolved" or estimate.estimate is None:
        return "unresolved"
    if "sensitive" in dependence:
        return "prior_sensitive"
    evidence = (
        confidence.fit_and_data,
        confidence.detector_statistical,
        confidence.identifiability,
        confidence.calibration,
        confidence.forward_model,
        confidence.repeatability,
        confidence.relative_change,
    )
    if (
        estimate.status == "complete"
        and all(value in ("adequate", "not_applicable") for value in evidence)
        and all(value in ("stable", "not_applicable") for value in dependence)
    ):
        return "quantitatively_resolved"
    return "informative_but_inconclusive"


def _validate_pair_inputs(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    fit: IndependentEndpointPairFit,
) -> None:
    _validate_input_pair(inputs, fit.provenance)
    for item, endpoint in zip(inputs, fit.endpoints, strict=True):
        if item.raw_block.role_owner_ids != endpoint.role_owner_ids:
            raise ValueError("diagnostic raw ownership differs from the fit")
    _validate_fit_input_binding(inputs, fit)


def analyse_independent_endpoint_residuals(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    fit: IndependentEndpointPairFit,
) -> IndependentEndpointResidualSummary:
    """Report residuals separately for each endpoint-owned raw block."""

    _validate_pair_inputs(inputs, fit)
    endpoints: list[EndpointResidualDiagnostics] = []
    for item, endpoint in zip(inputs, fit.endpoints, strict=True):
        if endpoint.status != "success" or endpoint.fit_result is None:
            endpoints.append(
                EndpointResidualDiagnostics(
                    endpoint_label=endpoint.endpoint_label,
                    status="fit_failure",
                    roles=(),
                    message=endpoint.message,
                )
            )
            continue
        prediction = endpoint.fit_result.prediction
        raw = item.raw_block
        if prediction.role_names != raw.role_names:
            raise ValueError("residual raw-role order differs from the fitted prediction")
        roi = np.asarray(item.operator.grid.roi_mask, dtype=bool)
        records: list[EndpointRawRoleResidual] = []
        whitened_parts: list[FloatArray] = []
        for role_name, owner, observed, expected, variance in zip(
            raw.role_names,
            raw.role_owner_ids,
            raw.observed_electrons,
            prediction.expected_electrons,
            prediction.conditional_variance_electrons2,
            strict=True,
        ):
            expected_array = np.asarray(expected, dtype=float)
            variance_array = np.asarray(variance, dtype=float)
            observed_array = np.asarray(observed, dtype=float)
            if (
                expected_array.shape != observed_array.shape
                or variance_array.shape != observed_array.shape
                or np.any(~np.isfinite(expected_array))
                or np.any(~np.isfinite(variance_array))
                or np.any(variance_array <= 0.0)
            ):
                raise ValueError("raw residual prediction arrays are invalid")
            residual = (observed_array - expected_array) / np.sqrt(variance_array)
            roi_values = residual[roi]
            whitened_parts.append(roi_values)
            records.append(
                EndpointRawRoleResidual(
                    role_name=role_name,
                    role_owner_id=owner,
                    standardised_residual_map=residual,
                    roi_pixel_count=int(roi_values.size),
                    roi_mean=float(np.mean(roi_values)),
                    roi_rms=float(np.sqrt(np.mean(roi_values**2))),
                    roi_standard_deviation=float(np.std(roi_values)),
                )
            )
        stored = np.asarray(
            endpoint.fit_result.diagnostics.whitened_residual_vector,
            dtype=float,
        )
        recomputed = np.concatenate(whitened_parts)
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
            raise ValueError("raw block is not the observation that produced the fit")
        endpoints.append(
            EndpointResidualDiagnostics(
                endpoint_label=endpoint.endpoint_label,
                status="success",
                roles=tuple(records),
                message="endpoint-local raw residuals verified",
            )
        )
    return IndependentEndpointResidualSummary(
        endpoints=(endpoints[0], endpoints[1]),
        provenance=fit.provenance,
    )


def _flatten_endpoint_roles(
    arrays: tuple[ArrayLike, ...],
    roi_mask: BoolArray,
) -> FloatArray:
    return np.concatenate([np.asarray(array, dtype=float)[roi_mask] for array in arrays])


def _fit_zero_density_endpoint(
    item: IndependentEndpointFitInput,
    endpoint_fit: IndependentEndpointFit,
) -> EndpointZeroDensityNullDiagnostic:
    endpoint_label = item.raw_block.endpoint_label
    result = endpoint_fit.fit_result
    status = endpoint_fit.status
    message = endpoint_fit.message
    if status != "success" or result is None:
        return EndpointZeroDensityNullDiagnostic(
            endpoint_label=endpoint_label,
            status="fit_failure",
            nuisance_values=None,
            null_weighted_chi_square=None,
            fitted_weighted_chi_square=None,
            improvement_over_null=None,
            message=message,
        )
    lower = _bound_vector(item.nuisance_lower, size=2, name="nuisance_lower")
    upper = _bound_vector(item.nuisance_upper, size=2, name="nuisance_upper")
    if np.any(lower < 0.0) or np.any(upper <= lower):
        raise ValueError("zero-density nuisance bounds are invalid")
    current = np.asarray(result.nuisance_values, dtype=float)
    if current.shape != (2,) or np.any(current < lower) or np.any(current > upper):
        raise ValueError("fitted nuisance values lie outside null bounds")
    zero_coefficients = np.zeros(item.model.parameter_count, dtype=float)
    zero_density = item.model.column_density(zero_coefficients)
    null_role_names, unit_illumination = (
        item.operator.expected_linked_sequence_from_density_maps(
            [zero_density],
            PCINuisanceValues(1.0, 0.0),
        )
    )
    if null_role_names != RAW_ROLE_NAMES:
        raise ValueError("zero-density prediction changed PCI raw-role order")
    observed = _flatten_endpoint_roles(
        item.raw_block.observed_electrons,
        item.operator.grid.roi_mask,
    )
    fit_options = item.options or LinkedScalarFitOptions()
    iterations = fit_options.irls_iterations

    def prediction(
        values: FloatArray,
    ) -> tuple[tuple[FloatArray, ...], tuple[FloatArray, ...]]:
        nuisance = PCINuisanceValues(*values)
        expected = tuple(
            np.asarray(
                nuisance.i0_photoelectrons_per_pixel * unit_role
                + nuisance.dark_electrons_per_pixel,
                dtype=float,
            )
            for unit_role in unit_illumination
        )
        variance = tuple(
            np.asarray(
                (
                    role
                    + item.operator.read_noise_electrons**2
                )
                / item.operator.independent_exposures_by_role[
                    "atom" if role_index == 0 else null_role_names[role_index]
                ],
                dtype=float,
            )
            for role_index, role in enumerate(expected)
        )
        return expected, variance

    try:
        final = None
        for _ in range(iterations):
            _, initial_variance = prediction(current)
            standard_deviation = np.sqrt(
                _flatten_endpoint_roles(
                    initial_variance,
                    item.operator.grid.roi_mask,
                )
            )

            def residual(values: FloatArray) -> FloatArray:
                return (
                    observed
                    - _flatten_endpoint_roles(
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
                max_nfev=fit_options.max_nfev,
                xtol=fit_options.xtol,
                ftol=fit_options.ftol,
                gtol=fit_options.gtol,
            )
            current = np.asarray(final.x, dtype=float)
        if final is None or not final.success or np.any(~np.isfinite(current)):
            failure_message = "zero-density nuisance refit did not converge"
            if final is not None:
                failure_message = str(final.message)
            return EndpointZeroDensityNullDiagnostic(
                endpoint_label=endpoint_label,
                status="fit_failure",
                nuisance_values=None,
                null_weighted_chi_square=None,
                fitted_weighted_chi_square=None,
                improvement_over_null=None,
                message=failure_message,
            )
        null_expected, null_variance = prediction(current)
        null_standard_deviation = np.sqrt(
            _flatten_endpoint_roles(
                null_variance,
                item.operator.grid.roi_mask,
            )
        )
        null_residual = (
            observed
            - _flatten_endpoint_roles(
                null_expected,
                item.operator.grid.roi_mask,
            )
        ) / null_standard_deviation
        null_chi = float(null_residual @ null_residual)
        fitted_chi = float(result.diagnostics.weighted_chi_square)
        return EndpointZeroDensityNullDiagnostic(
            endpoint_label=endpoint_label,
            status="success",
            nuisance_values=(float(current[0]), float(current[1])),
            null_weighted_chi_square=null_chi,
            fitted_weighted_chi_square=fitted_chi,
            improvement_over_null=float(null_chi - fitted_chi),
            message="endpoint-local zero-density nuisance refit converged",
        )
    except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return EndpointZeroDensityNullDiagnostic(
            endpoint_label=endpoint_label,
            status="fit_failure",
            nuisance_values=None,
            null_weighted_chi_square=None,
            fitted_weighted_chi_square=None,
            improvement_over_null=None,
            message=f"{type(exc).__name__}: {exc}",
        )


def analyse_independent_zero_density_null(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    fit: IndependentEndpointPairFit,
) -> IndependentEndpointZeroDensitySummary:
    """Refit zero-density nuisances independently for By and Bz."""

    _validate_pair_inputs(inputs, fit)
    analyse_independent_endpoint_residuals(inputs, fit)
    records = tuple(
        _fit_zero_density_endpoint(item, endpoint)
        for item, endpoint in zip(inputs, fit.endpoints, strict=True)
    )
    return IndependentEndpointZeroDensitySummary(
        endpoints=(records[0], records[1]),
        provenance=fit.provenance,
    )


def _observable_values_and_gradients(
    item: IndependentEndpointFitInput,
    coefficients: FloatArray,
    support: ObservableIntegrationSupport,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    density = np.asarray(item.model.column_density(coefficients), dtype=float)
    summary = extract_density_observables(density, support)
    values = np.asarray(
        [
            summary.integrated_response,
            np.nan if summary.centroid_y_m is None else summary.centroid_y_m * 1e6,
            np.nan if summary.centroid_z_m is None else summary.centroid_z_m * 1e6,
            np.nan if summary.covariance_m2 is None else np.sqrt(summary.covariance_m2[0, 0]) * 1e6,
            np.nan if summary.covariance_m2 is None else np.sqrt(summary.covariance_m2[1, 1]) * 1e6,
        ],
        dtype=float,
    )
    supported = np.isfinite(values)
    mask = support.support_mask
    area = np.where(mask, support.cell_area_m2, 0.0)
    pixel_gradients = np.full((len(OBSERVABLE_NAMES), *support.shape), np.nan)
    pixel_gradients[0] = area
    if summary.centroid_m is not None and summary.covariance_m2 is not None:
        integrated = summary.integrated_response
        y_c, z_c = summary.centroid_m
        dy = support.y_grid_m - y_c
        dz = support.z_grid_m - z_c
        pixel_gradients[1] = area * dy / integrated * 1e6
        pixel_gradients[2] = area * dz / integrated * 1e6
        sigma_y = float(np.sqrt(summary.covariance_m2[0, 0]))
        sigma_z = float(np.sqrt(summary.covariance_m2[1, 1]))
        if sigma_y > 0.0:
            pixel_gradients[3] = (
                area * (dy**2 - summary.covariance_m2[0, 0])
                / (2.0 * sigma_y * integrated) * 1e6
            )
        else:
            supported[3] = False
            values[3] = np.nan
        if sigma_z > 0.0:
            pixel_gradients[4] = (
                area * (dz**2 - summary.covariance_m2[1, 1])
                / (2.0 * sigma_z * integrated) * 1e6
            )
        else:
            supported[4] = False
            values[4] = np.nan
    gradients = np.full((len(OBSERVABLE_NAMES), item.model.parameter_count), np.nan)
    populated = np.zeros(item.model.parameter_count, dtype=bool)
    for parameter_slice, derivatives in item.model.iter_column_density_jacobian(
        coefficients,
        item.operator.jacobian_batch_size,
    ):
        if parameter_slice.start is None or parameter_slice.stop is None:
            raise ValueError("observable-gradient slices require explicit bounds")
        if np.any(populated[parameter_slice]):
            raise ValueError("observable-gradient slices overlap")
        for observable_index in range(len(OBSERVABLE_NAMES)):
            if supported[observable_index]:
                gradients[observable_index, parameter_slice] = np.einsum(
                    "pij,ij->p",
                    derivatives,
                    pixel_gradients[observable_index],
                    optimize=True,
                )
        populated[parameter_slice] = True
    if not np.all(populated):
        raise ValueError("observable gradients do not cover every model parameter")
    return values, supported, gradients


@dataclass(frozen=True)
class _EndpointLinearGeometry:
    item: IndependentEndpointFitInput
    spans: FloatArray
    active: BoolArray
    singular_values: FloatArray
    right_vectors: FloatArray
    rank: int
    condition: float
    values: FloatArray
    supported: BoolArray
    gradients: FloatArray


def _endpoint_geometry(
    item: IndependentEndpointFitInput,
    endpoint: IndependentEndpointFit,
    *,
    relative_rank_tolerance: float,
    relative_active_bound_tolerance: float,
) -> _EndpointLinearGeometry:
    result = endpoint.fit_result
    if endpoint.status != "success" or result is None:
        raise ValueError("identifiability requires successful endpoint fits")
    density_lower = _bound_vector(
        item.density_parameter_lower,
        size=item.model.parameter_count,
        name="density_parameter_lower",
    )
    density_upper = _bound_vector(
        item.density_coefficient_upper,
        size=item.model.parameter_count,
        name="density_coefficient_upper",
    )
    nuisance_lower = _bound_vector(item.nuisance_lower, size=2, name="nuisance_lower")
    nuisance_upper = _bound_vector(item.nuisance_upper, size=2, name="nuisance_upper")
    if np.any(density_upper <= density_lower) or np.any(nuisance_upper <= nuisance_lower):
        raise ValueError("identifiability bounds must be strictly ordered")
    lower = np.concatenate([density_lower, nuisance_lower])
    upper = np.concatenate([density_upper, nuisance_upper])
    spans = upper - lower
    parameters = np.concatenate([result.density_coefficients[0], result.nuisance_values])
    if np.any(parameters < lower) or np.any(parameters > upper):
        raise ValueError("fitted endpoint parameters lie outside declared bounds")
    active = (parameters - lower <= relative_active_bound_tolerance * spans) | (
        upper - parameters <= relative_active_bound_tolerance * spans
    )
    variance = _flatten_endpoint_roles(
        result.prediction.conditional_variance_electrons2,
        item.operator.grid.roi_mask,
    )
    jacobian = np.asarray(result.prediction.jacobian, dtype=float)
    if jacobian.shape != (variance.size, spans.size) or np.any(variance <= 0.0):
        raise ValueError("endpoint likelihood Jacobian or variance is invalid")
    scaled = jacobian / np.sqrt(variance)[:, None] * spans[None, :]
    triangular = np.linalg.qr(scaled, mode="r")
    _, singular, right_transpose = np.linalg.svd(
        triangular,
        full_matrices=triangular.shape[0] < spans.size,
    )
    spectrum = np.zeros(spans.size, dtype=float)
    spectrum[: singular.size] = singular
    largest = float(spectrum[0]) if spectrum.size else 0.0
    rank = int(np.count_nonzero(spectrum > relative_rank_tolerance * largest)) if largest > 0.0 else 0
    condition = (
        float(spectrum[0] / spectrum[-1])
        if rank == spans.size and spectrum[-1] > 0.0
        else float("inf")
    )
    values, supported, gradients = _observable_values_and_gradients(
        item,
        result.density_coefficients[0],
        item.observable_support,
    )
    return _EndpointLinearGeometry(
        item=item,
        spans=spans,
        active=active,
        singular_values=spectrum,
        right_vectors=np.asarray(right_transpose.T, dtype=float),
        rank=rank,
        condition=condition,
        values=values,
        supported=supported,
        gradients=gradients,
    )


def _failed_endpoint_geometry(
    item: IndependentEndpointFitInput,
) -> _EndpointLinearGeometry:
    """Return an all-unsupported local block without fabricating fit evidence."""

    density_lower = _bound_vector(
        item.density_parameter_lower,
        size=item.model.parameter_count,
        name="density_parameter_lower",
    )
    density_upper = _bound_vector(
        item.density_coefficient_upper,
        size=item.model.parameter_count,
        name="density_coefficient_upper",
    )
    nuisance_lower = _bound_vector(item.nuisance_lower, size=2, name="nuisance_lower")
    nuisance_upper = _bound_vector(item.nuisance_upper, size=2, name="nuisance_upper")
    spans = np.concatenate(
        (density_upper - density_lower, nuisance_upper - nuisance_lower)
    )
    if np.any(spans <= 0.0):
        raise ValueError("failed-endpoint identifiability bounds must be ordered")
    parameter_count = spans.size
    return _EndpointLinearGeometry(
        item=item,
        spans=spans,
        active=np.zeros(parameter_count, dtype=bool),
        singular_values=np.zeros(parameter_count, dtype=float),
        right_vectors=np.eye(parameter_count, dtype=float),
        rank=0,
        condition=float("inf"),
        values=np.full(len(OBSERVABLE_NAMES), np.nan, dtype=float),
        supported=np.zeros(len(OBSERVABLE_NAMES), dtype=bool),
        gradients=np.full(
            (len(OBSERVABLE_NAMES), item.model.parameter_count),
            np.nan,
            dtype=float,
        ),
    )


def _local_record(
    *,
    observable_name: str,
    quantity: str,
    estimate: float | None,
    local_gradients: tuple[FloatArray | None, FloatArray | None],
    geometries: tuple[_EndpointLinearGeometry, _EndpointLinearGeometry],
    reason: str,
) -> ObservableIdentifiabilityRecord:
    if estimate is None or not np.isfinite(estimate) or any(
        gradient is None for gradient in local_gradients
    ):
        return ObservableIdentifiabilityRecord(
            observable_name=observable_name,
            quantity=quantity,
            estimate=None if estimate is None or not np.isfinite(estimate) else float(estimate),
            scaled_gradient_norm=None,
            data_null_space_fraction=None,
            active_bound_gradient_fraction=None,
            identified_subspace_standard_uncertainty=None,
            supported=False,
            reasons=(reason,),
        )
    scaled_parts: list[FloatArray] = []
    null_square = 0.0
    active_square = 0.0
    uncertainty_square = 0.0
    for local_gradient, geometry in zip(local_gradients, geometries, strict=True):
        assert local_gradient is not None
        full_gradient = np.zeros(geometry.spans.size, dtype=float)
        full_gradient[: geometry.item.model.parameter_count] = local_gradient
        scaled = full_gradient * geometry.spans
        scaled_parts.append(scaled)
        identified = geometry.right_vectors[:, : geometry.rank].T @ scaled
        null = geometry.right_vectors[:, geometry.rank :].T @ scaled
        null_square += float(null @ null)
        active_square += float(scaled[geometry.active] @ scaled[geometry.active])
        if geometry.rank:
            uncertainty_square += float(
                np.sum((identified / geometry.singular_values[: geometry.rank]) ** 2)
            )
    gradient_norm = float(np.sqrt(sum(float(part @ part) for part in scaled_parts)))
    if gradient_norm == 0.0:
        return ObservableIdentifiabilityRecord(
            observable_name=observable_name,
            quantity=quantity,
            estimate=float(estimate),
            scaled_gradient_norm=None,
            data_null_space_fraction=None,
            active_bound_gradient_fraction=None,
            identified_subspace_standard_uncertainty=None,
            supported=False,
            reasons=("observable_has_zero_local_gradient",),
        )
    return ObservableIdentifiabilityRecord(
        observable_name=observable_name,
        quantity=quantity,
        estimate=float(estimate),
        scaled_gradient_norm=gradient_norm,
        data_null_space_fraction=float(min(max(np.sqrt(null_square) / gradient_norm, 0.0), 1.0)),
        active_bound_gradient_fraction=float(min(max(np.sqrt(active_square) / gradient_norm, 0.0), 1.0)),
        identified_subspace_standard_uncertainty=float(np.sqrt(max(uncertainty_square, 0.0))),
        supported=True,
        reasons=(),
    )


def analyse_independent_endpoint_identifiability(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    fit: IndependentEndpointPairFit,
    *,
    relative_rank_tolerance: float = 1e-10,
    relative_active_bound_tolerance: float = 1e-6,
) -> IndependentEndpointIdentifiabilitySummary:
    """Analyse local data support using two separate endpoint likelihood blocks."""

    _validate_pair_inputs(inputs, fit)
    analyse_independent_endpoint_residuals(inputs, fit)
    if not 0.0 < relative_rank_tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie in (0, 1)")
    if not 0.0 < relative_active_bound_tolerance < 1.0:
        raise ValueError("relative_active_bound_tolerance must lie in (0, 1)")
    geometries = tuple(
        (
            _endpoint_geometry(
                item,
                endpoint,
                relative_rank_tolerance=relative_rank_tolerance,
                relative_active_bound_tolerance=relative_active_bound_tolerance,
            )
            if endpoint.status == "success"
            else _failed_endpoint_geometry(item)
        )
        for item, endpoint in zip(inputs, fit.endpoints, strict=True)
    )
    endpoint_components: list[EndpointIdentifiabilityComponent] = []
    endpoint_local_gradients: list[dict[str, FloatArray | None]] = []
    endpoint_values: list[dict[str, float | None]] = []
    for endpoint_index, geometry in enumerate(geometries):
        records: list[ObservableIdentifiabilityRecord] = []
        gradients_by_name: dict[str, FloatArray | None] = {}
        values_by_name: dict[str, float | None] = {}
        for observable_index, observable_name in enumerate(OBSERVABLE_NAMES):
            supported = bool(geometry.supported[observable_index])
            value = float(geometry.values[observable_index]) if supported else None
            gradient = geometry.gradients[observable_index] if supported else None
            gradients_by_name[observable_name] = gradient
            values_by_name[observable_name] = value
            zero = np.zeros(geometry.item.model.parameter_count, dtype=float)
            local_pair = (
                (gradient, zero) if endpoint_index == 0 else (zero, gradient)
            )
            records.append(
                _local_record(
                    observable_name=observable_name,
                    quantity=ENDPOINT_LABELS[endpoint_index],
                    estimate=value,
                    local_gradients=local_pair,
                    geometries=geometries,
                    reason="endpoint observable gradient is not supported",
                )
            )
        sigma_y = values_by_name["sigma_y_um"]
        sigma_z = values_by_name["sigma_z_um"]
        sigma_y_gradient = gradients_by_name["sigma_y_um"]
        sigma_z_gradient = gradients_by_name["sigma_z_um"]
        if (
            sigma_y is not None
            and sigma_z is not None
            and sigma_y > 0.0
            and sigma_z > 0.0
            and sigma_y_gradient is not None
            and sigma_z_gradient is not None
        ):
            aspect = sigma_y / sigma_z
            aspect_gradient = (
                sigma_y_gradient / sigma_z
                - sigma_y * sigma_z_gradient / sigma_z**2
            )
        else:
            aspect = None
            aspect_gradient = None
        gradients_by_name[DERIVED_OBSERVABLE_NAME] = aspect_gradient
        values_by_name[DERIVED_OBSERVABLE_NAME] = aspect
        zero = np.zeros(geometry.item.model.parameter_count, dtype=float)
        records.append(
            _local_record(
                observable_name=DERIVED_OBSERVABLE_NAME,
                quantity=ENDPOINT_LABELS[endpoint_index],
                estimate=aspect,
                local_gradients=(aspect_gradient, zero) if endpoint_index == 0 else (zero, aspect_gradient),
                geometries=geometries,
                reason="positive jointly supported widths are required for aspect ratio",
            )
        )
        endpoint_local_gradients.append(gradients_by_name)
        endpoint_values.append(values_by_name)
        endpoint_components.append(
            EndpointIdentifiabilityComponent(
                endpoint_label=ENDPOINT_LABELS[endpoint_index],
                status=fit.endpoints[endpoint_index].status,
                message=(
                    "endpoint-local likelihood geometry evaluated"
                    if fit.endpoints[endpoint_index].status == "success"
                    else fit.endpoints[endpoint_index].message
                ),
                parameter_count=int(geometry.spans.size),
                data_rank=geometry.rank,
                relative_rank_tolerance=relative_rank_tolerance,
                singular_values=geometry.singular_values,
                data_condition_number=geometry.condition,
                active_bound_parameter_count=int(np.count_nonzero(geometry.active)),
                records=tuple(records),
            )
        )

    paired: list[ObservableIdentifiabilityRecord] = []
    for name in (*OBSERVABLE_NAMES, DERIVED_OBSERVABLE_NAME):
        left_value = endpoint_values[0][name]
        right_value = endpoint_values[1][name]
        left_gradient = endpoint_local_gradients[0][name]
        right_gradient = endpoint_local_gradients[1][name]
        both = (
            left_value is not None
            and right_value is not None
            and left_gradient is not None
            and right_gradient is not None
        )
        paired.append(
            _local_record(
                observable_name=name,
                quantity="delta_Bz_minus_By",
                estimate=(float(right_value - left_value) if both else None),
                local_gradients=(
                    -left_gradient if left_gradient is not None else None,
                    right_gradient,
                ),
                geometries=geometries,
                reason="both endpoint gradients are required for the orientation difference",
            )
        )
        if name in POSITIVE_OBSERVABLE_NAMES:
            calibrated = name != "A"
            ratio_supported = bool(
                both
                and calibrated
                and left_value is not None
                and right_value is not None
                and left_value > 0.0
                and right_value > 0.0
            )
            if ratio_supported:
                assert left_value is not None and right_value is not None
                assert left_gradient is not None and right_gradient is not None
                ratio_value = right_value / left_value
                ratio_gradients = (
                    -right_value * left_gradient / left_value**2,
                    right_gradient / left_value,
                )
                reason = "positive endpoint quantities are required for the orientation ratio"
            else:
                ratio_value = None
                ratio_gradients = (None, None)
                reason = (
                    "cross_orientation_amplitude_calibration_not_supplied"
                    if name == "A" and not calibrated
                    else "positive endpoint quantities are required for the orientation ratio"
                )
            paired.append(
                _local_record(
                    observable_name=name,
                    quantity="ratio_Bz_over_By",
                    estimate=ratio_value,
                    local_gradients=ratio_gradients,
                    geometries=geometries,
                    reason=reason,
                )
            )
    return IndependentEndpointIdentifiabilitySummary(
        endpoints=(endpoint_components[0], endpoint_components[1]),
        paired_records=tuple(paired),
        provenance=fit.provenance,
        assumptions=(
            "local linearisation at each independent endpoint point fit",
            "noise-whitened raw-count Jacobians without curvature rows",
            "parameter coordinates scaled by their declared endpoint-local bounds",
            "paired gradients combine independent endpoint likelihood blocks only",
            "identified-subspace uncertainty excludes every data-null component",
            "continuous metrics are not converted to a confidence grade",
        ),
    )


__all__ = [
    "EndpointIdentifiabilityComponent",
    "EndpointRawRoleResidual",
    "EndpointResidualDiagnostics",
    "EndpointZeroDensityNullDiagnostic",
    "EvidenceGrade",
    "IndependentConfidenceComponents",
    "IndependentEndpointIdentifiabilitySummary",
    "IndependentEndpointResidualSummary",
    "IndependentEndpointZeroDensitySummary",
    "IndependentQuantityStatus",
    "InformationLevel",
    "ModelDependenceGrade",
    "ObservableIdentifiabilityRecord",
    "analyse_independent_endpoint_identifiability",
    "analyse_independent_endpoint_residuals",
    "analyse_independent_zero_density_null",
    "classify_independent_information_level",
]
