"""Information-first summaries for linked PCI/DGI observable fits.

This successor layer keeps the established linked raw-count operators and
per-frame nuisance-density fit, but does not require a truth-derived affine
calibration before reporting what the data contain.  It preserves every
requested bootstrap draw in its original row and records support separately
for each observable.  The latent density maps remain nuisance fields and are
never returned as recovered images by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .linked_scalar_fit import (
    LinkedRawObservation,
    LinkedScalarFitOptions,
    LinkedScalarFitResult,
    draw_linked_raw_observation,
    fit_linked_scalar_sequence,
    nuisance_from_vector,
    nuisance_vector,
)
from .object_models import (
    DifferentiableColumnDensityModel,
    NonnegativeBilinearDensityModel,
)
from .observable_calibration import OBSERVABLE_NAMES
from .observables import ObservableIntegrationSupport, extract_density_observables
from .regularisation import CurvatureRegularisation
from .scalar_measurements import (
    DGILinkedRawOperator,
    DGINuisanceValues,
    LinkedRawSequencePrediction,
    PCILinkedRawOperator,
    PCINuisanceValues,
)


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
LinkedScalarOperator: TypeAlias = PCILinkedRawOperator | DGILinkedRawOperator
LinkedNuisanceValues: TypeAlias = PCINuisanceValues | DGINuisanceValues
BootstrapDrawStatus = Literal["success", "fit_failure"]
LinkedNullEvidenceLevel = Literal[
    "model_only",
    "synthetic_blank_development_rank",
]

ConditionalEstimateStatus = Literal["complete", "partial", "unresolved"]
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
EstimateForm = Literal[
    "two_sided_interval",
    "point_only",
    "lower_bound",
    "upper_bound",
    "none",
]
InformationLevel = Literal[
    "quantitatively_resolved",
    "informative_but_inconclusive",
    "bounded",
    "prior_sensitive",
    "unresolved",
    "fit_or_data_failure",
]
InformationQuantity = Literal["q1", "q2", "delta_21", "ratio_21"]

POSITIVE_OBSERVABLE_NAMES: tuple[str, ...] = (
    "A",
    "sigma_y_um",
    "sigma_z_um",
)
DERIVED_OBSERVABLE_NAMES: tuple[str, ...] = ("aspect_ratio_y_over_z",)
OBSERVABLE_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "A": "response_integral",
        "y_c_um": "um",
        "z_c_um": "um",
        "sigma_y_um": "um",
        "sigma_z_um": "um",
        "aspect_ratio_y_over_z": "1",
    }
)


@dataclass(frozen=True)
class ReferenceLightInferenceProvenance:
    """Caller-declared provenance for the linked two-exposure inverse route.

    The code can verify the flexible model family and reproduce the supplied
    linked fit from that model and operator.  It cannot infer how an upstream
    initialiser, support or regulariser was selected, so those sources are
    required explicitly and travel with every reported result.
    """

    contract_label: str
    method: Literal["PCI", "DGI"]
    fluence_mw_us: float
    detuning_ghz: float
    selected_eigenmode: str
    exposure_indices: tuple[int, int]
    imaged_pre_pulse_state_indices: tuple[int, int]
    observation_source: str
    initialisation_source: str
    support_source: str
    regularisation_source: str
    regularisation_applied: bool
    thermodynamic_prediction_used: bool
    truth_template_used: bool
    reference_template_used: bool
    temporal_coupling_used: bool
    truth_derived_initialisation_used: bool
    target_derived_support_used: bool
    truth_derived_affine_calibration_used: bool

    def __post_init__(self) -> None:
        if self.contract_label != "linked_two_exposure_information_v1":
            raise ValueError("reference-light provenance contract label changed")
        if self.method not in ("PCI", "DGI"):
            raise ValueError("reference-light method must be PCI or DGI")
        fluence = float(self.fluence_mw_us)
        detuning = float(self.detuning_ghz)
        if not np.isfinite(fluence) or fluence != 300.0:
            raise ValueError("linked two-exposure provenance requires F=300 mW us")
        if not np.isfinite(detuning) or detuning != 1.5:
            raise ValueError("linked two-exposure provenance requires +1.5 GHz detuning")
        if self.selected_eigenmode != "perpendicular":
            raise ValueError("linked two-exposure provenance requires the perpendicular mode")
        exposure_indices = tuple(self.exposure_indices)
        state_indices = tuple(self.imaged_pre_pulse_state_indices)
        if exposure_indices != (1, 2) or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in exposure_indices
        ):
            raise ValueError("reference-light exposure indices must be (1, 2)")
        if state_indices != (0, 1) or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in state_indices
        ):
            raise ValueError("reference-light q1/q2 states must be (0, 1)")
        sources = (
            self.observation_source,
            self.initialisation_source,
            self.support_source,
            self.regularisation_source,
        )
        if any(not isinstance(value, str) or not value.strip() for value in sources):
            raise ValueError("reference-light provenance sources must be non-empty text")
        flags = (
            self.regularisation_applied,
            self.thermodynamic_prediction_used,
            self.truth_template_used,
            self.reference_template_used,
            self.temporal_coupling_used,
            self.truth_derived_initialisation_used,
            self.target_derived_support_used,
            self.truth_derived_affine_calibration_used,
        )
        if any(type(value) is not bool for value in flags):
            raise TypeError("reference-light provenance flags must be bool")
        if any(
            (
                self.thermodynamic_prediction_used,
                self.truth_template_used,
                self.reference_template_used,
                self.temporal_coupling_used,
                self.truth_derived_initialisation_used,
                self.target_derived_support_used,
                self.truth_derived_affine_calibration_used,
            )
        ):
            raise ValueError(
                "primary reference-light provenance cannot consume a prediction, "
                "truth/reference template, target-derived support/calibration or "
                "temporal coupling"
            )
        if self.regularisation_applied == (
            self.regularisation_source.strip().lower() == "none"
        ):
            raise ValueError(
                "regularisation_applied and regularisation_source are inconsistent"
            )
        object.__setattr__(self, "fluence_mw_us", fluence)
        object.__setattr__(self, "detuning_ghz", detuning)
        object.__setattr__(self, "exposure_indices", exposure_indices)
        object.__setattr__(self, "imaged_pre_pulse_state_indices", state_indices)

    @property
    def assumptions(self) -> tuple[str, ...]:
        """Return explicit, caller-declared route assumptions for reporting."""

        return (
            "caller-declared reference-light primary inference route",
            f"validated acquisition declaration: {self.method}, F=300 mW us, +1.5 GHz, perpendicular mode",
            "frame declaration: exposure 1/q1 images state 0; exposure 2/q2 images state 1",
            "verified flexible model family: NonnegativeBilinearDensityModel",
            f"observation source: {self.observation_source}",
            f"initialisation source: {self.initialisation_source}",
            f"support source: {self.support_source}",
            f"regularisation source: {self.regularisation_source}",
            "caller declares no thermodynamic prediction, truth/reference template, truth-derived initialisation, target-derived support, affine truth calibration or temporal coupling was used",
        )


def _immutable(values: ArrayLike, *, dtype: type = float) -> NDArray:
    array = np.array(values, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def bilinear_effective_support_mask(
    model: NonnegativeBilinearDensityModel,
) -> BoolArray:
    """Return cells where the frozen bilinear model can represent density.

    The declared support can extend beyond the knot basis. Evaluating all
    non-negative coefficients at one identifies the subset that the current
    bilinear basis can actually represent.
    """

    if not isinstance(model, NonnegativeBilinearDensityModel):
        raise TypeError(
            "effective support requires a NonnegativeBilinearDensityModel"
        )
    y_grid = np.asarray(model.y_grid_m, dtype=float)
    z_grid = np.asarray(model.z_grid_m, dtype=float)
    support = np.asarray(model.support_mask, dtype=bool)
    if y_grid.ndim != 2 or z_grid.shape != y_grid.shape:
        raise ValueError("bilinear model coordinates must be same-shape 2D arrays")
    if support.shape != y_grid.shape:
        raise ValueError("bilinear model support has the wrong grid shape")
    if not np.any(support):
        raise ValueError("bilinear model support must be non-empty")

    represented_density = np.asarray(
        model.column_density(np.ones(model.parameter_count, dtype=float)),
        dtype=float,
    )
    if represented_density.shape != support.shape:
        raise RuntimeError("bilinear model density has the wrong grid shape")
    if np.any(~np.isfinite(represented_density)):
        raise RuntimeError("bilinear model density must be finite")
    if np.any(represented_density < 0.0):
        raise RuntimeError("bilinear model density must be non-negative")
    effective = represented_density > 0.0
    if not np.any(effective):
        raise RuntimeError("bilinear model has no representable support")
    if np.any(effective & ~support):
        raise RuntimeError("effective bilinear support exceeds declared support")
    return _immutable(effective, dtype=bool)


def _strict_int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _text_tuple(
    values: object,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of strings, not one string")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of strings") from exc
    if (not allow_empty and not result) or any(
        not isinstance(item, str) or not item.strip() for item in result
    ):
        raise ValueError(f"{name} must contain non-empty text")
    return result


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _observable_vector_with_support(
    column_density_m2: ArrayLike,
    support: ObservableIntegrationSupport,
) -> tuple[FloatArray, BoolArray]:
    """Return the axis-observable vector and its per-observable support mask."""

    summary = extract_density_observables(column_density_m2, support)
    values = np.asarray(
        [
            summary.integrated_response,
            np.nan if summary.centroid_y_m is None else summary.centroid_y_m * 1e6,
            np.nan if summary.centroid_z_m is None else summary.centroid_z_m * 1e6,
            (
                np.nan
                if summary.covariance_m2 is None
                else np.sqrt(summary.covariance_m2[0, 0]) * 1e6
            ),
            (
                np.nan
                if summary.covariance_m2 is None
                else np.sqrt(summary.covariance_m2[1, 1]) * 1e6
            ),
        ],
        dtype=float,
    )
    supported = np.isfinite(values)
    return values, supported


def _validate_observable_support_contract(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    support: ObservableIntegrationSupport,
) -> None:
    """Require one exact object grid and an integration mask inside the model."""

    model_y = getattr(model, "y_grid_m", None)
    model_z = getattr(model, "z_grid_m", None)
    model_support = getattr(model, "support_mask", None)
    if (
        model_y is None
        or model_z is None
        or model_support is None
    ):
        raise TypeError(
            "information bootstrapping requires a model with explicit object "
            "coordinates and support_mask"
        )
    if not isinstance(model, NonnegativeBilinearDensityModel):
        raise TypeError(
            "effective support requires a NonnegativeBilinearDensityModel"
        )
    model_y_array = np.asarray(model_y, dtype=float)
    model_z_array = np.asarray(model_z, dtype=float)
    model_support_array = np.asarray(model_support, dtype=bool)
    effective_support_array = bilinear_effective_support_mask(model)
    if (
        not np.array_equal(model_y_array, operator.grid.y_grid_m)
        or not np.array_equal(model_z_array, operator.grid.z_grid_m)
    ):
        raise ValueError("density model and linked operator object grids differ")
    if (
        not np.array_equal(support.y_grid_m, model_y_array)
        or not np.array_equal(support.z_grid_m, model_z_array)
    ):
        raise ValueError("observable support coordinates must match the object grid")
    if model_support_array.shape != support.shape:
        raise ValueError("model support mask has the wrong object-grid shape")
    if effective_support_array.shape != support.shape:
        raise ValueError("effective model support has the wrong object-grid shape")
    if np.any(effective_support_array & ~model_support_array):
        raise ValueError("effective model support cannot exceed its declared support")
    if np.any(support.support_mask & ~model_support_array):
        raise ValueError("observable support cannot extend beyond latent model support")
    if np.any(support.support_mask & ~effective_support_array):
        raise ValueError(
            "observable support cannot extend into cells fixed to zero by the "
            "density basis"
        )


def _validate_reference_light_primary_inputs(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    fit_result: LinkedScalarFitResult,
    provenance: ReferenceLightInferenceProvenance,
    *,
    regularisation: CurvatureRegularisation | None,
) -> None:
    """Verify the model/fit route that code can check and retain declarations."""

    if not isinstance(provenance, ReferenceLightInferenceProvenance):
        raise TypeError("reference_light_provenance has the wrong type")
    if not isinstance(model, NonnegativeBilinearDensityModel):
        raise TypeError(
            "the primary reference-light route requires "
            "NonnegativeBilinearDensityModel; template models belong only to "
            "an explicitly separate sensitivity analysis"
        )
    if provenance.method == "PCI" and not isinstance(operator, PCILinkedRawOperator):
        raise TypeError("PCI provenance requires a PCILinkedRawOperator")
    if provenance.method == "DGI" and not isinstance(operator, DGILinkedRawOperator):
        raise TypeError("DGI provenance requires a DGILinkedRawOperator")
    if provenance.regularisation_applied is not (regularisation is not None):
        raise ValueError(
            "declared regularisation provenance disagrees with the supplied fit route"
        )
    coefficients = np.asarray(fit_result.density_coefficients, dtype=float)
    if coefficients.shape != (2, model.parameter_count):
        raise ValueError("primary reference-light fit must contain exactly q1 and q2")
    diagnostic_regularisation = float(fit_result.diagnostics.regularisation_objective)
    if not np.isfinite(diagnostic_regularisation) or diagnostic_regularisation < 0.0:
        raise ValueError("point-fit regularisation objective is invalid")
    if regularisation is None:
        expected_regularisation = 0.0
    else:
        if regularisation.parameter_count != model.parameter_count:
            raise ValueError("regularisation parameter count does not match the model")
        if (
            not np.array_equal(regularisation.knot_y_um, model.knot_y_um)
            or not np.array_equal(regularisation.knot_z_um, model.knot_z_um)
        ):
            raise ValueError("regularisation knots do not match the model")
        penalty = regularisation.matrix_for_coefficient_scale(
            model.coefficient_scale_m2
        )
        residual = np.concatenate(
            [penalty @ parameter_row for parameter_row in coefficients]
        )
        expected_regularisation = 0.5 * float(residual @ residual)
    regularisation_tolerance = (
        16.0
        * np.finfo(float).eps
        * max(1.0, abs(expected_regularisation))
    )
    if not np.isclose(
        diagnostic_regularisation,
        expected_regularisation,
        rtol=0.0,
        atol=regularisation_tolerance,
    ):
        raise ValueError(
            "point-fit regularisation objective disagrees with the declared regulariser"
        )
    for frame_index, (parameter_row, fitted_map) in enumerate(
        zip(coefficients, fit_result.column_density_m2, strict=True)
    ):
        recomputed_map = np.asarray(model.column_density(parameter_row), dtype=float)
        fitted_array = np.asarray(fitted_map, dtype=float)
        if fitted_array.shape != recomputed_map.shape or not np.allclose(
            fitted_array,
            recomputed_map,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                f"point-fit density map {frame_index} does not come from the supplied "
                "reference-light model"
            )

    stored_prediction = fit_result.prediction
    _validate_linked_prediction_topology(operator, stored_prediction)
    if fit_result.nuisance_names != tuple(operator.nuisance_names):
        raise ValueError("point-fit nuisance names do not belong to the operator")
    recomputed_prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        list(coefficients),
        nuisance_from_vector(operator, fit_result.nuisance_values),
    )
    if (
        stored_prediction.role_names != recomputed_prediction.role_names
        or stored_prediction.role_frame_indices
        != recomputed_prediction.role_frame_indices
        or stored_prediction.shared_role_names
        != recomputed_prediction.shared_role_names
        or stored_prediction.density_parameter_slices
        != recomputed_prediction.density_parameter_slices
    ):
        raise ValueError("point-fit linked raw-role topology does not match the operator")
    nonshared_frames = tuple(
        frame_index
        for frame_index in stored_prediction.role_frame_indices
        if frame_index is not None
    )
    if nonshared_frames != (0, 1):
        raise ValueError("primary linked prediction must contain one q1 and one q2 role")
    for stored, recomputed in zip(
        stored_prediction.expected_electrons,
        recomputed_prediction.expected_electrons,
        strict=True,
    ):
        if not np.allclose(stored, recomputed, rtol=1e-12, atol=0.0):
            raise ValueError("point-fit prediction does not match the supplied operator")
    for stored, recomputed in zip(
        stored_prediction.conditional_variance_electrons2,
        recomputed_prediction.conditional_variance_electrons2,
        strict=True,
    ):
        if not np.allclose(stored, recomputed, rtol=1e-12, atol=0.0):
            raise ValueError("point-fit noise model does not match the supplied operator")
    if not np.allclose(
        stored_prediction.prediction_vector,
        recomputed_prediction.prediction_vector,
        rtol=1e-12,
        atol=0.0,
    ):
        raise ValueError("point-fit flattened prediction does not match the operator")
    if not np.allclose(
        stored_prediction.jacobian,
        recomputed_prediction.jacobian,
        rtol=1e-12,
        atol=0.0,
    ):
        raise ValueError("point-fit Jacobian does not match the supplied model/operator")


def _validate_linked_prediction_topology(
    operator: LinkedScalarOperator,
    prediction: LinkedRawSequencePrediction,
) -> None:
    """Require the fitted raw roles to belong to the supplied PCI/DGI operator."""

    frame_indices = tuple(
        value for value in prediction.role_frame_indices if value is not None
    )
    frame_count = len(frame_indices)
    if frame_indices != tuple(range(frame_count)):
        raise ValueError("linked prediction frame-role indices are not consecutive")
    if isinstance(operator, PCILinkedRawOperator):
        base_atom_role = "atom"
    elif isinstance(operator, DGILinkedRawOperator):
        base_atom_role = "atom_stop"
    else:
        raise TypeError("unsupported linked scalar operator")
    expected_names = tuple(
        f"{base_atom_role}_{frame_index:03d}" for frame_index in range(frame_count)
    ) + tuple(operator.shared_role_names)
    expected_indices = tuple(range(frame_count)) + (None,) * len(
        operator.shared_role_names
    )
    if (
        prediction.role_names != expected_names
        or prediction.role_frame_indices != expected_indices
        or prediction.shared_role_names != tuple(operator.shared_role_names)
        or prediction.nuisance_names != tuple(operator.nuisance_names)
    ):
        raise ValueError("linked prediction raw-role topology does not belong to operator")


def _validate_observation_matches_fit(
    operator: LinkedScalarOperator,
    observation: LinkedRawObservation,
    fit_result: LinkedScalarFitResult,
) -> None:
    """Authenticate the supplied raw roles against the fit's stored residual."""

    prediction = fit_result.prediction
    if observation.role_names != prediction.role_names:
        raise ValueError("observation and fitted prediction role order differ")
    if len(observation.observed_electrons) != len(prediction.expected_electrons):
        raise ValueError("observation and fitted prediction role counts differ")
    roi = np.asarray(operator.grid.roi_mask, dtype=bool)
    residual_parts: list[FloatArray] = []
    for observed, expected, variance in zip(
        observation.observed_electrons,
        prediction.expected_electrons,
        prediction.conditional_variance_electrons2,
        strict=True,
    ):
        observed_array = np.asarray(observed, dtype=float)
        expected_array = np.asarray(expected, dtype=float)
        variance_array = np.asarray(variance, dtype=float)
        if (
            observed_array.shape != operator.grid.camera_shape
            or expected_array.shape != observed_array.shape
            or variance_array.shape != observed_array.shape
        ):
            raise ValueError("observation, prediction and operator camera shapes differ")
        if (
            np.any(~np.isfinite(observed_array))
            or np.any(~np.isfinite(expected_array))
            or np.any(~np.isfinite(variance_array))
            or np.any(variance_array <= 0.0)
        ):
            raise ValueError("observation identity check requires finite raw roles")
        residual_parts.append(
            ((observed_array - expected_array) / np.sqrt(variance_array))[roi]
        )
    recomputed = np.concatenate(residual_parts)
    stored = np.asarray(fit_result.diagnostics.whitened_residual_vector, dtype=float)
    tolerance = 16.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(stored))) if stored.size else 1.0,
    )
    if stored.shape != recomputed.shape or not np.allclose(
        recomputed,
        stored,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("observation is not the raw sequence that produced the fit")


@dataclass(frozen=True, eq=False)
class LinkedObservableInformationBootstrap:
    """Aligned conditional-bootstrap samples for every frame and observable.

    ``samples`` has shape ``(requested_draws, frame_count, observable_count)``.
    A failed refit retains an all-``NaN`` row.  A successful refit with an
    undefined moment retains only that observable as ``NaN``.  This alignment
    is required for correlated two-frame differences and ratios.
    """

    parameter_names: tuple[str, ...]
    requested_draws: int
    fit_success_mask: BoolArray
    point_estimates: FloatArray
    point_supported_mask: BoolArray
    samples: FloatArray
    supported_mask: BoolArray
    route_provenance: ReferenceLightInferenceProvenance
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        parameter_names = tuple(self.parameter_names)
        if parameter_names != OBSERVABLE_NAMES:
            raise ValueError("observable parameter order changed")
        requested = _strict_int(
            self.requested_draws,
            name="requested_draws",
            minimum=1,
        )
        assumptions = _text_tuple(
            self.assumptions,
            name="bootstrap assumptions",
            allow_empty=False,
        )
        fit_success = np.asarray(self.fit_success_mask, dtype=bool)
        point = np.asarray(self.point_estimates, dtype=float)
        point_supported = np.asarray(self.point_supported_mask, dtype=bool)
        samples = np.asarray(self.samples, dtype=float)
        supported = np.asarray(self.supported_mask, dtype=bool)
        if fit_success.shape != (requested,):
            raise ValueError("fit-success mask does not match requested draws")
        if point.ndim != 2 or point.shape[1] != len(OBSERVABLE_NAMES):
            raise ValueError("point estimates have the wrong shape")
        if point.shape[0] == 0:
            raise ValueError("at least one linked frame is required")
        if point_supported.shape != point.shape:
            raise ValueError("point support mask does not match point estimates")
        expected_sample_shape = (requested, *point.shape)
        if samples.shape != expected_sample_shape or supported.shape != samples.shape:
            raise ValueError("bootstrap samples or support mask have the wrong shape")
        if not np.array_equal(point_supported, np.isfinite(point)):
            raise ValueError("point support mask must identify finite values exactly")
        if not np.array_equal(supported, np.isfinite(samples)):
            raise ValueError("sample support mask must identify finite values exactly")
        if np.any(supported[~fit_success]):
            raise ValueError("failed refits cannot contain supported observables")
        if not isinstance(self.route_provenance, ReferenceLightInferenceProvenance):
            raise TypeError("bootstrap route provenance has the wrong type")
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "requested_draws", requested)
        object.__setattr__(self, "fit_success_mask", _immutable(fit_success, dtype=bool))
        object.__setattr__(self, "point_estimates", _immutable(point))
        object.__setattr__(
            self,
            "point_supported_mask",
            _immutable(point_supported, dtype=bool),
        )
        object.__setattr__(self, "samples", _immutable(samples))
        object.__setattr__(self, "supported_mask", _immutable(supported, dtype=bool))
        object.__setattr__(self, "assumptions", assumptions)

    @property
    def frame_count(self) -> int:
        """Number of independently parameterised object frames."""

        return int(self.point_estimates.shape[0])

    @property
    def successful_fit_draws(self) -> int:
        """Number of bootstrap sequences whose numerical fit converged."""

        return int(np.count_nonzero(self.fit_success_mask))


@dataclass(frozen=True, eq=False)
class LinkedObservableBootstrapDraw:
    """One linked-raw bootstrap refit evaluated on named supports.

    ``values`` and ``supported_mask`` have shape
    ``(support_count, frame_count, observable_count)``.  A numerical refit
    failure is retained as an explicit ``fit_failure`` record whose values are
    all ``NaN``.  Named supports share the same raw draw and the same refit.
    """

    support_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    status: BootstrapDrawStatus
    fit_message: str
    fit_nfev: int
    fit_irls_iterations: int
    values: FloatArray
    supported_mask: BoolArray
    route_provenance: ReferenceLightInferenceProvenance
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        support_names = _text_tuple(
            self.support_names,
            name="bootstrap-draw support names",
            allow_empty=False,
        )
        if len(set(support_names)) != len(support_names):
            raise ValueError("bootstrap-draw support names must be unique")
        parameter_names = tuple(self.parameter_names)
        if parameter_names != OBSERVABLE_NAMES:
            raise ValueError("observable parameter order changed")
        if self.status not in ("success", "fit_failure"):
            raise ValueError("unknown bootstrap-draw status")
        if not isinstance(self.fit_message, str) or not self.fit_message.strip():
            raise ValueError("bootstrap-draw fit message cannot be empty")
        nfev = _strict_int(self.fit_nfev, name="fit_nfev", minimum=0)
        irls_iterations = _strict_int(
            self.fit_irls_iterations,
            name="fit_irls_iterations",
            minimum=0,
        )
        values = np.asarray(self.values, dtype=float)
        supported = np.asarray(self.supported_mask, dtype=bool)
        if values.ndim != 3 or values.shape != (
            len(support_names),
            2,
            len(OBSERVABLE_NAMES),
        ):
            raise ValueError("bootstrap-draw values have the wrong shape")
        if supported.shape != values.shape:
            raise ValueError("bootstrap-draw support mask has the wrong shape")
        if not np.array_equal(supported, np.isfinite(values)):
            raise ValueError("bootstrap-draw support must identify finite values exactly")
        if self.status == "fit_failure" and (
            np.any(supported) or np.any(~np.isnan(values))
        ):
            raise ValueError("a failed bootstrap refit must retain an all-NaN result")
        if not isinstance(self.route_provenance, ReferenceLightInferenceProvenance):
            raise TypeError("bootstrap-draw route provenance has the wrong type")
        assumptions = _text_tuple(
            self.assumptions,
            name="bootstrap-draw assumptions",
            allow_empty=False,
        )
        object.__setattr__(self, "support_names", support_names)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "fit_nfev", nfev)
        object.__setattr__(self, "fit_irls_iterations", irls_iterations)
        object.__setattr__(self, "values", _immutable(values))
        object.__setattr__(self, "supported_mask", _immutable(supported, dtype=bool))
        object.__setattr__(self, "assumptions", assumptions)

    @property
    def fit_success(self) -> bool:
        """Whether the single numerical refit converged."""

        return self.status == "success"


@dataclass(frozen=True, eq=False)
class LinkedRawRoleDiagnostics:
    """Noise-scaled residual diagnostics for one actual linked raw role."""

    role_name: str
    frame_index: int | None
    shared_role: bool
    standardised_residual_map: FloatArray
    roi_pixel_count: int
    roi_mean: float
    roi_rms: float
    roi_standard_deviation: float
    lag_one_correlation_y: float | None
    lag_one_correlation_z: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.role_name, str) or not self.role_name.strip():
            raise ValueError("raw role name cannot be empty")
        if type(self.shared_role) is not bool:
            raise TypeError("shared_role must be bool")
        frame_index = self.frame_index
        if frame_index is not None:
            frame_index = _strict_int(
                frame_index,
                name="raw role frame index",
                minimum=0,
            )
        if self.shared_role != (self.frame_index is None):
            raise ValueError("shared-role flag disagrees with the frame index")
        residual = np.asarray(self.standardised_residual_map, dtype=float)
        if residual.ndim != 2 or np.any(~np.isfinite(residual)):
            raise ValueError("standardised residual map must be finite and two-dimensional")
        roi_pixel_count = _strict_int(
            self.roi_pixel_count,
            name="roi_pixel_count",
            minimum=1,
        )
        scalars = np.asarray(
            (self.roi_mean, self.roi_rms, self.roi_standard_deviation),
            dtype=float,
        )
        if np.any(~np.isfinite(scalars)) or self.roi_rms < 0.0 or self.roi_standard_deviation < 0.0:
            raise ValueError("raw-role residual statistics are invalid")
        for value in (self.lag_one_correlation_y, self.lag_one_correlation_z):
            if value is not None and (not np.isfinite(value) or not -1.0 <= value <= 1.0):
                raise ValueError("lag-one residual correlations must lie in [-1, 1]")
        object.__setattr__(self, "standardised_residual_map", _immutable(residual))
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "roi_pixel_count", roi_pixel_count)


@dataclass(frozen=True)
class LinkedRawDiagnosticsSummary:
    """Provenance-bearing raw-role fit diagnostics for one q1/q2 sequence."""

    roles: tuple[LinkedRawRoleDiagnostics, ...]
    route_provenance: ReferenceLightInferenceProvenance
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        roles = tuple(self.roles)
        assumptions = _text_tuple(
            self.assumptions,
            name="raw diagnostics assumptions",
            allow_empty=False,
        )
        if not isinstance(self.route_provenance, ReferenceLightInferenceProvenance):
            raise TypeError("raw diagnostics route provenance has the wrong type")
        if not roles or any(not isinstance(role, LinkedRawRoleDiagnostics) for role in roles):
            raise ValueError("raw diagnostics require role records")
        frame_indices = tuple(
            role.frame_index for role in roles if not role.shared_role
        )
        if frame_indices != (0, 1):
            raise ValueError("raw diagnostics require exactly q1 and q2 atom roles")
        expected_shared_count = 2 if self.route_provenance.method == "PCI" else 4
        if sum(role.shared_role for role in roles) != expected_shared_count:
            raise ValueError("raw diagnostics shared-role count disagrees with method")
        if len({role.role_name for role in roles}) != len(roles):
            raise ValueError("raw diagnostic role names must be unique")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "assumptions", assumptions)


def _masked_lag_one_correlation(
    values: FloatArray,
    mask: BoolArray,
    *,
    axis: int,
) -> float | None:
    if axis == 0:
        first = values[:-1, :]
        second = values[1:, :]
        pair_mask = mask[:-1, :] & mask[1:, :]
    elif axis == 1:
        first = values[:, :-1]
        second = values[:, 1:]
        pair_mask = mask[:, :-1] & mask[:, 1:]
    else:
        raise ValueError("lag axis must be zero or one")
    left = first[pair_mask]
    right = second[pair_mask]
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def analyse_linked_raw_residuals(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    observation: LinkedRawObservation,
    fit_result: LinkedScalarFitResult,
    *,
    reference_light_provenance: ReferenceLightInferenceProvenance,
    regularisation: CurvatureRegularisation | None,
) -> LinkedRawDiagnosticsSummary:
    """Report raw-role residuals after verifying the primary model/operator route."""

    prediction = fit_result.prediction
    _validate_reference_light_primary_inputs(
        operator,
        model,
        fit_result,
        reference_light_provenance,
        regularisation=regularisation,
    )
    _validate_observation_matches_fit(operator, observation, fit_result)
    if observation.role_names != prediction.role_names:
        raise ValueError("observation and fitted prediction role order differ")
    if len(observation.observed_electrons) != len(prediction.expected_electrons):
        raise ValueError("observation and fitted prediction role counts differ")
    roi = np.asarray(operator.grid.roi_mask, dtype=bool)
    diagnostics: list[LinkedRawRoleDiagnostics] = []
    for name, frame_index, observed, expected, variance in zip(
        prediction.role_names,
        prediction.role_frame_indices,
        observation.observed_electrons,
        prediction.expected_electrons,
        prediction.conditional_variance_electrons2,
        strict=True,
    ):
        observed_array = np.asarray(observed, dtype=float)
        expected_array = np.asarray(expected, dtype=float)
        variance_array = np.asarray(variance, dtype=float)
        if (
            observed_array.shape != operator.grid.camera_shape
            or expected_array.shape != observed_array.shape
            or variance_array.shape != observed_array.shape
        ):
            raise ValueError("linked raw role camera shapes are inconsistent")
        if np.any(~np.isfinite(observed_array)) or np.any(~np.isfinite(expected_array)):
            raise ValueError("linked raw role values must be finite")
        if np.any(~np.isfinite(variance_array)) or np.any(variance_array <= 0.0):
            raise ValueError("linked raw role variances must be finite and positive")
        standardised = (observed_array - expected_array) / np.sqrt(variance_array)
        roi_values = standardised[roi]
        diagnostics.append(
            LinkedRawRoleDiagnostics(
                role_name=name,
                frame_index=frame_index,
                shared_role=frame_index is None,
                standardised_residual_map=standardised,
                roi_pixel_count=int(roi_values.size),
                roi_mean=float(np.mean(roi_values)),
                roi_rms=float(np.sqrt(np.mean(roi_values**2))),
                roi_standard_deviation=float(np.std(roi_values)),
                lag_one_correlation_y=_masked_lag_one_correlation(
                    standardised,
                    roi,
                    axis=1,
                ),
                lag_one_correlation_z=_masked_lag_one_correlation(
                    standardised,
                    roi,
                    axis=0,
                ),
            )
        )
    return LinkedRawDiagnosticsSummary(
        roles=tuple(diagnostics),
        route_provenance=reference_light_provenance,
        assumptions=reference_light_provenance.assumptions
        + (
            "residuals are standardised by each raw role's conditional variance",
            "shared raw roles are reported once and are not assigned to either frame",
            "raw-role residuals are fit/data diagnostics, not cloud-presence evidence",
        ),
    )


def select_q1_observation_for_reference_sensitivity(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    observation: LinkedRawObservation,
    fit_result: LinkedScalarFitResult,
    *,
    reference_light_provenance: ReferenceLightInferenceProvenance,
    regularisation: CurvatureRegularisation | None,
) -> LinkedRawObservation:
    """Select q1 plus shared raw roles from an exact q1/q2 observation.

    This is the only raw-data adapter provided for the optional
    reference-informed sensitivity.  It refuses sequences other than q1/q2
    and therefore cannot expose the q2 atom-bearing role to a template fit.
    """

    _validate_reference_light_primary_inputs(
        operator,
        model,
        fit_result,
        reference_light_provenance,
        regularisation=regularisation,
    )
    _validate_observation_matches_fit(operator, observation, fit_result)
    prediction = fit_result.prediction
    if observation.role_names != prediction.role_names:
        raise ValueError("observation and prediction role order differ")
    if len(observation.observed_electrons) != len(prediction.role_frame_indices):
        raise ValueError("observation and prediction role counts differ")
    frame_indices = tuple(
        index for index in prediction.role_frame_indices if index is not None
    )
    if set(frame_indices) != {0, 1} or frame_indices.count(0) != 1 or frame_indices.count(1) != 1:
        raise ValueError("reference sensitivity requires exactly one q1 and one q2 atom role")
    keep = tuple(
        index
        for index, frame_index in enumerate(prediction.role_frame_indices)
        if frame_index in (0, None)
    )
    selected_names = tuple(observation.role_names[index] for index in keep)
    selected_arrays = tuple(observation.observed_electrons[index] for index in keep)
    if any(name not in selected_names for name in prediction.shared_role_names):
        raise ValueError("q1 sensitivity selection lost a shared raw role")
    if any(
        prediction.role_frame_indices[index] == 1
        for index in keep
    ):
        raise RuntimeError("q2 atom role leaked into q1 reference sensitivity")
    return LinkedRawObservation(selected_names, selected_arrays)


def _normalise_named_integration_supports(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    integration_supports: Mapping[str, ObservableIntegrationSupport],
) -> tuple[tuple[str, ObservableIntegrationSupport], ...]:
    if not isinstance(integration_supports, Mapping):
        raise TypeError("integration_supports must be a mapping")
    items = tuple(integration_supports.items())
    if not items:
        raise ValueError("integration_supports cannot be empty")
    names: list[str] = []
    normalised: list[tuple[str, ObservableIntegrationSupport]] = []
    for raw_name, support in items:
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("integration-support names must be non-empty text")
        if not isinstance(support, ObservableIntegrationSupport):
            raise TypeError("named integration supports have the wrong type")
        name = raw_name.strip()
        names.append(name)
        _validate_observable_support_contract(operator, model, support)
        normalised.append((name, support))
    if len(set(names)) != len(names):
        raise ValueError("integration-support names must remain unique after stripping")
    return tuple(normalised)


def _validate_bootstrap_refit_bounds(
    model: DifferentiableColumnDensityModel,
    point_fit: LinkedScalarFitResult,
    *,
    density_parameter_lower: float | ArrayLike,
    density_coefficient_upper: float | ArrayLike,
    nuisance_lower: ArrayLike,
    nuisance_upper: ArrayLike,
) -> None:
    density_lower = np.asarray(density_parameter_lower, dtype=float)
    if density_lower.ndim == 0:
        density_lower = np.full(model.parameter_count, float(density_lower))
    density_upper = np.asarray(density_coefficient_upper, dtype=float)
    if density_upper.ndim == 0:
        density_upper = np.full(model.parameter_count, float(density_upper))
    density_shape = (model.parameter_count,)
    coefficients = np.asarray(point_fit.density_coefficients, dtype=float)
    if density_lower.shape != density_shape or density_upper.shape != density_shape:
        raise ValueError("bootstrap density bounds have the wrong shape")
    if (
        np.any(~np.isfinite(density_lower))
        or np.any(~np.isfinite(density_upper))
        or np.any(density_upper <= density_lower)
        or np.any(coefficients < density_lower[None, :])
        or np.any(coefficients > density_upper[None, :])
    ):
        raise ValueError("bootstrap density bounds exclude the point fit or are invalid")
    nuisance_lower_array = np.asarray(nuisance_lower, dtype=float)
    nuisance_upper_array = np.asarray(nuisance_upper, dtype=float)
    nuisance_shape = (len(point_fit.nuisance_names),)
    nuisance_values = np.asarray(point_fit.nuisance_values, dtype=float)
    if (
        nuisance_lower_array.shape != nuisance_shape
        or nuisance_upper_array.shape != nuisance_shape
    ):
        raise ValueError("bootstrap nuisance bounds have the wrong shape")
    if (
        np.any(~np.isfinite(nuisance_lower_array))
        or np.any(~np.isfinite(nuisance_upper_array))
        or np.any(nuisance_lower_array < 0.0)
        or np.any(nuisance_upper_array <= nuisance_lower_array)
        or np.any(nuisance_values < nuisance_lower_array)
        or np.any(nuisance_values > nuisance_upper_array)
    ):
        raise ValueError("bootstrap nuisance bounds exclude the point fit or are invalid")


def _bootstrap_draw_failure(
    *,
    named_supports: tuple[tuple[str, ObservableIntegrationSupport], ...],
    message: str,
    nfev: int,
    irls_iterations: int,
    provenance: ReferenceLightInferenceProvenance,
) -> LinkedObservableBootstrapDraw:
    values = np.full(
        (len(named_supports), 2, len(OBSERVABLE_NAMES)),
        np.nan,
        dtype=float,
    )
    return LinkedObservableBootstrapDraw(
        support_names=tuple(name for name, _ in named_supports),
        parameter_names=OBSERVABLE_NAMES,
        status="fit_failure",
        fit_message=message,
        fit_nfev=nfev,
        fit_irls_iterations=irls_iterations,
        values=values,
        supported_mask=np.zeros(values.shape, dtype=bool),
        route_provenance=provenance,
        assumptions=provenance.assumptions
        + (
            "one conditional Poisson-read-noise draw of every linked raw role",
            "one shared numerical refit supplies every named observable support",
            "failed refits retain an explicit all-NaN result",
        ),
    )


def _refit_linked_observable_bootstrap_draw_core(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    point_fit: LinkedScalarFitResult,
    *,
    named_supports: tuple[tuple[str, ObservableIntegrationSupport], ...],
    density_parameter_lower: float | ArrayLike,
    density_coefficient_upper: float | ArrayLike,
    nuisance_lower: ArrayLike,
    nuisance_upper: ArrayLike,
    regularisation: CurvatureRegularisation | None,
    options: LinkedScalarFitOptions,
    rng: np.random.Generator,
    provenance: ReferenceLightInferenceProvenance,
) -> LinkedObservableBootstrapDraw:
    observation = draw_linked_raw_observation(operator, point_fit.prediction, rng)
    try:
        refit = fit_linked_scalar_sequence(
            operator,
            model,
            observation,
            initial_density_coefficients=point_fit.density_coefficients,
            density_parameter_lower=density_parameter_lower,
            density_coefficient_upper=density_coefficient_upper,
            initial_nuisance=nuisance_from_vector(
                operator,
                point_fit.nuisance_values,
            ),
            nuisance_lower=nuisance_lower,
            nuisance_upper=nuisance_upper,
            regularisation=regularisation,
            options=options,
        )
    except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
        return _bootstrap_draw_failure(
            named_supports=named_supports,
            message=f"numerical refit raised {type(exc).__name__}: {exc}",
            nfev=0,
            irls_iterations=0,
            provenance=provenance,
        )
    if not refit.diagnostics.success:
        return _bootstrap_draw_failure(
            named_supports=named_supports,
            message=refit.diagnostics.message or "numerical refit did not converge",
            nfev=refit.diagnostics.nfev,
            irls_iterations=refit.diagnostics.irls_iterations,
            provenance=provenance,
        )
    values = np.full(
        (len(named_supports), 2, len(OBSERVABLE_NAMES)),
        np.nan,
        dtype=float,
    )
    supported = np.zeros(values.shape, dtype=bool)
    for support_index, (_, support) in enumerate(named_supports):
        for frame_index, density in enumerate(refit.column_density_m2):
            row_values, row_supported = _observable_vector_with_support(
                density,
                support,
            )
            values[support_index, frame_index] = row_values
            supported[support_index, frame_index] = row_supported
    return LinkedObservableBootstrapDraw(
        support_names=tuple(name for name, _ in named_supports),
        parameter_names=OBSERVABLE_NAMES,
        status="success",
        fit_message=refit.diagnostics.message or "numerical refit converged",
        fit_nfev=refit.diagnostics.nfev,
        fit_irls_iterations=refit.diagnostics.irls_iterations,
        values=values,
        supported_mask=supported,
        route_provenance=provenance,
        assumptions=provenance.assumptions
        + (
            "one conditional Poisson-read-noise draw of every linked raw role",
            "one shared numerical refit supplies every named observable support",
            "support-specific moments are extracted only after that shared refit",
        ),
    )


def refit_linked_observable_bootstrap_draw(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    point_fit: LinkedScalarFitResult,
    *,
    reference_light_provenance: ReferenceLightInferenceProvenance,
    integration_supports: Mapping[str, ObservableIntegrationSupport],
    density_parameter_lower: float | ArrayLike = 0.0,
    density_coefficient_upper: float | ArrayLike,
    nuisance_lower: ArrayLike,
    nuisance_upper: ArrayLike,
    regularisation: CurvatureRegularisation | None,
    options: LinkedScalarFitOptions,
    rng: np.random.Generator,
) -> LinkedObservableBootstrapDraw:
    """Draw, refit and extract one q1/q2 bootstrap sequence.

    The caller supplies an independent random generator.  All named supports
    are evaluated from one successful refit, so adding a support does not draw
    new raw roles or repeat the inverse solve.  Numerical refit failures are
    returned explicitly and retain no finite observable values.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(options, LinkedScalarFitOptions):
        raise TypeError("options must be LinkedScalarFitOptions")
    if not point_fit.diagnostics.success:
        raise ValueError("point fit must succeed before conditional bootstrapping")
    _validate_reference_light_primary_inputs(
        operator,
        model,
        point_fit,
        reference_light_provenance,
        regularisation=regularisation,
    )
    named_supports = _normalise_named_integration_supports(
        operator,
        model,
        integration_supports,
    )
    _validate_bootstrap_refit_bounds(
        model,
        point_fit,
        density_parameter_lower=density_parameter_lower,
        density_coefficient_upper=density_coefficient_upper,
        nuisance_lower=nuisance_lower,
        nuisance_upper=nuisance_upper,
    )
    return _refit_linked_observable_bootstrap_draw_core(
        operator,
        model,
        point_fit,
        named_supports=named_supports,
        density_parameter_lower=density_parameter_lower,
        density_coefficient_upper=density_coefficient_upper,
        nuisance_lower=nuisance_lower,
        nuisance_upper=nuisance_upper,
        regularisation=regularisation,
        options=options,
        rng=rng,
        provenance=reference_light_provenance,
    )


def bootstrap_linked_observable_information(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    point_fit: LinkedScalarFitResult,
    *,
    reference_light_provenance: ReferenceLightInferenceProvenance,
    integration_support: ObservableIntegrationSupport,
    density_parameter_lower: float | ArrayLike = 0.0,
    density_coefficient_upper: float | ArrayLike,
    nuisance_lower: ArrayLike,
    nuisance_upper: ArrayLike,
    regularisation: CurvatureRegularisation | None,
    options: LinkedScalarFitOptions,
    draws: int,
    rng: np.random.Generator,
) -> LinkedObservableInformationBootstrap:
    """Refit aligned linked-raw draws without truth-derived calibration.

    Every draw redraws each shared reference or dark role once for the whole
    sequence.  The result is conditional on the fitted forward operator,
    nuisance-field basis, integration support and regulariser.  It does not
    include identifiability, response-calibration or model/reference
    sensitivity, which must be reported as separate confidence components.
    """

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if not isinstance(options, LinkedScalarFitOptions):
        raise TypeError("options must be LinkedScalarFitOptions")
    requested = _strict_int(draws, name="draws", minimum=1)
    if not point_fit.diagnostics.success:
        raise ValueError("point fit must succeed before conditional bootstrapping")
    _validate_reference_light_primary_inputs(
        operator,
        model,
        point_fit,
        reference_light_provenance,
        regularisation=regularisation,
    )
    coefficients = np.asarray(point_fit.density_coefficients, dtype=float)
    named_supports = _normalise_named_integration_supports(
        operator,
        model,
        {"primary": integration_support},
    )
    _validate_bootstrap_refit_bounds(
        model,
        point_fit,
        density_parameter_lower=density_parameter_lower,
        density_coefficient_upper=density_coefficient_upper,
        nuisance_lower=nuisance_lower,
        nuisance_upper=nuisance_upper,
    )

    point_rows = [
        _observable_vector_with_support(density, integration_support)
        for density in point_fit.column_density_m2
    ]
    point_values = np.stack([row[0] for row in point_rows])
    point_supported = np.stack([row[1] for row in point_rows])
    sample_values = np.full(
        (requested, coefficients.shape[0], len(OBSERVABLE_NAMES)),
        np.nan,
        dtype=float,
    )
    sample_supported = np.zeros(sample_values.shape, dtype=bool)
    fit_success = np.zeros(requested, dtype=bool)

    for draw_index in range(requested):
        draw = _refit_linked_observable_bootstrap_draw_core(
            operator,
            model,
            point_fit,
            named_supports=named_supports,
            density_parameter_lower=density_parameter_lower,
            density_coefficient_upper=density_coefficient_upper,
            nuisance_lower=nuisance_lower,
            nuisance_upper=nuisance_upper,
            regularisation=regularisation,
            options=options,
            rng=rng,
            provenance=reference_light_provenance,
        )
        if not draw.fit_success:
            continue
        fit_success[draw_index] = True
        sample_values[draw_index] = draw.values[0]
        sample_supported[draw_index] = draw.supported_mask[0]

    return LinkedObservableInformationBootstrap(
        parameter_names=OBSERVABLE_NAMES,
        requested_draws=requested,
        fit_success_mask=fit_success,
        point_estimates=point_values,
        point_supported_mask=point_supported,
        samples=sample_values,
        supported_mask=sample_supported,
        route_provenance=reference_light_provenance,
        assumptions=reference_light_provenance.assumptions
        + (
            "conditional Poisson-read-noise bootstrap of the linked raw roles",
            "shared reference and dark roles are redrawn once per sequence",
            "fixed optical operator, nuisance-field basis, support and regulariser",
            "independent per-frame density blocks with no temporal penalty or transition law",
            "no affine truth calibration is applied by this bootstrap",
            "calibration, forward-model, basis, support, reference, regularisation, "
            "identifiability and repeatability evidence remain separate",
            "a positive non-negative nuisance fit is not by itself cloud-detection evidence",
        ),
    )


def _zero_density_linked_prediction(
    operator: LinkedScalarOperator,
    frame_count: int,
    nuisance: LinkedNuisanceValues,
) -> LinkedRawSequencePrediction:
    """Return the linked null prediction with density fixed exactly to zero."""

    frames = _strict_int(frame_count, name="frame_count", minimum=1)
    shape = operator.grid.camera_shape
    roi = np.asarray(operator.grid.roi_mask, dtype=bool)
    roi_pixels = operator.grid.roi_pixel_count
    expected: list[FloatArray] = []
    role_names: list[str] = []
    role_frame_indices: list[int | None] = []
    nuisance_jacobian: FloatArray

    if isinstance(operator, PCILinkedRawOperator):
        if not isinstance(nuisance, PCINuisanceValues):
            raise TypeError("PCI null prediction requires PCINuisanceValues")
        carrier_intensity = float(abs(operator.transfer.carrier_field) ** 2)
        atom_count = (
            nuisance.i0_photoelectrons_per_pixel * carrier_intensity
            + nuisance.dark_electrons_per_pixel
        )
        expected.extend(np.full(shape, atom_count, dtype=float) for _ in range(frames))
        role_names.extend(f"atom_{index:03d}" for index in range(frames))
        role_frame_indices.extend(range(frames))
        expected.extend(
            (
                np.full(shape, atom_count, dtype=float),
                np.full(shape, nuisance.dark_electrons_per_pixel, dtype=float),
            )
        )
        role_names.extend(operator.shared_role_names)
        role_frame_indices.extend((None, None))
        role_count = frames + 2
        nuisance_jacobian = np.zeros((role_count * roi_pixels, 2), dtype=float)
        for role_index in range(frames + 1):
            rows = slice(role_index * roi_pixels, (role_index + 1) * roi_pixels)
            nuisance_jacobian[rows, 0] = carrier_intensity
            nuisance_jacobian[rows, 1] = 1.0
        nuisance_jacobian[-roi_pixels:, 1] = 1.0
    elif isinstance(operator, DGILinkedRawOperator):
        if not isinstance(nuisance, DGINuisanceValues):
            raise TypeError("DGI null prediction requires DGINuisanceValues")
        leakage_intensity = float(operator.transfer.carrier_field**2)
        stopped_count = (
            nuisance.i0_photoelectrons_per_pixel
            * nuisance.open_to_stop_scale
            * leakage_intensity
            + nuisance.stop_dark_electrons_per_pixel
        )
        expected.extend(
            np.full(shape, stopped_count, dtype=float) for _ in range(frames)
        )
        role_names.extend(f"atom_stop_{index:03d}" for index in range(frames))
        role_frame_indices.extend(range(frames))
        expected.extend(
            (
                np.full(shape, stopped_count, dtype=float),
                np.full(
                    shape,
                    nuisance.stop_dark_electrons_per_pixel,
                    dtype=float,
                ),
                np.full(
                    shape,
                    nuisance.i0_photoelectrons_per_pixel
                    + nuisance.open_dark_electrons_per_pixel,
                    dtype=float,
                ),
                np.full(
                    shape,
                    nuisance.open_dark_electrons_per_pixel,
                    dtype=float,
                ),
            )
        )
        role_names.extend(operator.shared_role_names)
        role_frame_indices.extend((None, None, None, None))
        role_count = frames + 4
        nuisance_jacobian = np.zeros((role_count * roi_pixels, 4), dtype=float)
        stop_derivative_i0 = nuisance.open_to_stop_scale * leakage_intensity
        stop_derivative_scale = (
            nuisance.i0_photoelectrons_per_pixel * leakage_intensity
        )
        for role_index in range(frames + 1):
            rows = slice(role_index * roi_pixels, (role_index + 1) * roi_pixels)
            nuisance_jacobian[rows, 0] = stop_derivative_i0
            nuisance_jacobian[rows, 1] = 1.0
            nuisance_jacobian[rows, 3] = stop_derivative_scale
        cursor = (frames + 1) * roi_pixels
        nuisance_jacobian[cursor : cursor + roi_pixels, 1] = 1.0
        cursor += roi_pixels
        nuisance_jacobian[cursor : cursor + roi_pixels, 0] = 1.0
        nuisance_jacobian[cursor : cursor + roi_pixels, 2] = 1.0
        cursor += roi_pixels
        nuisance_jacobian[cursor : cursor + roi_pixels, 2] = 1.0
    else:
        raise TypeError("unsupported linked scalar operator")

    variances: list[FloatArray] = []
    for role_name, frame_index, role in zip(
        role_names,
        role_frame_indices,
        expected,
        strict=True,
    ):
        if frame_index is not None:
            base_role = "atom" if isinstance(operator, PCILinkedRawOperator) else "atom_stop"
        else:
            base_role = role_name
        exposure_count = operator.independent_exposures_by_role[base_role]
        variances.append(
            np.asarray(
                (role + operator.read_noise_electrons**2) / exposure_count,
                dtype=float,
            )
        )
    prediction_vector = np.concatenate([role[roi] for role in expected])
    return LinkedRawSequencePrediction(
        role_names=tuple(role_names),
        role_frame_indices=tuple(role_frame_indices),
        shared_role_names=tuple(operator.shared_role_names),
        expected_electrons=tuple(expected),
        conditional_variance_electrons2=tuple(variances),
        prediction_vector=prediction_vector,
        jacobian=nuisance_jacobian,
        density_parameter_slices=(),
        nuisance_names=tuple(operator.nuisance_names),
    )


def _linked_data_gaussian_quasi_deviance(
    operator: LinkedScalarOperator,
    observation: LinkedRawObservation,
    prediction: LinkedRawSequencePrediction,
) -> float:
    """Return the data-only linked Gaussian quasi-deviance over the fit ROI."""

    if observation.role_names != prediction.role_names:
        raise ValueError("observation and linked prediction role order differ")
    if len(observation.observed_electrons) != len(prediction.expected_electrons):
        raise ValueError("observation and linked prediction role counts differ")
    roi = np.asarray(operator.grid.roi_mask, dtype=bool)
    total = 0.0
    for observed, expected, variance in zip(
        observation.observed_electrons,
        prediction.expected_electrons,
        prediction.conditional_variance_electrons2,
        strict=True,
    ):
        observed_array = np.asarray(observed, dtype=float)
        expected_array = np.asarray(expected, dtype=float)
        variance_array = np.asarray(variance, dtype=float)
        if (
            observed_array.shape != operator.grid.camera_shape
            or expected_array.shape != observed_array.shape
            or variance_array.shape != observed_array.shape
        ):
            raise ValueError("linked quasi-deviance camera shapes differ")
        if (
            np.any(~np.isfinite(observed_array))
            or np.any(~np.isfinite(expected_array))
            or np.any(~np.isfinite(variance_array))
            or np.any(variance_array <= 0.0)
        ):
            raise ValueError("linked quasi-deviance requires finite positive variances")
        residual = observed_array[roi] - expected_array[roi]
        roi_variance = variance_array[roi]
        total += float(np.sum(residual**2 / roi_variance + np.log(roi_variance)))
    if not np.isfinite(total):
        raise ValueError("linked data-only Gaussian quasi-deviance is not finite")
    return total


@dataclass(frozen=True, eq=False)
class LinkedZeroDensityNullFit:
    """Nuisance-only linked fit with every density frame fixed to zero."""

    nuisance_names: tuple[str, ...]
    nuisance_values: FloatArray
    nuisance_lower: FloatArray
    nuisance_upper: FloatArray
    prediction: LinkedRawSequencePrediction
    success: bool
    message: str
    nfev: int
    irls_iterations: int
    data_gaussian_quasi_deviance: float
    route_provenance: ReferenceLightInferenceProvenance
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(self.nuisance_names)
        values = np.asarray(self.nuisance_values, dtype=float)
        lower = np.asarray(self.nuisance_lower, dtype=float)
        upper = np.asarray(self.nuisance_upper, dtype=float)
        if not names or len(set(names)) != len(names):
            raise ValueError("null-fit nuisance names must be non-empty and unique")
        if values.shape != (len(names),) or lower.shape != values.shape or upper.shape != values.shape:
            raise ValueError("null-fit nuisance vectors have the wrong shape")
        if (
            np.any(~np.isfinite(values))
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(lower < 0.0)
            or np.any(upper <= lower)
            or np.any(values < lower)
            or np.any(values > upper)
        ):
            raise ValueError("null-fit nuisance values or bounds are invalid")
        if type(self.success) is not bool:
            raise TypeError("null-fit success must be bool")
        message = _nonempty_text(self.message, name="null-fit message")
        nfev = _strict_int(self.nfev, name="null-fit nfev", minimum=0)
        iterations = _strict_int(
            self.irls_iterations,
            name="null-fit IRLS iterations",
            minimum=0,
        )
        score = float(self.data_gaussian_quasi_deviance)
        if not np.isfinite(score):
            raise ValueError("null-fit data Gaussian quasi-deviance must be finite")
        if self.prediction.density_parameter_slices:
            raise ValueError("zero-density null prediction cannot expose density blocks")
        if self.prediction.nuisance_names != names:
            raise ValueError("null-fit prediction nuisance order changed")
        if self.prediction.jacobian.shape[1] != len(names):
            raise ValueError("null-fit prediction must have nuisance-only derivatives")
        if not isinstance(self.route_provenance, ReferenceLightInferenceProvenance):
            raise TypeError("null-fit route provenance has the wrong type")
        assumptions = _text_tuple(
            self.assumptions,
            name="null-fit assumptions",
            allow_empty=False,
        )
        object.__setattr__(self, "nuisance_names", names)
        object.__setattr__(self, "nuisance_values", _immutable(values))
        object.__setattr__(self, "nuisance_lower", _immutable(lower))
        object.__setattr__(self, "nuisance_upper", _immutable(upper))
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "nfev", nfev)
        object.__setattr__(self, "irls_iterations", iterations)
        object.__setattr__(self, "data_gaussian_quasi_deviance", score)
        object.__setattr__(self, "assumptions", assumptions)


def _null_fit_failure(
    operator: LinkedScalarOperator,
    current: FloatArray,
    *,
    frame_count: int,
    nuisance_lower: FloatArray,
    nuisance_upper: FloatArray,
    observation: LinkedRawObservation,
    message: str,
    nfev: int,
    irls_iterations: int,
    provenance: ReferenceLightInferenceProvenance,
) -> LinkedZeroDensityNullFit:
    prediction = _zero_density_linked_prediction(
        operator,
        frame_count,
        nuisance_from_vector(operator, current),
    )
    return LinkedZeroDensityNullFit(
        nuisance_names=tuple(operator.nuisance_names),
        nuisance_values=current,
        nuisance_lower=nuisance_lower,
        nuisance_upper=nuisance_upper,
        prediction=prediction,
        success=False,
        message=message,
        nfev=nfev,
        irls_iterations=irls_iterations,
        data_gaussian_quasi_deviance=_linked_data_gaussian_quasi_deviance(
            operator,
            observation,
            prediction,
        ),
        route_provenance=provenance,
        assumptions=provenance.assumptions
        + (
            "all density frames are fixed exactly to zero",
            "shared nuisance parameters are re-fitted under the declared common bounds",
            "the null fit uses the same linked raw roles and IRLS count as the alternative",
            "the reported score excludes the alternative's spatial regularisation penalty",
        ),
    )


def fit_linked_zero_density_null(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    observation: LinkedRawObservation,
    point_fit: LinkedScalarFitResult,
    *,
    reference_light_provenance: ReferenceLightInferenceProvenance,
    initial_nuisance: LinkedNuisanceValues,
    nuisance_lower: ArrayLike,
    nuisance_upper: ArrayLike,
    regularisation: CurvatureRegularisation | None,
    options: LinkedScalarFitOptions,
) -> LinkedZeroDensityNullFit:
    """Fit the linked zero-density null while re-estimating shared nuisances.

    Density is not represented by a near-zero coefficient or an active bound;
    it is removed from the parameter vector.  The nuisance bounds, linked raw
    roles and number of IRLS updates are the caller-declared counterparts of
    the alternative fit.  Spatial regularisation is validated for the
    alternative route but contributes no term to this data-only null fit.
    """

    if not isinstance(options, LinkedScalarFitOptions):
        raise TypeError("options must be LinkedScalarFitOptions")
    if not point_fit.diagnostics.success:
        raise ValueError("point fit must succeed before fitting the zero-density null")
    _validate_reference_light_primary_inputs(
        operator,
        model,
        point_fit,
        reference_light_provenance,
        regularisation=regularisation,
    )
    _validate_observation_matches_fit(operator, observation, point_fit)
    if options.irls_iterations != point_fit.diagnostics.irls_iterations:
        raise ValueError("null and alternative fits must use the same IRLS count")
    current = nuisance_vector(initial_nuisance)
    if (
        isinstance(operator, PCILinkedRawOperator)
        and not isinstance(initial_nuisance, PCINuisanceValues)
    ) or (
        isinstance(operator, DGILinkedRawOperator)
        and not isinstance(initial_nuisance, DGINuisanceValues)
    ):
        raise TypeError("initial nuisance type does not match the linked operator")
    lower = np.asarray(nuisance_lower, dtype=float)
    upper = np.asarray(nuisance_upper, dtype=float)
    shape = (len(operator.nuisance_names),)
    if lower.shape != shape or upper.shape != shape:
        raise ValueError("null-fit nuisance bounds have the wrong shape")
    point_nuisance = np.asarray(point_fit.nuisance_values, dtype=float)
    if (
        np.any(~np.isfinite(lower))
        or np.any(~np.isfinite(upper))
        or np.any(lower < 0.0)
        or np.any(upper <= lower)
        or np.any(current < lower)
        or np.any(current > upper)
        or np.any(point_nuisance < lower)
        or np.any(point_nuisance > upper)
    ):
        raise ValueError("null-fit nuisance bounds or initial values are invalid")

    frame_count = int(point_fit.density_coefficients.shape[0])
    first_prediction = _zero_density_linked_prediction(
        operator,
        frame_count,
        initial_nuisance,
    )
    if first_prediction.role_names != observation.role_names:
        raise ValueError("zero-density null and alternative raw roles differ")
    roi = np.asarray(operator.grid.roi_mask, dtype=bool)
    observed_vector = np.concatenate(
        [np.asarray(values, dtype=float)[roi] for values in observation.observed_electrons]
    )
    final_result = None
    completed_irls = 0
    try:
        for outer in range(options.irls_iterations):
            initial_prediction = _zero_density_linked_prediction(
                operator,
                frame_count,
                nuisance_from_vector(operator, current),
            )
            standard_deviation = np.sqrt(
                np.concatenate(
                    [values[roi] for values in initial_prediction.conditional_variance_electrons2]
                )
            )
            cached_vector: FloatArray | None = None
            cached_prediction: LinkedRawSequencePrediction | None = None

            def evaluate(vector: FloatArray) -> LinkedRawSequencePrediction:
                nonlocal cached_vector, cached_prediction
                if cached_vector is None or not np.array_equal(vector, cached_vector):
                    cached_prediction = _zero_density_linked_prediction(
                        operator,
                        frame_count,
                        nuisance_from_vector(operator, vector),
                    )
                    cached_vector = np.array(vector, copy=True)
                if cached_prediction is None:
                    raise RuntimeError("zero-density prediction cache was not populated")
                return cached_prediction

            def residual(vector: FloatArray) -> FloatArray:
                return (
                    observed_vector - evaluate(vector).prediction_vector
                ) / standard_deviation

            def jacobian(vector: FloatArray) -> FloatArray:
                return -evaluate(vector).jacobian / standard_deviation[:, None]

            tr_options = None
            if options.trust_region_solver == "lsmr":
                tr_options = {
                    "atol": options.lsmr_atol,
                    "btol": options.lsmr_btol,
                    "conlim": options.lsmr_conlim,
                    "maxiter": options.lsmr_maxiter,
                    "regularize": options.lsmr_regularize,
                }
            final_result = least_squares(
                residual,
                current,
                jac=jacobian,
                bounds=(lower, upper),
                method=options.method,
                loss=options.loss,
                x_scale=options.x_scale,
                max_nfev=options.max_nfev,
                xtol=options.xtol,
                ftol=options.ftol,
                gtol=options.gtol,
                tr_solver=options.trust_region_solver,
                tr_options=tr_options,
            )
            current = np.asarray(final_result.x, dtype=float)
            completed_irls = outer + 1
    except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as exc:
        return _null_fit_failure(
            operator,
            current,
            frame_count=frame_count,
            nuisance_lower=lower,
            nuisance_upper=upper,
            observation=observation,
            message=f"null fit raised {type(exc).__name__}: {exc}",
            nfev=0,
            irls_iterations=completed_irls,
            provenance=reference_light_provenance,
        )
    if final_result is None:
        raise RuntimeError("zero-density linked fit did not execute")
    prediction = _zero_density_linked_prediction(
        operator,
        frame_count,
        nuisance_from_vector(operator, current),
    )
    success = bool(final_result.success and np.all(np.isfinite(current)))
    result = LinkedZeroDensityNullFit(
        nuisance_names=tuple(operator.nuisance_names),
        nuisance_values=current,
        nuisance_lower=lower,
        nuisance_upper=upper,
        prediction=prediction,
        success=success,
        message=str(final_result.message) or "zero-density fit returned no message",
        nfev=int(final_result.nfev),
        irls_iterations=completed_irls,
        data_gaussian_quasi_deviance=_linked_data_gaussian_quasi_deviance(
            operator,
            observation,
            prediction,
        ),
        route_provenance=reference_light_provenance,
        assumptions=reference_light_provenance.assumptions
        + (
            "all density frames are fixed exactly to zero",
            "shared nuisance parameters are re-fitted under the declared common bounds",
            "the null fit uses the same linked raw roles and IRLS count as the alternative",
            "the reported score excludes the alternative's spatial regularisation penalty",
        ),
    )
    return result


@dataclass(frozen=True, eq=False)
class LinkedSyntheticBlankReference:
    """Synthetic blank statistics used only for diagnostic ranking."""

    delta_data_gaussian_quasi_deviance: FloatArray
    case_ids: tuple[str, ...]
    attempted_count: int
    failed_case_ids: tuple[str, ...]
    pipeline_fingerprint: str
    condition_fingerprint: str

    def __post_init__(self) -> None:
        values = np.asarray(self.delta_data_gaussian_quasi_deviance, dtype=float)
        if values.ndim != 1 or np.any(~np.isfinite(values)):
            raise ValueError("synthetic blank statistics must be a finite vector")
        raw_case_ids = _text_tuple(
            self.case_ids,
            name="synthetic blank case IDs",
            allow_empty=True,
        )
        raw_failed_ids = _text_tuple(
            self.failed_case_ids,
            name="failed synthetic blank case IDs",
            allow_empty=True,
        )
        case_ids = tuple(
            _nonempty_text(value, name="synthetic blank case ID")
            for value in raw_case_ids
        )
        failed_ids = tuple(
            _nonempty_text(value, name="failed synthetic blank case ID")
            for value in raw_failed_ids
        )
        attempted = _strict_int(
            self.attempted_count,
            name="synthetic blank attempted_count",
            minimum=1,
        )
        if len(case_ids) != values.size:
            raise ValueError("synthetic blank IDs must match successful statistics")
        if attempted != len(case_ids) + len(failed_ids):
            raise ValueError("synthetic blank attempts must equal success plus failure")
        if len(set(case_ids + failed_ids)) != attempted:
            raise ValueError("synthetic blank case IDs must be unique")
        object.__setattr__(
            self,
            "delta_data_gaussian_quasi_deviance",
            _immutable(values),
        )
        object.__setattr__(self, "case_ids", case_ids)
        object.__setattr__(self, "failed_case_ids", failed_ids)
        object.__setattr__(self, "attempted_count", attempted)
        object.__setattr__(
            self,
            "pipeline_fingerprint",
            _nonempty_text(self.pipeline_fingerprint, name="pipeline_fingerprint"),
        )
        object.__setattr__(
            self,
            "condition_fingerprint",
            _nonempty_text(self.condition_fingerprint, name="condition_fingerprint"),
        )


@dataclass(frozen=True)
class LinkedZeroDensityEvidence:
    """Data-only zero-density comparison without a p value or threshold."""

    statistic_name: Literal["delta_data_gaussian_quasi_deviance"]
    null_model: Literal["linked_zero_density_with_refitted_shared_nuisance"]
    pipeline_fingerprint: str
    condition_fingerprint: str
    alternative_data_gaussian_quasi_deviance: float
    null_data_gaussian_quasi_deviance: float
    delta_data_gaussian_quasi_deviance: float
    alternative_is_regularised: bool
    evidence_level: LinkedNullEvidenceLevel
    reference_count: int
    failed_reference_count: int
    development_rank_from_largest: int | None
    route_provenance: ReferenceLightInferenceProvenance
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.statistic_name != "delta_data_gaussian_quasi_deviance":
            raise ValueError("linked null-evidence statistic changed")
        if self.null_model != "linked_zero_density_with_refitted_shared_nuisance":
            raise ValueError("linked zero-density null model changed")
        pipeline = _nonempty_text(
            self.pipeline_fingerprint,
            name="pipeline_fingerprint",
        )
        condition = _nonempty_text(
            self.condition_fingerprint,
            name="condition_fingerprint",
        )
        alternative = float(self.alternative_data_gaussian_quasi_deviance)
        null = float(self.null_data_gaussian_quasi_deviance)
        delta = float(self.delta_data_gaussian_quasi_deviance)
        if np.any(~np.isfinite([alternative, null, delta])):
            raise ValueError("linked null-evidence scores must be finite")
        tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(null), abs(alternative))
        if not np.isclose(delta, null - alternative, rtol=0.0, atol=tolerance):
            raise ValueError("linked null-evidence score difference is inconsistent")
        if type(self.alternative_is_regularised) is not bool:
            raise TypeError("alternative_is_regularised must be bool")
        if self.evidence_level not in (
            "model_only",
            "synthetic_blank_development_rank",
        ):
            raise ValueError("unknown linked null-evidence level")
        reference_count = _strict_int(
            self.reference_count,
            name="reference_count",
            minimum=0,
        )
        failed_count = _strict_int(
            self.failed_reference_count,
            name="failed_reference_count",
            minimum=0,
        )
        rank = self.development_rank_from_largest
        if self.evidence_level == "model_only":
            if rank is not None:
                raise ValueError("model-only evidence cannot report a development rank")
        else:
            if reference_count == 0:
                raise ValueError("development-rank evidence requires successful blanks")
            rank = _strict_int(rank, name="development rank", minimum=1)
            if rank > reference_count + 1:
                raise ValueError("development rank exceeds the target-plus-reference set")
        if not isinstance(self.route_provenance, ReferenceLightInferenceProvenance):
            raise TypeError("linked null-evidence provenance has the wrong type")
        assumptions = _text_tuple(
            self.assumptions,
            name="linked null-evidence assumptions",
            allow_empty=False,
        )
        object.__setattr__(self, "pipeline_fingerprint", pipeline)
        object.__setattr__(self, "condition_fingerprint", condition)
        object.__setattr__(
            self,
            "alternative_data_gaussian_quasi_deviance",
            alternative,
        )
        object.__setattr__(self, "null_data_gaussian_quasi_deviance", null)
        object.__setattr__(self, "delta_data_gaussian_quasi_deviance", delta)
        object.__setattr__(self, "reference_count", reference_count)
        object.__setattr__(self, "failed_reference_count", failed_count)
        object.__setattr__(self, "development_rank_from_largest", rank)
        object.__setattr__(self, "assumptions", assumptions)


def _assert_same_linked_prediction(
    stored: LinkedRawSequencePrediction,
    recomputed: LinkedRawSequencePrediction,
) -> None:
    metadata_equal = (
        stored.role_names == recomputed.role_names
        and stored.role_frame_indices == recomputed.role_frame_indices
        and stored.shared_role_names == recomputed.shared_role_names
        and stored.density_parameter_slices == recomputed.density_parameter_slices
        and stored.nuisance_names == recomputed.nuisance_names
    )
    arrays_equal = all(
        np.allclose(left, right, rtol=1e-12, atol=0.0)
        for left, right in zip(
            stored.expected_electrons
            + stored.conditional_variance_electrons2
            + (stored.prediction_vector, stored.jacobian),
            recomputed.expected_electrons
            + recomputed.conditional_variance_electrons2
            + (recomputed.prediction_vector, recomputed.jacobian),
            strict=True,
        )
    )
    if not metadata_equal or not arrays_equal:
        raise ValueError("stored linked prediction does not match its declared null fit")


def analyse_linked_zero_density_evidence(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    observation: LinkedRawObservation,
    point_fit: LinkedScalarFitResult,
    null_fit: LinkedZeroDensityNullFit,
    *,
    reference_light_provenance: ReferenceLightInferenceProvenance,
    regularisation: CurvatureRegularisation | None,
    pipeline_fingerprint: str,
    condition_fingerprint: str,
    synthetic_blank_reference: LinkedSyntheticBlankReference | None = None,
) -> LinkedZeroDensityEvidence:
    """Compare the fitted sequence with a nuisance-refitted zero-density null.

    The signed statistic is the null data-only Gaussian quasi-deviance minus
    the alternative data-only Gaussian quasi-deviance.  Synthetic blanks can
    provide a descriptive development rank only; this API has no p-value,
    acceptance level or threshold field.
    """

    pipeline = _nonempty_text(pipeline_fingerprint, name="pipeline_fingerprint")
    condition = _nonempty_text(condition_fingerprint, name="condition_fingerprint")
    if not isinstance(null_fit, LinkedZeroDensityNullFit):
        raise TypeError("null_fit has the wrong type")
    if not point_fit.diagnostics.success:
        raise ValueError("alternative point fit must succeed before evidence analysis")
    if not null_fit.success:
        raise ValueError("zero-density null fit must succeed before evidence analysis")
    _validate_reference_light_primary_inputs(
        operator,
        model,
        point_fit,
        reference_light_provenance,
        regularisation=regularisation,
    )
    _validate_observation_matches_fit(operator, observation, point_fit)
    if null_fit.route_provenance != reference_light_provenance:
        raise ValueError("null-fit and alternative route provenance differ")
    if null_fit.nuisance_names != tuple(operator.nuisance_names):
        raise ValueError("null-fit nuisance order does not belong to the operator")
    if null_fit.irls_iterations != point_fit.diagnostics.irls_iterations:
        raise ValueError("null-fit and alternative IRLS counts differ")
    point_nuisance = np.asarray(point_fit.nuisance_values, dtype=float)
    if np.any(point_nuisance < null_fit.nuisance_lower) or np.any(
        point_nuisance > null_fit.nuisance_upper
    ):
        raise ValueError("declared common nuisance bounds exclude the alternative fit")
    frame_count = int(point_fit.density_coefficients.shape[0])
    recomputed_null = _zero_density_linked_prediction(
        operator,
        frame_count,
        nuisance_from_vector(operator, null_fit.nuisance_values),
    )
    _assert_same_linked_prediction(null_fit.prediction, recomputed_null)
    if null_fit.prediction.role_names != point_fit.prediction.role_names:
        raise ValueError("null and alternative linked raw roles differ")
    alternative_score = _linked_data_gaussian_quasi_deviance(
        operator,
        observation,
        point_fit.prediction,
    )
    null_score = _linked_data_gaussian_quasi_deviance(
        operator,
        observation,
        null_fit.prediction,
    )
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(null_score))
    if not np.isclose(
        null_fit.data_gaussian_quasi_deviance,
        null_score,
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("null-fit stored data score does not match its prediction")
    delta = null_score - alternative_score
    level: LinkedNullEvidenceLevel = "model_only"
    reference_count = 0
    failed_reference_count = 0
    development_rank: int | None = None
    assumptions = list(reference_light_provenance.assumptions)
    assumptions.extend(
        (
            "the null fixes both q1 and q2 density exactly to zero and re-fits shared nuisances",
            "the signed statistic contains raw-role data terms only; spatial regularisation is excluded",
            "bounded or regularised alternatives do not inherit a chi-square or Wilks interpretation",
            "a positive score difference alone is not a calibrated cloud-detection claim",
        )
    )
    if synthetic_blank_reference is not None:
        if not isinstance(synthetic_blank_reference, LinkedSyntheticBlankReference):
            raise TypeError("synthetic_blank_reference has the wrong type")
        reference_count = int(
            synthetic_blank_reference.delta_data_gaussian_quasi_deviance.size
        )
        failed_reference_count = len(synthetic_blank_reference.failed_case_ids)
        if synthetic_blank_reference.pipeline_fingerprint != pipeline:
            assumptions.append(
                "the synthetic blank pipeline fingerprint differs, so its values were not ranked"
            )
        elif synthetic_blank_reference.condition_fingerprint != condition:
            assumptions.append(
                "the synthetic blank condition fingerprint differs, so its values were not ranked"
            )
        elif reference_count:
            development_rank = 1 + int(
                np.count_nonzero(
                    synthetic_blank_reference.delta_data_gaussian_quasi_deviance
                    >= delta
                )
            )
            level = "synthetic_blank_development_rank"
            assumptions.append(
                "the synthetic blank comparison is a development rank only and defines no p value or threshold"
            )
            if failed_reference_count:
                assumptions.append(
                    "failed synthetic blank fits are retained, so the development rank is incomplete"
                )
        else:
            assumptions.append(
                "all supplied synthetic blank fits failed, so no development rank is reported"
            )
    return LinkedZeroDensityEvidence(
        statistic_name="delta_data_gaussian_quasi_deviance",
        null_model="linked_zero_density_with_refitted_shared_nuisance",
        pipeline_fingerprint=pipeline,
        condition_fingerprint=condition,
        alternative_data_gaussian_quasi_deviance=alternative_score,
        null_data_gaussian_quasi_deviance=null_score,
        delta_data_gaussian_quasi_deviance=delta,
        alternative_is_regularised=regularisation is not None,
        evidence_level=level,
        reference_count=reference_count,
        failed_reference_count=failed_reference_count,
        development_rank_from_largest=development_rank,
        route_provenance=reference_light_provenance,
        assumptions=tuple(assumptions),
    )


@dataclass(frozen=True)
class ConditionalObservableEstimate:
    """One point estimate and strictly labelled conditional-bootstrap status."""

    observable_name: str
    quantity: InformationQuantity
    unit: str
    estimate: float | None
    status: ConditionalEstimateStatus
    confidence_level: float
    requested_draws: int
    successful_fit_draws: int
    supported_draws: int
    lower: float | None
    upper: float | None
    null_value: float | None
    excludes_null: bool | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.observable_name not in (*OBSERVABLE_NAMES, *DERIVED_OBSERVABLE_NAMES):
            raise ValueError("unknown conditional-estimate observable")
        if self.quantity not in ("q1", "q2", "delta_21", "ratio_21"):
            raise ValueError("unknown conditional-estimate quantity")
        if self.quantity == "ratio_21" and self.observable_name not in (
            *POSITIVE_OBSERVABLE_NAMES,
            *DERIVED_OBSERVABLE_NAMES,
        ):
            raise ValueError("ratios are restricted to positive observables")
        expected_unit = (
            "1" if self.quantity == "ratio_21" else OBSERVABLE_UNITS[self.observable_name]
        )
        if self.unit != expected_unit:
            raise ValueError("conditional-estimate unit changed")
        level = float(self.confidence_level)
        requested_draws = _strict_int(
            self.requested_draws,
            name="requested_draws",
            minimum=1,
        )
        successful_fit_draws = _strict_int(
            self.successful_fit_draws,
            name="successful_fit_draws",
            minimum=0,
        )
        supported_draws = _strict_int(
            self.supported_draws,
            name="supported_draws",
            minimum=0,
        )
        reasons = _text_tuple(
            self.reasons,
            name="conditional estimate reasons",
            allow_empty=True,
        )
        if self.status not in ("complete", "partial", "unresolved"):
            raise ValueError("unknown conditional estimate status")
        if self.excludes_null is not None and type(self.excludes_null) is not bool:
            raise TypeError("excludes_null must be bool or None")
        if not 0.0 < level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not 0 <= successful_fit_draws <= requested_draws:
            raise ValueError("successful_fit_draws must lie within requested_draws")
        if not 0 <= supported_draws <= successful_fit_draws:
            raise ValueError("supported_draws must lie within successful fits")
        if self.estimate is not None and not np.isfinite(self.estimate):
            raise ValueError("estimate must be finite or None")
        if self.null_value is not None and not np.isfinite(self.null_value):
            raise ValueError("null_value must be finite or None")
        expected_null = (
            0.0
            if self.quantity == "delta_21"
            else 1.0
            if self.quantity == "ratio_21"
            else None
        )
        if self.null_value != expected_null:
            raise ValueError("conditional-estimate null value changed")
        if self.status == "complete":
            if (
                self.estimate is None
                or requested_draws < 2
                or supported_draws != requested_draws
                or successful_fit_draws != requested_draws
                or self.lower is None
                or self.upper is None
            ):
                raise ValueError("complete estimates require all requested draws")
            if not np.isfinite(self.lower) or not np.isfinite(self.upper):
                raise ValueError("complete bounds must be finite")
            if self.lower > self.upper or reasons:
                raise ValueError("complete estimate bounds or reasons are inconsistent")
            expected_exclusion = (
                None
                if self.null_value is None
                else bool(self.lower > self.null_value or self.upper < self.null_value)
            )
            if self.excludes_null != expected_exclusion:
                raise ValueError("null-exclusion flag disagrees with complete bounds")
        elif self.status == "partial":
            if self.estimate is None or supported_draws == 0:
                raise ValueError("partial status requires a point and supported draws")
            if (
                self.lower is not None
                or self.upper is not None
                or self.excludes_null is not None
            ):
                raise ValueError(
                    "partial estimates cannot report empirical bounds or null exclusion"
                )
            if (
                supported_draws == requested_draws
                and requested_draws >= 2
            ):
                raise ValueError("full multi-draw support must use complete status")
            if not reasons:
                raise ValueError("partial estimates require explicit reasons")
        else:
            if self.lower is not None or self.upper is not None or self.excludes_null is not None:
                raise ValueError("unresolved estimates cannot report empirical bounds")
            if self.estimate is not None and supported_draws > 0:
                raise ValueError(
                    "a point with supported draws must use partial or complete status"
                )
            if not reasons:
                raise ValueError("incomplete estimates require explicit reasons")
        object.__setattr__(self, "confidence_level", level)
        object.__setattr__(self, "requested_draws", requested_draws)
        object.__setattr__(self, "successful_fit_draws", successful_fit_draws)
        object.__setattr__(self, "supported_draws", supported_draws)
        object.__setattr__(self, "reasons", reasons)

    @property
    def estimate_form(self) -> EstimateForm:
        """Return the evidence form represented by this estimate."""

        if self.status == "complete":
            return "two_sided_interval"
        if self.estimate is not None:
            return "point_only"
        return "none"


@dataclass(frozen=True)
class OneSidedObservableBound:
    """A genuine one-sided bound from an explicit predeclared construction.

    This type is deliberately separate from conditional point/interval output,
    so a two-sided result cannot be promoted to ``bounded`` by changing a label.
    The construction and its assumptions must be supplied by the later
    observable-specific calibration/evaluation contract.
    """

    observable_name: str
    quantity: InformationQuantity
    unit: str
    bound_value: float
    direction: Literal["lower", "upper"]
    confidence_level: float
    construction: str
    predeclared_rule_id: str
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        bound = float(self.bound_value)
        level = float(self.confidence_level)
        assumptions = _text_tuple(
            self.assumptions,
            name="one-sided bound assumptions",
            allow_empty=False,
        )
        allowed_observables = (*OBSERVABLE_NAMES, *DERIVED_OBSERVABLE_NAMES)
        if self.observable_name not in allowed_observables:
            raise ValueError("unknown one-sided-bound observable")
        if self.quantity not in ("q1", "q2", "delta_21", "ratio_21"):
            raise ValueError("unknown one-sided-bound quantity")
        if self.quantity == "ratio_21":
            if self.observable_name not in (
                *POSITIVE_OBSERVABLE_NAMES,
                *DERIVED_OBSERVABLE_NAMES,
            ):
                raise ValueError("ratio bounds are restricted to positive observables")
            expected_unit = "1"
        else:
            expected_unit = OBSERVABLE_UNITS[self.observable_name]
        if self.unit != expected_unit:
            raise ValueError("one-sided-bound unit changed")
        if not np.isfinite(bound):
            raise ValueError("one-sided bound value must be finite")
        nonnegative_domain = self.quantity == "ratio_21" or (
            self.quantity in ("q1", "q2")
            and self.observable_name
            in (*POSITIVE_OBSERVABLE_NAMES, *DERIVED_OBSERVABLE_NAMES)
        )
        if nonnegative_domain and bound < 0.0:
            raise ValueError("one-sided bound lies outside the non-negative domain")
        if self.direction not in ("lower", "upper"):
            raise ValueError("one-sided bound direction must be lower or upper")
        if not 0.0 < level < 1.0:
            raise ValueError("one-sided confidence_level must lie in (0, 1)")
        if (
            not isinstance(self.construction, str)
            or not self.construction.strip()
            or not isinstance(self.predeclared_rule_id, str)
            or not self.predeclared_rule_id.strip()
        ):
            raise ValueError("one-sided bound construction and rule id are required")
        object.__setattr__(self, "bound_value", bound)
        object.__setattr__(self, "confidence_level", level)
        object.__setattr__(self, "assumptions", assumptions)

    @property
    def estimate_form(self) -> EstimateForm:
        """Return the one-sided evidence form without any caller override."""

        return "lower_bound" if self.direction == "lower" else "upper_bound"


@dataclass(frozen=True)
class TwoFrameObservableSummary:
    """Per-frame, change and optional ratio information for one observable."""

    observable_name: str
    unit: str
    q1: ConditionalObservableEstimate
    q2: ConditionalObservableEstimate
    delta_21: ConditionalObservableEstimate
    ratio_21: ConditionalObservableEstimate | None

    def __post_init__(self) -> None:
        if self.observable_name not in (*OBSERVABLE_NAMES, "aspect_ratio_y_over_z"):
            raise ValueError("unknown two-frame observable name")
        expected_unit = OBSERVABLE_UNITS[self.observable_name]
        if self.unit != expected_unit:
            raise ValueError("two-frame observable unit changed")
        if self.ratio_21 is not None and self.observable_name not in (
            *POSITIVE_OBSERVABLE_NAMES,
            "aspect_ratio_y_over_z",
        ):
            raise ValueError("ratios are restricted to positive observables")
        records = (self.q1, self.q2, self.delta_21)
        expected_quantities = ("q1", "q2", "delta_21")
        if any(
            record.observable_name != self.observable_name
            or record.unit != self.unit
            or record.quantity != quantity
            for record, quantity in zip(records, expected_quantities, strict=True)
        ):
            raise ValueError("two-frame estimate identity changed")
        if self.ratio_21 is not None and (
            self.ratio_21.observable_name != self.observable_name
            or self.ratio_21.unit != "1"
            or self.ratio_21.quantity != "ratio_21"
        ):
            raise ValueError("two-frame ratio identity changed")


@dataclass(frozen=True)
class TwoFrameInformationSummary:
    """Information carried by q1, q2 and their paired changes."""

    frame_labels: tuple[str, str]
    observables: Mapping[str, TwoFrameObservableSummary]
    derived_aspect_ratio: TwoFrameObservableSummary | None
    route_provenance: ReferenceLightInferenceProvenance
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        frame_labels = tuple(self.frame_labels)
        assumptions = _text_tuple(
            self.assumptions,
            name="two-frame assumptions",
            allow_empty=False,
        )
        if frame_labels != ("q1", "q2"):
            raise ValueError("two-frame labels must be exactly ('q1', 'q2')")
        summaries = dict(self.observables)
        if tuple(summaries) != OBSERVABLE_NAMES:
            raise ValueError("two-frame summaries must follow the canonical order")
        if any(name != value.observable_name for name, value in summaries.items()):
            raise ValueError("observable summary keys do not match their records")
        if self.derived_aspect_ratio is not None and (
            self.derived_aspect_ratio.observable_name != "aspect_ratio_y_over_z"
        ):
            raise ValueError("derived aspect-ratio record has the wrong name")
        if not isinstance(self.route_provenance, ReferenceLightInferenceProvenance):
            raise TypeError("two-frame route provenance has the wrong type")
        object.__setattr__(self, "frame_labels", frame_labels)
        object.__setattr__(self, "observables", MappingProxyType(summaries))
        object.__setattr__(self, "assumptions", assumptions)


def _conditional_estimate(
    *,
    observable_name: str,
    quantity: InformationQuantity,
    unit: str,
    point_value: float,
    point_supported: bool,
    sample_values: FloatArray,
    sample_supported: BoolArray,
    fit_success_mask: BoolArray,
    confidence_level: float,
    null_value: float | None,
) -> ConditionalObservableEstimate:
    requested = int(fit_success_mask.size)
    successful = int(np.count_nonzero(fit_success_mask))
    supported_mask = np.asarray(sample_supported, dtype=bool) & fit_success_mask
    supported = int(np.count_nonzero(supported_mask))
    estimate = float(point_value) if point_supported and np.isfinite(point_value) else None
    reasons: list[str] = []
    if estimate is None:
        reasons.append("point_estimate_not_numerically_supported")
    if successful < requested:
        reasons.append("incomplete_bootstrap_refits")
    if supported < successful:
        reasons.append("observable_not_supported_in_every_successful_refit")
    if supported == 0:
        reasons.append("no_supported_bootstrap_draws")
    if 0 < supported < 2:
        reasons.append("fewer_than_two_supported_draws_for_empirical_interval")

    interval: tuple[float, float] | None = None
    if (
        estimate is not None
        and requested >= 2
        and supported == requested
        and successful == requested
    ):
        alpha = 0.5 * (1.0 - confidence_level)
        lower, upper = np.quantile(
            np.asarray(sample_values, dtype=float)[supported_mask],
            [alpha, 1.0 - alpha],
        )
        interval = (float(lower), float(upper))
    if estimate is not None and supported == requested and requested >= 2:
        assert interval is not None
        lower_value, upper_value = interval
        exclusion = (
            None
            if null_value is None
            else bool(lower_value > null_value or upper_value < null_value)
        )
        return ConditionalObservableEstimate(
            observable_name=observable_name,
            quantity=quantity,
            unit=unit,
            estimate=estimate,
            status="complete",
            confidence_level=confidence_level,
            requested_draws=requested,
            successful_fit_draws=successful,
            supported_draws=supported,
            lower=lower_value,
            upper=upper_value,
            null_value=null_value,
            excludes_null=exclusion,
            reasons=(),
        )
    status: ConditionalEstimateStatus = (
        "partial" if estimate is not None and supported > 0 else "unresolved"
    )
    return ConditionalObservableEstimate(
        observable_name=observable_name,
        quantity=quantity,
        unit=unit,
        estimate=estimate,
        status=status,
        confidence_level=confidence_level,
        requested_draws=requested,
        successful_fit_draws=successful,
        supported_draws=supported,
        lower=None,
        upper=None,
        null_value=null_value,
        excludes_null=None,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _summarise_values(
    *,
    observable_name: str,
    point_values: FloatArray,
    point_supported: BoolArray,
    sample_values: FloatArray,
    sample_supported: BoolArray,
    fit_success_mask: BoolArray,
    confidence_level: float,
    allow_ratio: bool,
) -> TwoFrameObservableSummary:
    unit = OBSERVABLE_UNITS[observable_name]
    frame_estimates = tuple(
        _conditional_estimate(
            observable_name=observable_name,
            quantity="q1" if index == 0 else "q2",
            unit=unit,
            point_value=float(point_values[index]),
            point_supported=bool(point_supported[index]),
            sample_values=sample_values[:, index],
            sample_supported=sample_supported[:, index],
            fit_success_mask=fit_success_mask,
            confidence_level=confidence_level,
            null_value=None,
        )
        for index in range(2)
    )
    delta_point_supported = bool(np.all(point_supported))
    delta_samples_supported = np.all(sample_supported, axis=1)
    delta = _conditional_estimate(
        observable_name=observable_name,
        quantity="delta_21",
        unit=unit,
        point_value=float(point_values[1] - point_values[0]),
        point_supported=delta_point_supported,
        sample_values=sample_values[:, 1] - sample_values[:, 0],
        sample_supported=delta_samples_supported,
        fit_success_mask=fit_success_mask,
        confidence_level=confidence_level,
        null_value=0.0,
    )
    ratio: ConditionalObservableEstimate | None = None
    if allow_ratio:
        ratio_point_supported = bool(
            np.all(point_supported) and point_values[0] > 0.0
        )
        ratio_samples_supported = (
            np.all(sample_supported, axis=1) & (sample_values[:, 0] > 0.0)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_samples = sample_values[:, 1] / sample_values[:, 0]
            ratio_point = point_values[1] / point_values[0]
        ratio = _conditional_estimate(
            observable_name=observable_name,
            quantity="ratio_21",
            unit="1",
            point_value=float(ratio_point),
            point_supported=ratio_point_supported,
            sample_values=ratio_samples,
            sample_supported=ratio_samples_supported,
            fit_success_mask=fit_success_mask,
            confidence_level=confidence_level,
            null_value=1.0,
        )
    return TwoFrameObservableSummary(
        observable_name=observable_name,
        unit=OBSERVABLE_UNITS[observable_name],
        q1=frame_estimates[0],
        q2=frame_estimates[1],
        delta_21=delta,
        ratio_21=ratio,
    )


def summarise_two_frame_information(
    bootstrap: LinkedObservableInformationBootstrap,
    *,
    confidence_level: float,
) -> TwoFrameInformationSummary:
    """Summarise q1, q2 and paired changes without using simulated truth."""

    if not isinstance(bootstrap, LinkedObservableInformationBootstrap):
        raise TypeError("bootstrap has the wrong type")
    if bootstrap.frame_count != 2:
        raise ValueError("two-frame information requires exactly q1 and q2")
    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    summaries: dict[str, TwoFrameObservableSummary] = {}
    for index, name in enumerate(OBSERVABLE_NAMES):
        summaries[name] = _summarise_values(
            observable_name=name,
            point_values=bootstrap.point_estimates[:, index],
            point_supported=bootstrap.point_supported_mask[:, index],
            sample_values=bootstrap.samples[:, :, index],
            sample_supported=bootstrap.supported_mask[:, :, index],
            fit_success_mask=bootstrap.fit_success_mask,
            confidence_level=level,
            allow_ratio=name in POSITIVE_OBSERVABLE_NAMES,
        )

    sigma_y_index = OBSERVABLE_NAMES.index("sigma_y_um")
    sigma_z_index = OBSERVABLE_NAMES.index("sigma_z_um")
    aspect_point_supported = (
        bootstrap.point_supported_mask[:, sigma_y_index]
        & bootstrap.point_supported_mask[:, sigma_z_index]
        & (bootstrap.point_estimates[:, sigma_z_index] > 0.0)
    )
    aspect_sample_supported = (
        bootstrap.supported_mask[:, :, sigma_y_index]
        & bootstrap.supported_mask[:, :, sigma_z_index]
        & (bootstrap.samples[:, :, sigma_z_index] > 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_points = (
            bootstrap.point_estimates[:, sigma_y_index]
            / bootstrap.point_estimates[:, sigma_z_index]
        )
        aspect_samples = (
            bootstrap.samples[:, :, sigma_y_index]
            / bootstrap.samples[:, :, sigma_z_index]
        )
    aspect = _summarise_values(
        observable_name="aspect_ratio_y_over_z",
        point_values=aspect_points,
        point_supported=aspect_point_supported,
        sample_values=aspect_samples,
        sample_supported=aspect_sample_supported,
        fit_success_mask=bootstrap.fit_success_mask,
        confidence_level=level,
        allow_ratio=True,
    )
    if not np.any(aspect_point_supported):
        aspect = None

    return TwoFrameInformationSummary(
        frame_labels=("q1", "q2"),
        observables=summaries,
        derived_aspect_ratio=aspect,
        route_provenance=bootstrap.route_provenance,
        assumptions=bootstrap.assumptions
        + (
            "delta_21 and ratio_21 use aligned q1/q2 values from the same linked draw",
            "reported intervals are conditional detector-noise intervals, not posteriors",
            "partial and unresolved outputs suppress empirical bounds and null exclusion",
            "A is a response integral until an absolute response calibration is supplied",
            "aspect ratio is derived only from jointly supported axial and radial widths",
        ),
    )


def _bound_vector(
    value: float | ArrayLike,
    *,
    size: int,
    name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=float)
    if array.shape != (size,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be a finite scalar or length-{size} vector")
    return np.asarray(array, dtype=float)


def _scaled_jacobian_subspaces(
    scaled_jacobian: ArrayLike,
    *,
    parameter_count: int,
    relative_rank_tolerance: float,
) -> tuple[FloatArray, FloatArray, int, float]:
    """Return stable right-singular subspaces without forming J-transpose-J."""

    jacobian = np.asarray(scaled_jacobian, dtype=float)
    if (
        jacobian.ndim != 2
        or jacobian.shape[1] != parameter_count
        or np.any(~np.isfinite(jacobian))
    ):
        raise ValueError("scaled Jacobian has the wrong shape or non-finite values")
    tolerance = float(relative_rank_tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie in (0, 1)")
    # Economy QR in ``mode='r'`` avoids materialising the enormous left
    # singular-vector matrix for the usual tall raw-pixel Jacobian.  SVD of R
    # retains J's singular values and right singular vectors without squaring
    # its condition number.
    triangular = np.linalg.qr(jacobian, mode="r")
    _, compact_singular_values, right_transpose = np.linalg.svd(
        triangular,
        full_matrices=triangular.shape[0] < parameter_count,
    )
    singular_values = np.zeros(parameter_count, dtype=float)
    singular_values[: compact_singular_values.size] = compact_singular_values
    vectors = right_transpose.T
    largest = float(singular_values[0])
    threshold = tolerance * largest
    rank = int(np.count_nonzero(singular_values > threshold)) if largest > 0.0 else 0
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == parameter_count and singular_values[-1] > 0.0
        else float("inf")
    )
    return (
        np.asarray(singular_values, dtype=float),
        np.asarray(vectors, dtype=float),
        rank,
        condition,
    )


def _axis_observable_parameter_gradients(
    model: DifferentiableColumnDensityModel,
    parameter_vector: ArrayLike,
    support: ObservableIntegrationSupport,
    *,
    jacobian_batch_size: int,
) -> tuple[FloatArray, BoolArray, FloatArray]:
    """Return values, support and analytic gradients of the five observables."""

    parameters = np.asarray(parameter_vector, dtype=float)
    if parameters.shape != (model.parameter_count,) or np.any(~np.isfinite(parameters)):
        raise ValueError("observable-gradient parameter vector has the wrong shape")
    if jacobian_batch_size <= 0:
        raise ValueError("jacobian_batch_size must be positive")
    density = np.asarray(model.column_density(parameters), dtype=float)
    values, supported = _observable_vector_with_support(density, support)
    summary = extract_density_observables(density, support)
    mask = support.support_mask
    area = np.where(mask, support.cell_area_m2, 0.0)
    pixel_gradients = np.full((len(OBSERVABLE_NAMES), *support.shape), np.nan)
    pixel_gradients[0] = area
    if summary.centroid_m is not None and summary.covariance_m2 is not None:
        integrated = summary.integrated_response
        y_c, z_c = summary.centroid_m
        y_displacement = support.y_grid_m - y_c
        z_displacement = support.z_grid_m - z_c
        pixel_gradients[1] = area * y_displacement / integrated * 1e6
        pixel_gradients[2] = area * z_displacement / integrated * 1e6
        sigma_y = float(np.sqrt(summary.covariance_m2[0, 0]))
        sigma_z = float(np.sqrt(summary.covariance_m2[1, 1]))
        if sigma_y > 0.0:
            pixel_gradients[3] = (
                area
                * (y_displacement**2 - summary.covariance_m2[0, 0])
                / (2.0 * sigma_y * integrated)
                * 1e6
            )
        else:
            supported[3] = False
            values[3] = np.nan
        if sigma_z > 0.0:
            pixel_gradients[4] = (
                area
                * (z_displacement**2 - summary.covariance_m2[1, 1])
                / (2.0 * sigma_z * integrated)
                * 1e6
            )
        else:
            supported[4] = False
            values[4] = np.nan

    gradients = np.full((len(OBSERVABLE_NAMES), model.parameter_count), np.nan)
    populated = np.zeros(model.parameter_count, dtype=bool)
    for parameter_slice, derivative_batch in model.iter_column_density_jacobian(
        parameters,
        jacobian_batch_size,
    ):
        if parameter_slice.start is None or parameter_slice.stop is None:
            raise ValueError("observable Jacobian slices require explicit bounds")
        if (
            parameter_slice.start < 0
            or parameter_slice.stop > model.parameter_count
            or parameter_slice.start >= parameter_slice.stop
            or np.any(populated[parameter_slice])
        ):
            raise ValueError("observable Jacobian batches overlap or exceed the model")
        derivatives = np.asarray(derivative_batch, dtype=float)
        expected_shape = (
            parameter_slice.stop - parameter_slice.start,
            *support.shape,
        )
        if derivatives.shape != expected_shape or np.any(~np.isfinite(derivatives)):
            raise ValueError("model density-Jacobian batch has the wrong shape or values")
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
        raise ValueError("observable Jacobian batches do not cover every model parameter")
    return values, supported, gradients


@dataclass(frozen=True)
class ObservableIdentifiabilityRecord:
    """Continuous local data-support metrics for one two-frame quantity."""

    observable_name: str
    quantity: InformationQuantity
    estimate: float | None
    scaled_gradient_norm: float | None
    data_null_space_fraction: float | None
    active_bound_gradient_fraction: float | None
    identified_subspace_standard_uncertainty: float | None
    supported: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        reasons = _text_tuple(
            self.reasons,
            name="identifiability reasons",
            allow_empty=True,
        )
        if type(self.supported) is not bool:
            raise TypeError("identifiability supported flag must be bool")
        if self.observable_name not in (*OBSERVABLE_NAMES, *DERIVED_OBSERVABLE_NAMES):
            raise ValueError("unknown identifiability observable name")
        if self.quantity not in ("q1", "q2", "delta_21", "ratio_21"):
            raise ValueError("unknown identifiability quantity")
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
                raise ValueError("supported observable gradient must be non-zero")
            if not 0.0 <= self.data_null_space_fraction <= 1.0:
                raise ValueError("data-null fraction must lie in [0, 1]")
            if not 0.0 <= self.active_bound_gradient_fraction <= 1.0:
                raise ValueError("active-bound fraction must lie in [0, 1]")
            if self.identified_subspace_standard_uncertainty < 0.0 or reasons:
                raise ValueError("supported identifiability record is inconsistent")
        else:
            if any(value is not None for value in metrics):
                raise ValueError("unsupported identifiability cannot report metrics")
            if self.estimate is not None and not np.isfinite(self.estimate):
                raise ValueError("unsupported estimate must be finite or None")
            if not reasons:
                raise ValueError("unsupported identifiability requires a reason")
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True, eq=False)
class TwoFrameIdentifiabilitySummary:
    """Local data-only identifiability for q1, q2 and paired quantities."""

    parameter_count: int
    data_rank: int
    relative_rank_tolerance: float
    relative_active_bound_tolerance: float
    singular_values: FloatArray
    data_condition_number: float
    active_bound_parameter_count: int
    records: tuple[ObservableIdentifiabilityRecord, ...]
    route_provenance: ReferenceLightInferenceProvenance
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        parameter_count = _strict_int(
            self.parameter_count,
            name="parameter_count",
            minimum=1,
        )
        data_rank = _strict_int(self.data_rank, name="data_rank", minimum=0)
        active_bound_parameter_count = _strict_int(
            self.active_bound_parameter_count,
            name="active_bound_parameter_count",
            minimum=0,
        )
        records = tuple(self.records)
        assumptions = _text_tuple(
            self.assumptions,
            name="identifiability assumptions",
            allow_empty=False,
        )
        if any(not isinstance(record, ObservableIdentifiabilityRecord) for record in records):
            raise TypeError("identifiability records have the wrong type")
        if not 0 <= data_rank <= parameter_count:
            raise ValueError("identifiability rank and parameter count are inconsistent")
        tolerance = float(self.relative_rank_tolerance)
        if not 0.0 < tolerance < 1.0:
            raise ValueError("relative_rank_tolerance must lie in (0, 1)")
        active_tolerance = float(self.relative_active_bound_tolerance)
        if not 0.0 < active_tolerance < 1.0:
            raise ValueError("relative_active_bound_tolerance must lie in (0, 1)")
        singular = np.asarray(self.singular_values, dtype=float)
        if singular.shape != (parameter_count,):
            raise ValueError("identifiability singular spectrum has the wrong length")
        if np.any(~np.isfinite(singular)) or np.any(singular < 0.0):
            raise ValueError("identifiability singular values must be finite and non-negative")
        if np.any(np.diff(singular) > 0.0):
            raise ValueError("identifiability singular values must be descending")
        if not np.isfinite(self.data_condition_number) and self.data_condition_number != float("inf"):
            raise ValueError("data condition number must be finite or positive infinity")
        if not 0 <= active_bound_parameter_count <= parameter_count:
            raise ValueError("active-bound count is inconsistent")
        if not records:
            raise ValueError("identifiability summary requires observable records")
        expected_keys: list[tuple[str, InformationQuantity]] = []
        for observable_name in OBSERVABLE_NAMES:
            expected_keys.extend(
                (observable_name, quantity)
                for quantity in ("q1", "q2", "delta_21")
            )
            if observable_name in POSITIVE_OBSERVABLE_NAMES:
                expected_keys.append((observable_name, "ratio_21"))
        expected_keys.extend(
            ("aspect_ratio_y_over_z", quantity)
            for quantity in ("q1", "q2", "delta_21", "ratio_21")
        )
        actual_keys = [
            (record.observable_name, record.quantity) for record in records
        ]
        if actual_keys != expected_keys:
            raise ValueError("identifiability records changed order or membership")
        if not isinstance(self.route_provenance, ReferenceLightInferenceProvenance):
            raise TypeError("identifiability route provenance has the wrong type")
        object.__setattr__(self, "parameter_count", parameter_count)
        object.__setattr__(self, "data_rank", data_rank)
        object.__setattr__(
            self,
            "active_bound_parameter_count",
            active_bound_parameter_count,
        )
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "relative_rank_tolerance", tolerance)
        object.__setattr__(
            self,
            "relative_active_bound_tolerance",
            active_tolerance,
        )
        object.__setattr__(self, "singular_values", _immutable(singular))
        object.__setattr__(self, "assumptions", assumptions)


def analyse_two_frame_observable_identifiability(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    fit_result: LinkedScalarFitResult,
    *,
    reference_light_provenance: ReferenceLightInferenceProvenance,
    integration_support: ObservableIntegrationSupport,
    density_parameter_lower: float | ArrayLike,
    density_coefficient_upper: float | ArrayLike,
    nuisance_lower: ArrayLike,
    nuisance_upper: ArrayLike,
    regularisation: CurvatureRegularisation | None,
    relative_rank_tolerance: float,
    relative_active_bound_tolerance: float,
) -> TwoFrameIdentifiabilitySummary:
    """Measure local observable support from the likelihood Jacobian alone.

    Parameter coordinates are scaled by their declared bound spans before the
    right-singular decomposition.  Curvature rows are deliberately absent, so the
    reported rank and null-space fractions are not regularisation support.
    No threshold is applied to convert these continuous metrics into a paper
    confidence grade.
    """

    if fit_result.density_coefficients.shape[0] != 2:
        raise ValueError("two-frame identifiability requires exactly q1 and q2")
    if not fit_result.diagnostics.success:
        raise ValueError("identifiability requires a successful linked point fit")
    _validate_reference_light_primary_inputs(
        operator,
        model,
        fit_result,
        reference_light_provenance,
        regularisation=regularisation,
    )
    _validate_observable_support_contract(operator, model, integration_support)
    tolerance = float(relative_rank_tolerance)
    if not 0.0 < tolerance < 1.0:
        raise ValueError("relative_rank_tolerance must lie in (0, 1)")
    active_relative_tolerance = float(relative_active_bound_tolerance)
    if not 0.0 < active_relative_tolerance < 1.0:
        raise ValueError("relative_active_bound_tolerance must lie in (0, 1)")

    density_lower = _bound_vector(
        density_parameter_lower,
        size=model.parameter_count,
        name="density_parameter_lower",
    )
    density_upper = _bound_vector(
        density_coefficient_upper,
        size=model.parameter_count,
        name="density_coefficient_upper",
    )
    nuisance_lower_array = _bound_vector(
        nuisance_lower,
        size=len(fit_result.nuisance_names),
        name="nuisance_lower",
    )
    nuisance_upper_array = _bound_vector(
        nuisance_upper,
        size=len(fit_result.nuisance_names),
        name="nuisance_upper",
    )
    if np.any(density_upper <= density_lower) or np.any(
        nuisance_upper_array <= nuisance_lower_array
    ):
        raise ValueError("identifiability bounds must be strictly ordered")
    spans = np.concatenate(
        [
            np.tile(density_upper - density_lower, 2),
            nuisance_upper_array - nuisance_lower_array,
        ]
    )
    parameter_values = np.concatenate(
        [fit_result.density_coefficients.ravel(), fit_result.nuisance_values]
    )
    lower = np.concatenate(
        [np.tile(density_lower, 2), nuisance_lower_array]
    )
    upper = np.concatenate(
        [np.tile(density_upper, 2), nuisance_upper_array]
    )
    if parameter_values.shape != spans.shape:
        raise ValueError("fit result parameter count disagrees with declared bounds")
    if np.any(parameter_values < lower) or np.any(parameter_values > upper):
        raise ValueError("fitted parameters lie outside identifiability bounds")
    active_tolerance = active_relative_tolerance * spans
    active = (parameter_values - lower <= active_tolerance) | (
        upper - parameter_values <= active_tolerance
    )

    prediction = fit_result.prediction
    variance = np.concatenate(
        [
            np.asarray(item, dtype=float)[operator.grid.roi_mask]
            for item in prediction.conditional_variance_electrons2
        ]
    )
    jacobian = np.asarray(prediction.jacobian, dtype=float)
    if jacobian.shape != (variance.size, spans.size):
        raise ValueError("prediction Jacobian does not match identifiability bounds")
    if np.any(~np.isfinite(jacobian)) or np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
        raise ValueError("identifiability Jacobian or variance is invalid")
    scaled_jacobian = jacobian / np.sqrt(variance)[:, None] * spans[None, :]
    singular_values, vectors, rank, condition = _scaled_jacobian_subspaces(
        scaled_jacobian,
        parameter_count=spans.size,
        relative_rank_tolerance=tolerance,
    )

    frame_values: list[FloatArray] = []
    frame_supported: list[BoolArray] = []
    frame_gradients: list[FloatArray] = []
    for coefficients in fit_result.density_coefficients:
        values, supported, gradients = _axis_observable_parameter_gradients(
            model,
            coefficients,
            integration_support,
            jacobian_batch_size=operator.jacobian_batch_size,
        )
        frame_values.append(values)
        frame_supported.append(supported)
        frame_gradients.append(gradients)

    def record(
        observable_name: str,
        quantity: InformationQuantity,
        estimate: float | None,
        gradient: FloatArray | None,
        reason: str | None = None,
    ) -> ObservableIdentifiabilityRecord:
        if gradient is None or estimate is None or not np.isfinite(estimate):
            return ObservableIdentifiabilityRecord(
                observable_name=observable_name,
                quantity=quantity,
                estimate=None if estimate is None or not np.isfinite(estimate) else float(estimate),
                scaled_gradient_norm=None,
                data_null_space_fraction=None,
                active_bound_gradient_fraction=None,
                identified_subspace_standard_uncertainty=None,
                supported=False,
                reasons=(reason or "observable_gradient_not_supported",),
            )
        full_gradient = np.asarray(gradient, dtype=float)
        if full_gradient.shape != spans.shape or np.any(~np.isfinite(full_gradient)):
            raise ValueError("observable identifiability gradient has the wrong shape")
        scaled_gradient = full_gradient * spans
        gradient_norm = float(np.linalg.norm(scaled_gradient))
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
        identified_vectors = vectors[:, :rank]
        null_vectors = vectors[:, rank:]
        identified_projection = identified_vectors.T @ scaled_gradient
        null_projection = null_vectors.T @ scaled_gradient
        null_fraction = float(np.linalg.norm(null_projection) / gradient_norm)
        active_fraction = float(np.linalg.norm(scaled_gradient[active]) / gradient_norm)
        identified_variance = float(
            np.sum(
                (identified_projection / singular_values[:rank]) ** 2
            )
        ) if rank else 0.0
        return ObservableIdentifiabilityRecord(
            observable_name=observable_name,
            quantity=quantity,
            estimate=float(estimate),
            scaled_gradient_norm=gradient_norm,
            data_null_space_fraction=min(max(null_fraction, 0.0), 1.0),
            active_bound_gradient_fraction=min(max(active_fraction, 0.0), 1.0),
            identified_subspace_standard_uncertainty=float(np.sqrt(max(identified_variance, 0.0))),
            supported=True,
            reasons=(),
        )

    records: list[ObservableIdentifiabilityRecord] = []
    parameter_count = model.parameter_count
    for observable_index, observable_name in enumerate(OBSERVABLE_NAMES):
        full_gradients: list[FloatArray | None] = []
        for frame_index in range(2):
            gradient: FloatArray | None = None
            if frame_supported[frame_index][observable_index]:
                gradient = np.zeros(spans.size, dtype=float)
                start = frame_index * parameter_count
                gradient[start : start + parameter_count] = frame_gradients[frame_index][
                    observable_index
                ]
            full_gradients.append(gradient)
            records.append(
                record(
                    observable_name,
                    "q1" if frame_index == 0 else "q2",
                    (
                        float(frame_values[frame_index][observable_index])
                        if frame_supported[frame_index][observable_index]
                        else None
                    ),
                    gradient,
                )
            )
        if full_gradients[0] is not None and full_gradients[1] is not None:
            delta_gradient = full_gradients[1] - full_gradients[0]
            delta_estimate = float(
                frame_values[1][observable_index]
                - frame_values[0][observable_index]
            )
        else:
            delta_gradient = None
            delta_estimate = None
        records.append(
            record(
                observable_name,
                "delta_21",
                delta_estimate,
                delta_gradient,
                "both frame gradients are required for delta_21",
            )
        )
        if observable_name in POSITIVE_OBSERVABLE_NAMES:
            denominator = frame_values[0][observable_index]
            if (
                full_gradients[0] is not None
                and full_gradients[1] is not None
                and denominator > 0.0
            ):
                numerator = frame_values[1][observable_index]
                ratio_estimate = float(numerator / denominator)
                ratio_gradient = (
                    full_gradients[1] / denominator
                    - numerator * full_gradients[0] / denominator**2
                )
            else:
                ratio_estimate = None
                ratio_gradient = None
            records.append(
                record(
                    observable_name,
                    "ratio_21",
                    ratio_estimate,
                    ratio_gradient,
                    "positive supported q1 and q2 gradients are required for ratio_21",
                )
            )

    sigma_y_index = OBSERVABLE_NAMES.index("sigma_y_um")
    sigma_z_index = OBSERVABLE_NAMES.index("sigma_z_um")
    aspect_values: list[float | None] = []
    aspect_gradients: list[FloatArray | None] = []
    for frame_index in range(2):
        sigma_y = float(frame_values[frame_index][sigma_y_index])
        sigma_z = float(frame_values[frame_index][sigma_z_index])
        if (
            frame_supported[frame_index][sigma_y_index]
            and frame_supported[frame_index][sigma_z_index]
            and sigma_y > 0.0
            and sigma_z > 0.0
        ):
            aspect_value = sigma_y / sigma_z
            local_gradient = (
                frame_gradients[frame_index][sigma_y_index] / sigma_z
                - sigma_y
                * frame_gradients[frame_index][sigma_z_index]
                / sigma_z**2
            )
            full_gradient = np.zeros(spans.size, dtype=float)
            start = frame_index * parameter_count
            full_gradient[start : start + parameter_count] = local_gradient
        else:
            aspect_value = None
            full_gradient = None
        aspect_values.append(aspect_value)
        aspect_gradients.append(full_gradient)
        records.append(
            record(
                "aspect_ratio_y_over_z",
                "q1" if frame_index == 0 else "q2",
                aspect_value,
                full_gradient,
                "positive jointly supported widths are required for aspect ratio",
            )
        )
    if aspect_gradients[0] is not None and aspect_gradients[1] is not None:
        assert aspect_values[0] is not None and aspect_values[1] is not None
        aspect_delta = float(aspect_values[1] - aspect_values[0])
        aspect_delta_gradient = aspect_gradients[1] - aspect_gradients[0]
    else:
        aspect_delta = None
        aspect_delta_gradient = None
    records.append(
        record(
            "aspect_ratio_y_over_z",
            "delta_21",
            aspect_delta,
            aspect_delta_gradient,
            "both frame aspect-ratio gradients are required for delta_21",
        )
    )
    if (
        aspect_gradients[0] is not None
        and aspect_gradients[1] is not None
        and aspect_values[0] is not None
        and aspect_values[1] is not None
        and aspect_values[0] > 0.0
    ):
        aspect_ratio_21 = aspect_values[1] / aspect_values[0]
        aspect_ratio_gradient = (
            aspect_gradients[1] / aspect_values[0]
            - aspect_values[1]
            * aspect_gradients[0]
            / aspect_values[0] ** 2
        )
    else:
        aspect_ratio_21 = None
        aspect_ratio_gradient = None
    records.append(
        record(
            "aspect_ratio_y_over_z",
            "ratio_21",
            aspect_ratio_21,
            aspect_ratio_gradient,
            "positive q1/q2 aspect-ratio gradients are required for ratio_21",
        )
    )

    return TwoFrameIdentifiabilitySummary(
        parameter_count=int(spans.size),
        data_rank=rank,
        relative_rank_tolerance=tolerance,
        relative_active_bound_tolerance=active_relative_tolerance,
        singular_values=singular_values,
        data_condition_number=condition,
        active_bound_parameter_count=int(np.count_nonzero(active)),
        records=tuple(records),
        route_provenance=reference_light_provenance,
        assumptions=reference_light_provenance.assumptions
        + (
            "local linearisation at the linked point fit",
            "noise-whitened raw-count Jacobian without curvature rows",
            "parameter coordinates scaled by their declared bound spans",
            "identified-subspace uncertainty excludes any data-null component",
            "inequality-bound truncation is not applied; active-bound gradient dependence is reported separately",
            "continuous metrics are not converted to a confidence grade here",
            "fixed optical operator and observable support",
            "aspect-ratio support uses its direct joint width gradient",
        ),
    )


@dataclass(frozen=True)
class ConfidenceDecomposition:
    """Separate evidence axes used to interpret one reported quantity."""

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
        reasons = _text_tuple(
            self.reasons,
            name="confidence reasons",
            allow_empty=True,
        )
        evidence_values = (
            self.fit_and_data,
            self.detector_statistical,
            self.identifiability,
            self.calibration,
            self.forward_model,
            self.repeatability,
            self.relative_change,
        )
        if any(
            value
            not in (
                "adequate",
                "limited",
                "not_assessed",
                "not_applicable",
                "failed",
            )
            for value in evidence_values
        ):
            raise ValueError("unknown confidence evidence grade")
        mandatory_evidence = (
            self.fit_and_data,
            self.detector_statistical,
            self.identifiability,
            self.calibration,
            self.forward_model,
            self.repeatability,
        )
        if "not_applicable" in mandatory_evidence:
            raise ValueError(
                "fit/data, detector, identifiability, calibration, forward-model "
                "and repeatability evidence are always applicable"
            )
        dependence_values = (
            self.basis_model,
            self.support,
            self.reference,
            self.regularisation,
        )
        if any(
            value
            not in (
                "stable",
                "sensitive",
                "not_assessed",
                "not_applicable",
                "failed",
            )
            for value in dependence_values
        ):
            raise ValueError("unknown model-dependence grade")
        if self.basis_model == "not_applicable" or self.support == "not_applicable":
            raise ValueError("basis-model and support dependence are always applicable")
        fully_assessed = (
            all(value in ("adequate", "not_applicable") for value in evidence_values)
            and all(
                value in ("stable", "not_applicable")
                for value in dependence_values
            )
        )
        if not fully_assessed and not reasons:
            raise ValueError("non-adequate confidence requires explicit reasons")
        object.__setattr__(self, "reasons", reasons)


def classify_information_level(
    estimate: ConditionalObservableEstimate | OneSidedObservableBound,
    confidence: ConfidenceDecomposition,
) -> InformationLevel:
    """Classify one quantity without collapsing its confidence components.

    The classifier contains no numerical threshold.  Upstream analyses must
    assign the component grades using observable-specific rules frozen before
    formal numerical evidence is inspected.  ``bounded`` requires an explicit
    :class:`OneSidedObservableBound`; a point or two-sided interval cannot be
    relabelled into that category.
    """

    if not isinstance(
        estimate,
        (ConditionalObservableEstimate, OneSidedObservableBound),
    ):
        raise TypeError("estimate has the wrong type")
    if not isinstance(confidence, ConfidenceDecomposition):
        raise TypeError("confidence has the wrong type")
    if estimate.quantity in ("q1", "q2"):
        if confidence.relative_change != "not_applicable":
            raise ValueError(
                "relative_change must be not_applicable for a single-frame quantity"
            )
    elif confidence.relative_change == "not_applicable":
        raise ValueError(
            "relative_change must be assessed or explicitly not_assessed for "
            "delta_21 and ratio_21"
        )
    if confidence.fit_and_data == "failed":
        return "fit_or_data_failure"
    failed_quantitative_component = (
        confidence.detector_statistical == "failed"
        or confidence.identifiability == "failed"
        or confidence.calibration == "failed"
        or confidence.forward_model == "failed"
    )
    dependence_values = (
        confidence.basis_model,
        confidence.support,
        confidence.reference,
        confidence.regularisation,
    )
    if failed_quantitative_component or "failed" in dependence_values:
        return "unresolved"
    if isinstance(estimate, ConditionalObservableEstimate) and (
        estimate.status == "unresolved"
        or estimate.estimate_form == "none"
        or estimate.estimate is None
    ):
        return "unresolved"
    if "sensitive" in dependence_values:
        return "prior_sensitive"
    evidence_values = (
        confidence.fit_and_data,
        confidence.detector_statistical,
        confidence.identifiability,
        confidence.calibration,
        confidence.forward_model,
        confidence.repeatability,
        confidence.relative_change,
    )
    evidence_adequate = all(
        value in ("adequate", "not_applicable") for value in evidence_values
    )
    dependence_stable = all(
        value in ("stable", "not_applicable") for value in dependence_values
    )
    if isinstance(estimate, OneSidedObservableBound):
        if evidence_adequate and dependence_stable:
            return "bounded"
        return "informative_but_inconclusive"

    form = estimate.estimate_form
    if (
        estimate.status == "complete"
        and form == "two_sided_interval"
        and evidence_adequate
        and dependence_stable
    ):
        return "quantitatively_resolved"
    return "informative_but_inconclusive"


__all__ = [
    "BootstrapDrawStatus",
    "InformationQuantity",
    "ConditionalEstimateStatus",
    "ConditionalObservableEstimate",
    "ConfidenceDecomposition",
    "DERIVED_OBSERVABLE_NAMES",
    "EstimateForm",
    "EvidenceGrade",
    "InformationLevel",
    "LinkedObservableInformationBootstrap",
    "LinkedObservableBootstrapDraw",
    "LinkedNullEvidenceLevel",
    "LinkedRawDiagnosticsSummary",
    "LinkedRawRoleDiagnostics",
    "LinkedSyntheticBlankReference",
    "LinkedZeroDensityEvidence",
    "LinkedZeroDensityNullFit",
    "ModelDependenceGrade",
    "OBSERVABLE_UNITS",
    "OneSidedObservableBound",
    "POSITIVE_OBSERVABLE_NAMES",
    "ReferenceLightInferenceProvenance",
    "TwoFrameInformationSummary",
    "TwoFrameIdentifiabilitySummary",
    "TwoFrameObservableSummary",
    "ObservableIdentifiabilityRecord",
    "analyse_linked_raw_residuals",
    "analyse_linked_zero_density_evidence",
    "analyse_two_frame_observable_identifiability",
    "bootstrap_linked_observable_information",
    "classify_information_level",
    "fit_linked_zero_density_null",
    "refit_linked_observable_bootstrap_draw",
    "select_q1_observation_for_reference_sensitivity",
    "summarise_two_frame_information",
]
