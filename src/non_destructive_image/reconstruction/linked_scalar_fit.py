"""Joint linked-raw fitting for scalar PCI/DGI column-density sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from .object_models import (
    DifferentiableColumnDensityModel,
    NonnegativeBilinearDensityModel,
)
from .regularisation import CurvatureRegularisation
from .scalar_measurements import (
    DGILinkedRawOperator,
    DGINuisanceValues,
    LinkedRawSequencePrediction,
    PCILinkedRawOperator,
    PCINuisanceValues,
)


FloatArray = NDArray[np.floating]
LinkedScalarOperator: TypeAlias = PCILinkedRawOperator | DGILinkedRawOperator
LinkedNuisanceValues: TypeAlias = PCINuisanceValues | DGINuisanceValues


def _immutable(array: ArrayLike, *, dtype: type = float) -> NDArray:
    source = np.asarray(array, dtype=dtype)
    result = np.frombuffer(source.tobytes(order="C"), dtype=np.dtype(dtype))
    return result.reshape(source.shape)


@dataclass(frozen=True, eq=False)
class LinkedRawObservation:
    """Observed electron-valued raw roles in the operator's linked order."""

    role_names: tuple[str, ...]
    observed_electrons: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        if not self.role_names or len(set(self.role_names)) != len(self.role_names):
            raise ValueError("observation role names must be non-empty and unique")
        if len(self.observed_electrons) != len(self.role_names):
            raise ValueError("observation role names and arrays must have equal length")
        arrays = tuple(_immutable(value) for value in self.observed_electrons)
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("observed raw roles must be two-dimensional")
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("observed raw roles must share one camera shape")
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("observed raw electron values must be finite")
        object.__setattr__(self, "observed_electrons", arrays)


@dataclass(frozen=True)
class LinkedScalarFitOptions:
    """Numerical controls for one joint linked-raw sequence fit."""

    method: Literal["trf"] = "trf"
    loss: Literal["linear"] = "linear"
    x_scale: Literal["jac"] = "jac"
    irls_iterations: int = 2
    max_nfev: int = 60
    xtol: float = 1e-7
    ftol: float = 1e-7
    gtol: float = 1e-7
    trust_region_solver: Literal["exact", "lsmr"] = "exact"
    lsmr_atol: float = 1e-6
    lsmr_btol: float = 1e-6
    lsmr_conlim: float = 1e8
    lsmr_maxiter: int | None = None
    lsmr_regularize: bool = True

    def __post_init__(self) -> None:
        if self.method != "trf" or self.loss != "linear" or self.x_scale != "jac":
            raise ValueError(
                "linked scalar fits require method='trf', loss='linear' and "
                "x_scale='jac'"
            )
        if self.irls_iterations <= 0 or self.max_nfev <= 0:
            raise ValueError("IRLS iterations and max_nfev must be positive")
        if min(self.xtol, self.ftol, self.gtol) <= 0.0:
            raise ValueError("fit tolerances must be positive")
        if self.trust_region_solver not in ("exact", "lsmr"):
            raise ValueError("trust_region_solver must be 'exact' or 'lsmr'")
        lsmr_values = np.asarray(
            (self.lsmr_atol, self.lsmr_btol, self.lsmr_conlim),
            dtype=float,
        )
        if np.any(~np.isfinite(lsmr_values)) or np.any(lsmr_values <= 0.0):
            raise ValueError("LSMR atol, btol and conlim must be finite and positive")
        if self.lsmr_maxiter is not None and (
            isinstance(self.lsmr_maxiter, bool)
            or not isinstance(self.lsmr_maxiter, int)
            or self.lsmr_maxiter <= 0
        ):
            raise ValueError("lsmr_maxiter must be a positive integer or None")
        if type(self.lsmr_regularize) is not bool:
            raise TypeError("lsmr_regularize must be bool")


@dataclass(frozen=True)
class LinkedScalarFitDiagnostics:
    """Numerical and raw-data support diagnostics for a linked fit."""

    success: bool
    message: str
    weighted_chi_square: float
    reduced_chi_square: float
    degrees_of_freedom: int
    data_jacobian_rank: int
    data_jacobian_condition: float
    nfev: int
    irls_iterations: int
    regularisation_objective: float
    active_lower_density_coefficients: int
    active_upper_density_coefficients: int
    active_nuisance_bounds: int
    whitened_residual_vector: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "whitened_residual_vector",
            _immutable(self.whitened_residual_vector),
        )


@dataclass(frozen=True, eq=False)
class LinkedScalarFitResult:
    """Recovered per-frame nuisance fields and shared sequence nuisances."""

    density_coefficients: FloatArray
    column_density_m2: tuple[FloatArray, ...]
    nuisance_names: tuple[str, ...]
    nuisance_values: FloatArray
    prediction: LinkedRawSequencePrediction
    diagnostics: LinkedScalarFitDiagnostics

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.density_coefficients, dtype=float)
        nuisance = np.asarray(self.nuisance_values, dtype=float)
        if coefficients.ndim != 2 or np.any(~np.isfinite(coefficients)):
            raise ValueError("density_coefficients must be a finite frame-by-parameter array")
        if nuisance.shape != (len(self.nuisance_names),) or np.any(~np.isfinite(nuisance)):
            raise ValueError("nuisance values do not match their names")
        maps = tuple(_immutable(value) for value in self.column_density_m2)
        if len(maps) != coefficients.shape[0]:
            raise ValueError("one fitted density map is required per coefficient row")
        object.__setattr__(self, "density_coefficients", _immutable(coefficients))
        object.__setattr__(self, "nuisance_values", _immutable(nuisance))
        object.__setattr__(self, "column_density_m2", maps)


def nuisance_vector(nuisance: LinkedNuisanceValues) -> FloatArray:
    """Return nuisance values in the corresponding operator's fixed order."""

    if isinstance(nuisance, PCINuisanceValues):
        return np.asarray(
            [
                nuisance.i0_photoelectrons_per_pixel,
                nuisance.dark_electrons_per_pixel,
            ],
            dtype=float,
        )
    if isinstance(nuisance, DGINuisanceValues):
        return np.asarray(
            [
                nuisance.i0_photoelectrons_per_pixel,
                nuisance.stop_dark_electrons_per_pixel,
                nuisance.open_dark_electrons_per_pixel,
                nuisance.open_to_stop_scale,
            ],
            dtype=float,
        )
    raise TypeError("unsupported linked nuisance type")


def nuisance_from_vector(
    operator: LinkedScalarOperator,
    values: ArrayLike,
) -> LinkedNuisanceValues:
    """Construct the operator-specific nuisance record from its fixed vector."""

    vector = np.asarray(values, dtype=float)
    if vector.shape != (len(operator.nuisance_names),):
        raise ValueError("nuisance vector has the wrong shape")
    if isinstance(operator, PCILinkedRawOperator):
        return PCINuisanceValues(*vector)
    if isinstance(operator, DGILinkedRawOperator):
        return DGINuisanceValues(*vector)
    raise TypeError("unsupported linked scalar operator")


def draw_linked_raw_observation(
    operator: LinkedScalarOperator,
    prediction: LinkedRawSequencePrediction,
    rng: np.random.Generator,
) -> LinkedRawObservation:
    """Draw every fresh/shared raw role once with its declared multiplicity."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    observed: list[FloatArray] = []
    for role_name, frame_index, expected in zip(
        prediction.role_names,
        prediction.role_frame_indices,
        prediction.expected_electrons,
        strict=True,
    ):
        if frame_index is None:
            base_role = role_name
        elif isinstance(operator, PCILinkedRawOperator):
            base_role = "atom"
        else:
            base_role = "atom_stop"
        exposure_count = operator.independent_exposures_by_role[base_role]
        draws = []
        for _ in range(exposure_count):
            poisson = rng.poisson(expected)
            read = rng.normal(0.0, operator.read_noise_electrons, expected.shape)
            draws.append(np.asarray(poisson + read, dtype=float))
        observed.append(np.mean(draws, axis=0))
    return LinkedRawObservation(prediction.role_names, tuple(observed))


def estimate_linked_nuisance_from_references(
    operator: LinkedScalarOperator,
    observation: LinkedRawObservation,
) -> LinkedNuisanceValues:
    """Estimate a bounded-fit starting point from shared reference roles only.

    The estimate is an initializer, not an uncertainty calculation. Negative
    sample means from read noise are projected to the physical zero boundary;
    the subsequent joint fit still consumes every raw role exactly once.
    """

    if len(observation.role_names) != len(observation.observed_electrons):
        raise ValueError("observation role metadata is inconsistent")
    role_map = dict(
        zip(observation.role_names, observation.observed_electrons, strict=True)
    )

    def mean(role: str) -> float:
        if role not in role_map:
            raise ValueError(f"observation is missing shared role {role!r}")
        return float(np.mean(role_map[role][operator.grid.roi_mask]))

    minimum_positive = np.finfo(float).tiny
    if isinstance(operator, PCILinkedRawOperator):
        dark = max(mean("dark"), 0.0)
        i0 = max(
            (mean("bright_reference") - dark)
            / operator.transfer.phase_plate_transmittance**2,
            minimum_positive,
        )
        return PCINuisanceValues(i0, dark)
    if isinstance(operator, DGILinkedRawOperator):
        stop_dark = max(mean("stop_dark"), 0.0)
        open_dark = max(mean("open_dark"), 0.0)
        i0 = max(mean("open_reference") - open_dark, minimum_positive)
        leakage = operator.transfer.carrier_field**2
        open_to_stop = max(
            (mean("leakage_stop") - stop_dark) / (i0 * leakage),
            minimum_positive,
        )
        return DGINuisanceValues(i0, stop_dark, open_dark, open_to_stop)
    raise TypeError("unsupported linked scalar operator")


def _flatten_observation(
    operator: LinkedScalarOperator,
    observation: LinkedRawObservation,
    expected_role_names: tuple[str, ...],
) -> FloatArray:
    if observation.role_names != expected_role_names:
        raise ValueError("observation role order does not match the linked operator")
    if any(array.shape != operator.grid.camera_shape for array in observation.observed_electrons):
        raise ValueError("observation camera shape does not match the operator")
    return np.concatenate(
        [array[operator.grid.roi_mask] for array in observation.observed_electrons]
    )


def _flatten_prediction_variance(
    operator: LinkedScalarOperator,
    prediction: LinkedRawSequencePrediction,
) -> FloatArray:
    return np.concatenate(
        [array[operator.grid.roi_mask] for array in prediction.conditional_variance_electrons2]
    )


def _regularisation_matrix(
    model: DifferentiableColumnDensityModel,
    regularisation: CurvatureRegularisation | None,
) -> FloatArray:
    if regularisation is None:
        return np.zeros((0, model.parameter_count), dtype=float)
    if not isinstance(model, NonnegativeBilinearDensityModel):
        raise TypeError("curvature regularisation requires the bilinear density model")
    if regularisation.parameter_count != model.parameter_count:
        raise ValueError("regularisation parameter count does not match the density model")
    if not np.array_equal(regularisation.knot_y_um, model.knot_y_um) or not np.array_equal(
        regularisation.knot_z_um,
        model.knot_z_um,
    ):
        raise ValueError("regularisation knots do not match the density model")
    return np.asarray(
        regularisation.matrix_for_coefficient_scale(model.coefficient_scale_m2),
        dtype=float,
    )


def fit_linked_scalar_sequence(
    operator: LinkedScalarOperator,
    model: DifferentiableColumnDensityModel,
    observation: LinkedRawObservation,
    *,
    initial_density_coefficients: ArrayLike,
    density_parameter_lower: float | ArrayLike = 0.0,
    density_coefficient_upper: float | ArrayLike,
    initial_nuisance: LinkedNuisanceValues,
    nuisance_lower: ArrayLike,
    nuisance_upper: ArrayLike,
    regularisation: CurvatureRegularisation | None,
    options: LinkedScalarFitOptions | None = None,
) -> LinkedScalarFitResult:
    """Fit independent per-frame density fields with shared raw nuisances.

    Spatial curvature is applied separately to every frame.  There is no
    temporal penalty or transition law.  Shared reference/dark roles occur once
    in the observation and therefore once in the likelihood.
    """

    fit_options = options or LinkedScalarFitOptions()
    initial_density = np.asarray(initial_density_coefficients, dtype=float)
    if initial_density.ndim != 2 or initial_density.shape[1] != model.parameter_count:
        raise ValueError(
            "initial_density_coefficients must have shape "
            f"(n_frame, {model.parameter_count})"
        )
    if initial_density.shape[0] == 0 or np.any(~np.isfinite(initial_density)):
        raise ValueError("initial density coefficients must be finite and non-empty")
    density_lower = np.asarray(density_parameter_lower, dtype=float)
    if density_lower.ndim == 0:
        density_lower = np.full(model.parameter_count, float(density_lower))
    if density_lower.shape != (model.parameter_count,):
        raise ValueError("density parameter lower bound has the wrong shape")
    density_upper = np.asarray(density_coefficient_upper, dtype=float)
    if density_upper.ndim == 0:
        density_upper = np.full(model.parameter_count, float(density_upper))
    if density_upper.shape != (model.parameter_count,):
        raise ValueError("density coefficient upper bound has the wrong shape")
    if (
        np.any(~np.isfinite(density_lower))
        or np.any(~np.isfinite(density_upper))
        or np.any(density_upper <= density_lower)
    ):
        raise ValueError("density parameter bounds must be finite and ordered")
    if np.any(initial_density < density_lower) or np.any(initial_density > density_upper):
        raise ValueError("initial density parameters lie outside their bounds")

    initial_nuisance_vector = nuisance_vector(initial_nuisance)
    if tuple(operator.nuisance_names) != tuple(
        PCILinkedRawOperator.nuisance_names
        if isinstance(initial_nuisance, PCINuisanceValues)
        else DGILinkedRawOperator.nuisance_names
    ):
        raise TypeError("initial nuisance type does not match the operator")
    nuisance_lower_array = np.asarray(nuisance_lower, dtype=float)
    nuisance_upper_array = np.asarray(nuisance_upper, dtype=float)
    nuisance_shape = (len(operator.nuisance_names),)
    if nuisance_lower_array.shape != nuisance_shape or nuisance_upper_array.shape != nuisance_shape:
        raise ValueError("nuisance bounds have the wrong shape")
    if (
        np.any(~np.isfinite(nuisance_lower_array))
        or np.any(~np.isfinite(nuisance_upper_array))
        or np.any(nuisance_lower_array < 0.0)
        or np.any(nuisance_upper_array <= nuisance_lower_array)
        or np.any(initial_nuisance_vector < nuisance_lower_array)
        or np.any(initial_nuisance_vector > nuisance_upper_array)
    ):
        raise ValueError("nuisance bounds or initial values are invalid")

    frame_count = initial_density.shape[0]
    density_parameter_count = frame_count * model.parameter_count
    lower = np.concatenate(
        [np.tile(density_lower, frame_count), nuisance_lower_array]
    )
    upper = np.concatenate(
        [np.tile(density_upper, frame_count), nuisance_upper_array]
    )
    current = np.concatenate([initial_density.ravel(), initial_nuisance_vector])
    penalty = _regularisation_matrix(model, regularisation)
    if penalty.size:
        block_penalty = np.zeros(
            (frame_count * penalty.shape[0], current.size),
            dtype=float,
        )
        for frame_index in range(frame_count):
            rows = slice(
                frame_index * penalty.shape[0],
                (frame_index + 1) * penalty.shape[0],
            )
            columns = slice(
                frame_index * model.parameter_count,
                (frame_index + 1) * model.parameter_count,
            )
            block_penalty[rows, columns] = penalty
    else:
        block_penalty = np.zeros((0, current.size), dtype=float)

    def predict(vector: FloatArray) -> LinkedRawSequencePrediction:
        density_vectors = vector[:density_parameter_count].reshape(
            frame_count,
            model.parameter_count,
        )
        nuisance = nuisance_from_vector(operator, vector[density_parameter_count:])
        return operator.expected_linked_sequence_and_jacobian_model(
            model,
            list(density_vectors),
            nuisance,
        )

    first_prediction = predict(current)
    observed_vector = _flatten_observation(
        operator,
        observation,
        first_prediction.role_names,
    )
    final_result = None
    completed_irls = 0
    for outer in range(fit_options.irls_iterations):
        initial_prediction = predict(current)
        standard_deviation = np.sqrt(
            _flatten_prediction_variance(operator, initial_prediction)
        )
        cached_vector: FloatArray | None = None
        cached_prediction: LinkedRawSequencePrediction | None = None

        def evaluate(vector: FloatArray) -> LinkedRawSequencePrediction:
            nonlocal cached_vector, cached_prediction
            if cached_vector is None or not np.array_equal(vector, cached_vector):
                cached_prediction = predict(vector)
                cached_vector = np.array(vector, copy=True)
            if cached_prediction is None:
                raise RuntimeError("linked scalar prediction cache was not populated")
            return cached_prediction

        def residual(vector: FloatArray) -> FloatArray:
            data = (observed_vector - evaluate(vector).prediction_vector) / standard_deviation
            if block_penalty.size:
                return np.concatenate([data, block_penalty @ vector])
            return data

        def jacobian(vector: FloatArray) -> FloatArray:
            data = -evaluate(vector).jacobian / standard_deviation[:, None]
            if block_penalty.size:
                return np.vstack([data, block_penalty])
            return data

        final_result = least_squares(
            residual,
            current,
            jac=jacobian,
            bounds=(lower, upper),
            method=fit_options.method,
            loss=fit_options.loss,
            x_scale=fit_options.x_scale,
            max_nfev=fit_options.max_nfev,
            xtol=fit_options.xtol,
            ftol=fit_options.ftol,
            gtol=fit_options.gtol,
            tr_solver=fit_options.trust_region_solver,
            tr_options=(
                {
                    "atol": fit_options.lsmr_atol,
                    "btol": fit_options.lsmr_btol,
                    "conlim": fit_options.lsmr_conlim,
                    "maxiter": fit_options.lsmr_maxiter,
                    "regularize": fit_options.lsmr_regularize,
                }
                if fit_options.trust_region_solver == "lsmr"
                else None
            ),
        )
        current = np.asarray(final_result.x, dtype=float)
        completed_irls = outer + 1
    if final_result is None:
        raise RuntimeError("linked scalar fit did not execute")

    prediction = predict(current)
    standard_deviation = np.sqrt(_flatten_prediction_variance(operator, prediction))
    whitened = (observed_vector - prediction.prediction_vector) / standard_deviation
    weighted_jacobian = prediction.jacobian / standard_deviation[:, None]
    singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
    threshold = (
        np.finfo(float).eps * max(weighted_jacobian.shape) * singular_values[0]
        if singular_values.size
        else float("inf")
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size
        and rank == weighted_jacobian.shape[1]
        and singular_values[-1] > threshold
        else float("inf")
    )
    degrees_of_freedom = max(observed_vector.size - rank, 1)
    weighted_chi_square = float(whitened @ whitened)
    density_coefficients = current[:density_parameter_count].reshape(
        frame_count,
        model.parameter_count,
    )
    nuisance_values = current[density_parameter_count:]
    density_tolerance = 1e-6 * np.maximum(density_upper - density_lower, 1.0)
    nuisance_tolerance = 1e-6 * np.maximum(nuisance_upper_array, 1.0)
    regularisation_residual = block_penalty @ current
    diagnostics = LinkedScalarFitDiagnostics(
        success=bool(final_result.success and np.all(np.isfinite(current))),
        message=str(final_result.message),
        weighted_chi_square=weighted_chi_square,
        reduced_chi_square=weighted_chi_square / degrees_of_freedom,
        degrees_of_freedom=degrees_of_freedom,
        data_jacobian_rank=rank,
        data_jacobian_condition=condition,
        nfev=int(final_result.nfev),
        irls_iterations=completed_irls,
        regularisation_objective=0.5 * float(
            regularisation_residual @ regularisation_residual
        ),
        active_lower_density_coefficients=int(
            np.count_nonzero(
                density_coefficients - density_lower[None, :] <= density_tolerance
            )
        ),
        active_upper_density_coefficients=int(
            np.count_nonzero(
                density_upper[None, :] - density_coefficients <= density_tolerance
            )
        ),
        active_nuisance_bounds=int(
            np.count_nonzero(
                (nuisance_values - nuisance_lower_array <= nuisance_tolerance)
                | (nuisance_upper_array - nuisance_values <= nuisance_tolerance)
            )
        ),
        whitened_residual_vector=whitened,
    )
    return LinkedScalarFitResult(
        density_coefficients=density_coefficients,
        column_density_m2=tuple(
            model.column_density(vector) for vector in density_coefficients
        ),
        nuisance_names=tuple(operator.nuisance_names),
        nuisance_values=nuisance_values,
        prediction=prediction,
        diagnostics=diagnostics,
    )


__all__ = [
    "LinkedRawObservation",
    "LinkedScalarFitDiagnostics",
    "LinkedScalarFitOptions",
    "LinkedScalarFitResult",
    "draw_linked_raw_observation",
    "estimate_linked_nuisance_from_references",
    "fit_linked_scalar_sequence",
    "nuisance_from_vector",
    "nuisance_vector",
]
