"""Calibration-only reportability and uncertainty for scalar observables.

The latent density map remains a nuisance representation.  This module maps
its low-order moments onto physical observables using a calibration split and
keeps unsupported quantities explicit.  It never consumes held-out truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


OBSERVABLE_NAMES: tuple[str, ...] = (
    "A",
    "y_c_um",
    "z_c_um",
    "sigma_y_um",
    "sigma_z_um",
)
CalibrationErrorKind = Literal["relative", "absolute"]
IntervalStatus = Literal["complete", "partial", "unsupported"]


def _immutable(values: ArrayLike, *, dtype: type = float) -> NDArray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class AffineObservableCalibration:
    """One affine simulation-calibration map and its leave-one-out audit."""

    observable_name: str
    slope: float
    intercept: float
    error_kind: CalibrationErrorKind
    error_scale: float
    leave_one_out_predictions: NDArray[np.floating]
    leave_one_out_normalised_errors: NDArray[np.floating]
    median_normalised_error: float
    maximum_normalised_error: float
    supported: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.observable_name not in OBSERVABLE_NAMES:
            raise ValueError("unknown scalar observable name")
        if self.error_kind not in ("relative", "absolute"):
            raise ValueError("error_kind must be 'relative' or 'absolute'")
        scalars = np.asarray(
            (
                self.slope,
                self.intercept,
                self.error_scale,
                self.median_normalised_error,
                self.maximum_normalised_error,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(scalars)) or self.error_scale <= 0.0:
            raise ValueError("calibration scalars must be finite with positive scale")
        predictions = np.asarray(self.leave_one_out_predictions, dtype=float)
        errors = np.asarray(self.leave_one_out_normalised_errors, dtype=float)
        if predictions.ndim != 1 or errors.shape != predictions.shape:
            raise ValueError("leave-one-out arrays must be equal-length vectors")
        if predictions.size < 4 or np.any(~np.isfinite(predictions)):
            raise ValueError("at least four finite leave-one-out predictions are required")
        if np.any(~np.isfinite(errors)) or np.any(errors < 0.0):
            raise ValueError("leave-one-out errors must be finite and non-negative")
        if self.supported != (len(self.reasons) == 0):
            raise ValueError("supported flag must agree with calibration reasons")
        object.__setattr__(
            self,
            "leave_one_out_predictions",
            _immutable(predictions),
        )
        object.__setattr__(
            self,
            "leave_one_out_normalised_errors",
            _immutable(errors),
        )

    def apply(self, raw_value: float) -> float:
        """Apply the frozen affine map to one raw inverse observable."""

        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError("raw observable must be finite")
        return float(self.slope * value + self.intercept)

    def apply_array(self, raw_values: ArrayLike) -> NDArray[np.floating]:
        """Apply the frozen affine map to a finite array."""

        values = np.asarray(raw_values, dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError("raw observable array must be finite")
        return np.asarray(self.slope * values + self.intercept, dtype=float)


@dataclass(frozen=True)
class CalibratedObservableInterval:
    """A calibrated point estimate with separated uncertainty components.

    ``lower`` and ``upper`` bound the conditional bootstrap estimator
    distribution.  ``combined_lower`` and ``combined_upper`` add the frozen
    leave-one-out calibration envelope conservatively around the point
    estimate.  The combined band is not a posterior or a nominal-confidence
    interval.
    """

    observable_name: str
    status: IntervalStatus
    estimate: float | None
    requested_draws: int
    successful_draws: int
    bootstrap_mean: float | None
    bootstrap_standard_deviation: float | None
    lower: float | None
    upper: float | None
    detector_bootstrap_half_span: float | None
    calibration_error_bound: float | None
    combined_lower: float | None
    combined_upper: float | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.observable_name not in OBSERVABLE_NAMES:
            raise ValueError("unknown scalar observable name")
        if self.requested_draws <= 0:
            raise ValueError("requested_draws must be positive")
        if not 0 <= self.successful_draws <= self.requested_draws:
            raise ValueError("successful_draws must lie within requested_draws")
        if self.status == "complete":
            if self.successful_draws != self.requested_draws or self.estimate is None:
                raise ValueError("complete interval requires every requested draw")
            values = (
                self.estimate,
                self.bootstrap_mean,
                self.bootstrap_standard_deviation,
                self.lower,
                self.upper,
                self.detector_bootstrap_half_span,
                self.calibration_error_bound,
                self.combined_lower,
                self.combined_upper,
            )
            if any(value is None or not np.isfinite(value) for value in values):
                raise ValueError("complete interval statistics must be finite")
            assert self.bootstrap_standard_deviation is not None
            assert self.lower is not None and self.upper is not None
            assert self.detector_bootstrap_half_span is not None
            assert self.calibration_error_bound is not None
            assert self.combined_lower is not None and self.combined_upper is not None
            if (
                self.bootstrap_standard_deviation < 0.0
                or self.detector_bootstrap_half_span < 0.0
                or self.calibration_error_bound < 0.0
                or self.lower > self.upper
                or self.combined_lower > self.combined_upper
            ):
                raise ValueError("complete interval bounds are invalid")
        else:
            if any(
                value is not None
                for value in (
                    self.bootstrap_mean,
                    self.bootstrap_standard_deviation,
                    self.lower,
                    self.upper,
                    self.detector_bootstrap_half_span,
                    self.calibration_error_bound,
                    self.combined_lower,
                    self.combined_upper,
                )
            ):
                raise ValueError("incomplete intervals cannot report bounds")
            if self.status == "partial" and self.estimate is None:
                raise ValueError("partial interval requires a point estimate")
            if self.status == "unsupported" and self.estimate is not None:
                raise ValueError("unsupported interval cannot report an estimate")
        if self.status != "complete" and not self.reasons:
            raise ValueError("incomplete interval requires a reason")


def fit_affine_observable_calibration(
    observable_name: str,
    raw_values: ArrayLike,
    truth_values: ArrayLike,
    *,
    error_kind: CalibrationErrorKind,
    error_scale: float,
    maximum_median_normalised_error: float = 1.0,
    maximum_normalised_error: float = 1.0,
) -> AffineObservableCalibration:
    """Fit one affine map and assess it by leave-one-out prediction.

    Relative errors are normalised as ``abs(prediction / truth - 1)`` divided
    by ``error_scale``.  Absolute errors are divided directly by
    ``error_scale``.  The support decision is frozen on the calibration split.
    """

    if observable_name not in OBSERVABLE_NAMES:
        raise ValueError("unknown scalar observable name")
    raw = np.asarray(raw_values, dtype=float)
    truth = np.asarray(truth_values, dtype=float)
    if raw.ndim != 1 or truth.shape != raw.shape or raw.size < 4:
        raise ValueError("raw and truth values must be equal vectors of length at least four")
    if np.any(~np.isfinite(raw)) or np.any(~np.isfinite(truth)):
        raise ValueError("calibration values must be finite")
    scale = float(error_scale)
    median_limit = float(maximum_median_normalised_error)
    maximum_limit = float(maximum_normalised_error)
    if (
        error_kind not in ("relative", "absolute")
        or not np.isfinite(scale)
        or scale <= 0.0
        or not np.isfinite(median_limit)
        or median_limit < 0.0
        or not np.isfinite(maximum_limit)
        or maximum_limit < median_limit
    ):
        raise ValueError("calibration error contract is invalid")
    if error_kind == "relative" and np.any(truth == 0.0):
        raise ValueError("relative calibration requires non-zero truth values")

    design = np.column_stack([raw, np.ones(raw.size, dtype=float)])
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("raw calibration values do not span an affine map")
    slope, intercept = np.linalg.lstsq(design, truth, rcond=None)[0]
    predictions = np.empty(raw.size, dtype=float)
    for index in range(raw.size):
        keep = np.arange(raw.size) != index
        fold_design = design[keep]
        if np.linalg.matrix_rank(fold_design) < 2:
            raise ValueError("a leave-one-out fold does not span an affine map")
        fold_slope, fold_intercept = np.linalg.lstsq(
            fold_design,
            truth[keep],
            rcond=None,
        )[0]
        predictions[index] = fold_slope * raw[index] + fold_intercept
    if error_kind == "relative":
        errors = np.abs(predictions / truth - 1.0) / scale
    else:
        errors = np.abs(predictions - truth) / scale
    median_error = float(np.median(errors))
    maximum_error = float(np.max(errors))
    reasons: list[str] = []
    if median_error > median_limit:
        reasons.append("leave_one_out_median_error")
    if maximum_error > maximum_limit:
        reasons.append("leave_one_out_maximum_error")
    return AffineObservableCalibration(
        observable_name=observable_name,
        slope=float(slope),
        intercept=float(intercept),
        error_kind=error_kind,
        error_scale=scale,
        leave_one_out_predictions=predictions,
        leave_one_out_normalised_errors=errors,
        median_normalised_error=median_error,
        maximum_normalised_error=maximum_error,
        supported=not reasons,
        reasons=tuple(reasons),
    )


def summarise_calibrated_bootstrap(
    calibration: AffineObservableCalibration,
    raw_point_estimate: float,
    successful_raw_samples: ArrayLike,
    *,
    requested_draws: int,
    confidence_level: float,
) -> CalibratedObservableInterval:
    """Transform successful raw bootstrap samples without hiding failures."""

    requested = int(requested_draws)
    level = float(confidence_level)
    samples = np.asarray(successful_raw_samples, dtype=float)
    if samples.ndim != 1 or np.any(~np.isfinite(samples)):
        raise ValueError("successful bootstrap samples must be a finite vector")
    if requested <= 0 or samples.size > requested:
        raise ValueError("bootstrap sample count is inconsistent")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not calibration.supported:
        return CalibratedObservableInterval(
            observable_name=calibration.observable_name,
            status="unsupported",
            estimate=None,
            requested_draws=requested,
            successful_draws=int(samples.size),
            bootstrap_mean=None,
            bootstrap_standard_deviation=None,
            lower=None,
            upper=None,
            detector_bootstrap_half_span=None,
            calibration_error_bound=None,
            combined_lower=None,
            combined_upper=None,
            reasons=("calibration_not_reportable",) + calibration.reasons,
        )
    estimate = calibration.apply(raw_point_estimate)
    if samples.size != requested:
        return CalibratedObservableInterval(
            observable_name=calibration.observable_name,
            status="partial",
            estimate=estimate,
            requested_draws=requested,
            successful_draws=int(samples.size),
            bootstrap_mean=None,
            bootstrap_standard_deviation=None,
            lower=None,
            upper=None,
            detector_bootstrap_half_span=None,
            calibration_error_bound=None,
            combined_lower=None,
            combined_upper=None,
            reasons=("incomplete_bootstrap_refits",),
        )
    transformed = calibration.apply_array(samples)
    alpha = 0.5 * (1.0 - level)
    lower = float(np.quantile(transformed, alpha))
    upper = float(np.quantile(transformed, 1.0 - alpha))
    detector_half_span = max(abs(lower - estimate), abs(upper - estimate))
    if calibration.error_kind == "relative":
        calibration_bound = (
            abs(estimate)
            * calibration.error_scale
            * calibration.maximum_normalised_error
        )
    else:
        calibration_bound = (
            calibration.error_scale * calibration.maximum_normalised_error
        )
    combined_half_span = detector_half_span + calibration_bound
    return CalibratedObservableInterval(
        observable_name=calibration.observable_name,
        status="complete",
        estimate=estimate,
        requested_draws=requested,
        successful_draws=requested,
        bootstrap_mean=float(np.mean(transformed)),
        bootstrap_standard_deviation=float(
            np.std(transformed, ddof=1 if requested > 1 else 0)
        ),
        lower=lower,
        upper=upper,
        detector_bootstrap_half_span=float(detector_half_span),
        calibration_error_bound=float(calibration_bound),
        combined_lower=float(estimate - combined_half_span),
        combined_upper=float(estimate + combined_half_span),
        reasons=(),
    )


__all__ = [
    "OBSERVABLE_NAMES",
    "AffineObservableCalibration",
    "CalibratedObservableInterval",
    "fit_affine_observable_calibration",
    "summarise_calibrated_bootstrap",
]
