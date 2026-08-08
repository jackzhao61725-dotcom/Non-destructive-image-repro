from __future__ import annotations

import numpy as np
import pytest

from non_destructive_image.reconstruction.observable_calibration import (
    AffineObservableCalibration,
    fit_affine_observable_calibration,
    summarise_calibrated_bootstrap,
)


def _calibration(*, supported: bool = True) -> AffineObservableCalibration:
    raw = np.asarray([1.0, 2.0, 3.0, 4.0])
    truth = 2.0 * raw + 1.0
    if not supported:
        truth = np.asarray([3.0, 5.0, 7.0, 40.0])
    return fit_affine_observable_calibration(
        "A",
        raw,
        truth,
        error_kind="relative",
        error_scale=0.2,
    )


def test_affine_observable_calibration_uses_leave_one_out_support() -> None:
    calibration = _calibration()

    assert calibration.supported is True
    assert calibration.slope == pytest.approx(2.0)
    assert calibration.intercept == pytest.approx(1.0)
    assert calibration.apply(2.5) == pytest.approx(6.0)
    np.testing.assert_allclose(
        calibration.apply_array([1.0, 4.0]),
        [3.0, 9.0],
    )
    assert calibration.maximum_normalised_error < 1e-12

    rejected = _calibration(supported=False)
    assert rejected.supported is False
    assert "leave_one_out_maximum_error" in rejected.reasons


def test_calibrated_bootstrap_keeps_complete_partial_and_unsupported_distinct() -> None:
    calibration = _calibration()
    complete = summarise_calibrated_bootstrap(
        calibration,
        2.5,
        [2.0, 2.5, 3.0, 3.5],
        requested_draws=4,
        confidence_level=0.95,
    )
    assert complete.status == "complete"
    assert complete.estimate == pytest.approx(6.0)
    assert complete.lower is not None
    assert complete.upper is not None
    assert complete.calibration_error_bound == pytest.approx(0.0, abs=1e-12)
    assert complete.combined_lower is not None
    assert complete.combined_upper is not None

    partial = summarise_calibrated_bootstrap(
        calibration,
        2.5,
        [2.0, 2.5, 3.0],
        requested_draws=4,
        confidence_level=0.95,
    )
    assert partial.status == "partial"
    assert partial.estimate == pytest.approx(6.0)
    assert partial.lower is None
    assert partial.combined_lower is None

    unsupported = summarise_calibrated_bootstrap(
        _calibration(supported=False),
        2.5,
        [2.0, 2.5, 3.0, 3.5],
        requested_draws=4,
        confidence_level=0.95,
    )
    assert unsupported.status == "unsupported"
    assert unsupported.estimate is None


@pytest.mark.parametrize(
    ("raw", "truth", "match"),
    [
        ([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], "span"),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "length at least four"),
        ([1.0, 2.0, 3.0, np.nan], [1.0, 2.0, 3.0, 4.0], "finite"),
    ],
)
def test_affine_calibration_rejects_invalid_inputs(
    raw: list[float],
    truth: list[float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        fit_affine_observable_calibration(
            "A",
            raw,
            truth,
            error_kind="relative",
            error_scale=0.2,
        )
