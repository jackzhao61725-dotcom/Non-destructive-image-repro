"""Joint visual and thermodynamic screening for repeated probe exposures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .multiframe_thermodynamics import ThermodynamicExposure


ScreenFailureReason = Literal[
    "visual_screen",
    "condensate_depletion_threshold",
    "visual_and_condensate_depletion",
]
NDepStatus = Literal["observed_crossing", "censored_without_crossing"]


@dataclass(frozen=True)
class ScreenedExposure:
    """One exposure evaluated against both approved screening conditions."""

    exposure_index: int
    expected_snr_5x5: float
    visual_screen_passed: bool
    thermodynamic_screen_passed: bool
    meets_joint_conditions: bool
    included_in_contiguous_screen: bool
    failure_reason: ScreenFailureReason | None


@dataclass(frozen=True)
class MultiframeScreenResult:
    """A consecutive usable prefix and its independent depletion ceiling."""

    visual_threshold_snr_5x5: float
    n_screen: int
    n_dep: int | None
    n_dep_status: NDepStatus
    accepted_thermodynamic_exposures: int
    depletion_failure_observed: bool
    first_visual_failure_exposure: int | None
    first_depletion_failure_exposure: int | None
    first_screen_failure_exposure: int | None
    first_screen_failure_reason: ScreenFailureReason | None
    exposures: tuple[ScreenedExposure, ...]


def evaluate_multiframe_screen(
    exposures: Sequence[ThermodynamicExposure],
    expected_snr_5x5: ArrayLike,
    *,
    visual_threshold_snr_5x5: float,
) -> MultiframeScreenResult:
    """Evaluate the consecutive-frame screen without changing thermodynamics.

    Exposure ``q`` uses its pre-exposure optical state.  It is included only
    when the expected method-specific acquisition SNR passes the visual screen
    and the post-exposure state remains below the depletion limit.  The first
    failed exposure terminates ``N_screen``.  ``N_dep`` is derived independently
    from the thermodynamic flags only after a crossing is observed; otherwise
    the finite simulated prefix is reported explicitly as censored.
    """

    threshold = float(visual_threshold_snr_5x5)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("visual_threshold_snr_5x5 must be positive and finite")
    snr = np.asarray(expected_snr_5x5, dtype=float)
    if snr.ndim != 1:
        raise ValueError("expected_snr_5x5 must be one-dimensional")
    if len(exposures) != snr.size:
        raise ValueError("one expected SNR value is required for every exposure")
    if not np.isfinite(snr).all() or np.any(snr < 0.0):
        raise ValueError("expected_snr_5x5 must be finite and non-negative")

    records: list[ScreenedExposure] = []
    contiguous_open = True
    n_screen = 0
    accepted_thermodynamic_exposures = 0
    first_visual: int | None = None
    first_depletion: int | None = None
    first_screen: int | None = None
    first_reason: ScreenFailureReason | None = None
    thermodynamic_failure_seen = False
    previous_index = 0

    for exposure, exposure_snr in zip(exposures, snr, strict=True):
        if not isinstance(exposure, ThermodynamicExposure):
            raise TypeError("exposures must contain ThermodynamicExposure records")
        if exposure.exposure_index != previous_index + 1:
            raise ValueError("thermodynamic exposure indices must be consecutive from one")
        previous_index = exposure.exposure_index

        thermodynamic_ok = bool(exposure.accepted_by_thermodynamics)
        if thermodynamic_failure_seen and thermodynamic_ok:
            raise ValueError("thermodynamic acceptance cannot resume after a failure")
        if thermodynamic_ok:
            accepted_thermodynamic_exposures += 1
        else:
            thermodynamic_failure_seen = True
            if first_depletion is None:
                first_depletion = exposure.exposure_index

        visual_ok = bool(exposure_snr >= threshold)
        if not visual_ok and first_visual is None:
            first_visual = exposure.exposure_index
        joint_ok = visual_ok and thermodynamic_ok
        included = contiguous_open and joint_ok
        if included:
            n_screen += 1
        elif contiguous_open:
            contiguous_open = False

        if visual_ok and thermodynamic_ok:
            reason: ScreenFailureReason | None = None
        elif not visual_ok and not thermodynamic_ok:
            reason = "visual_and_condensate_depletion"
        elif not visual_ok:
            reason = "visual_screen"
        else:
            reason = "condensate_depletion_threshold"
        if first_screen is None and not joint_ok:
            first_screen = exposure.exposure_index
            first_reason = reason

        records.append(
            ScreenedExposure(
                exposure_index=exposure.exposure_index,
                expected_snr_5x5=float(exposure_snr),
                visual_screen_passed=visual_ok,
                thermodynamic_screen_passed=thermodynamic_ok,
                meets_joint_conditions=joint_ok,
                included_in_contiguous_screen=included,
                failure_reason=reason,
            )
        )

    n_dep = (
        accepted_thermodynamic_exposures
        if first_depletion is not None
        else None
    )
    return MultiframeScreenResult(
        visual_threshold_snr_5x5=threshold,
        n_screen=n_screen,
        n_dep=n_dep,
        n_dep_status=(
            "observed_crossing"
            if first_depletion is not None
            else "censored_without_crossing"
        ),
        accepted_thermodynamic_exposures=accepted_thermodynamic_exposures,
        depletion_failure_observed=first_depletion is not None,
        first_visual_failure_exposure=first_visual,
        first_depletion_failure_exposure=first_depletion,
        first_screen_failure_exposure=first_screen,
        first_screen_failure_reason=first_reason,
        exposures=tuple(records),
    )
