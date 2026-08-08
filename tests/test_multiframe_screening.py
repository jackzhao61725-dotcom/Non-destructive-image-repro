from __future__ import annotations

import numpy as np
import pytest

from non_destructive_image import (
    ThermodynamicExposure,
    ThermodynamicState,
    evaluate_multiframe_screen,
)


def _state(frame: int, *, accepted: bool = True) -> ThermodynamicState:
    return ThermodynamicState(
        frame_index=frame,
        temperature_nk=100.0 + frame,
        condensate_atoms=1000.0 - 100.0 * frame,
        thermal_atoms=100.0 + 100.0 * frame,
        condensate_fraction=1.0 - 0.1 * frame,
        cumulative_scattered_photons_per_atom=0.01 * frame,
        cumulative_recoil_energy_j_per_trapped_atom=1e-32 * frame,
        number_conservation_relative_residual=0.0,
        energy_equation_relative_residual=0.0,
        closure_residual_atoms=0.0,
        accepted_frame=accepted,
        failure_reason=None if accepted else "condensate_depletion_threshold",
    )


def _exposures(
    count: int,
    *,
    crossing: int | None,
) -> tuple[ThermodynamicExposure, ...]:
    records = []
    for index in range(1, count + 1):
        accepted = crossing is None or index < crossing
        records.append(
            ThermodynamicExposure(
                exposure_index=index,
                pre_state=_state(index - 1),
                post_state=_state(index, accepted=accepted),
                accepted_by_thermodynamics=accepted,
                failure_reason=(
                    None if accepted else "condensate_depletion_threshold"
                ),
            )
        )
    return tuple(records)


def test_screen_counts_visual_prefix_and_depletion_ceiling_independently() -> None:
    result = evaluate_multiframe_screen(
        _exposures(5, crossing=5),
        [8.0, 6.0, 5.9, 6.2, 5.0],
        visual_threshold_snr_5x5=6.0,
    )

    assert result.n_screen == 2
    assert result.n_dep == 4
    assert result.n_dep_status == "observed_crossing"
    assert result.accepted_thermodynamic_exposures == 4
    assert result.first_visual_failure_exposure == 3
    assert result.first_depletion_failure_exposure == 5
    assert result.first_screen_failure_exposure == 3
    assert result.first_screen_failure_reason == "visual_screen"
    assert [record.included_in_contiguous_screen for record in result.exposures] == [
        True,
        True,
        False,
        False,
        False,
    ]


def test_screen_labels_simultaneous_first_failure_and_includes_threshold() -> None:
    result = evaluate_multiframe_screen(
        _exposures(3, crossing=3),
        np.array([7.0, 6.0, 5.5]),
        visual_threshold_snr_5x5=6.0,
    )

    assert result.n_screen == 2
    assert result.n_dep == 2
    assert result.depletion_failure_observed is True
    assert result.first_screen_failure_exposure == 3
    assert result.first_screen_failure_reason == "visual_and_condensate_depletion"
    assert result.exposures[1].visual_screen_passed is True


def test_screen_censors_unobserved_depletion_without_inventing_a_ceiling() -> None:
    result = evaluate_multiframe_screen(
        _exposures(3, crossing=None),
        [1.0, 2.0, 3.0],
        visual_threshold_snr_5x5=6.0,
    )

    assert result.n_screen == 0
    assert result.n_dep is None
    assert result.n_dep_status == "censored_without_crossing"
    assert result.accepted_thermodynamic_exposures == 3
    assert result.depletion_failure_observed is False
    assert result.first_depletion_failure_exposure is None
    assert result.first_screen_failure_reason == "visual_screen"


@pytest.mark.parametrize(
    ("snr", "threshold", "match"),
    [
        ([1.0], 0.0, "positive"),
        ([[1.0]], 6.0, "one-dimensional"),
        ([np.nan], 6.0, "finite"),
        ([-1.0], 6.0, "non-negative"),
    ],
)
def test_screen_rejects_invalid_numeric_contract(
    snr: list[float],
    threshold: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_multiframe_screen(
            _exposures(1, crossing=None),
            snr,
            visual_threshold_snr_5x5=threshold,
        )


def test_screen_rejects_nonconsecutive_or_resumed_thermodynamic_records() -> None:
    first, second, third = _exposures(3, crossing=None)
    with pytest.raises(ValueError, match="consecutive"):
        evaluate_multiframe_screen(
            (
                first,
                ThermodynamicExposure(
                    3,
                    second.pre_state,
                    second.post_state,
                    True,
                    None,
                ),
            ),
            [7.0, 7.0],
            visual_threshold_snr_5x5=6.0,
        )

    failed_second = ThermodynamicExposure(
        exposure_index=2,
        pre_state=second.pre_state,
        post_state=_state(2, accepted=False),
        accepted_by_thermodynamics=False,
        failure_reason="condensate_depletion_threshold",
    )
    with pytest.raises(ValueError, match="cannot resume"):
        evaluate_multiframe_screen(
            (first, failed_second, third),
            [7.0, 7.0, 7.0],
            visual_threshold_snr_5x5=6.0,
        )
