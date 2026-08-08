from __future__ import annotations

import numpy as np
import pytest

from non_destructive_image.reconstruction import (
    DensityObservableSummary,
    ObservableIntegrationSupport,
    ObservableSupportFlags,
    RelativeSignalSummary,
    extract_density_observables,
    relative_signal_and_depletion,
)


def _square_support(
    *,
    half_width: int = 6,
    support_mask: np.ndarray | None = None,
) -> ObservableIntegrationSupport:
    axis_m = np.arange(-half_width, half_width + 1, dtype=float) * 1e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    return ObservableIntegrationSupport(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        support_mask=support_mask,
    )


def _asymmetric_cloud(shape: tuple[int, int]) -> np.ndarray:
    density = np.zeros(shape, dtype=float)
    centre_z = shape[0] // 2
    centre_y = shape[1] // 2
    density[centre_z, centre_y - 2 : centre_y + 3] = [1.0, 2.0, 4.0, 2.0, 1.0]
    density[centre_z - 1, centre_y - 1 : centre_y + 2] += [0.3, 0.8, 0.2]
    return density


def _assert_optional_float_equal(left: float | None, right: float | None) -> None:
    assert left is not None
    assert right is not None
    assert left == pytest.approx(right, rel=1e-13, abs=1e-18)


def test_translation_changes_only_the_centroid() -> None:
    support = _square_support()
    original_map = _asymmetric_cloud(support.shape)
    translated_map = np.roll(original_map, shift=(2, -1), axis=(0, 1))

    original = extract_density_observables(original_map, support)
    translated = extract_density_observables(translated_map, support)

    assert original.support_flags.positive_integrated_response
    assert translated.integrated_response == pytest.approx(original.integrated_response)
    assert original.centroid_m is not None
    assert translated.centroid_m is not None
    np.testing.assert_allclose(
        translated.centroid_m - original.centroid_m,
        [-1e-6, 2e-6],
        rtol=0.0,
        atol=1e-20,
    )
    np.testing.assert_allclose(translated.covariance_m2, original.covariance_m2, atol=1e-27)
    _assert_optional_float_equal(translated.major_rms_width_m, original.major_rms_width_m)
    _assert_optional_float_equal(translated.minor_rms_width_m, original.minor_rms_width_m)
    _assert_optional_float_equal(translated.aspect_ratio, original.aspect_ratio)
    _assert_optional_float_equal(
        translated.principal_axis_angle_rad,
        original.principal_axis_angle_rad,
    )


def test_positive_amplitude_scaling_leaves_shape_observables_invariant() -> None:
    support = _square_support()
    density = _asymmetric_cloud(support.shape)
    original = extract_density_observables(density, support)
    scaled = extract_density_observables(7.25 * density, support)

    assert scaled.integrated_response == pytest.approx(7.25 * original.integrated_response)
    np.testing.assert_allclose(scaled.centroid_m, original.centroid_m, atol=1e-20)
    np.testing.assert_allclose(scaled.covariance_m2, original.covariance_m2, atol=1e-27)
    _assert_optional_float_equal(scaled.major_rms_width_m, original.major_rms_width_m)
    _assert_optional_float_equal(scaled.minor_rms_width_m, original.minor_rms_width_m)
    _assert_optional_float_equal(scaled.aspect_ratio, original.aspect_ratio)
    _assert_optional_float_equal(
        scaled.principal_axis_angle_rad,
        original.principal_axis_angle_rad,
    )


def test_rotated_cloud_has_expected_covariance_eigenvectors_and_widths() -> None:
    support = _square_support(half_width=4)
    density = np.zeros(support.shape, dtype=float)
    centre = support.shape[0] // 2
    # Equal point weights at +/- 2 sqrt(2) um along the 45-degree major
    # axis and +/- sqrt(2) um along its orthogonal minor axis.
    for delta_z, delta_y in ((2, 2), (-2, -2), (1, -1), (-1, 1)):
        density[centre + delta_z, centre + delta_y] = 1.0

    summary = extract_density_observables(density, support)

    expected_covariance_m2 = np.asarray([[2.5, 1.5], [1.5, 2.5]]) * 1e-12
    np.testing.assert_allclose(summary.centroid_m, [0.0, 0.0], atol=1e-21)
    np.testing.assert_allclose(summary.covariance_m2, expected_covariance_m2, atol=1e-27)
    assert summary.major_rms_width_m == pytest.approx(2e-6)
    assert summary.minor_rms_width_m == pytest.approx(1e-6)
    assert summary.aspect_ratio == pytest.approx(2.0)
    assert summary.principal_axis_angle_rad == pytest.approx(np.pi / 4)
    assert summary.support_flags.principal_axis_angle_supported
    assert summary.principal_axis_unit_vector_yz is not None
    expected_axis = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    assert abs(float(summary.principal_axis_unit_vector_yz @ expected_axis)) == pytest.approx(1.0)


def test_swapping_axes_swaps_tensor_components_but_not_ordered_widths() -> None:
    support = _square_support()
    density = _asymmetric_cloud(support.shape)
    original = extract_density_observables(density, support)
    swapped = extract_density_observables(density.T, support)

    assert original.centroid_m is not None
    assert original.covariance_m2 is not None
    np.testing.assert_allclose(swapped.centroid_m, original.centroid_m[::-1], atol=1e-20)
    np.testing.assert_allclose(
        swapped.covariance_m2,
        original.covariance_m2[np.ix_([1, 0], [1, 0])],
        atol=1e-27,
    )
    _assert_optional_float_equal(swapped.major_rms_width_m, original.major_rms_width_m)
    _assert_optional_float_equal(swapped.minor_rms_width_m, original.minor_rms_width_m)
    _assert_optional_float_equal(swapped.aspect_ratio, original.aspect_ratio)


def test_zero_and_support_blank_maps_are_explicitly_unsupported() -> None:
    support = _square_support(half_width=3)
    zero = extract_density_observables(np.zeros(support.shape), support)

    assert zero.integrated_response == 0.0
    assert zero.centroid_m is None
    assert zero.covariance_m2 is None
    assert zero.major_rms_width_m is None
    assert zero.minor_rms_width_m is None
    assert zero.aspect_ratio is None
    assert zero.principal_axis_angle_rad is None
    assert not zero.support_flags.positive_integrated_response
    assert not zero.support_flags.centroid_supported
    assert not zero.support_flags.covariance_supported
    assert not zero.support_flags.widths_supported
    assert not zero.support_flags.aspect_ratio_supported
    assert not zero.support_flags.principal_axis_angle_supported

    mask = np.zeros(support.shape, dtype=bool)
    mask[2:5, 2:5] = True
    central_support = ObservableIntegrationSupport(
        y_grid_m=support.y_grid_m,
        z_grid_m=support.z_grid_m,
        support_mask=mask,
    )
    blank_on_support = np.full(support.shape, np.nan)
    blank_on_support[mask] = 0.0
    blank = extract_density_observables(blank_on_support, central_support)
    assert blank.integrated_response == 0.0
    assert not blank.support_flags.positive_integrated_response
    assert blank.support_flags.reasons == (
        "blank_or_below_minimum_integrated_response",
    )


def test_near_circular_cloud_withholds_principal_axis_angle() -> None:
    support = _square_support(half_width=3)
    density = np.zeros(support.shape, dtype=float)
    centre = support.shape[0] // 2
    density[centre, centre - 2] = 1.0
    density[centre, centre + 2] = 1.0
    density[centre - 2, centre] = 1.0
    density[centre + 2, centre] = 1.0

    summary = extract_density_observables(
        density,
        support,
        angle_anisotropy_threshold=0.05,
    )

    assert summary.aspect_ratio == pytest.approx(1.0)
    assert summary.fractional_anisotropy == pytest.approx(0.0)
    assert summary.principal_axis_angle_rad is None
    assert not summary.support_flags.principal_axis_angle_supported
    assert "near_circular_below_angle_anisotropy_threshold" in summary.support_flags.reasons


def test_physical_cell_area_and_declared_support_weight_every_moment() -> None:
    y_axis_m = np.asarray([-3.0, -1.0, 2.0]) * 1e-6
    z_axis_m = np.asarray([-2.0, 1.0, 5.0]) * 1e-6
    y_grid_m, z_grid_m = np.meshgrid(y_axis_m, z_axis_m)
    mask = np.asarray(
        [
            [True, True, False],
            [True, False, False],
            [True, True, True],
        ]
    )
    support = ObservableIntegrationSupport(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        support_mask=mask,
    )
    density = np.ones(support.shape)
    density[~mask] = 1e12  # Outside the declared support must not enter any moment.

    summary = extract_density_observables(density, support)

    expected_weights = support.cell_area_m2[mask]
    expected_total = float(np.sum(expected_weights))
    expected_centroid = np.asarray(
        [
            np.dot(expected_weights, y_grid_m[mask]) / expected_total,
            np.dot(expected_weights, z_grid_m[mask]) / expected_total,
        ]
    )
    displacement = np.column_stack(
        [y_grid_m[mask] - expected_centroid[0], z_grid_m[mask] - expected_centroid[1]]
    )
    expected_covariance = (displacement.T * expected_weights) @ displacement / expected_total

    assert support.physical_area_m2 == pytest.approx(expected_total)
    assert summary.integrated_response == pytest.approx(expected_total)
    np.testing.assert_allclose(summary.centroid_m, expected_centroid, atol=1e-21)
    np.testing.assert_allclose(summary.covariance_m2, expected_covariance, atol=1e-27)
    with pytest.raises(ValueError, match="support mask must match"):
        ObservableIntegrationSupport(
            y_grid_m=y_grid_m,
            z_grid_m=z_grid_m,
            support_mask=np.ones((2, 2), dtype=bool),
        )


def test_constant_map_integral_is_consistent_across_cell_centre_resolutions() -> None:
    field_width_m = 18e-6
    integrated_responses = []
    for cell_count in (6, 15, 36):
        spacing_m = field_width_m / cell_count
        axis_m = (
            np.arange(cell_count, dtype=float) + 0.5
        ) * spacing_m - 0.5 * field_width_m
        y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
        support = ObservableIntegrationSupport(
            y_grid_m=y_grid_m,
            z_grid_m=z_grid_m,
        )
        summary = extract_density_observables(np.full(support.shape, 3.25), support)
        integrated_responses.append(summary.integrated_response)
        assert support.physical_area_m2 == pytest.approx(field_width_m**2)
        np.testing.assert_allclose(
            summary.centroid_m,
            [0.0, 0.0],
            atol=2.0 * np.finfo(float).eps * field_width_m,
        )

    np.testing.assert_allclose(
        integrated_responses,
        np.full(3, 3.25 * field_width_m**2),
        rtol=1e-14,
        atol=0.0,
    )


def test_relative_signal_and_depletion_require_identical_grid_and_support() -> None:
    support = _square_support(half_width=3)
    reference_map = _asymmetric_cloud(support.shape)
    reference = extract_density_observables(reference_map, support)
    current = extract_density_observables(0.6 * reference_map, support)

    relative = relative_signal_and_depletion(reference, current)
    assert isinstance(relative, RelativeSignalSummary)
    assert relative.integrated_response_ratio == pytest.approx(0.6)
    assert relative.reconstructed_depletion == pytest.approx(0.4)
    assert relative.a_q_over_a_0 == relative.integrated_response_ratio
    assert relative.l_rec == relative.reconstructed_depletion
    assert any("not verified" in assumption for assumption in relative.assumptions)

    changed_mask = np.asarray(support.support_mask).copy()
    changed_mask[0, 0] = False
    mask_mismatch = ObservableIntegrationSupport(
        y_grid_m=support.y_grid_m,
        z_grid_m=support.z_grid_m,
        support_mask=changed_mask,
        cell_area_m2=support.cell_area_m2,
    )
    current_mask_mismatch = extract_density_observables(
        0.6 * reference_map,
        mask_mismatch,
    )
    with pytest.raises(ValueError, match="identical coordinate grids"):
        relative_signal_and_depletion(reference, current_mask_mismatch)

    area_mismatch = ObservableIntegrationSupport(
        y_grid_m=support.y_grid_m,
        z_grid_m=support.z_grid_m,
        support_mask=support.support_mask,
        cell_area_m2=1.01 * support.cell_area_m2,
    )
    current_area_mismatch = extract_density_observables(
        0.6 * reference_map,
        area_mismatch,
    )
    with pytest.raises(ValueError, match="cell areas"):
        relative_signal_and_depletion(reference, current_area_mismatch)

    shifted_grid = ObservableIntegrationSupport(
        y_grid_m=support.y_grid_m + 0.1e-6,
        z_grid_m=support.z_grid_m,
        support_mask=support.support_mask,
        cell_area_m2=support.cell_area_m2,
    )
    current_grid_mismatch = extract_density_observables(
        0.6 * reference_map,
        shifted_grid,
    )
    with pytest.raises(ValueError, match="coordinate grids"):
        relative_signal_and_depletion(reference, current_grid_mismatch)


def test_relative_signal_allows_a_blank_current_map_but_not_blank_reference() -> None:
    support = _square_support(half_width=3)
    reference = extract_density_observables(_asymmetric_cloud(support.shape), support)
    blank = extract_density_observables(np.zeros(support.shape), support)

    relative = relative_signal_and_depletion(reference, blank)
    assert relative.integrated_response_ratio == 0.0
    assert relative.reconstructed_depletion == 1.0
    with pytest.raises(ValueError, match="reference integrated response"):
        relative_signal_and_depletion(blank, reference)


def test_observable_summary_and_support_types_are_public_api() -> None:
    support = _square_support(half_width=2)
    summary = extract_density_observables(_asymmetric_cloud(support.shape), support)
    independently_extracted = extract_density_observables(
        _asymmetric_cloud(support.shape),
        support,
    )
    assert isinstance(summary, DensityObservableSummary)
    assert isinstance(summary.support_flags, ObservableSupportFlags)
    assert summary != independently_extracted


def test_observable_contract_rejects_invalid_maps_and_is_immutably_snapshotted() -> None:
    support = _square_support(half_width=2)
    density = _asymmetric_cloud(support.shape)
    summary = extract_density_observables(density, support)

    with pytest.raises(ValueError, match="must match"):
        extract_density_observables(np.ones((2, 2)), support)
    negative = density.copy()
    negative[0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        extract_density_observables(negative, support)
    nonfinite = density.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_density_observables(nonfinite, support)
    with pytest.raises(ValueError, match="real-valued"):
        extract_density_observables(density.astype(complex) + 1j, support)
    with pytest.raises(ValueError, match="anisotropy threshold"):
        extract_density_observables(density, support, angle_anisotropy_threshold=1.1)

    below_threshold = extract_density_observables(
        density,
        support,
        minimum_integrated_response=2.0 * summary.integrated_response,
    )
    assert below_threshold.support_flags.positive_integrated_response
    assert not below_threshold.support_flags.moments_numerically_supported
    assert below_threshold.centroid_m is None

    for array in (
        support.y_grid_m,
        support.z_grid_m,
        support.cell_area_m2,
        support.support_mask,
        summary.centroid_m,
        summary.covariance_m2,
    ):
        assert array is not None
        with pytest.raises(ValueError):
            array.setflags(write=True)
