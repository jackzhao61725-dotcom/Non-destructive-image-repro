"""Tests for the physically normalised curvature penalty."""

from __future__ import annotations

import numpy as np
import pytest

from non_destructive_image.reconstruction.regularisation import (
    CurvatureAxisWeights,
    build_curvature_regularisation,
)


DENSITY_SCALE_M2 = 2.5e15


def _grid_values(
    knot_y_um: np.ndarray,
    knot_z_um: np.ndarray,
    function,
) -> np.ndarray:
    y_grid, z_grid = np.meshgrid(knot_y_um, knot_z_um)
    return DENSITY_SCALE_M2 * function(y_grid, z_grid)


@pytest.fixture
def nonuniform_knots() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([-2.0, -1.2, -0.25, 0.8, 2.0]),
        np.asarray([-1.5, -0.55, 0.2, 1.5]),
    )


def test_zero_weight_produces_no_residual_rows(
    nonuniform_knots: tuple[np.ndarray, np.ndarray],
) -> None:
    y, z = nonuniform_knots
    regularisation = build_curvature_regularisation(
        y,
        z,
        density_scale_m2=DENSITY_SCALE_M2,
        weight_um2=0.0,
    )

    assert regularisation.physical_density_matrix.shape == (0, y.size * z.size)
    assert regularisation.residual_count == 0
    assert regularisation.residual_from_density(np.ones((z.size, y.size))).size == 0


def test_fixed_zero_ghost_policy_has_explicit_boundary_cost(
    nonuniform_knots: tuple[np.ndarray, np.ndarray],
) -> None:
    y, z = nonuniform_knots
    regularisation = build_curvature_regularisation(
        y,
        z,
        density_scale_m2=DENSITY_SCALE_M2,
        weight_um2=3.0,
    )
    constant = _grid_values(y, z, lambda yy, zz: np.ones_like(yy))
    linear = _grid_values(y, z, lambda yy, zz: 1.0 + 0.05 * yy + 0.04 * zz)

    np.testing.assert_allclose(
        regularisation.interior_residual_from_density(constant),
        0.0,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        regularisation.interior_residual_from_density(linear),
        0.0,
        atol=1e-13,
    )
    assert np.linalg.norm(regularisation.residual_from_density(constant)) > 0.0
    assert np.linalg.norm(regularisation.residual_from_density(linear)) > 0.0


@pytest.mark.parametrize(
    ("function", "axis_weights", "expected_curvature"),
    [
        (
            lambda yy, zz: yy**2,
            CurvatureAxisWeights(yy=1.0, yz=0.0, zz=0.0),
            2.0,
        ),
        (
            lambda yy, zz: zz**2,
            CurvatureAxisWeights(yy=0.0, yz=0.0, zz=1.0),
            2.0,
        ),
        (
            lambda yy, zz: yy * zz,
            CurvatureAxisWeights(yy=0.0, yz=1.0, zz=0.0),
            1.0,
        ),
    ],
)
def test_nonuniform_divided_differences_reproduce_quadratic_components(
    nonuniform_knots: tuple[np.ndarray, np.ndarray],
    function,
    axis_weights: CurvatureAxisWeights,
    expected_curvature: float,
) -> None:
    y, z = nonuniform_knots
    weight_um2 = 2.4
    regularisation = build_curvature_regularisation(
        y,
        z,
        density_scale_m2=DENSITY_SCALE_M2,
        weight_um2=weight_um2,
        axis_weights=axis_weights,
    )
    values = _grid_values(y, z, function)
    norm_squared = float(
        regularisation.interior_residual_from_density(values)
        @ regularisation.interior_residual_from_density(values)
    )

    if axis_weights.yz:
        integrated_area = (y[-1] - y[0]) * (z[-1] - z[0])
        expected = 2.0 * weight_um2 * expected_curvature**2 * integrated_area
    elif axis_weights.yy:
        y_extended = np.concatenate(
            ([y[0] - (y[1] - y[0])], y, [y[-1] + (y[-1] - y[-2])])
        )
        z_extended = np.concatenate(
            ([z[0] - (z[1] - z[0])], z, [z[-1] + (z[-1] - z[-2])])
        )
        y_weights = 0.5 * (y_extended[2:] - y_extended[:-2])
        z_weights = 0.5 * (z_extended[2:] - z_extended[:-2])
        integrated_area = np.sum(y_weights[1:-1]) * np.sum(z_weights)
        expected = weight_um2 * expected_curvature**2 * integrated_area
    else:
        y_extended = np.concatenate(
            ([y[0] - (y[1] - y[0])], y, [y[-1] + (y[-1] - y[-2])])
        )
        z_extended = np.concatenate(
            ([z[0] - (z[1] - z[0])], z, [z[-1] + (z[-1] - z[-2])])
        )
        y_weights = 0.5 * (y_extended[2:] - y_extended[:-2])
        z_weights = 0.5 * (z_extended[2:] - z_extended[:-2])
        integrated_area = np.sum(y_weights) * np.sum(z_weights[1:-1])
        expected = weight_um2 * expected_curvature**2 * integrated_area
    assert norm_squared == pytest.approx(expected, rel=2e-13, abs=2e-13)


def test_optimizer_coefficient_scale_does_not_change_physical_penalty(
    nonuniform_knots: tuple[np.ndarray, np.ndarray],
) -> None:
    y, z = nonuniform_knots
    regularisation = build_curvature_regularisation(
        y,
        z,
        density_scale_m2=DENSITY_SCALE_M2,
        weight_um2=7.0,
    )
    physical_density = _grid_values(
        y,
        z,
        lambda yy, zz: (1.0 - (yy / 2.5) ** 2) ** 2 * (1.0 - (zz / 2.0) ** 2) ** 2,
    )
    direct = regularisation.residual_from_density(physical_density)

    for coefficient_scale_m2 in (1e14, 8e14, 3e15):
        coefficients = physical_density / coefficient_scale_m2
        scaled = regularisation.residual_from_coefficients(
            coefficients,
            coefficient_scale_m2=coefficient_scale_m2,
        )
        np.testing.assert_allclose(scaled, direct, rtol=2e-14, atol=2e-14)


def test_transposing_axes_preserves_penalty_when_axis_weights_are_swapped(
    nonuniform_knots: tuple[np.ndarray, np.ndarray],
) -> None:
    y, z = nonuniform_knots
    weights = CurvatureAxisWeights(yy=0.7, yz=1.4, zz=2.3)
    regularisation = build_curvature_regularisation(
        y,
        z,
        density_scale_m2=DENSITY_SCALE_M2,
        weight_um2=4.0,
        axis_weights=weights,
    )
    transposed_regularisation = build_curvature_regularisation(
        z,
        y,
        density_scale_m2=DENSITY_SCALE_M2,
        weight_um2=4.0,
        axis_weights=CurvatureAxisWeights(yy=weights.zz, yz=weights.yz, zz=weights.yy),
    )
    values = _grid_values(
        y,
        z,
        lambda yy, zz: 0.3 + 0.04 * yy + 0.02 * zz + 0.08 * yy**2 + 0.03 * yy * zz,
    )

    original_norm = np.linalg.norm(regularisation.residual_from_density(values))
    transposed_norm = np.linalg.norm(
        transposed_regularisation.residual_from_density(values.T)
    )
    assert original_norm == pytest.approx(transposed_norm, rel=2e-14, abs=2e-14)


def test_nonuniform_refinement_gives_a_stable_physical_penalty() -> None:
    coarse_y = np.linspace(-1.0, 1.0, 10)
    coarse_z = np.linspace(-1.0, 1.0, 9)
    coarse_y[1:-1] += 0.08 * (2.0 / 9.0) * np.sin(
        1.7 * np.arange(1, coarse_y.size - 1)
    )
    coarse_z[1:-1] += 0.08 * (2.0 / 8.0) * np.sin(
        1.3 * np.arange(1, coarse_z.size - 1)
    )
    fine_y = np.sort(np.concatenate((coarse_y, 0.5 * (coarse_y[:-1] + coarse_y[1:]))))
    fine_z = np.sort(np.concatenate((coarse_z, 0.5 * (coarse_z[:-1] + coarse_z[1:]))))

    def compact_profile(yy: np.ndarray, zz: np.ndarray) -> np.ndarray:
        return (1.0 - yy**2) ** 2 * (1.0 - zz**2) ** 2

    penalties = []
    for y, z in ((coarse_y, coarse_z), (fine_y, fine_z)):
        regularisation = build_curvature_regularisation(
            y,
            z,
            density_scale_m2=DENSITY_SCALE_M2,
            weight_um2=3.0,
        )
        density = _grid_values(y, z, compact_profile)
        penalties.append(np.linalg.norm(regularisation.residual_from_density(density)) ** 2)

    assert penalties[1] == pytest.approx(penalties[0], rel=0.18)


def test_builder_validates_grid_scales_and_boundary_policy() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        build_curvature_regularisation(
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 2.0],
            density_scale_m2=DENSITY_SCALE_M2,
            weight_um2=1.0,
        )
    with pytest.raises(ValueError, match="boundary policy"):
        build_curvature_regularisation(
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            density_scale_m2=DENSITY_SCALE_M2,
            weight_um2=1.0,
            boundary_policy="free",  # type: ignore[arg-type]
        )
