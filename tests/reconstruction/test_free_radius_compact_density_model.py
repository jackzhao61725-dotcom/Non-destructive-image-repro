from __future__ import annotations

import numpy as np
import pytest

from non_destructive_image.reconstruction import (
    DifferentiableColumnDensityModel,
    FreeRadiusCompactDensityModel,
    SmoothTFParameters,
    smooth_tf_density_and_internal_jacobian,
)
from non_destructive_image.reconstruction.parameters import to_internal


def _grid() -> tuple[np.ndarray, np.ndarray]:
    y_um = np.linspace(-12.0, 12.0, 49)
    z_um = np.linspace(-6.0, 6.0, 37)
    y_grid_um, z_grid_um = np.meshgrid(y_um, z_um, indexing="xy")
    return y_grid_um * 1e-6, z_grid_um * 1e-6


def _parameters() -> SmoothTFParameters:
    return SmoothTFParameters(
        column_density_peak_m2=4.2e14,
        y0_um=0.7,
        z0_um=-0.3,
        radius_y_um=8.3,
        radius_z_um=3.4,
    )


def test_primary_exponent_matches_the_existing_projected_tf_functions() -> None:
    y_grid_m, z_grid_m = _grid()
    parameters = _parameters()
    vector = to_internal(parameters)
    model = FreeRadiusCompactDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        profile_exponent=1.5,
    )
    expected_density, expected_jacobian = smooth_tf_density_and_internal_jacobian(
        y_grid_m,
        z_grid_m,
        parameters,
    )
    density, jacobian = model.column_density_and_jacobian(vector)

    assert isinstance(model, DifferentiableColumnDensityModel)
    np.testing.assert_allclose(density, expected_density, rtol=3e-15, atol=0.0)
    np.testing.assert_allclose(jacobian, expected_jacobian, rtol=3e-15, atol=0.0)


@pytest.mark.parametrize("profile_exponent", [1.25, 1.5, 2.0])
def test_analytic_jacobian_matches_finite_differences(profile_exponent: float) -> None:
    y_grid_m, z_grid_m = _grid()
    model = FreeRadiusCompactDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        profile_exponent=profile_exponent,
    )
    vector = to_internal(_parameters())
    density, jacobian = model.column_density_and_jacobian(vector)
    stable = density > 0.02 * np.max(density)
    step = 1e-6

    for index in range(model.parameter_count):
        delta = np.zeros(model.parameter_count)
        delta[index] = step
        finite_difference = (
            model.column_density(vector + delta)
            - model.column_density(vector - delta)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            jacobian[index][stable],
            finite_difference[stable],
            rtol=2e-5,
            atol=1e-5 * np.max(np.abs(jacobian[index][stable])),
        )


def test_model_validates_inputs_and_batches_derivatives() -> None:
    y_grid_m, z_grid_m = _grid()
    with pytest.raises(ValueError, match="at least one"):
        FreeRadiusCompactDensityModel.from_grid(
            y_grid_m=y_grid_m,
            z_grid_m=z_grid_m,
            profile_exponent=0.9,
        )
    with pytest.raises(ValueError, match="equal non-empty 2D"):
        FreeRadiusCompactDensityModel.from_grid(
            y_grid_m=y_grid_m,
            z_grid_m=z_grid_m[:, :-1],
        )

    model = FreeRadiusCompactDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
    )
    vector = to_internal(_parameters())
    with pytest.raises(ValueError, match="five finite"):
        model.column_density(np.ones(4))
    with pytest.raises(ValueError, match="batch size"):
        tuple(model.iter_column_density_jacobian(vector, 0))

    batches = tuple(model.iter_column_density_jacobian(vector, 2))
    assert [item[0] for item in batches] == [slice(0, 2), slice(2, 4), slice(4, 5)]
    assert np.concatenate([item[1] for item in batches]).shape == (
        model.parameter_count,
        *y_grid_m.shape,
    )
