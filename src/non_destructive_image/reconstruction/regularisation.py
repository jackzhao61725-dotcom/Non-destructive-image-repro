r"""Physically scaled curvature regularisation for nodal density models.

The operator in this module approximates the thin-plate seminorm

.. math::

   \ell^2 \int \left[w_{yy} n_{,yy}^2
   + 2 w_{yz} n_{,yz}^2 + w_{zz} n_{,zz}^2\right]
   / n_{\rm scale}^2\,\mathrm{d}y\,\mathrm{d}z,

where coordinates are measured in micrometres.  Consequently
``weight_um2`` has units of square micrometres and the residual is
dimensionless.  The physical knot-density scale is deliberately separate
from the numerical scale used for optimiser coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


BoundaryPolicy = Literal["fixed_zero_ghost_knots"]


@dataclass(frozen=True)
class CurvatureAxisWeights:
    """Relative weights of the three independent Hessian components."""

    yy: float = 1.0
    yz: float = 1.0
    zz: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray((self.yy, self.yz, self.zz), dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("curvature axis weights must be finite and non-negative")


@dataclass(frozen=True)
class CurvatureRegularisation:
    """Linear curvature penalty acting on physical nodal densities.

    ``physical_density_matrix`` maps a C-order flattened ``(z, y)`` grid of
    knot densities in m^-2 to dimensionless least-squares residuals.  Use
    :meth:`matrix_for_coefficient_scale` when the optimiser instead stores
    dimensionless coefficients whose physical values equal
    ``coefficient_scale_m2 * coefficients``.

    The supported boundary policy adds one fixed, zero-valued ghost knot
    beyond every edge, at the spacing of the adjacent real knot.  It therefore
    penalises a non-zero constant or linear field at the boundary.  The
    ``interior_physical_density_matrix`` omits every ghost-dependent row and
    is exposed for diagnostics of the interior discretisation; it is not the
    regulariser to use for the constrained reconstruction.
    """

    knot_y_um: NDArray[np.floating]
    knot_z_um: NDArray[np.floating]
    density_scale_m2: float
    weight_um2: float
    boundary_policy: BoundaryPolicy
    axis_weights: CurvatureAxisWeights
    physical_density_matrix: NDArray[np.floating]
    interior_physical_density_matrix: NDArray[np.floating]
    yy_physical_density_matrix: NDArray[np.floating]
    yz_physical_density_matrix: NDArray[np.floating]
    zz_physical_density_matrix: NDArray[np.floating]

    @property
    def parameter_count(self) -> int:
        """Number of nodal density parameters expected by the operator."""

        return int(self.knot_y_um.size * self.knot_z_um.size)

    @property
    def residual_count(self) -> int:
        """Number of residual rows in the complete boundary-aware operator."""

        return int(self.physical_density_matrix.shape[0])

    def matrix_for_coefficient_scale(
        self,
        coefficient_scale_m2: float,
    ) -> NDArray[np.floating]:
        """Return a matrix acting directly on dimensionless coefficients."""

        scale = float(coefficient_scale_m2)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("coefficient scale must be finite and positive")
        return self.physical_density_matrix * scale

    def residual_from_density(self, knot_density_m2: ArrayLike) -> NDArray[np.floating]:
        """Evaluate the residual from physical knot densities in m^-2."""

        density = _flatten_parameter_grid(knot_density_m2, self.knot_z_um.size, self.knot_y_um.size)
        return self.physical_density_matrix @ density

    def interior_residual_from_density(
        self,
        knot_density_m2: ArrayLike,
    ) -> NDArray[np.floating]:
        """Evaluate only rows that do not depend on fixed ghost knots."""

        density = _flatten_parameter_grid(knot_density_m2, self.knot_z_um.size, self.knot_y_um.size)
        return self.interior_physical_density_matrix @ density

    def residual_from_coefficients(
        self,
        coefficients: ArrayLike,
        *,
        coefficient_scale_m2: float,
    ) -> NDArray[np.floating]:
        """Evaluate the same physical penalty from optimiser coefficients."""

        vector = _flatten_parameter_grid(
            coefficients,
            self.knot_z_um.size,
            self.knot_y_um.size,
        )
        return self.matrix_for_coefficient_scale(coefficient_scale_m2) @ vector

    def penalty_from_density(self, knot_density_m2: ArrayLike) -> float:
        """Return one half of the squared residual norm."""

        residual = self.residual_from_density(knot_density_m2)
        return 0.5 * float(residual @ residual)


def build_curvature_regularisation(
    knot_y_um: ArrayLike,
    knot_z_um: ArrayLike,
    *,
    density_scale_m2: float,
    weight_um2: float,
    boundary_policy: BoundaryPolicy = "fixed_zero_ghost_knots",
    axis_weights: CurvatureAxisWeights | tuple[float, float, float] = CurvatureAxisWeights(),
) -> CurvatureRegularisation:
    """Build a non-uniform-grid, physically normalised curvature operator.

    The mixed derivative is integrated cellwise and carries the conventional
    factor of two from the Frobenius norm of a two-dimensional Hessian.  Pure
    second derivatives use non-uniform three-point divided differences and
    lumped nodal quadrature.
    """

    y = _validated_knots(knot_y_um, "y")
    z = _validated_knots(knot_z_um, "z")
    density_scale = float(density_scale_m2)
    weight = float(weight_um2)
    if not np.isfinite(density_scale) or density_scale <= 0.0:
        raise ValueError("density scale must be finite and positive")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("regularisation weight must be finite and non-negative")
    if boundary_policy != "fixed_zero_ghost_knots":
        raise ValueError(
            "boundary policy must be 'fixed_zero_ghost_knots'"
        )
    weights = (
        axis_weights
        if isinstance(axis_weights, CurvatureAxisWeights)
        else CurvatureAxisWeights(*axis_weights)
    )

    parameter_count = y.size * z.size
    if weight == 0.0 or (weights.yy == weights.yz == weights.zz == 0.0):
        empty = np.zeros((0, parameter_count), dtype=float)
        return CurvatureRegularisation(
            knot_y_um=_immutable_copy(y),
            knot_z_um=_immutable_copy(z),
            density_scale_m2=density_scale,
            weight_um2=weight,
            boundary_policy=boundary_policy,
            axis_weights=weights,
            physical_density_matrix=empty,
            interior_physical_density_matrix=empty.copy(),
            yy_physical_density_matrix=empty.copy(),
            yz_physical_density_matrix=empty.copy(),
            zz_physical_density_matrix=empty.copy(),
        )

    y_extended = _extend_with_zero_ghost_positions(y)
    z_extended = _extend_with_zero_ghost_positions(z)
    y_quadrature = 0.5 * (y_extended[2:] - y_extended[:-2])
    z_quadrature = 0.5 * (z_extended[2:] - z_extended[:-2])

    common_scale = np.sqrt(weight) / density_scale
    yy = _pure_second_derivative_rows(
        derivative_axis=y,
        transverse_axis=z,
        derivative_extended=y_extended,
        derivative_quadrature=y_quadrature,
        transverse_quadrature=z_quadrature,
        derivative_is_y=True,
        parameter_count=parameter_count,
        component_scale=common_scale * np.sqrt(weights.yy),
        include_boundary=True,
    )
    zz = _pure_second_derivative_rows(
        derivative_axis=z,
        transverse_axis=y,
        derivative_extended=z_extended,
        derivative_quadrature=z_quadrature,
        transverse_quadrature=y_quadrature,
        derivative_is_y=False,
        parameter_count=parameter_count,
        component_scale=common_scale * np.sqrt(weights.zz),
        include_boundary=True,
    )
    yz = _mixed_derivative_rows(
        y,
        z,
        y_extended,
        z_extended,
        parameter_count=parameter_count,
        component_scale=common_scale * np.sqrt(2.0 * weights.yz),
        include_ghost_cells=True,
    )

    yy_interior = _pure_second_derivative_rows(
        derivative_axis=y,
        transverse_axis=z,
        derivative_extended=y_extended,
        derivative_quadrature=y_quadrature,
        transverse_quadrature=z_quadrature,
        derivative_is_y=True,
        parameter_count=parameter_count,
        component_scale=common_scale * np.sqrt(weights.yy),
        include_boundary=False,
    )
    zz_interior = _pure_second_derivative_rows(
        derivative_axis=z,
        transverse_axis=y,
        derivative_extended=z_extended,
        derivative_quadrature=z_quadrature,
        transverse_quadrature=y_quadrature,
        derivative_is_y=False,
        parameter_count=parameter_count,
        component_scale=common_scale * np.sqrt(weights.zz),
        include_boundary=False,
    )
    yz_interior = _mixed_derivative_rows(
        y,
        z,
        y_extended,
        z_extended,
        parameter_count=parameter_count,
        component_scale=common_scale * np.sqrt(2.0 * weights.yz),
        include_ghost_cells=False,
    )

    matrix = _stack_nonempty(yy, yz, zz, column_count=parameter_count)
    interior = _stack_nonempty(
        yy_interior,
        yz_interior,
        zz_interior,
        column_count=parameter_count,
    )
    return CurvatureRegularisation(
        knot_y_um=_immutable_copy(y),
        knot_z_um=_immutable_copy(z),
        density_scale_m2=density_scale,
        weight_um2=weight,
        boundary_policy=boundary_policy,
        axis_weights=weights,
        physical_density_matrix=_immutable_copy(matrix),
        interior_physical_density_matrix=_immutable_copy(interior),
        yy_physical_density_matrix=_immutable_copy(yy),
        yz_physical_density_matrix=_immutable_copy(yz),
        zz_physical_density_matrix=_immutable_copy(zz),
    )


def _validated_knots(values: ArrayLike, axis_name: str) -> NDArray[np.floating]:
    knots = np.asarray(values, dtype=float)
    if knots.ndim != 1 or knots.size < 3:
        raise ValueError(f"{axis_name} knots must be one-dimensional with at least three values")
    if np.any(~np.isfinite(knots)) or np.any(np.diff(knots) <= 0.0):
        raise ValueError(f"{axis_name} knots must be finite and strictly increasing")
    return np.array(knots, copy=True)


def _extend_with_zero_ghost_positions(knots: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.concatenate(
        ([knots[0] - (knots[1] - knots[0])], knots, [knots[-1] + (knots[-1] - knots[-2])])
    )


def _flatten_parameter_grid(
    values: ArrayLike,
    z_count: int,
    y_count: int,
) -> NDArray[np.floating]:
    array = np.asarray(values, dtype=float)
    if array.shape == (z_count, y_count):
        vector = array.ravel()
    elif array.shape == (z_count * y_count,):
        vector = array
    else:
        raise ValueError(
            f"nodal values must have shape ({z_count}, {y_count}) or ({z_count * y_count},)"
        )
    if np.any(~np.isfinite(vector)):
        raise ValueError("nodal values must be finite")
    return vector


def _parameter_index(z_index: int, y_index: int, y_count: int) -> int:
    return z_index * y_count + y_index


def _second_divided_difference_coefficients(
    left_spacing: float,
    right_spacing: float,
) -> tuple[float, float, float]:
    normalisation = 2.0 / (left_spacing + right_spacing)
    return (
        normalisation / left_spacing,
        -normalisation * (1.0 / left_spacing + 1.0 / right_spacing),
        normalisation / right_spacing,
    )


def _pure_second_derivative_rows(
    *,
    derivative_axis: NDArray[np.floating],
    transverse_axis: NDArray[np.floating],
    derivative_extended: NDArray[np.floating],
    derivative_quadrature: NDArray[np.floating],
    transverse_quadrature: NDArray[np.floating],
    derivative_is_y: bool,
    parameter_count: int,
    component_scale: float,
    include_boundary: bool,
) -> NDArray[np.floating]:
    if component_scale == 0.0:
        return np.zeros((0, parameter_count), dtype=float)
    derivative_count = derivative_axis.size
    transverse_count = transverse_axis.size
    derivative_indices = (
        range(derivative_count)
        if include_boundary
        else range(1, derivative_count - 1)
    )
    y_count = derivative_count if derivative_is_y else transverse_count
    rows: list[NDArray[np.floating]] = []
    for transverse_index in range(transverse_count):
        for derivative_index in derivative_indices:
            extended_index = derivative_index + 1
            left_spacing = derivative_extended[extended_index] - derivative_extended[extended_index - 1]
            right_spacing = derivative_extended[extended_index + 1] - derivative_extended[extended_index]
            coefficients = _second_divided_difference_coefficients(left_spacing, right_spacing)
            row = np.zeros(parameter_count, dtype=float)
            for relative_index, coefficient in zip((-1, 0, 1), coefficients, strict=True):
                neighbour = derivative_index + relative_index
                if neighbour < 0 or neighbour >= derivative_count:
                    continue  # The corresponding fixed ghost value is zero.
                if derivative_is_y:
                    z_index, y_index = transverse_index, neighbour
                else:
                    z_index, y_index = neighbour, transverse_index
                row[_parameter_index(z_index, y_index, y_count)] = coefficient
            area = derivative_quadrature[derivative_index] * transverse_quadrature[transverse_index]
            rows.append(component_scale * np.sqrt(area) * row)
    return np.stack(rows) if rows else np.zeros((0, parameter_count), dtype=float)


def _mixed_derivative_rows(
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    y_extended: NDArray[np.floating],
    z_extended: NDArray[np.floating],
    *,
    parameter_count: int,
    component_scale: float,
    include_ghost_cells: bool,
) -> NDArray[np.floating]:
    if component_scale == 0.0:
        return np.zeros((0, parameter_count), dtype=float)
    y_count = y.size
    y_cells = range(y_extended.size - 1) if include_ghost_cells else range(1, y.size)
    z_cells = range(z_extended.size - 1) if include_ghost_cells else range(1, z.size)
    rows: list[NDArray[np.floating]] = []
    for z_cell in z_cells:
        dz = z_extended[z_cell + 1] - z_extended[z_cell]
        for y_cell in y_cells:
            dy = y_extended[y_cell + 1] - y_extended[y_cell]
            row = np.zeros(parameter_count, dtype=float)
            coefficient = 1.0 / (dy * dz)
            for extended_z, extended_y, sign in (
                (z_cell, y_cell, 1.0),
                (z_cell, y_cell + 1, -1.0),
                (z_cell + 1, y_cell, -1.0),
                (z_cell + 1, y_cell + 1, 1.0),
            ):
                real_z = extended_z - 1
                real_y = extended_y - 1
                if 0 <= real_z < z.size and 0 <= real_y < y.size:
                    row[_parameter_index(real_z, real_y, y_count)] += sign * coefficient
            rows.append(component_scale * np.sqrt(dy * dz) * row)
    return np.stack(rows) if rows else np.zeros((0, parameter_count), dtype=float)


def _stack_nonempty(
    *matrices: NDArray[np.floating],
    column_count: int,
) -> NDArray[np.floating]:
    nonempty = [matrix for matrix in matrices if matrix.shape[0] > 0]
    return np.vstack(nonempty) if nonempty else np.zeros((0, column_count), dtype=float)


def _immutable_copy(values: NDArray[np.floating]) -> NDArray[np.floating]:
    copied = np.array(values, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied
