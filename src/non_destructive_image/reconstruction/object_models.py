"""Object-space condensate models used by the inverse calculation.

The optical measurement layer consumes a column-density map and is deliberately
independent of the family that produced it.  The smooth Thomas--Fermi-like
profile remains a useful low-dimensional reference, while the non-negative
bilinear model provides a shape-flexible baseline that does not force an image
to be single-peaked or parabolic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .parameters import SmoothTFParameters


@runtime_checkable
class DifferentiableColumnDensityModel(Protocol):
    """Contract for a finite-dimensional column-density representation.

    Optimiser coordinates are model-specific and dimensionless.  The returned
    density and its derivatives always use SI column-density units, ``m^-2``.
    """

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the optimiser coordinates in their stored order."""

        ...

    @property
    def parameter_count(self) -> int:
        """Number of optimiser coordinates."""

        ...

    def column_density(self, parameter_vector: ArrayLike) -> NDArray[np.floating]:
        """Return the high-resolution column-density map in ``m^-2``."""

        ...

    def column_density_and_jacobian(
        self,
        parameter_vector: ArrayLike,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return density and derivatives shaped ``(n_parameter, ny, nx)``."""

        ...

    def iter_column_density_jacobian(
        self,
        parameter_vector: ArrayLike,
        batch_size: int,
    ) -> Iterator[tuple[slice, NDArray[np.floating]]]:
        """Yield parameter slices and density derivatives in bounded batches."""

        ...


def smooth_tf_column_density(
    y_grid_m: ArrayLike,
    z_grid_m: ArrayLike,
    parameters: SmoothTFParameters,
) -> NDArray[np.floating]:
    """Return the projected inverted-parabola column-density profile.

    The radii are free fit parameters. The function is therefore a smooth
    object family and does not assert that the real condensate is contact-only.
    """

    y_grid = np.asarray(y_grid_m, dtype=float)
    z_grid = np.asarray(z_grid_m, dtype=float)
    if y_grid.shape != z_grid.shape:
        raise ValueError("y and z coordinate grids must have the same shape")
    y = (y_grid * 1e6 - parameters.y0_um) / parameters.radius_y_um
    z = (z_grid * 1e6 - parameters.z0_um) / parameters.radius_z_um
    return parameters.column_density_peak_m2 * np.clip(1.0 - y**2 - z**2, 0.0, None) ** 1.5


def smooth_tf_density_and_internal_jacobian(
    y_grid_m: ArrayLike,
    z_grid_m: ArrayLike,
    parameters: SmoothTFParameters,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return the profile and derivatives in the five fit coordinates."""

    y_grid = np.asarray(y_grid_m, dtype=float)
    z_grid = np.asarray(z_grid_m, dtype=float)
    if y_grid.shape != z_grid.shape:
        raise ValueError("y and z coordinate grids must have the same shape")
    y = (y_grid * 1e6 - parameters.y0_um) / parameters.radius_y_um
    z = (z_grid * 1e6 - parameters.z0_um) / parameters.radius_z_um
    q = 1.0 - y**2 - z**2
    inside = q > 0.0
    root = np.sqrt(np.clip(q, 0.0, None))
    profile = np.clip(q, 0.0, None) ** 1.5
    density = parameters.column_density_peak_m2 * profile
    common = 1.5 * parameters.column_density_peak_m2 * root * inside
    derivatives = np.stack(
        [
            density,
            common * (2.0 * y / parameters.radius_y_um),
            common * (2.0 * z / parameters.radius_z_um),
            common * (2.0 * y**2),
            common * (2.0 * z**2),
        ]
    )
    return density, derivatives


def _piecewise_linear_basis(
    coordinates_um: NDArray[np.floating],
    knots_um: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return one-dimensional nodal hat functions on arbitrary increasing knots."""

    basis = np.zeros((knots_um.size, *coordinates_um.shape), dtype=float)
    for index, centre in enumerate(knots_um):
        if index > 0:
            left = knots_um[index - 1]
            mask = (coordinates_um >= left) & (coordinates_um <= centre)
            basis[index, mask] = (coordinates_um[mask] - left) / (centre - left)
        if index < knots_um.size - 1:
            right = knots_um[index + 1]
            mask = (coordinates_um >= centre) & (coordinates_um <= right)
            basis[index, mask] = np.maximum(
                basis[index, mask],
                (right - coordinates_um[mask]) / (right - centre),
            )
    return basis


@dataclass(frozen=True)
class NonnegativeBilinearDensityModel:
    """Shape-flexible column density on a non-negative nodal basis.

    The coefficients specify the density at a rectangular set of object-plane
    knots in units of ``coefficient_scale_m2``.  Piecewise-linear interpolation
    permits resolved multi-peak and asymmetric profiles.  The basis is exactly
    zero outside the knot rectangle and an optional fixed support mask, so the
    fit cannot create unconstrained density elsewhere in the field of view.

    This is a controlled baseline, not a claim that unresolved structure has
    been recovered.  Aperture-null modes must still be diagnosed from the data
    Jacobian, and any curvature regularisation must be reported explicitly.
    """

    y_grid_m: NDArray[np.floating]
    z_grid_m: NDArray[np.floating]
    knot_y_um: NDArray[np.floating]
    knot_z_um: NDArray[np.floating]
    coefficient_scale_m2: float
    support_mask: NDArray[np.bool_]
    _y_basis_1d: NDArray[np.floating] = field(init=False, repr=False)
    _z_basis_1d: NDArray[np.floating] = field(init=False, repr=False)

    @classmethod
    def from_grid(
        cls,
        *,
        y_grid_m: ArrayLike,
        z_grid_m: ArrayLike,
        knot_y_um: ArrayLike,
        knot_z_um: ArrayLike,
        coefficient_scale_m2: float,
        support_mask: ArrayLike | None = None,
    ) -> "NonnegativeBilinearDensityModel":
        y_grid = np.asarray(y_grid_m, dtype=float)
        z_grid = np.asarray(z_grid_m, dtype=float)
        if y_grid.ndim != 2 or y_grid.shape != z_grid.shape:
            raise ValueError("y and z coordinate grids must be same-shape 2D arrays")
        support = (
            np.ones(y_grid.shape, dtype=bool)
            if support_mask is None
            else np.asarray(support_mask, dtype=bool)
        )
        if support.shape != y_grid.shape:
            raise ValueError("support mask must have the reconstruction-grid shape")
        return cls(
            y_grid_m=y_grid,
            z_grid_m=z_grid,
            knot_y_um=np.asarray(knot_y_um, dtype=float),
            knot_z_um=np.asarray(knot_z_um, dtype=float),
            coefficient_scale_m2=float(coefficient_scale_m2),
            support_mask=support,
        )

    def __post_init__(self) -> None:
        y_grid = np.asarray(self.y_grid_m, dtype=float)
        z_grid = np.asarray(self.z_grid_m, dtype=float)
        knot_y = np.asarray(self.knot_y_um, dtype=float)
        knot_z = np.asarray(self.knot_z_um, dtype=float)
        support = np.asarray(self.support_mask, dtype=bool)
        if y_grid.ndim != 2 or y_grid.shape != z_grid.shape:
            raise ValueError("y and z coordinate grids must be same-shape 2D arrays")
        if support.shape != y_grid.shape or not np.any(support):
            raise ValueError("support mask must be non-empty and match the coordinate grids")
        if knot_y.ndim != 1 or knot_z.ndim != 1:
            raise ValueError("basis knots must be one-dimensional")
        if knot_y.size < 2 or knot_z.size < 2:
            raise ValueError("at least two knots are required along each axis")
        if np.any(~np.isfinite(knot_y)) or np.any(np.diff(knot_y) <= 0):
            raise ValueError("y knots must be finite and strictly increasing")
        if np.any(~np.isfinite(knot_z)) or np.any(np.diff(knot_z) <= 0):
            raise ValueError("z knots must be finite and strictly increasing")
        if not np.isfinite(self.coefficient_scale_m2) or self.coefficient_scale_m2 <= 0:
            raise ValueError("coefficient scale must be finite and positive")

        y_axis_um = y_grid[0] * 1e6
        z_axis_um = z_grid[:, 0] * 1e6
        if not np.allclose(y_grid * 1e6, y_axis_um[None, :], rtol=0.0, atol=1e-12):
            raise ValueError("bilinear density model requires a rectilinear y meshgrid")
        if not np.allclose(z_grid * 1e6, z_axis_um[:, None], rtol=0.0, atol=1e-12):
            raise ValueError("bilinear density model requires a rectilinear z meshgrid")
        y_basis = _piecewise_linear_basis(y_axis_um, knot_y)
        z_basis = _piecewise_linear_basis(z_axis_um, knot_z)
        partition = np.outer(np.sum(z_basis, axis=0), np.sum(y_basis, axis=0)) * support
        if np.any(partition > 1.0 + 1e-12):
            raise RuntimeError("bilinear basis does not form a valid partition")
        if not np.any(partition):
            raise ValueError("the fixed support does not overlap the density basis")

        object.__setattr__(self, "y_grid_m", y_grid)
        object.__setattr__(self, "z_grid_m", z_grid)
        object.__setattr__(self, "knot_y_um", knot_y)
        object.__setattr__(self, "knot_z_um", knot_z)
        object.__setattr__(self, "support_mask", support)
        object.__setattr__(self, "_y_basis_1d", y_basis)
        object.__setattr__(self, "_z_basis_1d", z_basis)

    @property
    def parameter_count(self) -> int:
        return int(self.knot_y_um.size * self.knot_z_um.size)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(
            f"density_z{z_index}_y{y_index}"
            for z_index in range(self.knot_z_um.size)
            for y_index in range(self.knot_y_um.size)
        )

    @property
    def basis(self) -> NDArray[np.floating]:
        """Dimensionless high-resolution basis functions in parameter order."""

        return np.concatenate(
            [batch for _, batch in self.iter_column_density_jacobian(
                np.zeros(self.parameter_count),
                max(self.parameter_count, 1),
            )],
            axis=0,
        ) / self.coefficient_scale_m2

    def _validate_vector(self, parameter_vector: ArrayLike) -> NDArray[np.floating]:
        vector = np.asarray(parameter_vector, dtype=float)
        if vector.shape != (self.parameter_count,):
            raise ValueError(
                f"bilinear density vector must have shape ({self.parameter_count},)"
            )
        if np.any(~np.isfinite(vector)):
            raise ValueError("bilinear density coefficients must be finite")
        if np.any(vector < 0):
            raise ValueError("bilinear density coefficients must be non-negative")
        return vector

    def column_density(self, parameter_vector: ArrayLike) -> NDArray[np.floating]:
        vector = self._validate_vector(parameter_vector)
        coefficient_grid = vector.reshape(self.knot_z_um.size, self.knot_y_um.size)
        dimensionless = self._z_basis_1d.T @ coefficient_grid @ self._y_basis_1d
        return self.coefficient_scale_m2 * dimensionless * self.support_mask

    def column_density_and_jacobian(
        self,
        parameter_vector: ArrayLike,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        density = self.column_density(parameter_vector)
        derivatives = np.concatenate(
            [batch for _, batch in self.iter_column_density_jacobian(
                parameter_vector,
                max(self.parameter_count, 1),
            )],
            axis=0,
        )
        return density, derivatives

    def iter_column_density_jacobian(
        self,
        parameter_vector: ArrayLike,
        batch_size: int,
    ) -> Iterator[tuple[slice, NDArray[np.floating]]]:
        """Yield nodal basis maps without materialising the complete tensor."""

        self._validate_vector(parameter_vector)
        if batch_size <= 0:
            raise ValueError("density Jacobian batch size must be positive")
        y_count = self.knot_y_um.size
        for start in range(0, self.parameter_count, batch_size):
            stop = min(start + batch_size, self.parameter_count)
            batch = np.empty((stop - start, *self.y_grid_m.shape), dtype=float)
            for local_index, parameter_index in enumerate(range(start, stop)):
                z_index, y_index = divmod(parameter_index, y_count)
                batch[local_index] = (
                    self.coefficient_scale_m2
                    * np.outer(
                        self._z_basis_1d[z_index],
                        self._y_basis_1d[y_index],
                    )
                    * self.support_mask
                )
            yield slice(start, stop), batch

    def basis_matvec(self, parameter_vector: ArrayLike) -> NDArray[np.floating]:
        """Apply the dimensionless basis and return a flattened density map."""

        vector = np.asarray(parameter_vector, dtype=float)
        if vector.shape != (self.parameter_count,) or np.any(~np.isfinite(vector)):
            raise ValueError(
                f"basis matrix input must be finite with shape ({self.parameter_count},)"
            )
        coefficient_grid = vector.reshape(self.knot_z_um.size, self.knot_y_um.size)
        dimensionless = self._z_basis_1d.T @ coefficient_grid @ self._y_basis_1d
        return (dimensionless * self.support_mask).ravel()

    def basis_rmatvec(self, flattened_map: ArrayLike) -> NDArray[np.floating]:
        """Apply the transpose of the dimensionless basis."""

        values = np.asarray(flattened_map, dtype=float)
        if values.shape != (self.y_grid_m.size,):
            raise ValueError(f"basis transpose input must have shape ({self.y_grid_m.size},)")
        image = values.reshape(self.y_grid_m.shape) * self.support_mask
        coefficient_grid = self._z_basis_1d @ image @ self._y_basis_1d.T
        return coefficient_grid.ravel()
