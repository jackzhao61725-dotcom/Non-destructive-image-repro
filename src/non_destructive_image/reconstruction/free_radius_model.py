"""Differentiable free-radius compact profiles for low-order PCI inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .parameters import from_internal


@dataclass(frozen=True)
class FreeRadiusCompactDensityModel:
    """Free elliptical compact profile in absolute physical parameters.

    Optimiser coordinates are ``(log(n_peak), y0_um, z0_um, log(R_y_um),
    log(R_z_um))``. The primary projected Thomas--Fermi model uses
    ``profile_exponent=1.5``; fixed alternative exponents provide a bounded
    profile-shape sensitivity without changing the optical likelihood.
    """

    y_grid_m: NDArray[np.floating]
    z_grid_m: NDArray[np.floating]
    profile_exponent: float = 1.5

    @classmethod
    def from_grid(
        cls,
        *,
        y_grid_m: ArrayLike,
        z_grid_m: ArrayLike,
        profile_exponent: float = 1.5,
    ) -> "FreeRadiusCompactDensityModel":
        """Construct the profile family on one fixed object-plane grid."""

        return cls(
            y_grid_m=np.asarray(y_grid_m, dtype=float),
            z_grid_m=np.asarray(z_grid_m, dtype=float),
            profile_exponent=float(profile_exponent),
        )

    def __post_init__(self) -> None:
        y_grid = np.asarray(self.y_grid_m, dtype=float)
        z_grid = np.asarray(self.z_grid_m, dtype=float)
        exponent = float(self.profile_exponent)
        if y_grid.ndim != 2 or z_grid.shape != y_grid.shape or y_grid.size == 0:
            raise ValueError("compact-profile grids must be equal non-empty 2D arrays")
        if np.any(~np.isfinite(y_grid)) or np.any(~np.isfinite(z_grid)):
            raise ValueError("compact-profile coordinates must be finite")
        if not np.isfinite(exponent) or exponent < 1.0:
            raise ValueError("profile exponent must be finite and at least one")
        y_copy = np.array(y_grid, copy=True)
        z_copy = np.array(z_grid, copy=True)
        y_copy.setflags(write=False)
        z_copy.setflags(write=False)
        object.__setattr__(self, "y_grid_m", y_copy)
        object.__setattr__(self, "z_grid_m", z_copy)
        object.__setattr__(self, "profile_exponent", exponent)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return (
            "log_column_density_peak_m2",
            "centre_y_um",
            "centre_z_um",
            "log_radius_y_um",
            "log_radius_z_um",
        )

    @property
    def parameter_count(self) -> int:
        return 5

    def _density_and_jacobian(
        self,
        parameter_vector: ArrayLike,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        vector = np.asarray(parameter_vector, dtype=float)
        if vector.shape != (self.parameter_count,) or np.any(~np.isfinite(vector)):
            raise ValueError(
                "compact-profile parameter vector must contain five finite values"
            )
        parameters = from_internal(vector)
        y = (
            self.y_grid_m * 1e6 - parameters.y0_um
        ) / parameters.radius_y_um
        z = (
            self.z_grid_m * 1e6 - parameters.z0_um
        ) / parameters.radius_z_um
        q = 1.0 - y**2 - z**2
        inside = q > 0.0
        clipped = np.clip(q, 0.0, None)
        profile = clipped**self.profile_exponent
        density = parameters.column_density_peak_m2 * profile
        slope = (
            self.profile_exponent
            * parameters.column_density_peak_m2
            * clipped ** (self.profile_exponent - 1.0)
            * inside
        )
        derivatives = np.stack(
            [
                density,
                slope * (2.0 * y / parameters.radius_y_um),
                slope * (2.0 * z / parameters.radius_z_um),
                slope * (2.0 * y**2),
                slope * (2.0 * z**2),
            ]
        )
        return np.asarray(density, dtype=float), np.asarray(derivatives, dtype=float)

    def column_density(self, parameter_vector: ArrayLike) -> NDArray[np.floating]:
        """Return the compact column-density profile in ``m^-2``."""

        return self._density_and_jacobian(parameter_vector)[0]

    def column_density_and_jacobian(
        self,
        parameter_vector: ArrayLike,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return the density and analytic derivatives in optimiser order."""

        return self._density_and_jacobian(parameter_vector)

    def iter_column_density_jacobian(
        self,
        parameter_vector: ArrayLike,
        batch_size: int,
    ) -> Iterator[tuple[slice, NDArray[np.floating]]]:
        """Yield analytic profile derivatives in bounded batches."""

        if batch_size <= 0:
            raise ValueError("density Jacobian batch size must be positive")
        derivatives = self._density_and_jacobian(parameter_vector)[1]
        for start in range(0, self.parameter_count, batch_size):
            stop = min(start + batch_size, self.parameter_count)
            yield slice(start, stop), derivatives[start:stop]


__all__ = ["FreeRadiusCompactDensityModel"]
