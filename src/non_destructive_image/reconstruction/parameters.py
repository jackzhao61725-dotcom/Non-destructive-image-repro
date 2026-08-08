"""Physical parameters and fit-coordinate transforms for reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SmoothTFParameters:
    """Parameters of a projected, free-radius Thomas-Fermi-like profile."""

    column_density_peak_m2: float
    y0_um: float
    z0_um: float
    radius_y_um: float
    radius_z_um: float

    def __post_init__(self) -> None:
        values = np.asarray(tuple(self.as_dict().values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("all smooth-TF parameters must be finite")
        if self.column_density_peak_m2 <= 0:
            raise ValueError("column_density_peak_m2 must be positive")
        if self.radius_y_um <= 0 or self.radius_z_um <= 0:
            raise ValueError("smooth-TF radii must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "column_density_peak_m2": float(self.column_density_peak_m2),
            "y0_um": float(self.y0_um),
            "z0_um": float(self.z0_um),
            "radius_y_um": float(self.radius_y_um),
            "radius_z_um": float(self.radius_z_um),
        }


def to_internal(parameters: SmoothTFParameters) -> NDArray[np.floating]:
    """Transform positive scale parameters to logarithmic fit coordinates."""

    return np.asarray(
        [
            np.log(parameters.column_density_peak_m2),
            parameters.y0_um,
            parameters.z0_um,
            np.log(parameters.radius_y_um),
            np.log(parameters.radius_z_um),
        ],
        dtype=float,
    )


def from_internal(vector: ArrayLike) -> SmoothTFParameters:
    """Transform a five-element optimiser vector to physical parameters."""

    values = np.asarray(vector, dtype=float)
    if values.shape != (5,):
        raise ValueError("the smooth-TF optimiser vector must contain five values")
    return SmoothTFParameters(
        column_density_peak_m2=float(np.exp(values[0])),
        y0_um=float(values[1]),
        z0_um=float(values[2]),
        radius_y_um=float(np.exp(values[3])),
        radius_z_um=float(np.exp(values[4])),
    )
