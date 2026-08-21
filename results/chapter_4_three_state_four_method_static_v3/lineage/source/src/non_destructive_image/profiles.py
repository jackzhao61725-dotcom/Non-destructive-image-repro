"""Projected Thomas--Fermi profile functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def thomas_fermi_profile_2d(
    coordinate_a: ArrayLike,
    coordinate_b: ArrayLike,
    radius_a: float,
    radius_b: float,
) -> NDArray[np.floating]:
    """Return a unit-peak projected parabolic Thomas--Fermi profile.

    Coordinates and radii must share a length unit. The exponent ``3/2`` is the
    analytic column-density exponent obtained by integrating a three-dimensional
    parabolic Thomas--Fermi density along the imaging axis.
    """

    for name, radius in (("radius_a", radius_a), ("radius_b", radius_b)):
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"{name} must be positive and finite")

    coordinate_a = np.asarray(coordinate_a, dtype=float)
    coordinate_b = np.asarray(coordinate_b, dtype=float)
    if not np.all(np.isfinite(coordinate_a)) or not np.all(np.isfinite(coordinate_b)):
        raise ValueError("profile coordinates must be finite")
    return np.maximum(
        0,
        1 - coordinate_a**2 / radius_a**2 - coordinate_b**2 / radius_b**2,
    ) ** 1.5
