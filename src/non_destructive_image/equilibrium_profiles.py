"""Transparent column-density models for representative dipolar equilibria.

The profiles in this module are analytic, source-informed comparison objects.
They are not outputs of an equilibrium eGPE solver and do not encode phase
coherence.  Every returned column density is normalised on its sampled grid to
the declared atom number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks

from .profiles import thomas_fermi_profile_2d


Morphology = Literal["smooth_bec", "connected_modulated", "separated_droplets"]


def _finite_positive(value: object, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return scalar


def _finite_nonnegative(value: object, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative scalar") from exc
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return scalar


def _immutable(values: ArrayLike) -> NDArray[np.floating]:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class EquilibriumProfileDefinition:
    """Parameters of one analytic representative equilibrium profile.

    Smooth profiles use ``radius_y_m`` and ``radius_z_m``.  Modulated profiles
    use a Gaussian chain defined by component centres, weights and widths.
    """

    state_id: str
    label: str
    morphology: Morphology
    atom_number: float
    radius_y_m: float | None = None
    radius_z_m: float | None = None
    component_centres_y_m: tuple[float, ...] = ()
    component_weights: tuple[float, ...] = ()
    component_sigma_y_m: float | None = None
    component_sigma_z_m: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id.strip():
            raise ValueError("state_id must be non-empty")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be non-empty")
        if self.morphology not in {
            "smooth_bec",
            "connected_modulated",
            "separated_droplets",
        }:
            raise ValueError("unsupported equilibrium morphology")
        object.__setattr__(
            self,
            "atom_number",
            _finite_positive(self.atom_number, "atom_number"),
        )

        if self.morphology == "smooth_bec":
            if any(
                value is not None
                for value in (self.component_sigma_y_m, self.component_sigma_z_m)
            ) or self.component_centres_y_m or self.component_weights:
                raise ValueError("smooth_bec must not define Gaussian components")
            object.__setattr__(
                self,
                "radius_y_m",
                _finite_positive(self.radius_y_m, "radius_y_m"),
            )
            object.__setattr__(
                self,
                "radius_z_m",
                _finite_positive(self.radius_z_m, "radius_z_m"),
            )
            return

        if self.radius_y_m is not None or self.radius_z_m is not None:
            raise ValueError("Gaussian-chain profiles must not define Thomas--Fermi radii")
        centres = tuple(float(value) for value in self.component_centres_y_m)
        weights = tuple(
            _finite_positive(value, "component weight")
            for value in self.component_weights
        )
        if len(centres) < 2 or len(centres) != len(weights):
            raise ValueError("Gaussian-chain centres and weights must have equal length >= 2")
        if not np.all(np.isfinite(centres)) or any(
            right <= left for left, right in zip(centres[:-1], centres[1:], strict=True)
        ):
            raise ValueError("component centres must be finite and strictly increasing")
        object.__setattr__(self, "component_centres_y_m", centres)
        object.__setattr__(self, "component_weights", weights)
        object.__setattr__(
            self,
            "component_sigma_y_m",
            _finite_positive(self.component_sigma_y_m, "component_sigma_y_m"),
        )
        object.__setattr__(
            self,
            "component_sigma_z_m",
            _finite_positive(self.component_sigma_z_m, "component_sigma_z_m"),
        )


@dataclass(frozen=True)
class MorphologyObservables:
    """Low-dimensional morphology measured from one non-negative 2D map."""

    integrated_weight: float
    rms_y_m: float
    rms_z_m: float
    peak_count: int
    peak_positions_y_m: tuple[float, ...]
    mean_peak_spacing_m: float | None
    modulation_contrast: float


@dataclass(frozen=True)
class EquilibriumProfile:
    """One sampled and atom-number-normalised representative profile."""

    definition: EquilibriumProfileDefinition
    y_grid_m: NDArray[np.floating]
    z_grid_m: NDArray[np.floating]
    column_density_m2: NDArray[np.floating]
    line_density_m1: NDArray[np.floating]
    observables: MorphologyObservables


def _validated_grid(
    y_grid_m: ArrayLike,
    z_grid_m: ArrayLike,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    float,
    float,
]:
    y = np.asarray(y_grid_m, dtype=float)
    z = np.asarray(z_grid_m, dtype=float)
    if y.ndim != 2 or y.shape != z.shape or min(y.shape) < 3:
        raise ValueError("y_grid_m and z_grid_m must be same-shape 2D arrays of size >= 3")
    if not np.isfinite(y).all() or not np.isfinite(z).all():
        raise ValueError("coordinate grids must be finite")
    y_axis = y[0, :]
    z_axis = z[:, 0]
    if not np.allclose(y, y_axis[None, :], rtol=0.0, atol=1e-15):
        raise ValueError("y_grid_m must be constant down each column")
    if not np.allclose(z, z_axis[:, None], rtol=0.0, atol=1e-15):
        raise ValueError("z_grid_m must be constant across each row")
    dy_values = np.diff(y_axis)
    dz_values = np.diff(z_axis)
    dy_m = float(np.mean(dy_values))
    dz_m = float(np.mean(dz_values))
    if (
        dy_m <= 0.0
        or dz_m <= 0.0
        or not np.allclose(dy_values, dy_m, rtol=1e-12, atol=0.0)
        or not np.allclose(dz_values, dz_m, rtol=1e-12, atol=0.0)
    ):
        raise ValueError("coordinate grids must be strictly increasing and uniform")
    return y, z, y_axis, z_axis, dy_m, dz_m


def measure_morphology(
    map_values: ArrayLike,
    y_axis_m: ArrayLike,
    z_axis_m: ArrayLike,
    *,
    minimum_peak_distance_m: float,
    peak_prominence_fraction: float,
) -> MorphologyObservables:
    """Measure rms widths, axial maxima, spacing and inter-peak contrast.

    ``map_values`` may be a column density or any non-negative image-derived
    weight.  The axial profile is obtained by integrating over ``z``.  For a
    single-peaked profile the modulation contrast is exactly zero by definition
    and the peak spacing is unsupported (``None``).
    """

    values = np.asarray(map_values, dtype=float)
    y_axis = np.asarray(y_axis_m, dtype=float)
    z_axis = np.asarray(z_axis_m, dtype=float)
    if values.ndim != 2 or values.shape != (z_axis.size, y_axis.size):
        raise ValueError("map_values shape must be (len(z_axis_m), len(y_axis_m))")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("map_values must be finite and non-negative")
    if y_axis.size < 3 or z_axis.size < 3:
        raise ValueError("morphology measurement requires at least a 3x3 grid")
    if not np.isfinite(y_axis).all() or not np.isfinite(z_axis).all():
        raise ValueError("morphology axes must be finite")
    dy_values = np.diff(y_axis)
    dz_values = np.diff(z_axis)
    dy_m = float(np.mean(dy_values))
    dz_m = float(np.mean(dz_values))
    if (
        dy_m <= 0.0
        or dz_m <= 0.0
        or not np.allclose(dy_values, dy_m, rtol=1e-12, atol=0.0)
        or not np.allclose(dz_values, dz_m, rtol=1e-12, atol=0.0)
    ):
        raise ValueError("morphology axes must be strictly increasing and uniform")
    minimum_distance = _finite_positive(
        minimum_peak_distance_m,
        "minimum_peak_distance_m",
    )
    prominence_fraction = _finite_nonnegative(
        peak_prominence_fraction,
        "peak_prominence_fraction",
    )
    if prominence_fraction >= 1.0:
        raise ValueError("peak_prominence_fraction must be smaller than one")

    total = float(np.sum(values) * dy_m * dz_m)
    if total <= 0.0:
        raise ValueError("map_values must contain positive integrated weight")
    y_grid, z_grid = np.meshgrid(y_axis, z_axis)
    mean_y = float(np.sum(values * y_grid) * dy_m * dz_m / total)
    mean_z = float(np.sum(values * z_grid) * dy_m * dz_m / total)
    rms_y = float(
        np.sqrt(np.sum(values * (y_grid - mean_y) ** 2) * dy_m * dz_m / total)
    )
    rms_z = float(
        np.sqrt(np.sum(values * (z_grid - mean_z) ** 2) * dy_m * dz_m / total)
    )

    line_density = np.sum(values, axis=0) * dz_m
    minimum_samples = max(1, int(np.ceil(minimum_distance / dy_m)))
    prominence = prominence_fraction * float(np.max(line_density))
    peak_indices, _ = find_peaks(
        line_density,
        distance=minimum_samples,
        prominence=prominence,
    )
    if peak_indices.size == 0:
        peak_indices = np.asarray([int(np.argmax(line_density))], dtype=int)
    positions = tuple(float(y_axis[index]) for index in peak_indices)
    if len(positions) < 2:
        spacing = None
        contrast = 0.0
    else:
        spacing = float(np.mean(np.diff(positions)))
        pair_contrasts: list[float] = []
        for left, right in zip(peak_indices[:-1], peak_indices[1:], strict=True):
            peak_level = 0.5 * (line_density[left] + line_density[right])
            trough_level = float(np.min(line_density[left : right + 1]))
            denominator = peak_level + trough_level
            pair_contrasts.append(
                0.0 if denominator <= 0.0 else float((peak_level - trough_level) / denominator)
            )
        contrast = float(np.mean(pair_contrasts))

    return MorphologyObservables(
        integrated_weight=total,
        rms_y_m=rms_y,
        rms_z_m=rms_z,
        peak_count=len(positions),
        peak_positions_y_m=positions,
        mean_peak_spacing_m=spacing,
        modulation_contrast=contrast,
    )


def build_equilibrium_profile(
    definition: EquilibriumProfileDefinition,
    y_grid_m: ArrayLike,
    z_grid_m: ArrayLike,
    *,
    minimum_peak_distance_m: float,
    peak_prominence_fraction: float,
) -> EquilibriumProfile:
    """Build and normalise one source-informed representative column density."""

    if not isinstance(definition, EquilibriumProfileDefinition):
        raise TypeError("definition must be an EquilibriumProfileDefinition")
    y, z, y_axis, z_axis, dy_m, dz_m = _validated_grid(y_grid_m, z_grid_m)
    if definition.morphology == "smooth_bec":
        assert definition.radius_y_m is not None
        assert definition.radius_z_m is not None
        raw = thomas_fermi_profile_2d(
            y,
            z,
            definition.radius_y_m,
            definition.radius_z_m,
        )
    else:
        assert definition.component_sigma_y_m is not None
        assert definition.component_sigma_z_m is not None
        raw = np.zeros_like(y)
        for centre_m, weight in zip(
            definition.component_centres_y_m,
            definition.component_weights,
            strict=True,
        ):
            raw += weight * np.exp(
                -0.5 * ((y - centre_m) / definition.component_sigma_y_m) ** 2
                -0.5 * (z / definition.component_sigma_z_m) ** 2
            )
    if not np.isfinite(raw).all() or np.any(raw < 0.0):
        raise RuntimeError("analytic profile produced an invalid density")
    raw_integral = float(np.sum(raw) * dy_m * dz_m)
    if raw_integral <= 0.0:
        raise ValueError("profile support does not overlap the supplied grid")
    density = raw * definition.atom_number / raw_integral
    observables = measure_morphology(
        density,
        y_axis,
        z_axis,
        minimum_peak_distance_m=minimum_peak_distance_m,
        peak_prominence_fraction=peak_prominence_fraction,
    )
    line_density = np.sum(density, axis=0) * dz_m
    return EquilibriumProfile(
        definition=definition,
        y_grid_m=_immutable(y),
        z_grid_m=_immutable(z),
        column_density_m2=_immutable(density),
        line_density_m1=_immutable(line_density),
        observables=observables,
    )


__all__ = [
    "EquilibriumProfile",
    "EquilibriumProfileDefinition",
    "MorphologyObservables",
    "build_equilibrium_profile",
    "measure_morphology",
]
