"""Physical low-order observables of reconstructed column-density maps.

The routines in this module operate on a declared object-plane integration
support.  They do not infer cloud presence from camera data: the numerical
support flags only state whether a supplied non-negative map contains enough
integrated response to define its moments.  Raw-channel evidence for cloud
presence remains a separate reconstruction credibility question.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _read_only_array(value: ArrayLike, *, dtype: type) -> NDArray:
    """Return an array backed by immutable bytes for a public contract."""

    source = np.asarray(value, dtype=dtype)
    # ``setflags(write=False)`` on an owning ndarray is reversible by its
    # caller.  A bytes-backed view cannot have write access restored, which
    # keeps stored grid/support contracts stable after construction.
    immutable = np.frombuffer(source.tobytes(order="C"), dtype=np.dtype(dtype))
    return immutable.reshape(source.shape)


def _cell_widths_from_centres(axis_m: NDArray[np.floating]) -> NDArray[np.floating]:
    """Infer cell widths from strictly increasing one-dimensional centres."""

    if axis_m.ndim != 1 or axis_m.size < 2:
        raise ValueError("cell-area inference requires at least two points per axis")
    differences = np.diff(axis_m)
    if np.any(differences <= 0.0):
        raise ValueError("coordinate axes must be strictly increasing")
    edges = np.empty(axis_m.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (axis_m[:-1] + axis_m[1:])
    edges[0] = axis_m[0] - 0.5 * differences[0]
    edges[-1] = axis_m[-1] + 0.5 * differences[-1]
    return np.diff(edges)


def _infer_rectilinear_cell_areas(
    y_grid_m: NDArray[np.floating],
    z_grid_m: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Infer physical cell areas for a rectilinear cell-centre mesh."""

    y_axis_m = y_grid_m[0, :]
    z_axis_m = z_grid_m[:, 0]
    y_rectilinear = np.broadcast_to(y_axis_m, y_grid_m.shape)
    z_rectilinear = np.broadcast_to(z_axis_m[:, None], z_grid_m.shape)
    coordinate_scale = max(
        float(np.max(np.abs(y_grid_m))),
        float(np.max(np.abs(z_grid_m))),
        np.finfo(float).tiny,
    )
    absolute_tolerance = 8.0 * np.finfo(float).eps * coordinate_scale
    if not np.allclose(
        y_grid_m,
        y_rectilinear,
        rtol=8.0 * np.finfo(float).eps,
        atol=absolute_tolerance,
    ) or not np.allclose(
        z_grid_m,
        z_rectilinear,
        rtol=8.0 * np.finfo(float).eps,
        atol=absolute_tolerance,
    ):
        raise ValueError(
            "automatic cell-area inference requires a rectilinear coordinate grid"
        )
    widths_y_m = _cell_widths_from_centres(y_axis_m)
    widths_z_m = _cell_widths_from_centres(z_axis_m)
    return widths_z_m[:, None] * widths_y_m[None, :]


@dataclass(frozen=True, eq=False)
class ObservableIntegrationSupport:
    """Fixed physical grid, cell areas and mask used for every map in a series.

    Coordinates are object-plane cell centres in metres.  If ``cell_area_m2``
    is omitted, areas are inferred from a rectilinear grid by placing cell
    boundaries halfway between adjacent centres and extending the two outer
    cells by half of their nearest centre spacing.  Supplying cell areas
    explicitly permits other cell-centre grids.

    Arrays are copied and made read-only so that a summary retains the exact
    support definition used to calculate it.
    """

    y_grid_m: ArrayLike
    z_grid_m: ArrayLike
    support_mask: ArrayLike | None = None
    cell_area_m2: ArrayLike | float | None = None

    def __post_init__(self) -> None:
        y_grid = np.asarray(self.y_grid_m, dtype=float)
        z_grid = np.asarray(self.z_grid_m, dtype=float)
        if y_grid.ndim != 2 or z_grid.shape != y_grid.shape:
            raise ValueError("y and z coordinate grids must be same-shape 2D arrays")
        if y_grid.size == 0:
            raise ValueError("observable integration grid cannot be empty")
        if np.any(~np.isfinite(y_grid)) or np.any(~np.isfinite(z_grid)):
            raise ValueError("observable integration coordinates must be finite")

        if self.support_mask is None:
            mask = np.ones(y_grid.shape, dtype=bool)
        else:
            mask = np.asarray(self.support_mask, dtype=bool)
            if mask.shape != y_grid.shape:
                raise ValueError("integration support mask must match the coordinate grid")
        if not np.any(mask):
            raise ValueError("integration support mask cannot be empty")

        if self.cell_area_m2 is None:
            area = _infer_rectilinear_cell_areas(y_grid, z_grid)
        else:
            supplied_area = np.asarray(self.cell_area_m2, dtype=float)
            if supplied_area.ndim == 0:
                area = np.full(y_grid.shape, float(supplied_area), dtype=float)
            elif supplied_area.shape == y_grid.shape:
                area = supplied_area
            else:
                raise ValueError("cell areas must be scalar or match the coordinate grid")
        if np.any(~np.isfinite(area)) or np.any(area <= 0.0):
            raise ValueError("physical cell areas must be finite and positive")

        object.__setattr__(self, "y_grid_m", _read_only_array(y_grid, dtype=float))
        object.__setattr__(self, "z_grid_m", _read_only_array(z_grid, dtype=float))
        object.__setattr__(self, "support_mask", _read_only_array(mask, dtype=bool))
        object.__setattr__(self, "cell_area_m2", _read_only_array(area, dtype=float))

    @property
    def shape(self) -> tuple[int, int]:
        """Shape required of every response map on this support."""

        return self.y_grid_m.shape

    @property
    def physical_area_m2(self) -> float:
        """Total physical area of supported cells."""

        return float(np.sum(self.cell_area_m2[self.support_mask]))

    def is_identical_to(self, other: object) -> bool:
        """Return whether two definitions use exactly the same grid and support."""

        return (
            isinstance(other, ObservableIntegrationSupport)
            and np.array_equal(self.y_grid_m, other.y_grid_m)
            and np.array_equal(self.z_grid_m, other.z_grid_m)
            and np.array_equal(self.cell_area_m2, other.cell_area_m2)
            and np.array_equal(self.support_mask, other.support_mask)
        )


@dataclass(frozen=True)
class ObservableSupportFlags:
    """Explicit numerical support state for each family of observables.

    These flags describe a supplied map, not raw-camera evidence that a cloud
    is present.  In particular, ``positive_integrated_response`` can be true
    for an arbitrarily small positive map.
    """

    positive_integrated_response: bool
    moments_numerically_supported: bool
    centroid_supported: bool
    covariance_supported: bool
    widths_supported: bool
    aspect_ratio_supported: bool
    principal_axis_angle_supported: bool
    reasons: tuple[str, ...]

@dataclass(frozen=True, eq=False)
class DensityObservableSummary:
    """Physical moments and their numerical support for one response map.

    The centroid vector is ordered ``(y, z)`` and the covariance tensor uses
    the same order.  The principal-axis angle is measured from positive
    ``y`` towards positive ``z`` and represents an undirected axis modulo
    pi; the reported representative lies in
    ``[-pi/2, pi/2)``.  Unsupported quantities are ``None`` rather than
    fabricated zeros or NaNs.
    """

    integrated_response: float
    centroid_m: NDArray[np.floating] | None
    covariance_m2: NDArray[np.floating] | None
    major_rms_width_m: float | None
    minor_rms_width_m: float | None
    aspect_ratio: float | None
    principal_axis_angle_rad: float | None
    fractional_anisotropy: float | None
    angle_anisotropy_threshold: float
    minimum_integrated_response: float
    support_flags: ObservableSupportFlags
    integration_support: ObservableIntegrationSupport

    def __post_init__(self) -> None:
        integrated_response = float(self.integrated_response)
        minimum_response = float(self.minimum_integrated_response)
        angle_threshold = float(self.angle_anisotropy_threshold)
        if not np.isfinite(integrated_response) or integrated_response < 0.0:
            raise ValueError("integrated response must be finite and non-negative")
        if not np.isfinite(minimum_response) or minimum_response < 0.0:
            raise ValueError("minimum integrated response must be finite and non-negative")
        if not np.isfinite(angle_threshold) or not 0.0 <= angle_threshold <= 1.0:
            raise ValueError("angle anisotropy threshold must lie between zero and one")
        if not isinstance(self.integration_support, ObservableIntegrationSupport):
            raise TypeError("integration support has the wrong type")
        if not isinstance(self.support_flags, ObservableSupportFlags):
            raise TypeError("observable support flags have the wrong type")

        object.__setattr__(self, "integrated_response", integrated_response)
        object.__setattr__(self, "minimum_integrated_response", minimum_response)
        object.__setattr__(self, "angle_anisotropy_threshold", angle_threshold)
        if self.centroid_m is not None:
            centroid = np.asarray(self.centroid_m, dtype=float)
            if centroid.shape != (2,):
                raise ValueError("centroid must have shape (2,)")
            if np.any(~np.isfinite(centroid)):
                raise ValueError("supported centroid must be finite")
            object.__setattr__(
                self,
                "centroid_m",
                _read_only_array(centroid, dtype=float),
            )
        if self.covariance_m2 is not None:
            covariance = np.asarray(self.covariance_m2, dtype=float)
            if covariance.shape != (2, 2):
                raise ValueError("covariance tensor must have shape (2, 2)")
            if np.any(~np.isfinite(covariance)):
                raise ValueError("supported covariance must be finite")
            if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0.0):
                raise ValueError("covariance tensor must be symmetric")
            object.__setattr__(
                self,
                "covariance_m2",
                _read_only_array(covariance, dtype=float),
            )

        flags = self.support_flags
        positive_response = integrated_response > 0.0
        moments_supported = integrated_response > minimum_response
        if flags.positive_integrated_response != positive_response:
            raise ValueError("positive-response flag is inconsistent with the integral")
        if flags.moments_numerically_supported != moments_supported:
            raise ValueError("moment-support flag is inconsistent with the integral threshold")
        if flags.centroid_supported != (self.centroid_m is not None):
            raise ValueError("centroid support flag is inconsistent with the centroid")
        if flags.covariance_supported != (self.covariance_m2 is not None):
            raise ValueError("covariance support flag is inconsistent with the covariance")
        widths_present = (
            self.major_rms_width_m is not None and self.minor_rms_width_m is not None
        )
        if flags.widths_supported != widths_present:
            raise ValueError("width support flag is inconsistent with the rms widths")
        if flags.aspect_ratio_supported != (self.aspect_ratio is not None):
            raise ValueError("aspect-ratio support flag is inconsistent with the value")
        if flags.principal_axis_angle_supported != (
            self.principal_axis_angle_rad is not None
        ):
            raise ValueError("angle support flag is inconsistent with the angle")
        for name, value in (
            ("major rms width", self.major_rms_width_m),
            ("minor rms width", self.minor_rms_width_m),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if widths_present and self.major_rms_width_m < self.minor_rms_width_m:
            raise ValueError("major rms width cannot be smaller than minor rms width")
        if self.aspect_ratio is not None and (
            not np.isfinite(self.aspect_ratio) or self.aspect_ratio < 1.0
        ):
            raise ValueError("supported aspect ratio must be finite and at least one")
        if self.fractional_anisotropy is not None and (
            not np.isfinite(self.fractional_anisotropy)
            or not 0.0 <= self.fractional_anisotropy <= 1.0
        ):
            raise ValueError("fractional anisotropy must lie between zero and one")
        if self.principal_axis_angle_rad is not None and (
            not np.isfinite(self.principal_axis_angle_rad)
            or not -0.5 * np.pi <= self.principal_axis_angle_rad < 0.5 * np.pi
        ):
            raise ValueError("principal-axis angle must lie in [-pi/2, pi/2)")

    @property
    def centroid_y_m(self) -> float | None:
        return None if self.centroid_m is None else float(self.centroid_m[0])

    @property
    def centroid_z_m(self) -> float | None:
        return None if self.centroid_m is None else float(self.centroid_m[1])

    @property
    def principal_axis_unit_vector_yz(self) -> NDArray[np.floating] | None:
        """Unit vector along the reported major axis, when it is supported."""

        if self.principal_axis_angle_rad is None:
            return None
        angle = self.principal_axis_angle_rad
        vector = np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
        vector.setflags(write=False)
        return vector


@dataclass(frozen=True)
class RelativeSignalSummary:
    """Relative integrated response and inferred reconstructed depletion."""

    reference_integrated_response: float
    current_integrated_response: float
    integrated_response_ratio: float
    reconstructed_depletion: float
    assumptions: tuple[str, ...]

    @property
    def a_q_over_a_0(self) -> float:
        """Alias matching the mathematical ``A_q / A_0`` notation."""

        return self.integrated_response_ratio

    @property
    def l_rec(self) -> float:
        """Alias matching the mathematical ``L_rec(q)`` notation."""

        return self.reconstructed_depletion


def extract_density_observables(
    response_map: ArrayLike,
    integration_support: ObservableIntegrationSupport,
    *,
    angle_anisotropy_threshold: float = 0.05,
    minimum_integrated_response: float = 0.0,
) -> DensityObservableSummary:
    """Extract physical low-order moments on one declared support.

    ``response_map`` may be a calibrated column-density map or any
    non-negative map with a stable common response scale.  Every integral and
    moment is weighted by the declared physical cell areas.  Values outside
    ``support_mask`` are deliberately ignored, including their finiteness,
    because they are not part of the integration contract.

    ``fractional_anisotropy`` is ``(lambda_major - lambda_minor) /
    (lambda_major + lambda_minor)``.  The principal-axis angle is withheld
    when this value does not exceed ``angle_anisotropy_threshold``.
    """

    if not isinstance(integration_support, ObservableIntegrationSupport):
        raise TypeError("integration_support must be ObservableIntegrationSupport")
    threshold = float(angle_anisotropy_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("angle anisotropy threshold must lie between zero and one")
    minimum_response = float(minimum_integrated_response)
    if not np.isfinite(minimum_response) or minimum_response < 0.0:
        raise ValueError("minimum integrated response must be finite and non-negative")

    unconverted_density = np.asarray(response_map)
    if np.iscomplexobj(unconverted_density):
        raise ValueError("response map must be real-valued")
    density = np.asarray(unconverted_density, dtype=float)
    if density.shape != integration_support.shape:
        raise ValueError(
            "response map must match the declared observable integration grid"
        )
    mask = integration_support.support_mask
    supported_density = density[mask]
    if np.any(~np.isfinite(supported_density)) or np.any(supported_density < 0.0):
        raise ValueError("response map must be finite and non-negative on its support")
    weights = supported_density * integration_support.cell_area_m2[mask]
    integrated_response = float(np.sum(weights))
    if not np.isfinite(integrated_response):
        raise ValueError("area-weighted integrated response is not finite")

    if integrated_response <= minimum_response:
        return DensityObservableSummary(
            integrated_response=integrated_response,
            centroid_m=None,
            covariance_m2=None,
            major_rms_width_m=None,
            minor_rms_width_m=None,
            aspect_ratio=None,
            principal_axis_angle_rad=None,
            fractional_anisotropy=None,
            angle_anisotropy_threshold=threshold,
            minimum_integrated_response=minimum_response,
            support_flags=ObservableSupportFlags(
                positive_integrated_response=integrated_response > 0.0,
                moments_numerically_supported=False,
                centroid_supported=False,
                covariance_supported=False,
                widths_supported=False,
                aspect_ratio_supported=False,
                principal_axis_angle_supported=False,
                reasons=("blank_or_below_minimum_integrated_response",),
            ),
            integration_support=integration_support,
        )

    y_values_m = integration_support.y_grid_m[mask]
    z_values_m = integration_support.z_grid_m[mask]
    centroid = np.asarray(
        [
            np.dot(weights, y_values_m) / integrated_response,
            np.dot(weights, z_values_m) / integrated_response,
        ],
        dtype=float,
    )
    displacement_y = y_values_m - centroid[0]
    displacement_z = z_values_m - centroid[1]
    covariance = np.asarray(
        [
            [
                np.dot(weights, displacement_y * displacement_y),
                np.dot(weights, displacement_y * displacement_z),
            ],
            [
                np.dot(weights, displacement_y * displacement_z),
                np.dot(weights, displacement_z * displacement_z),
            ],
        ],
        dtype=float,
    ) / integrated_response
    covariance = 0.5 * (covariance + covariance.T)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    negative_tolerance = 64.0 * np.finfo(float).eps * largest_scale
    if float(np.min(eigenvalues)) < -negative_tolerance:
        raise RuntimeError("area-weighted covariance is unexpectedly non-positive")
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    minor_variance = float(eigenvalues[0])
    major_variance = float(eigenvalues[1])
    minor_width = float(np.sqrt(minor_variance))
    major_width = float(np.sqrt(major_variance))

    reasons: list[str] = []
    if minor_width > 0.0:
        aspect_ratio = major_width / minor_width
        aspect_ratio_supported = True
    else:
        aspect_ratio = None
        aspect_ratio_supported = False
        reasons.append("zero_minor_width")

    total_variance = major_variance + minor_variance
    if total_variance > 0.0:
        fractional_anisotropy = (major_variance - minor_variance) / total_variance
    else:
        fractional_anisotropy = None

    if fractional_anisotropy is None:
        angle = None
        reasons.append("zero_spatial_variance")
    elif fractional_anisotropy <= threshold:
        angle = None
        reasons.append("near_circular_below_angle_anisotropy_threshold")
    else:
        major_vector = eigenvectors[:, 1]
        raw_angle = float(np.arctan2(major_vector[1], major_vector[0]))
        angle = float((raw_angle + 0.5 * np.pi) % np.pi - 0.5 * np.pi)

    return DensityObservableSummary(
        integrated_response=integrated_response,
        centroid_m=centroid,
        covariance_m2=covariance,
        major_rms_width_m=major_width,
        minor_rms_width_m=minor_width,
        aspect_ratio=aspect_ratio,
        principal_axis_angle_rad=angle,
        fractional_anisotropy=fractional_anisotropy,
        angle_anisotropy_threshold=threshold,
        minimum_integrated_response=minimum_response,
        support_flags=ObservableSupportFlags(
            positive_integrated_response=True,
            moments_numerically_supported=True,
            centroid_supported=True,
            covariance_supported=True,
            widths_supported=True,
            aspect_ratio_supported=aspect_ratio_supported,
            principal_axis_angle_supported=angle is not None,
            reasons=tuple(reasons),
        ),
        integration_support=integration_support,
    )


def relative_signal_and_depletion(
    reference: DensityObservableSummary,
    current: DensityObservableSummary,
) -> RelativeSignalSummary:
    """Return ``A_q/A_0`` and ``1 - A_q/A_0`` on an identical support.

    The comparison intentionally requires exact equality of coordinate grids,
    physical cell areas and support masks.  This prevents grid or ROI drift
    from being misreported as physical depletion.  A blank current map is a
    valid zero signal; a blank or threshold-unsupported reference is not a
    valid denominator.
    """

    if not isinstance(reference, DensityObservableSummary) or not isinstance(
        current,
        DensityObservableSummary,
    ):
        raise TypeError("relative comparison requires density observable summaries")
    if not reference.integration_support.is_identical_to(
        current.integration_support
    ):
        raise ValueError(
            "relative signal requires identical coordinate grids, cell areas, "
            "and integration support masks"
        )
    if (
        not reference.support_flags.moments_numerically_supported
        or reference.integrated_response <= 0.0
    ):
        raise ValueError("reference integrated response must be numerically supported")
    ratio = current.integrated_response / reference.integrated_response
    return RelativeSignalSummary(
        reference_integrated_response=reference.integrated_response,
        current_integrated_response=current.integrated_response,
        integrated_response_ratio=float(ratio),
        reconstructed_depletion=float(1.0 - ratio),
        assumptions=(
            "identical coordinate grid, physical cell areas, and support mask",
            "stable common response scale, detector gain, and optical transfer",
            "integrated response represents the same physical component in both maps",
            "response-scale stability is assumed by this helper, not verified",
        ),
    )


__all__ = [
    "DensityObservableSummary",
    "ObservableIntegrationSupport",
    "ObservableSupportFlags",
    "RelativeSignalSummary",
    "extract_density_observables",
    "relative_signal_and_depletion",
]
