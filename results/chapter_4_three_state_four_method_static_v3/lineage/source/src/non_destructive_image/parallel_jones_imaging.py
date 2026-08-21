"""Exact two-polarisation PCI/DGI imaging for the three-state geometry.

This module is a successor path.  It leaves the admitted scalar Oxford and
legacy three-state helpers unchanged while exposing the Jones field required
when the probe propagates parallel to the magnetic field.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .atomic_response import (
    parallel_jones_column_response,
    parallel_jones_optical_response_from_config,
)
from .camera import resample_to_camera_pixels
from .equilibrium_imaging import OpticalTransfer
from .imaging import simulate_dgi_jones_fields, simulate_pci_jones_fields


def _finite_positive(value: object, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return scalar


def _immutable(values: ArrayLike, *, complex_values: bool = False) -> NDArray:
    array = np.array(values, dtype=complex if complex_values else float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class MatchedJonesPCIImage:
    """Camera-sampled PCI image of one exact three-state Jones response."""

    optical_transfer: OpticalTransfer
    input_pixel_size_m: float
    camera_pixel_size_m: float
    branch_phase_maps_rad: NDArray[np.floating]
    branch_optical_depth_maps: NDArray[np.floating]
    common_phase_map_rad: NDArray[np.floating]
    faraday_rotation_map_rad: NDArray[np.floating]
    common_optical_depth_map: NDArray[np.floating]
    co_polarised_object_field: NDArray[np.complexfloating]
    faraday_orthogonal_object_field: NDArray[np.complexfloating]
    total_object_intensity_fraction: NDArray[np.floating]
    camera_y_m: NDArray[np.floating]
    camera_z_m: NDArray[np.floating]
    camera_intensity_over_i0: NDArray[np.floating]
    atom_free_intensity_over_i0: float
    pci_signal_over_i0: NDArray[np.floating]


@dataclass(frozen=True)
class MatchedJonesDGIImage:
    """DGI image of the same Jones response, transfer and camera grid as PCI."""

    optical_transfer: OpticalTransfer
    input_pixel_size_m: float
    camera_pixel_size_m: float
    branch_phase_maps_rad: NDArray[np.floating]
    branch_optical_depth_maps: NDArray[np.floating]
    common_phase_map_rad: NDArray[np.floating]
    faraday_rotation_map_rad: NDArray[np.floating]
    common_optical_depth_map: NDArray[np.floating]
    co_polarised_object_field: NDArray[np.complexfloating]
    faraday_orthogonal_object_field: NDArray[np.complexfloating]
    total_object_intensity_fraction: NDArray[np.floating]
    camera_y_m: NDArray[np.floating]
    camera_z_m: NDArray[np.floating]
    camera_intensity_over_i0: NDArray[np.floating]
    atom_free_intensity_over_i0: float
    dgi_signal_over_i0: NDArray[np.floating]
    stop_optical_depth: float


def _validated_object_grid(
    column_density_m2: ArrayLike,
    y_axis_m: ArrayLike,
    z_axis_m: ArrayLike,
    optical_transfer: OpticalTransfer,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    float,
]:
    density = np.asarray(column_density_m2, dtype=float)
    y_axis = np.asarray(y_axis_m, dtype=float)
    z_axis = np.asarray(z_axis_m, dtype=float)
    if density.shape != (z_axis.size, y_axis.size) or min(density.shape) < 2:
        raise ValueError("column density shape must match the supplied z and y axes")
    if not np.isfinite(density).all() or np.any(density < 0.0):
        raise ValueError("column_density_m2 must be finite and non-negative")
    if not np.isfinite(y_axis).all() or not np.isfinite(z_axis).all():
        raise ValueError("object axes must be finite")
    dy = np.diff(y_axis)
    dz = np.diff(z_axis)
    input_pixel = float(np.mean(dy))
    if (
        input_pixel <= 0.0
        or not np.allclose(dy, input_pixel, rtol=1e-12, atol=0.0)
        or not np.allclose(dz, input_pixel, rtol=1e-12, atol=0.0)
    ):
        raise ValueError("Jones imaging requires a uniform square object grid")
    if not isinstance(optical_transfer, OpticalTransfer):
        raise TypeError("optical_transfer must be an OpticalTransfer")
    if optical_transfer.transfer.shape != density.shape:
        raise ValueError("optical transfer shape must match the column density")
    return density, y_axis, z_axis, input_pixel


def simulate_matched_jones_pci_image(
    column_density_m2: ArrayLike,
    y_axis_m: ArrayLike,
    z_axis_m: ArrayLike,
    *,
    model_config: Mapping[str, Any],
    orientation_config: Mapping[str, Any],
    optical_transfer: OpticalTransfer,
    detuning_hz: float,
    camera_pixel_size_m: float,
    phase_plate_transmittance: float,
    phase_plate_phase_rad: float,
) -> MatchedJonesPCIImage:
    """Apply the exact parallel-geometry response, PCI transfer and sampler."""

    density, _, _, input_pixel = _validated_object_grid(
        column_density_m2,
        y_axis_m,
        z_axis_m,
        optical_transfer,
    )
    if not isinstance(model_config, Mapping) or not isinstance(
        orientation_config,
        Mapping,
    ):
        raise ValueError("model_config and orientation_config must be mappings")
    try:
        atom = model_config["atom"]
    except (KeyError, TypeError) as exc:
        raise ValueError("model_config is missing the atomic contract") from exc

    response_contract = parallel_jones_optical_response_from_config(orientation_config)
    response = parallel_jones_column_response(
        density,
        detuning_hz,
        atom["resonant_cross_section_m2"],
        atom["natural_linewidth_rad_s"],
        response_contract,
    )
    optical_intensity = simulate_pci_jones_fields(
        response.co_polarised_field,
        response.faraday_orthogonal_field,
        optical_transfer.transfer,
        phase_plate_transmittance,
        phase_plate_phase_rad,
    )
    camera_pixel = _finite_positive(camera_pixel_size_m, "camera_pixel_size_m")
    camera_intensity = resample_to_camera_pixels(
        optical_intensity,
        input_pixel,
        camera_pixel,
    )
    rows, columns = camera_intensity.shape
    camera_y = (np.arange(columns, dtype=float) - (columns - 1) / 2.0) * camera_pixel
    camera_z = (np.arange(rows, dtype=float) - (rows - 1) / 2.0) * camera_pixel
    transmittance = float(phase_plate_transmittance)
    if not np.isfinite(transmittance) or not 0.0 <= transmittance <= 1.0:
        raise ValueError("phase_plate_transmittance must lie in [0, 1]")
    background = transmittance**2
    return MatchedJonesPCIImage(
        optical_transfer=optical_transfer,
        input_pixel_size_m=input_pixel,
        camera_pixel_size_m=camera_pixel,
        branch_phase_maps_rad=_immutable(response.branch_phase_maps_rad),
        branch_optical_depth_maps=_immutable(response.branch_optical_depth_maps),
        common_phase_map_rad=_immutable(response.common_phase_map_rad),
        faraday_rotation_map_rad=_immutable(response.faraday_rotation_map_rad),
        common_optical_depth_map=_immutable(response.common_optical_depth_map),
        co_polarised_object_field=_immutable(
            response.co_polarised_field,
            complex_values=True,
        ),
        faraday_orthogonal_object_field=_immutable(
            response.faraday_orthogonal_field,
            complex_values=True,
        ),
        total_object_intensity_fraction=_immutable(response.total_intensity_fraction),
        camera_y_m=_immutable(camera_y),
        camera_z_m=_immutable(camera_z),
        camera_intensity_over_i0=_immutable(camera_intensity),
        atom_free_intensity_over_i0=background,
        pci_signal_over_i0=_immutable(camera_intensity - background),
    )


def simulate_matched_jones_dgi_image(
    matched_pci: MatchedJonesPCIImage,
    *,
    stop_optical_depth: float,
) -> MatchedJonesDGIImage:
    """Apply DGI to the exact Jones fields and transfer already used by PCI."""

    if not isinstance(matched_pci, MatchedJonesPCIImage):
        raise TypeError("matched_pci must be a MatchedJonesPCIImage")
    optical_depth = float(stop_optical_depth)
    if not np.isfinite(optical_depth) or optical_depth < 0.0:
        raise ValueError("stop_optical_depth must be finite and non-negative")
    optical_intensity = simulate_dgi_jones_fields(
        matched_pci.co_polarised_object_field,
        matched_pci.faraday_orthogonal_object_field,
        matched_pci.optical_transfer.transfer,
        optical_depth,
    )
    camera_intensity = resample_to_camera_pixels(
        optical_intensity,
        matched_pci.input_pixel_size_m,
        matched_pci.camera_pixel_size_m,
        matched_pci.camera_intensity_over_i0.shape,
    )
    if camera_intensity.shape != matched_pci.camera_intensity_over_i0.shape:
        raise RuntimeError("matched Jones PCI and DGI camera shapes diverged")
    background = float(10.0 ** (-optical_depth))
    return MatchedJonesDGIImage(
        optical_transfer=matched_pci.optical_transfer,
        input_pixel_size_m=matched_pci.input_pixel_size_m,
        camera_pixel_size_m=matched_pci.camera_pixel_size_m,
        branch_phase_maps_rad=matched_pci.branch_phase_maps_rad,
        branch_optical_depth_maps=matched_pci.branch_optical_depth_maps,
        common_phase_map_rad=matched_pci.common_phase_map_rad,
        faraday_rotation_map_rad=matched_pci.faraday_rotation_map_rad,
        common_optical_depth_map=matched_pci.common_optical_depth_map,
        co_polarised_object_field=matched_pci.co_polarised_object_field,
        faraday_orthogonal_object_field=matched_pci.faraday_orthogonal_object_field,
        total_object_intensity_fraction=matched_pci.total_object_intensity_fraction,
        camera_y_m=matched_pci.camera_y_m,
        camera_z_m=matched_pci.camera_z_m,
        camera_intensity_over_i0=_immutable(camera_intensity),
        atom_free_intensity_over_i0=background,
        dgi_signal_over_i0=_immutable(camera_intensity - background),
        stop_optical_depth=optical_depth,
    )


__all__ = [
    "MatchedJonesDGIImage",
    "MatchedJonesPCIImage",
    "simulate_matched_jones_dgi_image",
    "simulate_matched_jones_pci_image",
]
