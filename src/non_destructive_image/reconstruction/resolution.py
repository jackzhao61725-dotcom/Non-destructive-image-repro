"""Uniform object grids with integer-block or physical-pixel camera sampling."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .contracts import ReconstructionGrid


def _uniform_object_arrays(
    *,
    ngrid: int,
    field_of_view_m: float,
    numerical_aperture: float,
    wavelength_m: float,
    coordinate_shift_m: float = 0.0,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
]:
    """Build the shared object coordinates and circular pupil."""

    if ngrid <= 0:
        raise ValueError("ngrid must be positive")
    if not np.isfinite(field_of_view_m) or field_of_view_m <= 0:
        raise ValueError("field of view must be finite and positive")
    if not np.isfinite(numerical_aperture) or numerical_aperture <= 0:
        raise ValueError("numerical aperture must be finite and positive")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength must be finite and positive")
    if not np.isfinite(coordinate_shift_m):
        raise ValueError("coordinate shift must be finite")

    spacing_m = field_of_view_m / ngrid
    coordinate_axis_m = (
        (np.arange(ngrid, dtype=float) - ngrid // 2) * spacing_m
        + coordinate_shift_m
    )
    y_grid_m, z_grid_m = np.meshgrid(coordinate_axis_m, coordinate_axis_m)
    frequency_axis_m_inv = np.fft.fftfreq(ngrid, d=spacing_m)
    frequency_y_m_inv, frequency_z_m_inv = np.meshgrid(
        frequency_axis_m_inv,
        frequency_axis_m_inv,
    )
    pupil = (
        np.sqrt(frequency_y_m_inv**2 + frequency_z_m_inv**2)
        <= numerical_aperture / wavelength_m
    ).astype(float)
    return y_grid_m, z_grid_m, pupil


def build_uniform_physical_camera_grid(
    *,
    ngrid: int,
    field_of_view_m: float,
    camera_pixel_size_m: float,
    camera_output_shape: tuple[int, int],
    numerical_aperture: float,
    wavelength_m: float,
    roi_half_width_y_um: float | None = None,
    roi_half_width_z_um: float | None = None,
) -> ReconstructionGrid:
    """Build an object grid sampled by centred physical camera pixels."""

    if len(camera_output_shape) != 2 or any(value <= 0 for value in camera_output_shape):
        raise ValueError("camera_output_shape must contain two positive dimensions")
    if not np.isfinite(camera_pixel_size_m) or camera_pixel_size_m <= 0:
        raise ValueError("camera_pixel_size_m must be finite and positive")
    if (roi_half_width_y_um is None) != (roi_half_width_z_um is None):
        raise ValueError("both ROI half-widths must be supplied together")
    if roi_half_width_y_um is not None and (
        not np.isfinite(roi_half_width_y_um)
        or not np.isfinite(roi_half_width_z_um)
        or roi_half_width_y_um <= 0
        or roi_half_width_z_um <= 0
    ):
        raise ValueError("ROI half-widths must be finite and positive")

    y_grid_m, z_grid_m, pupil = _uniform_object_arrays(
        ngrid=ngrid,
        field_of_view_m=field_of_view_m,
        numerical_aperture=numerical_aperture,
        wavelength_m=wavelength_m,
    )
    if roi_half_width_y_um is None:
        roi_mask = np.ones(camera_output_shape, dtype=bool)
    else:
        camera_y_um = (
            np.arange(camera_output_shape[1], dtype=float)
            - (camera_output_shape[1] - 1) / 2
        ) * camera_pixel_size_m * 1e6
        camera_z_um = (
            np.arange(camera_output_shape[0], dtype=float)
            - (camera_output_shape[0] - 1) / 2
        ) * camera_pixel_size_m * 1e6
        camera_y_grid_um, camera_z_grid_um = np.meshgrid(camera_y_um, camera_z_um)
        roi_mask = (
            (np.abs(camera_y_grid_um) <= float(roi_half_width_y_um))
            & (np.abs(camera_z_grid_um) <= float(roi_half_width_z_um))
        )
    return ReconstructionGrid.from_arrays(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        pupil=pupil,
        bin_size=None,
        roi_mask=roi_mask,
        camera_pixel_size_m=camera_pixel_size_m,
        camera_output_shape=camera_output_shape,
    )
