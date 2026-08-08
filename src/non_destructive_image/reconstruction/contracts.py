"""Explicit optical and detector contracts for reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..camera import (
    bin_to_camera_pixels,
    centered_camera_shape,
    resample_to_camera_pixels,
)


@dataclass(frozen=True)
class DetectorContract:
    """Camera quantities required by the Poisson-Gaussian count model."""

    photoelectrons_per_i0_pixel: float
    read_noise_electrons_per_pixel_per_readout: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.photoelectrons_per_i0_pixel):
            raise ValueError("photoelectron scale must be finite")
        if self.photoelectrons_per_i0_pixel <= 0:
            raise ValueError("photoelectron scale must be positive")
        if not np.isfinite(self.read_noise_electrons_per_pixel_per_readout):
            raise ValueError("read noise must be finite")
        if self.read_noise_electrons_per_pixel_per_readout < 0:
            raise ValueError("read noise cannot be negative")


@dataclass(frozen=True)
class ReconstructionGrid:
    """Object grid, pupil, camera sampler and fixed reconstruction ROI.

    Integer block sampling supplies ``bin_size``. Physical-pixel sampling sets
    it to ``None`` and supplies the object-plane camera pitch and output shape.
    Both modes expose the same camera-space contract to the operators.
    """

    y_grid_m: NDArray[np.floating]
    z_grid_m: NDArray[np.floating]
    pupil: NDArray[np.complexfloating]
    bin_size: int | None
    roi_mask: NDArray[np.bool_]
    camera_pixel_size_m: float | None = None
    camera_output_shape: tuple[int, int] | None = None

    @classmethod
    def from_arrays(
        cls,
        *,
        y_grid_m: ArrayLike,
        z_grid_m: ArrayLike,
        pupil: ArrayLike,
        bin_size: int | None,
        roi_mask: ArrayLike,
        camera_pixel_size_m: float | None = None,
        camera_output_shape: tuple[int, int] | None = None,
    ) -> "ReconstructionGrid":
        y = np.asarray(y_grid_m, dtype=float)
        z = np.asarray(z_grid_m, dtype=float)
        p = np.asarray(pupil, dtype=complex)
        roi = np.asarray(roi_mask, dtype=bool)
        if y.ndim != 2 or y.shape != z.shape or y.shape != p.shape:
            raise ValueError("coordinate grids and pupil must be same-shape 2D arrays")
        if bin_size is not None and camera_pixel_size_m is not None:
            raise ValueError("choose integer binning or physical-pixel sampling, not both")
        if bin_size is None and camera_pixel_size_m is None:
            raise ValueError("a camera sampling contract is required")

        if bin_size is not None:
            if bin_size <= 0:
                raise ValueError("bin_size must be positive")
            if camera_output_shape is not None:
                raise ValueError("camera_output_shape belongs to physical-pixel sampling")
            rows = (y.shape[0] // bin_size) * bin_size
            cols = (y.shape[1] // bin_size) * bin_size
            if rows == 0 or cols == 0:
                raise ValueError("bin_size is larger than the reconstruction grid")
            camera_shape = (rows // bin_size, cols // bin_size)
            resolved_pixel_size = None
            resolved_output_shape = None
        else:
            if not np.isfinite(camera_pixel_size_m) or camera_pixel_size_m <= 0:
                raise ValueError("camera_pixel_size_m must be finite and positive")
            spacing_y = np.diff(y[0, :])
            spacing_z = np.diff(z[:, 0])
            if spacing_y.size == 0 or spacing_z.size == 0:
                raise ValueError("physical-pixel sampling requires at least a 2x2 grid")
            object_spacing_m = float(np.mean(spacing_y))
            if (
                object_spacing_m <= 0
                or not np.allclose(spacing_y, object_spacing_m, rtol=1e-10, atol=0.0)
                or not np.allclose(spacing_z, object_spacing_m, rtol=1e-10, atol=0.0)
            ):
                raise ValueError("physical-pixel sampling requires a uniform square grid")
            centre = (y.shape[0] // 2, y.shape[1] // 2)
            if not np.isclose(y[centre], 0.0, rtol=0.0, atol=1e-12 * object_spacing_m):
                raise ValueError("physical-pixel sampling requires a grid centred on y=0")
            if not np.isclose(z[centre], 0.0, rtol=0.0, atol=1e-12 * object_spacing_m):
                raise ValueError("physical-pixel sampling requires a grid centred on z=0")
            camera_shape = (
                centered_camera_shape(y.shape, object_spacing_m, float(camera_pixel_size_m))
                if camera_output_shape is None
                else tuple(int(value) for value in camera_output_shape)
            )
            if len(camera_shape) != 2 or any(value <= 0 for value in camera_shape):
                raise ValueError("camera_output_shape must contain two positive dimensions")
            # Building the weights also verifies that the requested physical
            # camera area lies inside the numerical field.
            resample_to_camera_pixels(
                np.ones_like(y),
                object_spacing_m,
                float(camera_pixel_size_m),
                camera_shape,
            )
            resolved_pixel_size = float(camera_pixel_size_m)
            resolved_output_shape = camera_shape
        if roi.shape != camera_shape:
            raise ValueError(f"ROI mask must have camera shape {camera_shape}")
        if not np.any(roi):
            raise ValueError("ROI mask cannot be empty")
        return cls(
            y,
            z,
            p,
            None if bin_size is None else int(bin_size),
            roi,
            resolved_pixel_size,
            resolved_output_shape,
        )

    @property
    def camera_shape(self) -> tuple[int, int]:
        if self.bin_size is None:
            if self.camera_output_shape is None:
                raise RuntimeError("physical camera output shape was not resolved")
            return self.camera_output_shape
        rows = (self.y_grid_m.shape[0] // self.bin_size) * self.bin_size
        cols = (self.y_grid_m.shape[1] // self.bin_size) * self.bin_size
        return rows // self.bin_size, cols // self.bin_size

    @property
    def object_grid_spacing_m(self) -> float:
        """Uniform object-plane grid spacing used by the camera sampler."""

        return float(np.mean(np.diff(self.y_grid_m[0, :])))

    @property
    def sampling_mode(self) -> str:
        return "integer_block" if self.bin_size is not None else "physical_pixel"

    @property
    def roi_pixel_count(self) -> int:
        return int(np.count_nonzero(self.roi_mask))

    @property
    def camera_y_um(self) -> NDArray[np.floating]:
        if self.bin_size is None:
            if self.camera_pixel_size_m is None:
                raise RuntimeError("physical camera pitch was not resolved")
            columns = self.camera_shape[1]
            return (
                np.arange(columns, dtype=float) - (columns - 1) / 2
            ) * self.camera_pixel_size_m * 1e6
        return bin_to_camera_pixels(self.y_grid_m * 1e6, self.bin_size)[0, :]

    @property
    def camera_z_um(self) -> NDArray[np.floating]:
        if self.bin_size is None:
            if self.camera_pixel_size_m is None:
                raise RuntimeError("physical camera pitch was not resolved")
            rows = self.camera_shape[0]
            return (
                np.arange(rows, dtype=float) - (rows - 1) / 2
            ) * self.camera_pixel_size_m * 1e6
        return bin_to_camera_pixels(self.z_grid_m * 1e6, self.bin_size)[:, 0]

    def camera_average(self, image: ArrayLike) -> NDArray:
        """Average one object-grid image over the configured camera pixels."""

        array = np.asarray(image)
        if array.shape != self.y_grid_m.shape:
            raise ValueError(f"camera input must have shape {self.y_grid_m.shape}")
        if self.bin_size is not None:
            return bin_to_camera_pixels(array, self.bin_size)
        if self.camera_pixel_size_m is None:
            raise RuntimeError("physical camera pitch was not resolved")
        return resample_to_camera_pixels(
            array,
            self.object_grid_spacing_m,
            self.camera_pixel_size_m,
            self.camera_shape,
        )

    def camera_average_stack(self, images: ArrayLike) -> NDArray:
        """Average an array whose final two axes are object-grid images."""

        array = np.asarray(images)
        if array.ndim < 2 or array.shape[-2:] != self.y_grid_m.shape:
            raise ValueError(
                "camera stack input must end with object-grid shape "
                f"{self.y_grid_m.shape}"
            )
        leading = array.shape[:-2]
        if self.bin_size is not None:
            rows = (array.shape[-2] // self.bin_size) * self.bin_size
            cols = (array.shape[-1] // self.bin_size) * self.bin_size
            trimmed = array[..., :rows, :cols]
            return trimmed.reshape(
                *leading,
                rows // self.bin_size,
                self.bin_size,
                cols // self.bin_size,
                self.bin_size,
            ).mean(axis=(-3, -1))
        flattened = array.reshape((-1, *array.shape[-2:]))
        sampled = np.stack([self.camera_average(image) for image in flattened], axis=0)
        return sampled.reshape((*leading, *self.camera_shape))
