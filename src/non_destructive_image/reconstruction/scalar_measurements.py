"""Linked raw-count PCI and DGI operators for scalar dispersive imaging.

The operators in this module keep atom-bearing frames and shared reference or
dark blocks in one ordered likelihood vector.  A shared block is represented
once, irrespective of the number of atom-bearing frames that consume it.  The
column-density maps remain frame-specific nuisance objects, whereas incident
fluence and dark offsets are explicit sequence-level nuisance quantities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft as scipy_fft

from .contracts import DetectorContract, ReconstructionGrid
from .object_models import DifferentiableColumnDensityModel


FloatArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _immutable(array: ArrayLike, *, dtype: type) -> NDArray:
    source = np.asarray(array, dtype=dtype)
    result = np.frombuffer(source.tobytes(order="C"), dtype=np.dtype(dtype))
    return result.reshape(source.shape)


@dataclass(frozen=True)
class ScalarOpticalResponseContract:
    """Linear column-density coefficients of one complex scalar transmission."""

    phase_per_column_density_rad_m2: float
    optical_depth_per_column_density_m2: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase_per_column_density_rad_m2",
            _finite(
                self.phase_per_column_density_rad_m2,
                "phase_per_column_density_rad_m2",
            ),
        )
        object.__setattr__(
            self,
            "optical_depth_per_column_density_m2",
            _nonnegative(
                self.optical_depth_per_column_density_m2,
                "optical_depth_per_column_density_m2",
            ),
        )

    @property
    def complex_exponent_per_column_density_m2(self) -> complex:
        """Return ``-OD/2 + i*phi`` per unit column density."""

        return complex(
            -0.5 * self.optical_depth_per_column_density_m2,
            self.phase_per_column_density_rad_m2,
        )


@dataclass(frozen=True)
class PCITransferContract:
    """Fixed PCI phase-plate field transfer used by the forward operator."""

    phase_plate_transmittance: float
    phase_plate_phase_rad: float

    def __post_init__(self) -> None:
        transmittance = _finite(
            self.phase_plate_transmittance,
            "phase_plate_transmittance",
        )
        if not 0.0 <= transmittance <= 1.0:
            raise ValueError("phase_plate_transmittance must lie in [0, 1]")
        object.__setattr__(self, "phase_plate_transmittance", transmittance)
        object.__setattr__(
            self,
            "phase_plate_phase_rad",
            _finite(self.phase_plate_phase_rad, "phase_plate_phase_rad"),
        )

    @property
    def carrier_field(self) -> complex:
        """Return the atom-free reference field after the phase plate."""

        return self.phase_plate_transmittance * np.exp(
            1j * self.phase_plate_phase_rad
        )


@dataclass(frozen=True)
class DGITransferContract:
    """Fixed Fourier-stop field transfer used by the DGI forward operator."""

    stop_optical_depth: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stop_optical_depth",
            _nonnegative(self.stop_optical_depth, "stop_optical_depth"),
        )

    @property
    def carrier_field(self) -> float:
        """Return the atom-free stopped-carrier field amplitude."""

        return float(10.0 ** (-self.stop_optical_depth / 2.0))


@dataclass(frozen=True)
class PCINuisanceValues:
    """Sequence-level PCI illumination and detector-background quantities."""

    i0_photoelectrons_per_pixel: float
    dark_electrons_per_pixel: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "i0_photoelectrons_per_pixel",
            _positive(
                self.i0_photoelectrons_per_pixel,
                "i0_photoelectrons_per_pixel",
            ),
        )
        object.__setattr__(
            self,
            "dark_electrons_per_pixel",
            _nonnegative(self.dark_electrons_per_pixel, "dark_electrons_per_pixel"),
        )


@dataclass(frozen=True)
class DGINuisanceValues:
    """Sequence-level DGI illumination, background and stop-scale quantities."""

    i0_photoelectrons_per_pixel: float
    stop_dark_electrons_per_pixel: float = 0.0
    open_dark_electrons_per_pixel: float = 0.0
    open_to_stop_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "i0_photoelectrons_per_pixel",
            _positive(
                self.i0_photoelectrons_per_pixel,
                "i0_photoelectrons_per_pixel",
            ),
        )
        object.__setattr__(
            self,
            "stop_dark_electrons_per_pixel",
            _nonnegative(
                self.stop_dark_electrons_per_pixel,
                "stop_dark_electrons_per_pixel",
            ),
        )
        object.__setattr__(
            self,
            "open_dark_electrons_per_pixel",
            _nonnegative(
                self.open_dark_electrons_per_pixel,
                "open_dark_electrons_per_pixel",
            ),
        )
        object.__setattr__(
            self,
            "open_to_stop_scale",
            _positive(self.open_to_stop_scale, "open_to_stop_scale"),
        )


@dataclass(frozen=True, eq=False)
class LinkedRawSequencePrediction:
    """One linked raw-role prediction and its density/nuisance Jacobian."""

    role_names: tuple[str, ...]
    role_frame_indices: tuple[int | None, ...]
    shared_role_names: tuple[str, ...]
    expected_electrons: tuple[FloatArray, ...]
    conditional_variance_electrons2: tuple[FloatArray, ...]
    prediction_vector: FloatArray
    jacobian: FloatArray
    density_parameter_slices: tuple[slice, ...]
    nuisance_names: tuple[str, ...]

    def __post_init__(self) -> None:
        role_count = len(self.role_names)
        if role_count == 0:
            raise ValueError("linked raw prediction must contain at least one role")
        if len(set(self.role_names)) != role_count:
            raise ValueError("linked raw role names must be unique")
        if not (
            len(self.role_frame_indices)
            == len(self.expected_electrons)
            == len(self.conditional_variance_electrons2)
            == role_count
        ):
            raise ValueError("linked raw role metadata lengths do not match")
        if not set(self.shared_role_names).issubset(self.role_names):
            raise ValueError("shared raw roles must be present in the role vector")
        for name, frame_index in zip(
            self.role_names,
            self.role_frame_indices,
            strict=True,
        ):
            if frame_index is None and name not in self.shared_role_names:
                raise ValueError("a shared role must be named in shared_role_names")
            if frame_index is not None and (
                isinstance(frame_index, bool) or frame_index < 0
            ):
                raise ValueError("frame indices must be non-negative integers or None")
        expected = tuple(_immutable(value, dtype=float) for value in self.expected_electrons)
        variances = tuple(
            _immutable(value, dtype=float)
            for value in self.conditional_variance_electrons2
        )
        if any(array.shape != expected[0].shape for array in expected + variances):
            raise ValueError("all linked raw role arrays must share one camera shape")
        if any(np.any(~np.isfinite(array)) or np.any(array < 0.0) for array in expected):
            raise ValueError("expected linked raw counts must be finite and non-negative")
        if any(np.any(~np.isfinite(array)) or np.any(array <= 0.0) for array in variances):
            raise ValueError("linked raw conditional variances must be finite and positive")
        prediction = np.asarray(self.prediction_vector, dtype=float)
        jacobian = np.asarray(self.jacobian, dtype=float)
        if prediction.ndim != 1 or np.any(~np.isfinite(prediction)):
            raise ValueError("prediction_vector must be a finite one-dimensional array")
        if jacobian.ndim != 2 or jacobian.shape[0] != prediction.size:
            raise ValueError("linked raw Jacobian row count must match the prediction")
        if np.any(~np.isfinite(jacobian)):
            raise ValueError("linked raw Jacobian must be finite")
        if jacobian.shape[1] != (
            sum(item.stop - item.start for item in self.density_parameter_slices)
            + len(self.nuisance_names)
        ):
            raise ValueError("linked raw Jacobian column contract is inconsistent")
        object.__setattr__(self, "expected_electrons", expected)
        object.__setattr__(self, "conditional_variance_electrons2", variances)
        object.__setattr__(
            self,
            "prediction_vector",
            _immutable(prediction, dtype=float),
        )
        object.__setattr__(self, "jacobian", _immutable(jacobian, dtype=float))


class _LinkedScalarRawOperator:
    """Common coherent propagation and linked-role assembly."""

    base_role_names: ClassVar[tuple[str, ...]]
    shared_role_names: ClassVar[tuple[str, ...]]
    nuisance_names: ClassVar[tuple[str, ...]]

    def __init__(
        self,
        *,
        grid: ReconstructionGrid,
        detector: DetectorContract,
        response: ScalarOpticalResponseContract,
        independent_exposures_by_role: Mapping[str, int],
        jacobian_batch_size: int = 4,
    ) -> None:
        if jacobian_batch_size <= 0:
            raise ValueError("jacobian_batch_size must be positive")
        if set(independent_exposures_by_role) != set(self.base_role_names):
            raise ValueError("independent exposure roles do not match the operator")
        exposures: dict[str, int] = {}
        for name in self.base_role_names:
            value = independent_exposures_by_role[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("independent exposure counts must be positive integers")
            exposures[name] = value
        self.grid = grid
        self.detector = detector
        self.response = response
        self.independent_exposures_by_role = MappingProxyType(exposures)
        self.jacobian_batch_size = int(jacobian_batch_size)

    @property
    def read_noise_electrons(self) -> float:
        """Return the per-readout detector noise in electrons rms."""

        return self.detector.read_noise_electrons_per_pixel_per_readout

    def _validate_density(self, column_density_m2: ArrayLike) -> FloatArray:
        density = np.asarray(column_density_m2, dtype=float)
        if density.shape != self.grid.y_grid_m.shape:
            raise ValueError(
                "column density must have reconstruction-grid shape "
                f"{self.grid.y_grid_m.shape}"
            )
        if np.any(~np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("column density must be finite and non-negative")
        return density

    def _validate_derivatives(self, derivatives_m2: ArrayLike) -> FloatArray:
        derivatives = np.asarray(derivatives_m2, dtype=float)
        if derivatives.ndim != 3 or derivatives.shape[1:] != self.grid.y_grid_m.shape:
            raise ValueError(
                "density derivatives must have shape "
                f"(n_parameter, {self.grid.y_grid_m.shape[0]}, "
                f"{self.grid.y_grid_m.shape[1]})"
            )
        if np.any(~np.isfinite(derivatives)):
            raise ValueError("density derivatives must be finite")
        return derivatives

    def _image_field(self, density: FloatArray, carrier_field: complex) -> ComplexArray:
        exponent = self.response.complex_exponent_per_column_density_m2
        object_field = np.exp(exponent * density)
        propagated = scipy_fft.ifft2(
            scipy_fft.fft2(object_field - 1.0, workers=-1) * self.grid.pupil,
            workers=-1,
        )
        return np.asarray(carrier_field + propagated, dtype=complex)

    def _camera_intensity(self, density: FloatArray, carrier_field: complex) -> FloatArray:
        field = self._image_field(density, carrier_field)
        return np.asarray(self.grid.camera_average(np.abs(field) ** 2), dtype=float)

    def _camera_intensity_derivatives(
        self,
        density: FloatArray,
        derivatives_m2: FloatArray,
        carrier_field: complex,
    ) -> FloatArray:
        derivatives = self._validate_derivatives(derivatives_m2)
        exponent = self.response.complex_exponent_per_column_density_m2
        object_field = np.exp(exponent * density)
        field = self._image_field(density, carrier_field)
        object_derivatives = exponent * object_field[None, ...] * derivatives
        propagated = scipy_fft.ifft2(
            scipy_fft.fft2(object_derivatives, axes=(-2, -1), workers=-1)
            * self.grid.pupil[None, ...],
            axes=(-2, -1),
            workers=-1,
        )
        intensity_derivatives = 2.0 * np.real(np.conj(field)[None, ...] * propagated)
        return np.asarray(
            self.grid.camera_average_stack(intensity_derivatives),
            dtype=float,
        )

    def _conditional_variance(self, expected: FloatArray, base_role: str) -> FloatArray:
        exposures = self.independent_exposures_by_role[base_role]
        return np.asarray(
            (expected + self.read_noise_electrons**2) / exposures,
            dtype=float,
        )

    def _flatten_roles(self, roles: Sequence[FloatArray]) -> FloatArray:
        return np.concatenate(
            [np.asarray(role, dtype=float)[self.grid.roi_mask] for role in roles]
        )

    def _density_blocks(
        self,
        model: DifferentiableColumnDensityModel,
        parameter_vectors: Sequence[ArrayLike],
        *,
        carrier_field: complex,
        density_count_scale: float,
        role_count: int,
    ) -> tuple[list[FloatArray], list[FloatArray], tuple[slice, ...]]:
        if len(parameter_vectors) == 0:
            raise ValueError("a linked sequence must contain at least one density frame")
        frame_densities: list[FloatArray] = []
        frame_jacobians: list[FloatArray] = []
        slices: list[slice] = []
        cursor = 0
        roi_pixels = self.grid.roi_pixel_count
        for vector in parameter_vectors:
            density = self._validate_density(model.column_density(vector))
            frame_densities.append(density)
            image_jacobian = np.empty(
                (roi_pixels, model.parameter_count),
                dtype=float,
            )
            populated = np.zeros(model.parameter_count, dtype=bool)
            for parameter_slice, raw_derivatives in model.iter_column_density_jacobian(
                vector,
                self.jacobian_batch_size,
            ):
                if parameter_slice.start is None or parameter_slice.stop is None:
                    raise ValueError("Jacobian batch slices must have explicit bounds")
                if (
                    parameter_slice.start < 0
                    or parameter_slice.stop > model.parameter_count
                    or parameter_slice.start >= parameter_slice.stop
                    or np.any(populated[parameter_slice])
                ):
                    raise ValueError("invalid or overlapping Jacobian batch slice")
                derivatives = self._camera_intensity_derivatives(
                    density,
                    raw_derivatives,
                    carrier_field,
                )
                image_jacobian[:, parameter_slice] = (
                    density_count_scale * derivatives[:, self.grid.roi_mask]
                ).T
                populated[parameter_slice] = True
            if not np.all(populated):
                raise ValueError("Jacobian batches do not cover every density parameter")
            frame_jacobians.append(image_jacobian)
            slices.append(slice(cursor, cursor + model.parameter_count))
            cursor += model.parameter_count
        expected_rows = role_count * roi_pixels
        density_jacobian = np.zeros((expected_rows, cursor), dtype=float)
        for frame_index, (parameter_slice, block) in enumerate(
            zip(slices, frame_jacobians, strict=True)
        ):
            row_slice = slice(frame_index * roi_pixels, (frame_index + 1) * roi_pixels)
            density_jacobian[row_slice, parameter_slice] = block
        return frame_densities, [density_jacobian], tuple(slices)


class PCILinkedRawOperator(_LinkedScalarRawOperator):
    """PCI raw-count operator with one shared bright reference and dark block."""

    base_role_names = ("atom", "bright_reference", "dark")
    shared_role_names = ("bright_reference", "dark")
    nuisance_names = ("i0_photoelectrons_per_pixel", "dark_electrons_per_pixel")

    def __init__(
        self,
        *,
        grid: ReconstructionGrid,
        detector: DetectorContract,
        response: ScalarOpticalResponseContract,
        transfer: PCITransferContract,
        independent_exposures_by_role: Mapping[str, int],
        jacobian_batch_size: int = 4,
    ) -> None:
        super().__init__(
            grid=grid,
            detector=detector,
            response=response,
            independent_exposures_by_role=independent_exposures_by_role,
            jacobian_batch_size=jacobian_batch_size,
        )
        self.transfer = transfer

    def expected_linked_sequence_from_density_maps(
        self,
        column_density_maps_m2: Sequence[ArrayLike],
        nuisance: PCINuisanceValues,
    ) -> tuple[tuple[str, ...], tuple[FloatArray, ...]]:
        """Return fresh atom roles followed by one shared bright/dark pair."""

        if not column_density_maps_m2:
            raise ValueError("a linked PCI sequence must contain at least one frame")
        shape = self.grid.camera_shape
        roles: list[FloatArray] = []
        names: list[str] = []
        for frame_index, raw_density in enumerate(column_density_maps_m2):
            density = self._validate_density(raw_density)
            image = self._camera_intensity(density, self.transfer.carrier_field)
            roles.append(
                nuisance.i0_photoelectrons_per_pixel * image
                + nuisance.dark_electrons_per_pixel
            )
            names.append(f"atom_{frame_index:03d}")
        roles.extend(
            [
                np.full(
                    shape,
                    nuisance.i0_photoelectrons_per_pixel
                    * self.transfer.phase_plate_transmittance**2
                    + nuisance.dark_electrons_per_pixel,
                    dtype=float,
                ),
                np.full(shape, nuisance.dark_electrons_per_pixel, dtype=float),
            ]
        )
        names.extend(self.shared_role_names)
        return tuple(names), tuple(np.asarray(role, dtype=float) for role in roles)

    def expected_linked_sequence_and_jacobian_model(
        self,
        model: DifferentiableColumnDensityModel,
        parameter_vectors: Sequence[ArrayLike],
        nuisance: PCINuisanceValues,
    ) -> LinkedRawSequencePrediction:
        """Return a linked PCI sequence and derivatives for inverse fitting."""

        density_maps = [self._validate_density(model.column_density(v)) for v in parameter_vectors]
        role_names, expected = self.expected_linked_sequence_from_density_maps(
            density_maps,
            nuisance,
        )
        frame_count = len(density_maps)
        role_count = frame_count + len(self.shared_role_names)
        _, density_jacobians, slices = self._density_blocks(
            model,
            parameter_vectors,
            carrier_field=self.transfer.carrier_field,
            density_count_scale=nuisance.i0_photoelectrons_per_pixel,
            role_count=role_count,
        )
        density_jacobian = density_jacobians[0]
        roi_pixels = self.grid.roi_pixel_count
        nuisance_jacobian = np.zeros(
            (role_count * roi_pixels, len(self.nuisance_names)),
            dtype=float,
        )
        for frame_index, density in enumerate(density_maps):
            image = self._camera_intensity(density, self.transfer.carrier_field)
            rows = slice(frame_index * roi_pixels, (frame_index + 1) * roi_pixels)
            nuisance_jacobian[rows, 0] = image[self.grid.roi_mask]
            nuisance_jacobian[rows, 1] = 1.0
        reference_rows = slice(frame_count * roi_pixels, (frame_count + 1) * roi_pixels)
        dark_rows = slice((frame_count + 1) * roi_pixels, role_count * roi_pixels)
        nuisance_jacobian[reference_rows, 0] = self.transfer.phase_plate_transmittance**2
        nuisance_jacobian[reference_rows, 1] = 1.0
        nuisance_jacobian[dark_rows, 1] = 1.0
        variances = tuple(
            self._conditional_variance(
                role,
                "atom" if index < frame_count else self.shared_role_names[index - frame_count],
            )
            for index, role in enumerate(expected)
        )
        return LinkedRawSequencePrediction(
            role_names=role_names,
            role_frame_indices=tuple(range(frame_count)) + (None, None),
            shared_role_names=self.shared_role_names,
            expected_electrons=expected,
            conditional_variance_electrons2=variances,
            prediction_vector=self._flatten_roles(expected),
            jacobian=np.hstack([density_jacobian, nuisance_jacobian]),
            density_parameter_slices=slices,
            nuisance_names=self.nuisance_names,
        )


class DGILinkedRawOperator(_LinkedScalarRawOperator):
    """DGI raw-count operator with one shared stopped/open reference block."""

    base_role_names = (
        "atom_stop",
        "leakage_stop",
        "stop_dark",
        "open_reference",
        "open_dark",
    )
    shared_role_names = (
        "leakage_stop",
        "stop_dark",
        "open_reference",
        "open_dark",
    )
    nuisance_names = (
        "i0_photoelectrons_per_pixel",
        "stop_dark_electrons_per_pixel",
        "open_dark_electrons_per_pixel",
        "open_to_stop_scale",
    )

    def __init__(
        self,
        *,
        grid: ReconstructionGrid,
        detector: DetectorContract,
        response: ScalarOpticalResponseContract,
        transfer: DGITransferContract,
        independent_exposures_by_role: Mapping[str, int],
        jacobian_batch_size: int = 4,
    ) -> None:
        super().__init__(
            grid=grid,
            detector=detector,
            response=response,
            independent_exposures_by_role=independent_exposures_by_role,
            jacobian_batch_size=jacobian_batch_size,
        )
        self.transfer = transfer

    def expected_linked_sequence_from_density_maps(
        self,
        column_density_maps_m2: Sequence[ArrayLike],
        nuisance: DGINuisanceValues,
    ) -> tuple[tuple[str, ...], tuple[FloatArray, ...]]:
        """Return fresh stopped-atom roles followed by one shared reference block."""

        if not column_density_maps_m2:
            raise ValueError("a linked DGI sequence must contain at least one frame")
        shape = self.grid.camera_shape
        stop_i0 = (
            nuisance.i0_photoelectrons_per_pixel * nuisance.open_to_stop_scale
        )
        roles: list[FloatArray] = []
        names: list[str] = []
        for frame_index, raw_density in enumerate(column_density_maps_m2):
            density = self._validate_density(raw_density)
            image = self._camera_intensity(density, self.transfer.carrier_field)
            roles.append(stop_i0 * image + nuisance.stop_dark_electrons_per_pixel)
            names.append(f"atom_stop_{frame_index:03d}")
        leakage_intensity = self.transfer.carrier_field**2
        roles.extend(
            [
                np.full(
                    shape,
                    stop_i0 * leakage_intensity
                    + nuisance.stop_dark_electrons_per_pixel,
                    dtype=float,
                ),
                np.full(
                    shape,
                    nuisance.stop_dark_electrons_per_pixel,
                    dtype=float,
                ),
                np.full(
                    shape,
                    nuisance.i0_photoelectrons_per_pixel
                    + nuisance.open_dark_electrons_per_pixel,
                    dtype=float,
                ),
                np.full(
                    shape,
                    nuisance.open_dark_electrons_per_pixel,
                    dtype=float,
                ),
            ]
        )
        names.extend(self.shared_role_names)
        return tuple(names), tuple(np.asarray(role, dtype=float) for role in roles)

    def expected_linked_sequence_and_jacobian_model(
        self,
        model: DifferentiableColumnDensityModel,
        parameter_vectors: Sequence[ArrayLike],
        nuisance: DGINuisanceValues,
    ) -> LinkedRawSequencePrediction:
        """Return a linked DGI sequence and derivatives for inverse fitting."""

        density_maps = [self._validate_density(model.column_density(v)) for v in parameter_vectors]
        role_names, expected = self.expected_linked_sequence_from_density_maps(
            density_maps,
            nuisance,
        )
        frame_count = len(density_maps)
        role_count = frame_count + len(self.shared_role_names)
        stop_i0 = (
            nuisance.i0_photoelectrons_per_pixel * nuisance.open_to_stop_scale
        )
        _, density_jacobians, slices = self._density_blocks(
            model,
            parameter_vectors,
            carrier_field=self.transfer.carrier_field,
            density_count_scale=stop_i0,
            role_count=role_count,
        )
        density_jacobian = density_jacobians[0]
        roi_pixels = self.grid.roi_pixel_count
        nuisance_jacobian = np.zeros(
            (role_count * roi_pixels, len(self.nuisance_names)),
            dtype=float,
        )
        for frame_index, density in enumerate(density_maps):
            image = self._camera_intensity(density, self.transfer.carrier_field)
            rows = slice(frame_index * roi_pixels, (frame_index + 1) * roi_pixels)
            nuisance_jacobian[rows, 0] = (
                nuisance.open_to_stop_scale * image[self.grid.roi_mask]
            )
            nuisance_jacobian[rows, 1] = 1.0
            nuisance_jacobian[rows, 3] = (
                nuisance.i0_photoelectrons_per_pixel * image[self.grid.roi_mask]
            )
        cursor = frame_count * roi_pixels
        leakage = self.transfer.carrier_field**2
        leakage_rows = slice(cursor, cursor + roi_pixels)
        nuisance_jacobian[leakage_rows, 0] = nuisance.open_to_stop_scale * leakage
        nuisance_jacobian[leakage_rows, 1] = 1.0
        nuisance_jacobian[leakage_rows, 3] = nuisance.i0_photoelectrons_per_pixel * leakage
        cursor += roi_pixels
        stop_dark_rows = slice(cursor, cursor + roi_pixels)
        nuisance_jacobian[stop_dark_rows, 1] = 1.0
        cursor += roi_pixels
        open_reference_rows = slice(cursor, cursor + roi_pixels)
        nuisance_jacobian[open_reference_rows, 0] = 1.0
        nuisance_jacobian[open_reference_rows, 2] = 1.0
        cursor += roi_pixels
        open_dark_rows = slice(cursor, cursor + roi_pixels)
        nuisance_jacobian[open_dark_rows, 2] = 1.0
        variances = tuple(
            self._conditional_variance(
                role,
                "atom_stop" if index < frame_count else self.shared_role_names[index - frame_count],
            )
            for index, role in enumerate(expected)
        )
        return LinkedRawSequencePrediction(
            role_names=role_names,
            role_frame_indices=tuple(range(frame_count)) + (None, None, None, None),
            shared_role_names=self.shared_role_names,
            expected_electrons=expected,
            conditional_variance_electrons2=variances,
            prediction_vector=self._flatten_roles(expected),
            jacobian=np.hstack([density_jacobian, nuisance_jacobian]),
            density_parameter_slices=slices,
            nuisance_names=self.nuisance_names,
        )


__all__ = [
    "DGILinkedRawOperator",
    "DGINuisanceValues",
    "DGITransferContract",
    "LinkedRawSequencePrediction",
    "PCILinkedRawOperator",
    "PCINuisanceValues",
    "PCITransferContract",
    "ScalarOpticalResponseContract",
]
