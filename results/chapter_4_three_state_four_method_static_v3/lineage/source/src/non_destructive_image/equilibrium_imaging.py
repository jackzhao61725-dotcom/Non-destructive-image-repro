"""Matched PCI transfer for representative equilibrium column densities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .acquisition import paired_difference_diagnostic, simulate_intensity_frame
from .atomic_response import (
    complex_column_response,
    polarised_optical_response_from_config,
)
from .camera import resample_to_camera_pixels
from .equilibrium_profiles import MorphologyObservables, measure_morphology
from .imaging import (
    simulate_dgi_object_field,
    simulate_fourier_image,
    simulate_pci_object_field,
)
from .light_atom import intensity_at_atoms


OpticalCaseId = Literal["design", "measured_best", "alignment_sensitivity"]


def _finite_positive(value: object, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return scalar


def _immutable(values: ArrayLike, *, complex_values: bool = False) -> NDArray:
    dtype = complex if complex_values else float
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class OpticalTransfer:
    """One declared coherent optical-transfer case on the object grid."""

    case_id: OpticalCaseId
    model: str
    evidence_role: str
    transfer: NDArray[np.floating]
    numerical_aperture: float | None = None
    resolution_m: float | None = None
    gaussian_amplitude_sigma_m: float | None = None


@dataclass(frozen=True)
class MatchedPCIImage:
    """Object response and camera-sampled PCI result for one state and case."""

    optical_transfer: OpticalTransfer
    phase_map_rad: NDArray[np.floating]
    optical_depth_map: NDArray[np.floating]
    object_field: NDArray[np.complexfloating]
    camera_y_m: NDArray[np.floating]
    camera_z_m: NDArray[np.floating]
    camera_intensity_over_i0: NDArray[np.floating]
    atom_free_intensity_over_i0: float
    pci_signal_over_i0: NDArray[np.floating]


@dataclass(frozen=True)
class MatchedDGIImage:
    """DGI readout of the same complex object and camera grid as a PCI image."""

    optical_transfer: OpticalTransfer
    phase_map_rad: NDArray[np.floating]
    optical_depth_map: NDArray[np.floating]
    object_field: NDArray[np.complexfloating]
    camera_y_m: NDArray[np.floating]
    camera_z_m: NDArray[np.floating]
    camera_intensity_over_i0: NDArray[np.floating]
    atom_free_intensity_over_i0: float
    dgi_signal_over_i0: NDArray[np.floating]
    stop_optical_depth: float


@dataclass(frozen=True)
class CoherentIntensityDecomposition:
    """Exact reference--scattered-field intensity decomposition on two grids."""

    reference_field: complex
    propagated_scattered_field: NDArray[np.complexfloating]
    optical_grid_background_over_i0: NDArray[np.floating]
    optical_grid_interference_over_i0: NDArray[np.floating]
    optical_grid_self_over_i0: NDArray[np.floating]
    optical_grid_delta_intensity_over_i0: NDArray[np.floating]
    optical_grid_total_intensity_over_i0: NDArray[np.floating]
    camera_background_over_i0: NDArray[np.floating]
    camera_interference_over_i0: NDArray[np.floating]
    camera_self_over_i0: NDArray[np.floating]
    camera_delta_intensity_over_i0: NDArray[np.floating]
    camera_total_intensity_over_i0: NDArray[np.floating]
    optical_grid_closure_max_abs_error: float
    camera_grid_closure_max_abs_error: float


@dataclass(frozen=True)
class NoisyPCIImage:
    """One replayable atom-minus-reference PCI camera draw."""

    seed_components_prefix: tuple[int, ...]
    photoelectrons_per_i0_pixel: float
    read_noise_electrons_rms: float
    expected_signal_over_i0: NDArray[np.floating]
    observed_signal_over_i0: NDArray[np.floating]


def _frequency_grids(
    shape: tuple[int, int],
    input_pixel_size_m: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("shape must contain two dimensions >= 2")
    pixel = _finite_positive(input_pixel_size_m, "input_pixel_size_m")
    fy = np.fft.fftfreq(int(shape[1]), d=pixel)
    fz = np.fft.fftfreq(int(shape[0]), d=pixel)
    return np.meshgrid(fy, fz)


def hard_circular_coherent_transfer(
    shape: tuple[int, int],
    input_pixel_size_m: float,
    *,
    wavelength_m: float,
    numerical_aperture: float,
) -> NDArray[np.floating]:
    """Return an ideal circular coherent pupil with cutoff ``NA / wavelength``."""

    wavelength = _finite_positive(wavelength_m, "wavelength_m")
    na = _finite_positive(numerical_aperture, "numerical_aperture")
    if na > 1.0:
        raise ValueError("numerical_aperture must not exceed one")
    frequency_y, frequency_z = _frequency_grids(shape, input_pixel_size_m)
    transfer = (
        np.hypot(frequency_y, frequency_z) <= na / wavelength
    ).astype(float)
    return _immutable(transfer)


def gaussian_amplitude_psf_transfer(
    shape: tuple[int, int],
    input_pixel_size_m: float,
    *,
    resolution_m: float,
    resolution_to_sigma_factor: float = 2.9039,
) -> NDArray[np.floating]:
    """Return the coherent transfer of the report's Gaussian amplitude PSF.

    The source model is
    ``h_amp(r) = exp[-0.5 * (resolution_to_sigma_factor*r/R0)**2]``.
    Thus ``sigma = R0 / resolution_to_sigma_factor`` and the unit-DC Fourier
    transfer is ``exp[-2*pi**2*sigma**2*(fy**2+fz**2)]``.
    """

    resolution = _finite_positive(resolution_m, "resolution_m")
    factor = _finite_positive(
        resolution_to_sigma_factor,
        "resolution_to_sigma_factor",
    )
    frequency_y, frequency_z = _frequency_grids(shape, input_pixel_size_m)
    sigma_m = resolution / factor
    transfer = np.exp(
        -2.0 * np.pi**2 * sigma_m**2 * (frequency_y**2 + frequency_z**2)
    )
    return _immutable(transfer)


def optical_transfer_from_objective_config(
    objective_config: Mapping[str, Any],
    case_id: OpticalCaseId,
    shape: tuple[int, int],
    input_pixel_size_m: float,
) -> OpticalTransfer:
    """Build one of the three non-collapsed ErK objective transfer cases."""

    if not isinstance(objective_config, Mapping):
        raise ValueError("objective_config must be a mapping")
    if case_id not in {"design", "measured_best", "alignment_sensitivity"}:
        raise ValueError("unsupported optical case")
    try:
        wavelength_m = objective_config["system_identity"]["erbium_wavelength_m"]
        policy = objective_config["forward_model_policy"]
    except (KeyError, TypeError) as exc:
        raise ValueError("objective config is missing its forward-model contract") from exc

    if case_id == "design":
        record = policy["ideal_design_case"]
        na = _finite_positive(record["numerical_aperture"], "design numerical aperture")
        transfer = hard_circular_coherent_transfer(
            shape,
            input_pixel_size_m,
            wavelength_m=wavelength_m,
            numerical_aperture=na,
        )
        return OpticalTransfer(
            case_id=case_id,
            model=str(record["model"]),
            evidence_role=str(record["evidence_role"]),
            transfer=transfer,
            numerical_aperture=na,
        )

    key = (
        "measured_best_case"
        if case_id == "measured_best"
        else "alignment_sensitivity_case"
    )
    record = policy[key]
    resolution = _finite_positive(record["resolution_m"], f"{case_id} resolution_m")
    factor = 2.9039
    transfer = gaussian_amplitude_psf_transfer(
        shape,
        input_pixel_size_m,
        resolution_m=resolution,
        resolution_to_sigma_factor=factor,
    )
    return OpticalTransfer(
        case_id=case_id,
        model=str(record["model"]),
        evidence_role=str(record["evidence_role"]),
        transfer=transfer,
        resolution_m=resolution,
        gaussian_amplitude_sigma_m=resolution / factor,
    )


def simulate_matched_pci_image(
    column_density_m2: ArrayLike,
    y_axis_m: ArrayLike,
    z_axis_m: ArrayLike,
    *,
    model_config: Mapping[str, Any],
    optical_transfer: OpticalTransfer,
    detuning_hz: float,
    camera_pixel_size_m: float,
    phase_plate_transmittance: float,
    phase_plate_phase_rad: float,
) -> MatchedPCIImage:
    """Apply the maintained atomic response, PCI propagation and camera sampler."""

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
        raise ValueError("matched PCI imaging requires a uniform square object grid")
    if optical_transfer.transfer.shape != density.shape:
        raise ValueError("optical transfer shape must match the column density")
    if not isinstance(model_config, Mapping):
        raise ValueError("model_config must be a mapping")
    try:
        atom = model_config["atom"]
    except (KeyError, TypeError) as exc:
        raise ValueError("model_config is missing the atomic contract") from exc

    polarised = polarised_optical_response_from_config(model_config)
    response = complex_column_response(
        density,
        detuning_hz,
        atom["resonant_cross_section_m2"],
        atom["natural_linewidth_rad_s"],
        polarised,
    )
    object_intensity = simulate_pci_object_field(
        response.object_field,
        optical_transfer.transfer,
        phase_plate_transmittance,
        phase_plate_phase_rad,
    )
    camera_intensity = resample_to_camera_pixels(
        object_intensity,
        input_pixel,
        _finite_positive(camera_pixel_size_m, "camera_pixel_size_m"),
    )
    rows, columns = camera_intensity.shape
    camera_y = (
        np.arange(columns, dtype=float) - (columns - 1) / 2.0
    ) * float(camera_pixel_size_m)
    camera_z = (
        np.arange(rows, dtype=float) - (rows - 1) / 2.0
    ) * float(camera_pixel_size_m)
    transmittance = float(phase_plate_transmittance)
    if not np.isfinite(transmittance) or not 0.0 <= transmittance <= 1.0:
        raise ValueError("phase_plate_transmittance must lie in [0, 1]")
    background = transmittance**2
    signal = np.asarray(camera_intensity, dtype=float) - background
    return MatchedPCIImage(
        optical_transfer=optical_transfer,
        phase_map_rad=_immutable(response.phase_map_rad),
        optical_depth_map=_immutable(response.optical_depth_map),
        object_field=_immutable(response.object_field, complex_values=True),
        camera_y_m=_immutable(camera_y),
        camera_z_m=_immutable(camera_z),
        camera_intensity_over_i0=_immutable(camera_intensity),
        atom_free_intensity_over_i0=background,
        pci_signal_over_i0=_immutable(signal),
    )


def simulate_matched_dgi_image(
    matched_pci: MatchedPCIImage,
    *,
    input_pixel_size_m: float,
    camera_pixel_size_m: float,
    stop_optical_depth: float,
) -> MatchedDGIImage:
    """Apply DGI to exactly the complex object and transfer used by PCI.

    The PCI result is the bridge object deliberately: its complex transmission
    contains both phase and absorption, and its optical transfer and camera
    shape are reused without method-specific substitutions.
    """

    if not isinstance(matched_pci, MatchedPCIImage):
        raise TypeError("matched_pci must be a MatchedPCIImage")
    input_pixel = _finite_positive(input_pixel_size_m, "input_pixel_size_m")
    camera_pixel = _finite_positive(camera_pixel_size_m, "camera_pixel_size_m")
    optical_depth = float(stop_optical_depth)
    if not np.isfinite(optical_depth) or optical_depth < 0.0:
        raise ValueError("stop_optical_depth must be finite and non-negative")
    object_intensity = simulate_dgi_object_field(
        matched_pci.object_field,
        matched_pci.optical_transfer.transfer,
        optical_depth,
    )
    camera_intensity = resample_to_camera_pixels(
        object_intensity,
        input_pixel,
        camera_pixel,
        matched_pci.camera_intensity_over_i0.shape,
    )
    if camera_intensity.shape != matched_pci.camera_intensity_over_i0.shape:
        raise RuntimeError("matched PCI and DGI camera shapes diverged")
    background = float(10.0 ** (-optical_depth))
    return MatchedDGIImage(
        optical_transfer=matched_pci.optical_transfer,
        phase_map_rad=matched_pci.phase_map_rad,
        optical_depth_map=matched_pci.optical_depth_map,
        object_field=matched_pci.object_field,
        camera_y_m=matched_pci.camera_y_m,
        camera_z_m=matched_pci.camera_z_m,
        camera_intensity_over_i0=_immutable(camera_intensity),
        atom_free_intensity_over_i0=background,
        dgi_signal_over_i0=_immutable(camera_intensity - background),
        stop_optical_depth=optical_depth,
    )


def decompose_coherent_intensity(
    object_field: ArrayLike,
    optical_transfer: OpticalTransfer,
    *,
    reference_field: complex | float,
    input_pixel_size_m: float,
    camera_pixel_size_m: float,
    camera_output_shape: tuple[int, int],
) -> CoherentIntensityDecomposition:
    """Decompose a coherent image before and after camera pixel integration.

    For propagated scattered field ``Es`` and constant method reference
    ``Eref``, this evaluates the signed cross term
    ``2*Re(conj(Eref)*Es)``, the non-negative self term ``abs(Es)**2`` and
    their sum.  Every intensity term is independently passed through the same
    physical camera pixel-area integration.
    """

    if not isinstance(optical_transfer, OpticalTransfer):
        raise TypeError("optical_transfer must be an OpticalTransfer")
    field = np.asarray(object_field, dtype=complex)
    if field.ndim != 2 or field.shape != optical_transfer.transfer.shape:
        raise ValueError("object_field and optical transfer must share one 2D shape")
    if not np.isfinite(field).all():
        raise ValueError("object_field must be finite")
    reference = complex(reference_field)
    if not np.isfinite(reference.real) or not np.isfinite(reference.imag):
        raise ValueError("reference_field must be finite")
    input_pixel = _finite_positive(input_pixel_size_m, "input_pixel_size_m")
    camera_pixel = _finite_positive(camera_pixel_size_m, "camera_pixel_size_m")
    if (
        len(camera_output_shape) != 2
        or any(isinstance(value, bool) or int(value) <= 0 for value in camera_output_shape)
    ):
        raise ValueError("camera_output_shape must contain two positive integers")
    output_shape = tuple(int(value) for value in camera_output_shape)

    image_field = np.asarray(
        simulate_fourier_image(
            field,
            optical_transfer.transfer,
            reference,
            return_intensity=False,
        ),
        dtype=complex,
    )
    scattered = image_field - reference
    background = np.full(field.shape, abs(reference) ** 2, dtype=float)
    interference = 2.0 * np.real(np.conjugate(reference) * scattered)
    self_intensity = np.abs(scattered) ** 2
    total_intensity = np.abs(image_field) ** 2
    delta_intensity = total_intensity - background
    optical_closure = float(
        np.max(np.abs(delta_intensity - (interference + self_intensity)))
    )

    def camera_average(values: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(
            resample_to_camera_pixels(
                values,
                input_pixel,
                camera_pixel,
                output_shape,
            ),
            dtype=float,
        )

    camera_background = camera_average(background)
    camera_interference = camera_average(interference)
    camera_self = camera_average(self_intensity)
    camera_total = camera_average(total_intensity)
    camera_delta = camera_total - camera_background
    camera_closure = float(
        np.max(np.abs(camera_delta - (camera_interference + camera_self)))
    )
    return CoherentIntensityDecomposition(
        reference_field=reference,
        propagated_scattered_field=_immutable(scattered, complex_values=True),
        optical_grid_background_over_i0=_immutable(background),
        optical_grid_interference_over_i0=_immutable(interference),
        optical_grid_self_over_i0=_immutable(self_intensity),
        optical_grid_delta_intensity_over_i0=_immutable(delta_intensity),
        optical_grid_total_intensity_over_i0=_immutable(total_intensity),
        camera_background_over_i0=_immutable(camera_background),
        camera_interference_over_i0=_immutable(camera_interference),
        camera_self_over_i0=_immutable(camera_self),
        camera_delta_intensity_over_i0=_immutable(camera_delta),
        camera_total_intensity_over_i0=_immutable(camera_total),
        optical_grid_closure_max_abs_error=optical_closure,
        camera_grid_closure_max_abs_error=camera_closure,
    )


def incident_photoelectrons_per_i0_pixel(
    model_config: Mapping[str, Any],
    *,
    camera_pixel_size_m: float,
    probe_power_mw: float,
    pulse_duration_s: float,
    quantum_efficiency: float,
) -> float:
    """Calculate the incident photoelectron scale for one object-plane pixel."""

    try:
        constants = model_config["constants"]
        atom = model_config["atom"]
        geometry = model_config["imaging_geometry"]
    except (KeyError, TypeError) as exc:
        raise ValueError("model config lacks photon-count inputs") from exc
    pixel = _finite_positive(camera_pixel_size_m, "camera_pixel_size_m")
    power = _finite_positive(probe_power_mw, "probe_power_mw")
    duration = _finite_positive(pulse_duration_s, "pulse_duration_s")
    efficiency = _finite_positive(quantum_efficiency, "quantum_efficiency")
    if efficiency > 1.0:
        raise ValueError("quantum_efficiency must not exceed one")
    h_planck = 2.0 * np.pi * _finite_positive(constants["hbar"], "hbar")
    photon_energy_j = (
        h_planck
        * _finite_positive(constants["speed_of_light"], "speed_of_light")
        / _finite_positive(atom["transition_wavelength_m"], "transition_wavelength_m")
    )
    incident_intensity_w_m2 = intensity_at_atoms(
        power,
        _finite_positive(geometry["probe_diameter_m"], "probe_diameter_m"),
        use_peak_intensity=True,
    )
    return float(
        incident_intensity_w_m2
        * pixel**2
        * duration
        * efficiency
        / photon_energy_j
    )


def simulate_noisy_pci_difference(
    image: MatchedPCIImage,
    *,
    photoelectrons_per_i0_pixel: float,
    read_noise_electrons_rms: float,
    seed_components_prefix: tuple[int, ...],
    camera_contract_id: str,
    sampling_contract_id: str,
) -> NoisyPCIImage:
    """Draw independent atom and bright-reference frames and subtract them."""

    if not isinstance(image, MatchedPCIImage):
        raise TypeError("image must be a MatchedPCIImage")
    count_scale = _finite_positive(
        photoelectrons_per_i0_pixel,
        "photoelectrons_per_i0_pixel",
    )
    read_noise = float(read_noise_electrons_rms)
    if not np.isfinite(read_noise) or read_noise < 0.0:
        raise ValueError("read_noise_electrons_rms must be finite and non-negative")
    prefix = tuple(int(value) for value in seed_components_prefix)
    if not prefix or any(value < 0 for value in prefix):
        raise ValueError("seed_components_prefix must contain non-negative integers")
    atom = simulate_intensity_frame(
        image.camera_intensity_over_i0,
        photoelectrons_per_i0_pixel=count_scale,
        read_noise_electrons_rms=read_noise,
        role="pci_atom",
        camera_contract_id=camera_contract_id,
        sampling_contract_id=sampling_contract_id,
        seed_components=prefix + (0,),
    )
    reference = simulate_intensity_frame(
        np.full_like(
            image.camera_intensity_over_i0,
            image.atom_free_intensity_over_i0,
        ),
        photoelectrons_per_i0_pixel=count_scale,
        read_noise_electrons_rms=read_noise,
        role="pci_bright_reference",
        camera_contract_id=camera_contract_id,
        sampling_contract_id=sampling_contract_id,
        seed_components=prefix + (1,),
    )
    difference = paired_difference_diagnostic(atom, reference)
    return NoisyPCIImage(
        seed_components_prefix=prefix,
        photoelectrons_per_i0_pixel=count_scale,
        read_noise_electrons_rms=read_noise,
        expected_signal_over_i0=_immutable(difference.expected_value / count_scale),
        observed_signal_over_i0=_immutable(difference.observed_value / count_scale),
    )


def positive_pci_morphology_map(
    pci_signal_over_i0: ArrayLike,
    camera_y_m: ArrayLike,
    camera_z_m: ArrayLike,
    *,
    analysis_half_width_y_m: float,
    analysis_half_width_z_m: float,
) -> NDArray[np.floating]:
    """Return the declared positive, background-subtracted morphology weight.

    The fixed physical support is applied first.  Negative signed PCI excursions
    are then excluded from the moment and peak estimator; the operation is an
    explicit estimator definition, not a repair of the optical image.
    """

    signal = np.asarray(pci_signal_over_i0, dtype=float)
    y_axis = np.asarray(camera_y_m, dtype=float)
    z_axis = np.asarray(camera_z_m, dtype=float)
    if signal.shape != (z_axis.size, y_axis.size):
        raise ValueError("PCI signal shape must match the supplied camera axes")
    if not np.isfinite(signal).all():
        raise ValueError("PCI signal must be finite")
    half_y = _finite_positive(analysis_half_width_y_m, "analysis_half_width_y_m")
    half_z = _finite_positive(analysis_half_width_z_m, "analysis_half_width_z_m")
    support = (np.abs(y_axis)[None, :] <= half_y) & (
        np.abs(z_axis)[:, None] <= half_z
    )
    weight = np.where(support, np.maximum(signal, 0.0), 0.0)
    if not np.any(weight > 0.0):
        raise ValueError("PCI morphology support contains no positive signal")
    return _immutable(weight)


def recover_pci_morphology(
    pci_signal_over_i0: ArrayLike,
    camera_y_m: ArrayLike,
    camera_z_m: ArrayLike,
    *,
    analysis_half_width_y_m: float,
    analysis_half_width_z_m: float,
    minimum_peak_distance_m: float,
    peak_prominence_fraction: float,
) -> MorphologyObservables:
    """Extract the declared low-dimensional morphology from a PCI difference."""

    weight = positive_pci_morphology_map(
        pci_signal_over_i0,
        camera_y_m,
        camera_z_m,
        analysis_half_width_y_m=analysis_half_width_y_m,
        analysis_half_width_z_m=analysis_half_width_z_m,
    )
    return measure_morphology(
        weight,
        camera_y_m,
        camera_z_m,
        minimum_peak_distance_m=minimum_peak_distance_m,
        peak_prominence_fraction=peak_prominence_fraction,
    )


__all__ = [
    "CoherentIntensityDecomposition",
    "MatchedDGIImage",
    "MatchedPCIImage",
    "NoisyPCIImage",
    "OpticalTransfer",
    "decompose_coherent_intensity",
    "gaussian_amplitude_psf_transfer",
    "hard_circular_coherent_transfer",
    "incident_photoelectrons_per_i0_pixel",
    "optical_transfer_from_objective_config",
    "positive_pci_morphology_map",
    "recover_pci_morphology",
    "simulate_matched_dgi_image",
    "simulate_matched_pci_image",
    "simulate_noisy_pci_difference",
]
