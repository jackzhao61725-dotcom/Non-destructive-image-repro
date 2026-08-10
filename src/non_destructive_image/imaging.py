"""Shared coherent Fourier-imaging helpers for PCI/DGI-style paths."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fourier import propagate_scattered_field


def simulate_fourier_image(
    object_field: ArrayLike,
    pupil: ArrayLike,
    reference_field: complex | float,
    *,
    return_intensity: bool = True,
) -> NDArray[np.floating] | NDArray[np.complexfloating]:
    """Propagate the scattered field and recombine it with a carrier reference.

    Only ``object_field - 1`` passes through the pupil. The unscattered carrier
    is supplied separately by the readout model and added before evaluating
    ``abs(E)**2``. No shift or padding is applied implicitly.
    """

    object_array, pupil_array = _validated_complex_object(object_field, pupil)
    reference = np.asarray(reference_field)
    if reference.shape != () or not np.isfinite(reference).all():
        raise ValueError("reference_field must be a finite scalar field amplitude")
    scattered_field = object_array - 1
    image_field = reference.item() + propagate_scattered_field(scattered_field, pupil_array)
    if return_intensity:
        return np.abs(image_field) ** 2
    return image_field


def simulate_pci_image(
    phase_map: ArrayLike,
    pupil: ArrayLike,
    phase_plate_transmittance: float = 0.95,
    phase_plate_phase: float = np.pi / 2,
    *,
    return_intermediates: bool = False,
) -> NDArray[np.floating] | dict[str, NDArray[np.floating] | NDArray[np.complexfloating] | complex]:
    """Return the phase-only scalar PCI image intensity.

    This compatibility entry point constructs ``exp(1j * phase_map)``. Use
    :func:`simulate_pci_object_field` when absorption is part of the object.
    """

    phase_array, pupil_array = _validated_real_map(phase_map, pupil, "phase_map")
    object_field = np.exp(1j * phase_array)
    return _simulate_pci_object_field(
        object_field,
        pupil_array,
        phase_plate_transmittance,
        phase_plate_phase,
        return_intermediates=return_intermediates,
    )


def _simulate_pci_object_field(
    object_field: ArrayLike,
    pupil: ArrayLike,
    phase_plate_transmittance: float,
    phase_plate_phase: float,
    *,
    return_intermediates: bool,
) -> NDArray[np.floating] | dict[
    str,
    NDArray[np.floating] | NDArray[np.complexfloating] | complex,
]:
    pci_reference_field = phase_plate_transmittance * np.exp(1j * phase_plate_phase)
    image_field = simulate_fourier_image(
        object_field,
        pupil,
        pci_reference_field,
        return_intensity=False,
    )
    pci_image_intensity = np.abs(image_field) ** 2

    if return_intermediates:
        return {
            "object_field": object_field,
            "scattered_field": object_field - 1,
            "propagated_scattered_field": image_field - pci_reference_field,
            "pci_reference_field": pci_reference_field,
            "pci_image_intensity": pci_image_intensity,
        }
    return pci_image_intensity


def _validated_complex_object(
    object_field: ArrayLike,
    pupil: ArrayLike,
) -> tuple[NDArray[np.complexfloating], NDArray[np.complexfloating]]:
    object_array = np.asarray(object_field)
    pupil_array = np.asarray(pupil)
    if object_array.ndim != 2 or object_array.size == 0:
        raise ValueError("object_field must be a non-empty two-dimensional array")
    if pupil_array.shape != object_array.shape:
        raise ValueError("pupil must have the same two-dimensional shape as object_field")
    if not np.isfinite(object_array).all() or not np.isfinite(pupil_array).all():
        raise ValueError("object_field and pupil must be finite")
    return object_array, pupil_array


def _validated_real_map(
    value: ArrayLike,
    pupil: ArrayLike,
    name: str,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    real_array = np.asarray(array, dtype=float)
    _, pupil_array = _validated_complex_object(real_array, pupil)
    return real_array, pupil_array


def simulate_pci_object_field(
    object_field: ArrayLike,
    pupil: ArrayLike,
    phase_plate_transmittance: float = 0.95,
    phase_plate_phase: float = np.pi / 2,
    *,
    return_intermediates: bool = False,
) -> NDArray[np.floating] | dict[
    str,
    NDArray[np.floating] | NDArray[np.complexfloating] | complex,
]:
    """Return a PCI image from a validated complex object transmission.

    ``phase_plate_transmittance`` is a field-amplitude transmission, so the
    atom-free intensity relative to the unattenuated incident field is its
    square.
    """

    object_array, pupil_array = _validated_complex_object(object_field, pupil)
    transmittance = float(phase_plate_transmittance)
    phase = float(phase_plate_phase)
    if not np.isfinite(transmittance) or not 0.0 <= transmittance <= 1.0:
        raise ValueError("phase_plate_transmittance must be a finite field amplitude in [0, 1]")
    if not np.isfinite(phase):
        raise ValueError("phase_plate_phase must be finite")
    return _simulate_pci_object_field(
        object_array,
        pupil_array,
        transmittance,
        phase,
        return_intermediates=return_intermediates,
    )


def simulate_dgi_image(
    phase_map: ArrayLike,
    pupil: ArrayLike,
    stop_optical_depth: float = 4.0,
    *,
    return_intermediates: bool = False,
) -> NDArray[np.floating] | dict[str, NDArray[np.floating] | NDArray[np.complexfloating] | float]:
    """Return the phase-only scalar DGI image intensity.

    This compatibility entry point constructs ``exp(1j * phase_map)``. Use
    :func:`simulate_dgi_object_field` when absorption is part of the object.
    """

    phase_array, pupil_array = _validated_real_map(phase_map, pupil, "phase_map")
    object_field = np.exp(1j * phase_array)
    return _simulate_dgi_object_field(
        object_field,
        pupil_array,
        stop_optical_depth,
        return_intermediates=return_intermediates,
    )


def _simulate_dgi_object_field(
    object_field: ArrayLike,
    pupil: ArrayLike,
    stop_optical_depth: float,
    *,
    return_intermediates: bool,
) -> NDArray[np.floating] | dict[
    str,
    NDArray[np.floating] | NDArray[np.complexfloating] | float,
]:
    dgi_reference_field = 10 ** (-stop_optical_depth / 2)
    image_field = simulate_fourier_image(
        object_field,
        pupil,
        dgi_reference_field,
        return_intensity=False,
    )
    dgi_image_intensity = np.abs(image_field) ** 2

    if return_intermediates:
        return {
            "object_field": object_field,
            "scattered_field": object_field - 1,
            "propagated_scattered_field": image_field - dgi_reference_field,
            "dgi_reference_field": dgi_reference_field,
            "dgi_image_intensity": dgi_image_intensity,
        }
    return dgi_image_intensity


def simulate_dgi_object_field(
    object_field: ArrayLike,
    pupil: ArrayLike,
    stop_optical_depth: float = 4.0,
    *,
    return_intermediates: bool = False,
) -> NDArray[np.floating] | dict[
    str,
    NDArray[np.floating] | NDArray[np.complexfloating] | float,
]:
    """Return a DGI image from a validated complex object transmission.

    ``stop_optical_depth`` is a base-10 intensity optical density, giving the
    carrier reference field ``10**(-OD/2)`` and atom-free intensity
    ``10**(-OD)`` relative to the incident field.
    """

    object_array, pupil_array = _validated_complex_object(object_field, pupil)
    optical_depth = float(stop_optical_depth)
    if not np.isfinite(optical_depth) or optical_depth < 0.0:
        raise ValueError("stop_optical_depth must be finite and non-negative")
    return _simulate_dgi_object_field(
        object_array,
        pupil_array,
        optical_depth,
        return_intermediates=return_intermediates,
    )


def simulate_selected_scalar_readout(
    readout: Literal["pci", "dgi"],
    object_field: ArrayLike,
    pupil: ArrayLike,
    *,
    phase_plate_transmittance: float = 0.95,
    phase_plate_phase: float = np.pi / 2,
    stop_optical_depth: float = 4.0,
    return_intermediates: bool = False,
) -> NDArray[np.floating] | dict[
    str,
    NDArray[np.floating] | NDArray[np.complexfloating] | complex | float,
]:
    """Dispatch the selected tensor-eigenmode complex object to PCI or DGI."""

    if readout == "pci":
        return simulate_pci_object_field(
            object_field,
            pupil,
            phase_plate_transmittance,
            phase_plate_phase,
            return_intermediates=return_intermediates,
        )
    if readout == "dgi":
        return simulate_dgi_object_field(
            object_field,
            pupil,
            stop_optical_depth,
            return_intermediates=return_intermediates,
        )
    raise ValueError("readout must be exactly 'pci' or 'dgi'")


def simulate_faraday_image(
    theta_f_map: ArrayLike,
    pupil: ArrayLike,
    *,
    return_intermediates: bool = False,
) -> dict[str, NDArray[np.floating] | NDArray[np.complexfloating]]:
    """Return Faraday dark-field and signed dual-port outputs.

    The caller supplies the signed rotation map ``theta_F``. The helper applies
    opposite circular phase shifts, the common FFT/pupil propagation, and the
    circular-to-linear recombination. The ``u``/``v`` keys preserve the
    admitted forward-model convention; explicit analyser ``H``/``V`` aliases
    expose the scalar--tensor convention used by this model
    ``S = (I_H - I_V) / (I_H + I_V)``.
    """

    theta_f_map_array, pupil_array = _validated_real_map(
        theta_f_map,
        pupil,
        "theta_f_map",
    )
    sigma_plus_object_field = np.exp(1j * theta_f_map_array)
    sigma_minus_object_field = np.exp(-1j * theta_f_map_array)

    sigma_plus_scattered_field = sigma_plus_object_field - 1
    sigma_minus_scattered_field = sigma_minus_object_field - 1
    sigma_plus_propagated_scattered_field = propagate_scattered_field(
        sigma_plus_scattered_field,
        pupil_array,
    )
    sigma_minus_propagated_scattered_field = propagate_scattered_field(
        sigma_minus_scattered_field,
        pupil_array,
    )
    sigma_plus_field = 1 + sigma_plus_propagated_scattered_field
    sigma_minus_field = 1 + sigma_minus_propagated_scattered_field

    output_ex_field = (sigma_plus_field + sigma_minus_field) / 2
    output_ey_field = 1j * (sigma_plus_field - sigma_minus_field) / 2
    dark_field_intensity = np.abs(output_ey_field) ** 2
    dual_port_u_intensity = np.abs(output_ex_field + output_ey_field) ** 2 / 2
    dual_port_v_intensity = np.abs(output_ex_field - output_ey_field) ** 2 / 2
    dual_port_signal = (dual_port_v_intensity - dual_port_u_intensity) / (
        dual_port_v_intensity + dual_port_u_intensity
    )

    outputs = {
        "dark_field_intensity": dark_field_intensity,
        "dual_port_u_intensity": dual_port_u_intensity,
        "dual_port_v_intensity": dual_port_v_intensity,
        "analyser_h_intensity": dual_port_v_intensity,
        "analyser_v_intensity": dual_port_u_intensity,
        "dual_port_signal": dual_port_signal,
    }
    if return_intermediates:
        return {
            "theta_f_map_rad": theta_f_map_array,
            "sigma_plus_object_field": sigma_plus_object_field,
            "sigma_minus_object_field": sigma_minus_object_field,
            "sigma_plus_scattered_field": sigma_plus_scattered_field,
            "sigma_minus_scattered_field": sigma_minus_scattered_field,
            "sigma_plus_propagated_scattered_field": sigma_plus_propagated_scattered_field,
            "sigma_minus_propagated_scattered_field": sigma_minus_propagated_scattered_field,
            "sigma_plus_field": sigma_plus_field,
            "sigma_minus_field": sigma_minus_field,
            "output_ex_field": output_ex_field,
            "output_ey_field": output_ey_field,
            "parallel_field": output_ex_field,
            "perpendicular_field": -output_ey_field,
            **outputs,
        }
    return outputs
