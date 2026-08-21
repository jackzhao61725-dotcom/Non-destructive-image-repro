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


def _validated_jones_fields(
    co_polarised_field: ArrayLike,
    faraday_orthogonal_field: ArrayLike,
    pupil: ArrayLike,
) -> tuple[
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
    NDArray[np.complexfloating],
]:
    co_array, pupil_array = _validated_complex_object(co_polarised_field, pupil)
    faraday_array = np.asarray(faraday_orthogonal_field)
    if faraday_array.shape != co_array.shape:
        raise ValueError(
            "faraday_orthogonal_field must have the same 2D shape as co_polarised_field"
        )
    if not np.isfinite(faraday_array).all():
        raise ValueError("faraday_orthogonal_field must be finite")
    return co_array, faraday_array, pupil_array


def _simulate_jones_fourier_readout(
    co_polarised_field: ArrayLike,
    faraday_orthogonal_field: ArrayLike,
    pupil: ArrayLike,
    reference_field: complex | float,
    *,
    return_intermediates: bool,
) -> NDArray[np.floating] | dict[
    str,
    NDArray[np.floating] | NDArray[np.complexfloating] | complex,
]:
    """Propagate two orthogonal output polarisations and add their intensities.

    The co-polarised field contains the unit blank carrier, so its
    atom-dependent part is ``co_polarised_field - 1``.  The Faraday-orthogonal
    field is zero without atoms and is propagated in full.  The two image-plane
    fields are orthogonally polarised and therefore contribute incoherently to
    a camera without an output analyser.

    The scalar ``reference_field`` implements the idealised Fourier-mask model:
    the phase dot or stop acts on the uniform carrier but does not additionally
    filter either atom-dependent field.
    """

    co_array, faraday_array, pupil_array = _validated_jones_fields(
        co_polarised_field,
        faraday_orthogonal_field,
        pupil,
    )
    reference = np.asarray(reference_field)
    if reference.shape != () or not np.isfinite(reference).all():
        raise ValueError("reference_field must be a finite scalar field amplitude")
    reference_scalar = complex(reference.item())

    co_scattered = co_array - 1.0
    co_propagated = propagate_scattered_field(co_scattered, pupil_array)
    faraday_propagated = propagate_scattered_field(faraday_array, pupil_array)
    co_image_field = reference_scalar + co_propagated
    total_intensity = np.abs(co_image_field) ** 2 + np.abs(faraday_propagated) ** 2

    if return_intermediates:
        return {
            "co_polarised_object_field": co_array,
            "faraday_orthogonal_object_field": faraday_array,
            "co_polarised_scattered_field": co_scattered,
            "co_polarised_propagated_field": co_propagated,
            "faraday_propagated_field": faraday_propagated,
            "co_polarised_image_field": co_image_field,
            "reference_field": reference_scalar,
            "total_image_intensity": total_intensity,
        }
    return total_intensity


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


def simulate_pci_jones_fields(
    co_polarised_field: ArrayLike,
    faraday_orthogonal_field: ArrayLike,
    pupil: ArrayLike,
    phase_plate_transmittance: float = 0.95,
    phase_plate_phase: float = np.pi / 2,
    *,
    return_intermediates: bool = False,
) -> NDArray[np.floating] | dict[
    str,
    NDArray[np.floating] | NDArray[np.complexfloating] | complex,
]:
    """Return PCI intensity for the exact two-component Jones output field."""

    transmittance = float(phase_plate_transmittance)
    phase = float(phase_plate_phase)
    if not np.isfinite(transmittance) or not 0.0 <= transmittance <= 1.0:
        raise ValueError("phase_plate_transmittance must be a finite field amplitude in [0, 1]")
    if not np.isfinite(phase):
        raise ValueError("phase_plate_phase must be finite")
    reference = transmittance * np.exp(1j * phase)
    return _simulate_jones_fourier_readout(
        co_polarised_field,
        faraday_orthogonal_field,
        pupil,
        reference,
        return_intermediates=return_intermediates,
    )


def simulate_dgi_jones_fields(
    co_polarised_field: ArrayLike,
    faraday_orthogonal_field: ArrayLike,
    pupil: ArrayLike,
    stop_optical_depth: float = 4.0,
    *,
    return_intermediates: bool = False,
) -> NDArray[np.floating] | dict[
    str,
    NDArray[np.floating] | NDArray[np.complexfloating] | complex,
]:
    """Return DGI intensity for the exact two-component Jones output field."""

    optical_depth = float(stop_optical_depth)
    if not np.isfinite(optical_depth) or optical_depth < 0.0:
        raise ValueError("stop_optical_depth must be finite and non-negative")
    reference = float(10.0 ** (-optical_depth / 2.0))
    return _simulate_jones_fourier_readout(
        co_polarised_field,
        faraday_orthogonal_field,
        pupil,
        reference,
        return_intermediates=return_intermediates,
    )


def simulate_dffi_jones_fields(
    co_polarised_field: ArrayLike,
    faraday_orthogonal_field: ArrayLike,
    pupil: ArrayLike,
    *,
    return_intermediates: bool = False,
) -> NDArray[np.floating] | dict[
    str,
    NDArray[np.floating] | NDArray[np.complexfloating],
]:
    """Return the crossed-analyser dark-port intensity of a Jones field."""

    co_array, faraday_array, pupil_array = _validated_jones_fields(
        co_polarised_field,
        faraday_orthogonal_field,
        pupil,
    )
    faraday_image_field = propagate_scattered_field(faraday_array, pupil_array)
    intensity = np.abs(faraday_image_field) ** 2
    if return_intermediates:
        return {
            "co_polarised_object_field": co_array,
            "faraday_orthogonal_object_field": faraday_array,
            "faraday_image_field": faraday_image_field,
            "dark_port_intensity": intensity,
        }
    return intensity


def simulate_dpfi_jones_fields(
    co_polarised_field: ArrayLike,
    faraday_orthogonal_field: ArrayLike,
    pupil: ArrayLike,
    *,
    return_intermediates: bool = False,
) -> dict[str, NDArray[np.floating] | NDArray[np.complexfloating]]:
    """Return the two bright DPFI ports and their normalised difference."""

    co_array, faraday_array, pupil_array = _validated_jones_fields(
        co_polarised_field,
        faraday_orthogonal_field,
        pupil,
    )
    co_image_field = 1.0 + propagate_scattered_field(co_array - 1.0, pupil_array)
    faraday_image_field = propagate_scattered_field(faraday_array, pupil_array)
    h_intensity = np.abs(co_image_field + faraday_image_field) ** 2 / 2.0
    v_intensity = np.abs(co_image_field - faraday_image_field) ** 2 / 2.0
    denominator = h_intensity + v_intensity
    signal = np.divide(
        h_intensity - v_intensity,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=float),
        where=denominator > 0.0,
    )
    outputs: dict[str, NDArray[np.floating] | NDArray[np.complexfloating]] = {
        "analyser_h_intensity": h_intensity,
        "analyser_v_intensity": v_intensity,
        "dual_port_signal": signal,
    }
    if return_intermediates:
        return {
            "co_polarised_object_field": co_array,
            "faraday_orthogonal_object_field": faraday_array,
            "co_polarised_image_field": co_image_field,
            "faraday_image_field": faraday_image_field,
            **outputs,
        }
    return outputs


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
    expose the dissertation convention
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
