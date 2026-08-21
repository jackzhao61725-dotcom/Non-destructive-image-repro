"""Matched camera sampling for the four ideal Jones-field imaging readouts.

The atom-side response and the PCI/DGI camera path live in
``parallel_jones_imaging``.  This module adds the two Faraday readouts without
changing that established source: all optical-grid intensities are integrated
over the same physical camera pixels, and the DPFI ratio is formed only after
the two port intensities have been integrated separately.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .camera import resample_to_camera_pixels
from .equilibrium_imaging import OpticalTransfer
from .imaging import (
    simulate_dffi_jones_fields,
    simulate_dgi_jones_fields,
    simulate_dpfi_jones_fields,
    simulate_pci_jones_fields,
)
from .parallel_jones_imaging import (
    MatchedJonesDGIImage,
    MatchedJonesPCIImage,
    simulate_matched_jones_dgi_image,
    simulate_matched_jones_pci_image,
)


FloatArray = NDArray[np.floating]
ComplexArray = NDArray[np.complexfloating]


def _immutable(values: ArrayLike, *, complex_values: bool = False) -> NDArray:
    array = np.array(values, dtype=complex if complex_values else float, copy=True)
    array.setflags(write=False)
    return array


def _camera_resample(
    image: ArrayLike,
    matched_pci: MatchedJonesPCIImage,
) -> FloatArray:
    values = resample_to_camera_pixels(
        image,
        matched_pci.input_pixel_size_m,
        matched_pci.camera_pixel_size_m,
        matched_pci.camera_intensity_over_i0.shape,
    )
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class MatchedFourMethodJonesImages:
    """Noiseless, camera-sampled PCI, DGI, DFFI and DPFI outputs."""

    pci: MatchedJonesPCIImage
    dgi: MatchedJonesDGIImage
    circular_transmission_fields: ComplexArray
    co_polarised_propagated_field: ComplexArray
    faraday_propagated_field: ComplexArray
    pci_optical_grid_intensity_over_i0: FloatArray
    dgi_optical_grid_intensity_over_i0: FloatArray
    dffi_optical_grid_intensity_over_i0: FloatArray
    dpfi_h_optical_grid_intensity_over_i0: FloatArray
    dpfi_v_optical_grid_intensity_over_i0: FloatArray
    pci_co_camera_intensity_over_i0: FloatArray
    dgi_co_camera_intensity_over_i0: FloatArray
    open_co_camera_intensity_over_i0: FloatArray
    dffi_camera_intensity_over_i0: FloatArray
    dpfi_h_camera_intensity_over_i0: FloatArray
    dpfi_v_camera_intensity_over_i0: FloatArray
    dpfi_sum_camera_intensity_over_i0: FloatArray
    dpfi_difference_camera_intensity_over_i0: FloatArray
    dpfi_normalised_difference: FloatArray

    @property
    def camera_y_m(self) -> FloatArray:
        return self.pci.camera_y_m

    @property
    def camera_z_m(self) -> FloatArray:
        return self.pci.camera_z_m


def simulate_matched_four_method_jones_images(
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
    dgi_stop_optical_depth: float,
) -> MatchedFourMethodJonesImages:
    """Apply all four ideal readouts to one exact parallel-geometry Jones field.

    PCI and DGI are unanalysed: the two orthogonal image-plane intensities are
    added.  DFFI selects the crossed (Faraday) port.  DPFI forms the balanced
    ``H=+`` and ``V=-`` ports, integrates each intensity over the physical
    camera pixels, and only then forms their normalised difference.
    """

    pci = simulate_matched_jones_pci_image(
        column_density_m2,
        y_axis_m,
        z_axis_m,
        model_config=model_config,
        orientation_config=orientation_config,
        optical_transfer=optical_transfer,
        detuning_hz=detuning_hz,
        camera_pixel_size_m=camera_pixel_size_m,
        phase_plate_transmittance=phase_plate_transmittance,
        phase_plate_phase_rad=phase_plate_phase_rad,
    )
    dgi = simulate_matched_jones_dgi_image(
        pci,
        stop_optical_depth=dgi_stop_optical_depth,
    )

    co_object = pci.co_polarised_object_field
    faraday_object = pci.faraday_orthogonal_object_field
    pupil = pci.optical_transfer.transfer
    pci_fields = simulate_pci_jones_fields(
        co_object,
        faraday_object,
        pupil,
        phase_plate_transmittance,
        phase_plate_phase_rad,
        return_intermediates=True,
    )
    dgi_fields = simulate_dgi_jones_fields(
        co_object,
        faraday_object,
        pupil,
        dgi_stop_optical_depth,
        return_intermediates=True,
    )
    dffi_fields = simulate_dffi_jones_fields(
        co_object,
        faraday_object,
        pupil,
        return_intermediates=True,
    )
    dpfi_fields = simulate_dpfi_jones_fields(
        co_object,
        faraday_object,
        pupil,
        return_intermediates=True,
    )

    co_propagated = np.asarray(
        pci_fields["co_polarised_propagated_field"],
        dtype=complex,
    )
    faraday_propagated = np.asarray(
        pci_fields["faraday_propagated_field"],
        dtype=complex,
    )
    if not np.allclose(
        faraday_propagated,
        dffi_fields["faraday_image_field"],
        rtol=2e-14,
        atol=2e-15,
    ):
        raise RuntimeError("DFFI did not propagate the shared Faraday field")
    if not np.allclose(
        faraday_propagated,
        dpfi_fields["faraday_image_field"],
        rtol=2e-14,
        atol=2e-15,
    ):
        raise RuntimeError("DPFI did not propagate the shared Faraday field")

    pci_optical = np.asarray(pci_fields["total_image_intensity"], dtype=float)
    dgi_optical = np.asarray(dgi_fields["total_image_intensity"], dtype=float)
    dffi_optical = np.asarray(dffi_fields["dark_port_intensity"], dtype=float)
    dpfi_h_optical = np.asarray(dpfi_fields["analyser_h_intensity"], dtype=float)
    dpfi_v_optical = np.asarray(dpfi_fields["analyser_v_intensity"], dtype=float)

    pci_camera = _camera_resample(pci_optical, pci)
    dgi_camera = _camera_resample(dgi_optical, pci)
    if not np.allclose(
        pci_camera,
        pci.camera_intensity_over_i0,
        rtol=2e-14,
        atol=2e-15,
    ):
        raise RuntimeError("four-method PCI sampling diverged from the matched path")
    if not np.allclose(
        dgi_camera,
        dgi.camera_intensity_over_i0,
        rtol=2e-14,
        atol=2e-15,
    ):
        raise RuntimeError("four-method DGI sampling diverged from the matched path")

    pci_co_camera = _camera_resample(
        np.abs(np.asarray(pci_fields["co_polarised_image_field"])) ** 2,
        pci,
    )
    dgi_co_camera = _camera_resample(
        np.abs(np.asarray(dgi_fields["co_polarised_image_field"])) ** 2,
        pci,
    )
    open_co_camera = _camera_resample(
        np.abs(np.asarray(dpfi_fields["co_polarised_image_field"])) ** 2,
        pci,
    )
    dffi_camera = _camera_resample(dffi_optical, pci)
    dpfi_h_camera = _camera_resample(dpfi_h_optical, pci)
    dpfi_v_camera = _camera_resample(dpfi_v_optical, pci)
    dpfi_sum = dpfi_h_camera + dpfi_v_camera
    dpfi_difference = dpfi_h_camera - dpfi_v_camera
    if not np.all(np.isfinite(dpfi_sum)) or np.any(dpfi_sum <= 0.0):
        raise RuntimeError("DPFI camera denominator is not finite and positive")
    dpfi_signal = dpfi_difference / dpfi_sum

    circular = np.exp(
        -np.asarray(pci.branch_optical_depth_maps, dtype=float) / 2.0
        + 1j * np.asarray(pci.branch_phase_maps_rad, dtype=float)
    )
    return MatchedFourMethodJonesImages(
        pci=pci,
        dgi=dgi,
        circular_transmission_fields=_immutable(circular, complex_values=True),
        co_polarised_propagated_field=_immutable(
            co_propagated,
            complex_values=True,
        ),
        faraday_propagated_field=_immutable(
            faraday_propagated,
            complex_values=True,
        ),
        pci_optical_grid_intensity_over_i0=_immutable(pci_optical),
        dgi_optical_grid_intensity_over_i0=_immutable(dgi_optical),
        dffi_optical_grid_intensity_over_i0=_immutable(dffi_optical),
        dpfi_h_optical_grid_intensity_over_i0=_immutable(dpfi_h_optical),
        dpfi_v_optical_grid_intensity_over_i0=_immutable(dpfi_v_optical),
        pci_co_camera_intensity_over_i0=_immutable(pci_co_camera),
        dgi_co_camera_intensity_over_i0=_immutable(dgi_co_camera),
        open_co_camera_intensity_over_i0=_immutable(open_co_camera),
        dffi_camera_intensity_over_i0=_immutable(dffi_camera),
        dpfi_h_camera_intensity_over_i0=_immutable(dpfi_h_camera),
        dpfi_v_camera_intensity_over_i0=_immutable(dpfi_v_camera),
        dpfi_sum_camera_intensity_over_i0=_immutable(dpfi_sum),
        dpfi_difference_camera_intensity_over_i0=_immutable(dpfi_difference),
        dpfi_normalised_difference=_immutable(dpfi_signal),
    )


__all__ = [
    "MatchedFourMethodJonesImages",
    "simulate_matched_four_method_jones_images",
]
