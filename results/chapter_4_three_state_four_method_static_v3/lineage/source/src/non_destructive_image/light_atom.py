"""Two-level light--atom response and recoil-screening calculations.

All dimensional inputs are explicit. Multilevel branch weights and the shared
saturation denominator are implemented separately in :mod:`atomic_response`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _require_positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _column_density_array(value: ArrayLike, name: str) -> NDArray[np.floating]:
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be non-negative, finite, and non-empty")
    return array


def _scalar_or_array(value: NDArray[np.floating]) -> float | NDArray[np.floating]:
    return float(value) if value.ndim == 0 else value


def dimensionless_detuning(detuning_hz: float, gamma_rad_per_s: float) -> float:
    """Return ``delta = 2 Delta / Gamma`` from detuning in Hz and linewidth in rad/s."""

    if not np.isfinite(detuning_hz):
        raise ValueError("detuning_hz must be finite")
    _require_positive_finite(gamma_rad_per_s, "gamma_rad_per_s")

    return 2 * detuning_hz * 2 * np.pi / gamma_rad_per_s


def scalar_phase_shift(
    detuning_hz: float,
    column_density_peak: ArrayLike,
    resonant_cross_section: float,
    gamma_rad_per_s: float,
) -> float | NDArray[np.floating]:
    """Return the signed scalar phase shift for a column density in m^-2."""

    density = _column_density_array(column_density_peak, "column_density_peak")
    _require_positive_finite(resonant_cross_section, "resonant_cross_section")

    detuning = dimensionless_detuning(detuning_hz, gamma_rad_per_s)
    result = resonant_cross_section * density * detuning / (2 * (1 + detuning**2))
    return _scalar_or_array(result)


def residual_optical_depth(
    detuning_hz: float,
    column_density_peak: ArrayLike,
    resonant_cross_section: float,
    gamma_rad_per_s: float,
) -> float | NDArray[np.floating]:
    """Return the non-negative residual optical depth at the supplied detuning."""

    density = _column_density_array(column_density_peak, "column_density_peak")
    _require_positive_finite(resonant_cross_section, "resonant_cross_section")

    detuning = dimensionless_detuning(detuning_hz, gamma_rad_per_s)
    result = resonant_cross_section * density / (1 + detuning**2)
    return _scalar_or_array(result)


def intensity_at_atoms(
    probe_power_mw: float,
    probe_diameter_m: float,
    use_peak_intensity: bool = True,
) -> float:
    """Convert Gaussian-beam power in mW to peak or area-averaged intensity in W/m^2."""

    if not np.isfinite(probe_power_mw) or probe_power_mw < 0:
        raise ValueError("probe_power_mw must be non-negative and finite")
    _require_positive_finite(probe_diameter_m, "probe_diameter_m")
    if type(use_peak_intensity) is not bool:
        raise ValueError("use_peak_intensity must be boolean")

    area_averaged = (probe_power_mw * 1e-3) / (np.pi * (probe_diameter_m / 2) ** 2)
    return 2 * area_averaged if use_peak_intensity else area_averaged


def scattered_photons_per_atom(
    detuning_hz: float,
    probe_power_mw: float,
    pulse_duration_s: float,
    saturation_intensity: float,
    gamma_rad_per_s: float,
    probe_diameter_m: float,
    use_peak_intensity: bool = True,
) -> float:
    """Return the two-level scattered photons per atom during one probe pulse."""

    if not np.isfinite(pulse_duration_s) or pulse_duration_s < 0:
        raise ValueError("pulse_duration_s must be non-negative and finite")
    _require_positive_finite(saturation_intensity, "saturation_intensity")

    saturation_parameter = intensity_at_atoms(
        probe_power_mw,
        probe_diameter_m,
        use_peak_intensity=use_peak_intensity,
    ) / saturation_intensity
    detuning = dimensionless_detuning(detuning_hz, gamma_rad_per_s)
    return (gamma_rad_per_s / 2) * saturation_parameter / (1 + saturation_parameter + detuning**2) * pulse_duration_s


def faraday_rotation_angle(
    detuning_hz: float,
    column_density_peak: ArrayLike,
    resonant_cross_section: float,
    gamma_rad_per_s: float,
    kappa_f: float,
) -> float | NDArray[np.floating]:
    """Return the signed peak Faraday rotation ``theta_F = kappa_F * phi``.

    ``kappa_f`` is supplied by the atomic-response model; apparatus-level
    polarimetric calibration is handled separately.
    """

    if not np.isfinite(kappa_f):
        raise ValueError("kappa_f must be finite")

    return kappa_f * scalar_phase_shift(
        detuning_hz,
        column_density_peak,
        resonant_cross_section,
        gamma_rad_per_s,
    )


def reabsorption_fraction(
    detuning_hz: float,
    column_densities: ArrayLike,
    resonant_cross_section: float,
    gamma_rad_per_s: float,
) -> float:
    """Return the angle-averaged Rayleigh reabsorption fraction.

    The supplied array contains the peak column densities for the propagation
    directions being averaged.
    """

    densities = _column_density_array(column_densities, "column_densities")
    _require_positive_finite(resonant_cross_section, "resonant_cross_section")

    detuning = dimensionless_detuning(detuning_hz, gamma_rad_per_s)
    optical_depth = (
        resonant_cross_section * densities / (1 + detuning**2)
    )
    return float(np.mean(1 - np.exp(-optical_depth)))
