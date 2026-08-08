"""Contact-interaction Thomas--Fermi and single-photon recoil calculations.

Parameter values are supplied by the active configuration or calling model; this
module owns only the dimensionally explicit algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ThomasFermiState:
    """Derived three-dimensional contact Thomas--Fermi quantities in SI units.

    ``column_density[i]`` is the peak column density obtained by integrating
    along principal axis ``i``. It is not a camera-plane density map.
    """

    trap_angular_frequencies: NDArray[np.floating]
    geometric_mean_frequency: float
    harmonic_oscillator_length: float
    chemical_potential: float
    chemical_potential_temperature: float
    peak_density: float
    radii: NDArray[np.floating]
    column_density: NDArray[np.floating]
    atom_number_check: float


def build_thomas_fermi_state(
    atom_number: float,
    scattering_length: float,
    trap_frequencies_hz: ArrayLike,
    atomic_mass: float,
    hbar: float,
    boltzmann_constant: float,
) -> ThomasFermiState:
    """Return the contact Thomas--Fermi state for a triaxial harmonic trap.

    Frequencies are supplied in hertz; lengths, mass, and constants use SI
    units. The model assumes positive atom number and scattering length and
    returns angular frequencies (rad/s), energy (J), temperature (K), peak
    density (m^-3), radii (m), and peak column densities (m^-2). Dipolar
    corrections are handled by :mod:`dipolar_tf`.
    """

    scalar_inputs = {
        "atom_number": atom_number,
        "scattering_length": scattering_length,
        "atomic_mass": atomic_mass,
        "hbar": hbar,
        "boltzmann_constant": boltzmann_constant,
    }
    for name, value in scalar_inputs.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")

    trap_frequencies = np.asarray(trap_frequencies_hz, dtype=float)
    if trap_frequencies.shape != (3,):
        raise ValueError("trap_frequencies_hz must contain three principal-axis frequencies")
    if not np.all(np.isfinite(trap_frequencies)) or np.any(trap_frequencies <= 0):
        raise ValueError("trap_frequencies_hz must be positive and finite")

    trap_angular_frequencies = 2 * np.pi * trap_frequencies
    geometric_mean_frequency = float(trap_angular_frequencies.prod() ** (1 / 3))
    harmonic_oscillator_length = float(np.sqrt(hbar / (atomic_mass * geometric_mean_frequency)))
    chemical_potential = float(
        0.5
        * (15 * atom_number * scattering_length / harmonic_oscillator_length) ** (2 / 5)
        * hbar
        * geometric_mean_frequency
    )
    chemical_potential_temperature = float(chemical_potential / boltzmann_constant)
    peak_density = float(chemical_potential * atomic_mass / (4 * np.pi * hbar**2 * scattering_length))
    radii = np.sqrt(2 * chemical_potential / (atomic_mass * trap_angular_frequencies**2))
    column_density = (4 / 3) * peak_density * radii
    atom_number_check = float((8 * np.pi / 15) * peak_density * radii.prod())

    return ThomasFermiState(
        trap_angular_frequencies=trap_angular_frequencies,
        geometric_mean_frequency=geometric_mean_frequency,
        harmonic_oscillator_length=harmonic_oscillator_length,
        chemical_potential=chemical_potential,
        chemical_potential_temperature=chemical_potential_temperature,
        peak_density=peak_density,
        radii=radii,
        column_density=column_density,
        atom_number_check=atom_number_check,
    )


def recoil_quantities(
    hbar: float,
    wavevector: float,
    atomic_mass: float,
    boltzmann_constant: float,
) -> tuple[float, float, float]:
    """Return recoil energy (J), temperature (K), and velocity (m/s)."""

    scalar_inputs = {
        "hbar": hbar,
        "wavevector": wavevector,
        "atomic_mass": atomic_mass,
        "boltzmann_constant": boltzmann_constant,
    }
    for name, value in scalar_inputs.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")

    recoil_energy = (hbar * wavevector) ** 2 / (2 * atomic_mass)
    recoil_temperature = recoil_energy / boltzmann_constant
    recoil_velocity = hbar * wavevector / atomic_mass
    return float(recoil_energy), float(recoil_temperature), float(recoil_velocity)
