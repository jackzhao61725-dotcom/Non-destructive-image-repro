"""Deterministic recurrence used by the admitted multishot intensity builder."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def accumulate_snr(snr: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return root-sum-square accumulated SNR for consecutive frames."""

    return np.sqrt(np.nancumsum(np.where(np.isfinite(snr), np.asarray(snr, float) ** 2, 0.0)))


def simulate_multishot_sequence(
    photons_scattered_per_atom_per_shot: float,
    initial_condensate_atoms: float,
    loss_fraction_limit: float = 0.30,
    max_shots: int = 400,
    *,
    model: Literal["heating", "loss"] = "heating",
    total_atoms: float | None = None,
    initial_temperature_k: float | None = None,
    critical_temperature_k: float | None = None,
    energy_coefficient_j_per_k4: float | None = None,
    deposited_energy_per_atom_per_shot_j: float | None = None,
    eta_coll: float = 1.0,
    phase_from_n0: Callable[[float], float] | None = None,
    snr_from_phi: Callable[[float], float] | None = None,
) -> dict[str, NDArray[np.floating]]:
    """Step through the fixed-coefficient multishot intensity recurrence.

    Imaging, camera noise, and readout-specific SNR stay outside this helper.
    Callbacks supply the phase and deterministic SNR for each population state.
    """

    if model not in {"heating", "loss"}:
        raise ValueError("model must be 'heating' or 'loss'")
    if max_shots < 0:
        raise ValueError("max_shots must be non-negative")
    if not (0 <= loss_fraction_limit <= 1):
        raise ValueError("loss_fraction_limit must be between 0 and 1")
    if photons_scattered_per_atom_per_shot < 0:
        raise ValueError("photons_scattered_per_atom_per_shot must be non-negative")
    if initial_condensate_atoms <= 0:
        raise ValueError("initial_condensate_atoms must be positive")

    if model == "heating":
        required = {
            "total_atoms": total_atoms,
            "initial_temperature_k": initial_temperature_k,
            "critical_temperature_k": critical_temperature_k,
            "energy_coefficient_j_per_k4": energy_coefficient_j_per_k4,
            "deposited_energy_per_atom_per_shot_j": deposited_energy_per_atom_per_shot_j,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"heating model requires: {', '.join(missing)}")

    phase_fn = phase_from_n0 or (lambda _n0: np.nan)
    snr_fn = snr_from_phi or (lambda _phi: np.nan)

    out: dict[str, list[float]] = {
        "shot": [],
        "N0": [],
        "condensate_fraction": [],
        "loss_fraction": [],
        "frac": [],
        "T": [],
        "phi": [],
        "snr": [],
    }

    temperature = initial_temperature_k
    for shot in range(max_shots + 1):
        if model == "heating":
            assert total_atoms is not None
            assert temperature is not None
            assert critical_temperature_k is not None
            condensate_atoms = total_atoms * (1 - (temperature / critical_temperature_k) ** 3)
            temperature_now = temperature
        else:
            condensate_atoms = initial_condensate_atoms * np.exp(
                -eta_coll * photons_scattered_per_atom_per_shot * shot
            )
            temperature_now = np.nan

        if condensate_atoms <= 0:
            break

        loss_fraction = 1 - condensate_atoms / initial_condensate_atoms
        phi = phase_fn(condensate_atoms)
        snr = snr_fn(phi)

        out["shot"].append(float(shot))
        out["N0"].append(float(condensate_atoms))
        out["condensate_fraction"].append(float(condensate_atoms / initial_condensate_atoms))
        out["loss_fraction"].append(float(loss_fraction))
        out["frac"].append(float(loss_fraction))
        out["T"].append(float(temperature_now))
        out["phi"].append(float(phi))
        out["snr"].append(float(snr))

        if loss_fraction >= loss_fraction_limit:
            break

        if model == "heating":
            assert temperature is not None
            assert energy_coefficient_j_per_k4 is not None
            assert deposited_energy_per_atom_per_shot_j is not None
            temperature = (temperature**4 + deposited_energy_per_atom_per_shot_j / energy_coefficient_j_per_k4) ** 0.25

    return {key: np.asarray(value) for key, value in out.items()}
