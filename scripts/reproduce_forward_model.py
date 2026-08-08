"""Reproduce the static endpoint and repeated-exposure forward model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from non_destructive_image.atomic_model import recoil_quantities
from non_destructive_image.atomic_response import (
    branch_summed_scattered_photons_per_atom,
)
from non_destructive_image.multiframe_thermodynamics import (
    IDEAL_BOSE_ENERGY_COEFFICIENT,
    oxford_multiframe_contract_from_configs,
    simulate_polarised_thermodynamic_sequence,
)
from non_destructive_image.reconstruction.scalar_measurements import PCINuisanceValues

try:  # Package import in tests; direct import when executed as a file.
    from scripts._common import (
        REPOSITORY_ROOT,
        endpoint_products,
        input_identity,
        load_configs,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI acceptance
    from _common import (
        REPOSITORY_ROOT,
        endpoint_products,
        input_identity,
        load_configs,
        write_json,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=REPOSITORY_ROOT / "configs"
    )
    parser.add_argument(
        "--output", type=Path, default=REPOSITORY_ROOT / "outputs" / "forward_model.json"
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _disturbance(model: dict[str, Any], reproduction: dict[str, Any], response: Any) -> tuple[Any, float]:
    constants = model["constants"]
    atom = model["atom"]
    probe = reproduction["probe"]
    hbar = float(constants["hbar"])
    wavelength = float(atom["transition_wavelength_m"])
    gamma = float(atom["natural_linewidth_rad_s"])
    h_planck = 2.0 * np.pi * hbar
    saturation_intensity = (
        np.pi * h_planck * float(constants["speed_of_light"]) * gamma
        / (3.0 * wavelength**3)
    )
    atomic_mass = float(atom["mass_number"]) * float(constants["atomic_mass_unit"])
    recoil_energy, _, _ = recoil_quantities(
        hbar,
        2.0 * np.pi / wavelength,
        atomic_mass,
        float(constants["boltzmann_constant"]),
    )
    fluence = float(probe["reference_fluence_mw_us"])
    power = float(probe["power_mw"])
    scattering = branch_summed_scattered_photons_per_atom(
        float(probe["detuning_hz"]),
        power,
        fluence / power * 1e-6,
        float(saturation_intensity),
        gamma,
        float(model["imaging_geometry"]["probe_diameter_m"]),
        response,
        use_peak_intensity=True,
    )
    return scattering, recoil_energy


def reproduce(config_dir: Path) -> dict[str, Any]:
    """Return the forward-model payload without writing it."""

    model, reference, reproduction = load_configs(config_dir)
    products = endpoint_products(model, reference, reproduction)
    endpoint_rows: list[dict[str, Any]] = []
    for product in products:
        nuisance = PCINuisanceValues(
            float(reproduction["acquisition"]["photoelectrons_per_i0_pixel_at_300_mw_us"]),
            float(reproduction["acquisition"]["dark_expected_electrons_per_pixel"]),
        )
        role_names, expected = product.canonical_operator.expected_linked_sequence_from_density_maps(
            [product.canonical_density_m2], nuisance
        )
        radii_um = np.asarray(product.state.radii_m, dtype=float) * 1e6
        endpoint_rows.append(
            {
                "label": product.spec.label,
                "radii_um": radii_um.tolist(),
                "projected_rms_widths_um": {
                    "sigma_y": float(radii_um[1] / np.sqrt(7.0)),
                    "sigma_z": float(radii_um[2] / np.sqrt(7.0)),
                },
                "peak_column_density_m2": float(product.state.peak_column_density_m2[0]),
                "demagnetisation_factors": np.asarray(
                    product.state.demagnetisation_factors, dtype=float
                ).tolist(),
                "total_energy_j": float(product.state.total_energy_j),
                "virial_relative_residual": float(product.state.virial_relative_residual),
                "stationarity_max_abs": float(product.state.stationarity_max_abs),
                "scalar_response": asdict(product.scalar_response),
                "mean_raw_electrons": {
                    name: {
                        "minimum": float(np.min(values)),
                        "maximum": float(np.max(values)),
                        "sum": float(np.sum(values)),
                    }
                    for name, values in zip(role_names, expected, strict=True)
                },
            }
        )

    scattering, recoil_energy = _disturbance(model, reproduction, products[0].response)
    thermodynamic_contract = oxford_multiframe_contract_from_configs(reference, model)
    initial = thermodynamic_contract.initial_state(
        str(reproduction["reference"]["repetition_id"])
    )
    sequence = simulate_polarised_thermodynamic_sequence(
        imaging_method="pci",
        branch_scattering=scattering,
        initial_state=initial,
        energy_coefficient=IDEAL_BOSE_ENERGY_COEFFICIENT,
        recoil_energy_j=recoil_energy,
        maximum_pulses=15,
        cadence_name=thermodynamic_contract.cadence_name,
        cadence_period_s=thermodynamic_contract.cadence_minimum_period_s,
        contract=thermodynamic_contract,
        condensate_depletion_fraction=0.30,
        reabsorption_energy_fraction=0.0,
        include_condensate_core_energy=False,
        continue_after_depletion_for_diagnostic=True,
    )
    widths = [row["projected_rms_widths_um"] for row in endpoint_rows]
    return {
        "schema": "equilibrium_imaging_forward_reproduction_v1",
        "status": "model_conditional_reproduction",
        "identity": input_identity(config_dir),
        "source_commit": model["source"]["commit"],
        "endpoint_states": endpoint_rows,
        "orientation_contrast_um": {
            "delta_sigma_y": widths[1]["sigma_y"] - widths[0]["sigma_y"],
            "delta_sigma_z": widths[1]["sigma_z"] - widths[0]["sigma_z"],
        },
        "reference_probe": {
            "fluence_mw_us": reproduction["probe"]["reference_fluence_mw_us"],
            "detuning_hz": reproduction["probe"]["detuning_hz"],
            "dimensionless_detuning": scattering.dimensionless_detuning,
            "incident_saturation_parameter": scattering.incident_saturation_parameter,
            "total_saturation_parameter": scattering.total_saturation_parameter,
            "branch_labels": list(scattering.branch_labels),
            "scattered_photons_per_atom_by_branch": list(
                scattering.photons_per_atom_by_branch
            ),
            "total_scattered_photons_per_atom": scattering.total_photons_per_atom,
            "recoil_energy_j": recoil_energy,
        },
        "conditional_thermodynamic_sequence": {
            "cadence_status": sequence.cadence_status,
            "stop_reason": sequence.stop_reason,
            "first_excluded_frame": sequence.first_excluded_frame,
            "condensate_depletion_frame": sequence.condensate_depletion_frame,
            "states": [asdict(state) for state in sequence.states],
        },
        "limitations": [
            "independent equilibrium endpoints, not a field-rotation trajectory",
            "fixed-number recoil-only update conditional on complete re-equilibration",
            "core-only Thomas-Fermi state with no kinetic or thermal mean-field correction",
            "ideal isolated-line optical response, not an apparatus calibration",
        ],
    }


def main() -> None:
    args = _arguments()
    payload = reproduce(args.config_dir)
    if args.validate_only:
        print("configuration and forward-model contract validated")
        return
    write_json(args.output, payload)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
