"""Oxford-anchored thermodynamic states for repeated dispersive exposures.

This module implements the approved quasi-equilibrium, recoil-limited,
fixed-trapped-number screening model.  It deliberately does not replace the
historical notebook-aligned multi-shot helper in :mod:`non_destructive_image.multishot`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import zeta

from .dipolar_tf import (
    DipolarThomasFermiState,
    build_dipolar_thomas_fermi_state,
    scale_dipolar_thomas_fermi_state,
)
from .atomic_response import BranchScatteringResult
from .reference_state import (
    POLARISED_DIPOLAR_TF_CORE,
    ReferenceState,
    reference_state_from_config,
    reference_states_from_config,
    select_reference_condition,
)


FARADAY_SCATTERING_FACTOR = 46.0 / 91.0
FARADAY_ROTATION_FACTOR = -45.0 / 91.0
IDEAL_BOSE_ENERGY_COEFFICIENT = float(3.0 * zeta(4.0) / zeta(3.0))
CLASSICAL_HARMONIC_ENERGY_COEFFICIENT = 3.0

CadenceStatus = Literal[
    "conditional_on_rethermalisation",
    "unsupported_cadence",
]
SolveStatus = Literal["supported", "unsupported_domain"]
ThermodynamicImagingMethod = Literal["faraday", "pci", "dgi"]
ThermodynamicScatteringModel = Literal[
    "legacy_faraday_scaled_saturated_two_level",
    "polarised_branch_sum_common_detuning",
]


OxfordInitialState = ReferenceState


@dataclass(frozen=True)
class OxfordMultiframeContract:
    """Validated measured-scale closure and atomic inputs."""

    initial_states: tuple[OxfordInitialState, ...]
    trap_frequencies_hz: tuple[float, float, float]
    scattering_length_m: float
    dipolar_length_m: float
    epsilon_dd: float
    dipole_axis: int
    core_scaling_reference_atoms: float
    atomic_mass_kg: float
    hbar_j_s: float
    boltzmann_constant_j_k: float
    measured_scale_intercept: float
    measured_scale_slope: float
    source_fit_intercept: float
    source_fit_slope: float
    measured_x_domain: tuple[float, float]
    temperature_ratio_domain: tuple[float, float]
    closure_validation_xy: tuple[tuple[float, float], ...]
    cadence_name: str
    cadence_minimum_period_s: float

    def initial_state(self, state_id: str) -> OxfordInitialState:
        """Return one repetition without separating its measured triplet."""

        for state in self.initial_states:
            if state.state_id == state_id:
                return state
        raise ValueError(f"unknown Oxford initial-state id {state_id!r}")


@dataclass(frozen=True)
class ThermodynamicState:
    """One initial or post-pulse thermodynamic state."""

    frame_index: int
    temperature_nk: float
    condensate_atoms: float
    thermal_atoms: float
    condensate_fraction: float
    cumulative_scattered_photons_per_atom: float
    cumulative_recoil_energy_j_per_trapped_atom: float
    number_conservation_relative_residual: float
    energy_equation_relative_residual: float
    closure_residual_atoms: float
    accepted_frame: bool
    failure_reason: str | None


@dataclass(frozen=True)
class StateSolveResult:
    """Supported state or an explicit Oxford-domain stop."""

    status: SolveStatus
    state: ThermodynamicState | None
    reason: str | None


@dataclass(frozen=True)
class ThermodynamicPulseScattering:
    """Resolved per-pulse disturbance consumed by the state-update engine."""

    imaging_method: ThermodynamicImagingMethod
    model: ThermodynamicScatteringModel
    total_photons_per_atom_per_pulse: float
    branch_scattering: BranchScatteringResult | None = None
    unit_strength_two_level_photons_per_atom_per_pulse: float | None = None

    def __post_init__(self) -> None:
        if self.imaging_method not in ("faraday", "pci", "dgi"):
            raise ValueError("imaging_method must be faraday, pci or dgi")
        if self.model not in (
            "legacy_faraday_scaled_saturated_two_level",
            "polarised_branch_sum_common_detuning",
        ):
            raise ValueError("unknown thermodynamic scattering model")

        total = float(self.total_photons_per_atom_per_pulse)
        if not np.isfinite(total) or total < 0.0:
            raise ValueError("total scattered photons must be finite and non-negative")
        object.__setattr__(self, "total_photons_per_atom_per_pulse", total)

        if self.model == "legacy_faraday_scaled_saturated_two_level":
            if self.imaging_method != "faraday":
                raise ValueError("legacy Faraday scattering requires imaging_method='faraday'")
            if self.branch_scattering is not None:
                raise ValueError("legacy Faraday scattering must not carry a branch result")
            source = self.unit_strength_two_level_photons_per_atom_per_pulse
            if source is None:
                raise ValueError("legacy Faraday scattering requires its two-level source")
            source = float(source)
            if not np.isfinite(source) or source < 0.0:
                raise ValueError("two-level scattered photons must be finite and non-negative")
            expected = FARADAY_SCATTERING_FACTOR * source
            if not np.isclose(total, expected, rtol=1e-15, atol=0.0):
                raise ValueError("legacy Faraday total must apply 46/91 exactly once")
            object.__setattr__(
                self,
                "unit_strength_two_level_photons_per_atom_per_pulse",
                source,
            )
            return

        if self.imaging_method not in ("pci", "dgi"):
            raise ValueError("polarised branch scattering is restricted to PCI or DGI")
        if self.unit_strength_two_level_photons_per_atom_per_pulse is not None:
            raise ValueError("branch-summed scattering must not carry a pseudo two-level rate")
        if not isinstance(self.branch_scattering, BranchScatteringResult):
            raise ValueError("branch-summed scattering requires its replayable result")
        if not np.isclose(
            total,
            self.branch_scattering.total_photons_per_atom,
            rtol=1e-15,
            atol=0.0,
        ):
            raise ValueError("thermodynamic total disagrees with the bound branch result")

    @property
    def branch_labels(self) -> tuple[str, ...]:
        """Return replayable branch labels, or none for the legacy path."""

        if self.branch_scattering is None:
            return ()
        return self.branch_scattering.branch_labels

    @property
    def photons_per_atom_by_branch(self) -> tuple[float, ...]:
        """Return branch-resolved rates, or none for the legacy path."""

        if self.branch_scattering is None:
            return ()
        return self.branch_scattering.photons_per_atom_by_branch

    @property
    def selected_eigenmode(self) -> str | None:
        """Return the exact bound response eigenmode when available."""

        if self.branch_scattering is None:
            return None
        return self.branch_scattering.response.selected_eigenmode

    @property
    def response_species(self) -> str | None:
        """Return the exact bound response species when available."""

        if self.branch_scattering is None:
            return None
        return self.branch_scattering.response.species

    @property
    def response_transition_label(self) -> str | None:
        """Return the exact bound response transition when available."""

        if self.branch_scattering is None:
            return None
        return self.branch_scattering.response.transition_label


@dataclass(frozen=True)
class ThermodynamicSequence:
    """A bounded repeated-exposure trajectory and its stopping status."""

    initial_state_id: str
    energy_coefficient: float
    pulse_scattering: ThermodynamicPulseScattering
    recoil_energy_j_per_scattering_cycle: float
    reabsorption_energy_fraction: float
    include_condensate_core_energy: bool
    cadence_status: CadenceStatus
    states: tuple[ThermodynamicState, ...]
    condensate_depletion_frame: int | None
    first_excluded_frame: int | None
    stop_reason: str | None

    @property
    def imaging_method(self) -> ThermodynamicImagingMethod:
        """Return the optical method recorded for the atomic disturbance."""

        return self.pulse_scattering.imaging_method

    @property
    def scattering_model(self) -> ThermodynamicScatteringModel:
        """Return the provenance model for the consumed scattering rate."""

        return self.pulse_scattering.model

    @property
    def scattered_photons_per_atom_per_pulse(self) -> float:
        """Return the already-resolved rate consumed by thermodynamics."""

        return self.pulse_scattering.total_photons_per_atom_per_pulse

    @property
    def two_level_scattered_photons_per_atom_per_pulse(self) -> float | None:
        """Return the legacy unit-strength source, if this is a Faraday run."""

        return self.pulse_scattering.unit_strength_two_level_photons_per_atom_per_pulse

    @property
    def faraday_scattered_photons_per_atom_per_pulse(self) -> float | None:
        """Return the legacy Faraday total without mislabelling PCI or DGI."""

        if self.imaging_method != "faraday":
            return None
        return self.scattered_photons_per_atom_per_pulse


@dataclass(frozen=True)
class ThermodynamicExposure:
    """One probe exposure bracketed by its pre- and post-pulse states."""

    exposure_index: int
    pre_state: ThermodynamicState
    post_state: ThermodynamicState
    accepted_by_thermodynamics: bool
    failure_reason: str | None


def _finite_number(
    mapping: Mapping[str, object], key: str, *, positive: bool = False
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    number = float(value)
    if not np.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{key} must be {qualifier}")
    return number


def _mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _number_pair(mapping: Mapping[str, object], key: str) -> tuple[float, float]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must contain two finite numbers")
    numbers = tuple(float(item) for item in value)
    if len(numbers) != 2 or not np.all(np.isfinite(numbers)):
        raise ValueError(f"{key} must contain two finite numbers")
    if not numbers[0] < numbers[1]:
        raise ValueError(f"{key} lower bound must be less than its upper bound")
    return numbers[0], numbers[1]


def oxford_multiframe_contract_from_configs(
    initial_condition_config: Mapping[str, object],
    model_config: Mapping[str, object],
    *,
    condition_id: str = "oxford_bimodal_300ms",
    representative_repetition_id: str = "oxford_300ms_rep_1_central",
) -> OxfordMultiframeContract:
    """Build the one Oxford contract embedded in the existing condition record."""

    condition = select_reference_condition(initial_condition_config, condition_id)
    thermodynamics = _mapping(condition, "multiframe_thermodynamics")
    states = reference_states_from_config(
        initial_condition_config,
        condition_id=condition_id,
    )
    if len(states) != 3:
        raise ValueError("the Oxford contract requires three unique 300 ms repetitions")
    representative = reference_state_from_config(
        initial_condition_config,
        condition_id=condition_id,
        repetition_id=representative_repetition_id,
    )

    selector = model_config.get("reference_state_selector")
    if selector is not None:
        if not isinstance(selector, Mapping):
            raise ValueError("reference_state_selector must be a mapping")
        if selector.get("condition_id") != condition_id:
            raise ValueError("model selector and multiframe condition_id disagree")
        if selector.get("repetition_id") != representative_repetition_id:
            raise ValueError("model selector and representative repetition disagree")
        if selector.get("optical_forward_model") != POLARISED_DIPOLAR_TF_CORE:
            raise ValueError("multiframe coupling requires the polarised dipolar TF core")
        if selector.get("thermal_halo_rendered") is not False:
            raise ValueError("multiframe coupling requires thermal_halo_rendered=false")

    resolved_reference = model_config.get("resolved_reference_state")
    if resolved_reference is not None:
        if not isinstance(resolved_reference, Mapping):
            raise ValueError("resolved_reference_state must be a mapping")
        if not isinstance(selector, Mapping):
            raise ValueError(
                "resolved_reference_state requires its reference_state_selector"
            )
        expected_strings = {
            "source_config_path": selector.get("source_config_path"),
            "condition_id": representative.condition_id,
            "repetition_id": representative.repetition_id,
            "selection_basis": selector.get("selection_basis"),
            "temperature_source_units": representative.temperature_source_units,
            "temperature_runtime_units": representative.temperature_runtime_units,
            "dataset_doi": representative.dataset_doi,
            "dataset_zip_filename": representative.dataset_zip_filename,
            "dataset_zip_sha256": representative.dataset_zip_sha256,
            "raw_repetitions_zip_member": representative.raw_repetitions_zip_member,
            "raw_repetitions_sha256": representative.raw_repetitions_sha256,
            "optical_forward_model": selector.get("optical_forward_model"),
        }
        for key, expected in expected_strings.items():
            if resolved_reference.get(key) != expected:
                raise ValueError(
                    f"resolved reference state and Oxford source record disagree for {key}"
                )
        if resolved_reference.get("thermal_halo_rendered") is not False:
            raise ValueError(
                "resolved reference state and Oxford source record disagree for "
                "thermal_halo_rendered"
            )
        expected_numbers = {
            "condensate_atoms": representative.condensate_atoms,
            "thermal_atoms": representative.thermal_atoms,
            "trapped_atoms": representative.trapped_atoms,
            "temperature_nk": representative.temperature_nk,
            "temperature_k": representative.temperature_k,
            "scattering_length_bohr": representative.scattering_length_bohr,
            "dipolar_length_bohr": representative.dipolar_length_bohr,
            "epsilon_dd": representative.epsilon_dd,
            "dipolar_tf_scattering_length_bohr": (
                representative.dipolar_tf_scattering_length_bohr
            ),
            "runtime_scattering_length_bohr": (
                representative.dipolar_tf_scattering_length_bohr
            ),
        }
        for key, expected in expected_numbers.items():
            if _finite_number(resolved_reference, key) != expected:
                raise ValueError(
                    f"resolved reference state and Oxford source record disagree for {key}"
                )
        resolved_source_line = resolved_reference.get("source_csv_line")
        if (
            isinstance(resolved_source_line, bool)
            or not isinstance(resolved_source_line, int)
            or resolved_source_line != representative.source_csv_line
        ):
            raise ValueError(
                "resolved reference state and Oxford source record disagree for source_csv_line"
            )
        resolved_trap = resolved_reference.get("trap_frequencies_hz")
        if not isinstance(resolved_trap, Sequence) or isinstance(
            resolved_trap, (str, bytes)
        ):
            raise ValueError("resolved reference trap frequencies must be a sequence")
        if len(resolved_trap) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in resolved_trap
        ):
            raise ValueError(
                "resolved reference trap frequencies must contain three numeric values"
            )
        resolved_trap_tuple = tuple(float(value) for value in resolved_trap)
        if not all(np.isfinite(value) and value > 0.0 for value in resolved_trap_tuple):
            raise ValueError(
                "resolved reference trap frequencies must be positive and finite"
            )
        if resolved_trap_tuple != representative.trap_frequencies_hz:
            raise ValueError(
                "resolved reference state and Oxford source record disagree for trap_frequencies_hz"
            )

    closure = _mapping(thermodynamics, "closure")
    measured_scale = _mapping(closure, "measured_scale")
    source_fit = _mapping(closure, "source_fit")
    combined_calibration = _mapping(closure, "combined_calibration")
    domain = _mapping(closure, "domain")
    cadence = _mapping(thermodynamics, "cadence")
    validation_points = closure.get("accepted_average_xy")
    if not isinstance(validation_points, Sequence) or isinstance(
        validation_points, (str, bytes)
    ):
        raise ValueError("accepted_average_xy must be a sequence")
    xy: list[tuple[float, float]] = []
    for point in validation_points:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)):
            raise ValueError("each closure validation point must contain x and y")
        values = tuple(float(value) for value in point)
        if len(values) != 2 or not np.all(np.isfinite(values)):
            raise ValueError("each closure validation point must contain finite x and y")
        xy.append((values[0], values[1]))
    if not xy:
        raise ValueError("closure validation points must not be empty")

    constants = _mapping(model_config, "constants")
    atom = _mapping(model_config, "atom")
    hbar = _finite_number(constants, "hbar", positive=True)
    boltzmann = _finite_number(constants, "boltzmann_constant", positive=True)
    atomic_mass = _finite_number(atom, "mass_number", positive=True) * _finite_number(
        constants, "atomic_mass_unit", positive=True
    )
    bohr_radius_m = _finite_number(constants, "bohr_radius_m", positive=True)
    scattering_length_m = (
        representative.dipolar_tf_scattering_length_bohr * bohr_radius_m
    )
    dipolar_length_m = representative.dipolar_length_bohr * bohr_radius_m
    cadence_name = cadence.get("name")
    if not isinstance(cadence_name, str) or not cadence_name:
        raise ValueError("cadence name must be a non-empty string")

    beta = _finite_number(combined_calibration, "beta", positive=True)
    validation_x = np.asarray([point[0] for point in xy], dtype=float)
    source_calibrated_x_domain = (
        float(np.min(validation_x)),
        float(np.max(validation_x)),
    )
    legacy_rounded_x_domain = _number_pair(domain, "x_two_fifths")
    rounded_source_domain = tuple(
        round(value, 4) for value in source_calibrated_x_domain
    )
    if not np.allclose(
        legacy_rounded_x_domain,
        rounded_source_domain,
        rtol=0.0,
        atol=5e-15,
    ):
        raise ValueError(
            "legacy Oxford x domain must equal the accepted source extrema "
            "rounded to four decimal places"
        )
    measured_x_domain = tuple(
        beta ** (2.0 / 5.0) * value for value in source_calibrated_x_domain
    )

    return OxfordMultiframeContract(
        initial_states=states,
        trap_frequencies_hz=representative.trap_frequencies_hz,
        scattering_length_m=scattering_length_m,
        dipolar_length_m=dipolar_length_m,
        epsilon_dd=representative.epsilon_dd,
        dipole_axis=representative.dipole_axis_index,
        core_scaling_reference_atoms=representative.condensate_atoms,
        atomic_mass_kg=atomic_mass,
        hbar_j_s=hbar,
        boltzmann_constant_j_k=boltzmann,
        measured_scale_intercept=_finite_number(
            measured_scale, "intercept", positive=True
        ),
        measured_scale_slope=_finite_number(measured_scale, "slope", positive=True),
        source_fit_intercept=_finite_number(source_fit, "intercept", positive=True),
        source_fit_slope=_finite_number(source_fit, "slope", positive=True),
        measured_x_domain=measured_x_domain,
        temperature_ratio_domain=_number_pair(domain, "temperature_over_initial"),
        closure_validation_xy=tuple(xy),
        cadence_name=cadence_name,
        cadence_minimum_period_s=_finite_number(
            cadence, "minimum_period_s", positive=True
        ),
    )


def nominal_critical_atoms(
    temperature_nk: float,
    contract: OxfordMultiframeContract,
) -> float:
    """Return the nominal ideal critical number on the measured trap scale."""

    if not np.isfinite(temperature_nk) or temperature_nk <= 0.0:
        raise ValueError("temperature_nk must be positive and finite")
    omega = 2.0 * np.pi * np.asarray(contract.trap_frequencies_hz, dtype=float)
    omega_bar = float(np.prod(omega) ** (1.0 / 3.0))
    temperature_k = temperature_nk * 1e-9
    return float(
        zeta(3.0)
        * (
            contract.boltzmann_constant_j_k
            * temperature_k
            / (contract.hbar_j_s * omega_bar)
        )
        ** 3
    )


def measured_scale_non_saturation(
    temperature_nk: float,
    condensate_atoms: float,
    contract: OxfordMultiframeContract,
) -> float:
    """Return ``F_ns(T, N0)`` without the trajectory-specific anchor."""

    if not np.isfinite(condensate_atoms) or condensate_atoms < 0.0:
        raise ValueError("condensate_atoms must be finite and non-negative")
    critical = nominal_critical_atoms(temperature_nk, contract)
    x_two_fifths = (condensate_atoms / critical) ** (2.0 / 5.0)
    return float(
        critical
        * (
            contract.measured_scale_intercept
            + contract.measured_scale_slope * x_two_fifths
        )
    )


def anchored_thermal_atoms(
    temperature_nk: float,
    condensate_atoms: float,
    initial_state: OxfordInitialState,
    contract: OxfordMultiframeContract,
) -> float:
    """Return the anchored Oxford thermal population."""

    return float(
        initial_state.thermal_atoms
        + measured_scale_non_saturation(temperature_nk, condensate_atoms, contract)
        - measured_scale_non_saturation(
            initial_state.temperature_nk,
            initial_state.condensate_atoms,
            contract,
        )
    )


def closure_validation_residuals(
    contract: OxfordMultiframeContract,
) -> tuple[float, float, float]:
    """Return relative RMS, mean-absolute and maximum source-line residuals."""

    points = np.asarray(contract.closure_validation_xy, dtype=float)
    prediction = (
        contract.source_fit_intercept + contract.source_fit_slope * points[:, 0]
    )
    residual = (points[:, 1] - prediction) / points[:, 1]
    return (
        float(np.sqrt(np.mean(residual**2))),
        float(np.mean(np.abs(residual))),
        float(np.max(np.abs(residual))),
    )


def cadence_status(
    cadence_name: str,
    period_s: float,
    contract: OxfordMultiframeContract,
) -> CadenceStatus:
    """Return the sole supported conditional cadence status."""

    if not np.isfinite(period_s) or period_s <= 0.0:
        raise ValueError("period_s must be positive and finite")
    if (
        cadence_name != contract.cadence_name
        or period_s + 1e-15 < contract.cadence_minimum_period_s
    ):
        return "unsupported_cadence"
    return "conditional_on_rethermalisation"


@lru_cache(maxsize=8)
def _dipolar_core_scaling_reference(
    contract: OxfordMultiframeContract,
) -> DipolarThomasFermiState:
    return build_dipolar_thomas_fermi_state(
        atom_number=contract.core_scaling_reference_atoms,
        scattering_length_m=contract.scattering_length_m,
        dipolar_length_m=contract.dipolar_length_m,
        trap_frequencies_hz=contract.trap_frequencies_hz,
        dipole_axis=contract.dipole_axis,
        atomic_mass_kg=contract.atomic_mass_kg,
        hbar_j_s=contract.hbar_j_s,
        boltzmann_constant_j_k=contract.boltzmann_constant_j_k,
    )


def condensate_core_state(
    condensate_atoms: float,
    contract: OxfordMultiframeContract,
) -> DipolarThomasFermiState:
    """Return the Oxford dipolar-TF core using exact fixed-shape scaling."""

    if not np.isfinite(condensate_atoms) or condensate_atoms <= 0.0:
        raise ValueError("condensate_atoms must be positive and finite")
    return scale_dipolar_thomas_fermi_state(
        _dipolar_core_scaling_reference(contract),
        condensate_atoms,
    )


def condensate_core_energy_j(
    condensate_atoms: float,
    contract: OxfordMultiframeContract,
) -> float:
    """Return the total dipolar-TF condensate-core energy, ``5 mu N / 7``."""

    if condensate_atoms == 0.0:
        return 0.0
    core = condensate_core_state(condensate_atoms, contract)
    return float((5.0 / 7.0) * core.chemical_potential * condensate_atoms)


def _state_total_energy_j(
    condensate_atoms: float,
    thermal_atoms: float,
    temperature_nk: float,
    energy_coefficient: float,
    contract: OxfordMultiframeContract,
    *,
    include_condensate_core_energy: bool,
) -> float:
    total_energy = (
        energy_coefficient
        * thermal_atoms
        * contract.boltzmann_constant_j_k
        * temperature_nk
        * 1e-9
    )
    if include_condensate_core_energy:
        total_energy += condensate_core_energy_j(condensate_atoms, contract)
    return float(total_energy)


def solve_state_at_energy(
    *,
    frame_index: int,
    target_energy_j: float,
    initial_state: OxfordInitialState,
    previous_condensate_atoms: float,
    energy_coefficient: float,
    contract: OxfordMultiframeContract,
    cumulative_scattered_photons_per_atom: float = 0.0,
    cumulative_recoil_energy_j_per_trapped_atom: float = 0.0,
    include_condensate_core_energy: bool = False,
) -> StateSolveResult:
    """Solve the fixed-number, anchored-closure state with one bounded root."""

    scalars = (
        target_energy_j,
        previous_condensate_atoms,
        energy_coefficient,
        cumulative_scattered_photons_per_atom,
        cumulative_recoil_energy_j_per_trapped_atom,
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("state-solve inputs must be finite")
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if target_energy_j <= 0.0 or energy_coefficient <= 0.0:
        raise ValueError("energy values and coefficient must be positive")
    if not 0.0 < previous_condensate_atoms <= initial_state.trapped_atoms:
        raise ValueError("previous_condensate_atoms is outside the trapped population")
    if cumulative_scattered_photons_per_atom < 0.0:
        raise ValueError("cumulative scattered photons must be non-negative")
    if cumulative_recoil_energy_j_per_trapped_atom < 0.0:
        raise ValueError("cumulative recoil energy must be non-negative")

    trapped_atoms = initial_state.trapped_atoms

    def values(condensate_atoms: float) -> tuple[float, float, float]:
        thermal_atoms = trapped_atoms - condensate_atoms
        core_energy = (
            condensate_core_energy_j(condensate_atoms, contract)
            if include_condensate_core_energy
            else 0.0
        )
        thermal_energy = target_energy_j - core_energy
        if thermal_atoms <= 0.0 or thermal_energy <= 0.0:
            return thermal_atoms, float("nan"), float("nan")
        temperature_nk = (
            thermal_energy
            / (
                energy_coefficient
                * thermal_atoms
                * contract.boltzmann_constant_j_k
            )
            * 1e9
        )
        closure = anchored_thermal_atoms(
            temperature_nk,
            condensate_atoms,
            initial_state,
            contract,
        )
        return thermal_atoms, float(temperature_nk), float(thermal_atoms - closure)

    lower = 0.0
    upper = min(previous_condensate_atoms, trapped_atoms * (1.0 - 1e-15))
    lower_residual = values(lower)[2]
    upper_residual = values(upper)[2]
    if not np.isfinite(lower_residual) or not np.isfinite(upper_residual):
        return StateSolveResult(
            status="unsupported_domain",
            state=None,
            reason="bounded_state_solve_non_finite_bracket",
        )
    endpoint_tolerance_atoms = 1e-6
    if abs(lower_residual) <= endpoint_tolerance_atoms:
        root = lower
    elif abs(upper_residual) <= endpoint_tolerance_atoms:
        root = upper
    elif lower_residual * upper_residual > 0.0:
        return StateSolveResult(
            status="unsupported_domain",
            state=None,
            reason="bounded_state_solve_has_no_population_root",
        )
    else:
        root = float(
            brentq(
                lambda condensate_atoms: values(condensate_atoms)[2],
                lower,
                upper,
                xtol=1e-9,
                rtol=1e-14,
                maxiter=200,
            )
        )
    thermal_atoms, temperature_nk, closure_residual = values(root)
    critical = nominal_critical_atoms(temperature_nk, contract)
    x_two_fifths = (root / critical) ** (2.0 / 5.0)
    temperature_ratio = temperature_nk / initial_state.temperature_nk
    if not (
        contract.measured_x_domain[0]
        <= x_two_fifths
        <= contract.measured_x_domain[1]
    ):
        return StateSolveResult(
            status="unsupported_domain",
            state=None,
            reason="outside_oxford_measured_x_domain",
        )
    if not (
        contract.temperature_ratio_domain[0]
        <= temperature_ratio
        <= contract.temperature_ratio_domain[1]
    ):
        return StateSolveResult(
            status="unsupported_domain",
            state=None,
            reason="outside_oxford_temperature_domain",
        )
    if not np.all(np.isfinite((root, thermal_atoms, temperature_nk))):
        return StateSolveResult(
            status="unsupported_domain",
            state=None,
            reason="non_finite_population_or_temperature",
        )
    if root < 0.0 or thermal_atoms < 0.0:
        return StateSolveResult(
            status="unsupported_domain",
            state=None,
            reason="negative_population",
        )

    realised_energy = _state_total_energy_j(
        root,
        thermal_atoms,
        temperature_nk,
        energy_coefficient,
        contract,
        include_condensate_core_energy=include_condensate_core_energy,
    )
    energy_scale = max(
        abs(target_energy_j),
        contract.boltzmann_constant_j_k * temperature_nk * 1e-9,
    )
    state = ThermodynamicState(
        frame_index=frame_index,
        temperature_nk=float(temperature_nk),
        condensate_atoms=float(root),
        thermal_atoms=float(thermal_atoms),
        condensate_fraction=float(root / initial_state.condensate_atoms),
        cumulative_scattered_photons_per_atom=float(
            cumulative_scattered_photons_per_atom
        ),
        cumulative_recoil_energy_j_per_trapped_atom=float(
            cumulative_recoil_energy_j_per_trapped_atom
        ),
        number_conservation_relative_residual=float(
            abs(root + thermal_atoms - trapped_atoms) / trapped_atoms
        ),
        energy_equation_relative_residual=float(
            abs(realised_energy - target_energy_j) / energy_scale
        ),
        closure_residual_atoms=float(closure_residual),
        accepted_frame=True,
        failure_reason=None,
    )
    return StateSolveResult(status="supported", state=state, reason=None)


def thermodynamic_scattering_from_branch_result(
    *,
    imaging_method: Literal["pci", "dgi"],
    result: BranchScatteringResult,
) -> ThermodynamicPulseScattering:
    """Resolve a PCI/DGI branch result without applying another line factor."""

    if imaging_method not in ("pci", "dgi"):
        raise ValueError("branch-summed thermodynamics supports only PCI or DGI")
    if not isinstance(result, BranchScatteringResult):
        raise ValueError("result must be a BranchScatteringResult")
    return ThermodynamicPulseScattering(
        imaging_method=imaging_method,
        model="polarised_branch_sum_common_detuning",
        total_photons_per_atom_per_pulse=result.total_photons_per_atom,
        branch_scattering=result,
    )


def _legacy_faraday_pulse_scattering(
    two_level_scattered_photons_per_atom_per_pulse: float,
) -> ThermodynamicPulseScattering:
    source = float(two_level_scattered_photons_per_atom_per_pulse)
    return ThermodynamicPulseScattering(
        imaging_method="faraday",
        model="legacy_faraday_scaled_saturated_two_level",
        total_photons_per_atom_per_pulse=FARADAY_SCATTERING_FACTOR * source,
        unit_strength_two_level_photons_per_atom_per_pulse=source,
    )


def _simulate_resolved_thermodynamic_sequence(
    *,
    initial_state: OxfordInitialState,
    energy_coefficient: float,
    pulse_scattering: ThermodynamicPulseScattering,
    recoil_energy_j: float,
    maximum_pulses: int,
    cadence_name: str,
    cadence_period_s: float,
    contract: OxfordMultiframeContract,
    condensate_depletion_fraction: float = 0.30,
    reabsorption_energy_fraction: float = 0.0,
    include_condensate_core_energy: bool = False,
    continue_after_depletion_for_diagnostic: bool = False,
) -> ThermodynamicSequence:
    """Update states from an already-resolved non-negative scattering rate."""

    scalars = (
        energy_coefficient,
        recoil_energy_j,
        cadence_period_s,
        condensate_depletion_fraction,
        reabsorption_energy_fraction,
    )
    if not np.all(np.isfinite(scalars)):
        raise ValueError("sequence inputs must be finite")
    if not isinstance(pulse_scattering, ThermodynamicPulseScattering):
        raise ValueError("pulse_scattering must be a ThermodynamicPulseScattering")
    if energy_coefficient <= 0.0 or recoil_energy_j <= 0.0:
        raise ValueError("energy coefficient and recoil energy must be positive")
    if maximum_pulses < 0:
        raise ValueError("maximum_pulses must be non-negative")
    if not 0.0 < condensate_depletion_fraction < 1.0:
        raise ValueError("condensate depletion fraction must be between zero and one")
    if not 0.0 <= reabsorption_energy_fraction <= 0.05:
        raise ValueError("reabsorption energy fraction must lie between 0 and 5%")
    if not isinstance(continue_after_depletion_for_diagnostic, bool):
        raise ValueError("continue_after_depletion_for_diagnostic must be a bool")

    resolved_scattering = pulse_scattering.total_photons_per_atom_per_pulse
    recoil_cycle_energy = 2.0 * recoil_energy_j
    recoil_per_atom_per_pulse = (
        resolved_scattering
        * recoil_cycle_energy
        * (1.0 + reabsorption_energy_fraction)
    )
    initial_energy = _state_total_energy_j(
        initial_state.condensate_atoms,
        initial_state.thermal_atoms,
        initial_state.temperature_nk,
        energy_coefficient,
        contract,
        include_condensate_core_energy=include_condensate_core_energy,
    )
    cadence = cadence_status(cadence_name, cadence_period_s, contract)
    initial_result = solve_state_at_energy(
        frame_index=0,
        target_energy_j=initial_energy,
        initial_state=initial_state,
        previous_condensate_atoms=initial_state.condensate_atoms,
        energy_coefficient=energy_coefficient,
        contract=contract,
        include_condensate_core_energy=include_condensate_core_energy,
    )
    if initial_result.state is None:
        return ThermodynamicSequence(
            initial_state_id=initial_state.state_id,
            energy_coefficient=energy_coefficient,
            pulse_scattering=pulse_scattering,
            recoil_energy_j_per_scattering_cycle=recoil_cycle_energy,
            reabsorption_energy_fraction=reabsorption_energy_fraction,
            include_condensate_core_energy=include_condensate_core_energy,
            cadence_status=cadence,
            states=(),
            condensate_depletion_frame=None,
            first_excluded_frame=0,
            stop_reason=initial_result.reason,
        )
    states = [initial_result.state]
    if cadence == "unsupported_cadence":
        return ThermodynamicSequence(
            initial_state_id=initial_state.state_id,
            energy_coefficient=energy_coefficient,
            pulse_scattering=pulse_scattering,
            recoil_energy_j_per_scattering_cycle=recoil_cycle_energy,
            reabsorption_energy_fraction=reabsorption_energy_fraction,
            include_condensate_core_energy=include_condensate_core_energy,
            cadence_status=cadence,
            states=tuple(states),
            condensate_depletion_frame=None,
            first_excluded_frame=1,
            stop_reason="unsupported_cadence",
        )

    depletion_frame: int | None = None
    for pulse in range(1, maximum_pulses + 1):
        cumulative_recoil = pulse * recoil_per_atom_per_pulse
        target_energy = initial_energy + initial_state.trapped_atoms * cumulative_recoil
        solved = solve_state_at_energy(
            frame_index=pulse,
            target_energy_j=target_energy,
            initial_state=initial_state,
            previous_condensate_atoms=states[-1].condensate_atoms,
            energy_coefficient=energy_coefficient,
            contract=contract,
            cumulative_scattered_photons_per_atom=pulse * resolved_scattering,
            cumulative_recoil_energy_j_per_trapped_atom=cumulative_recoil,
            include_condensate_core_energy=include_condensate_core_energy,
        )
        if solved.state is None:
            return ThermodynamicSequence(
                initial_state_id=initial_state.state_id,
                energy_coefficient=energy_coefficient,
                pulse_scattering=pulse_scattering,
                recoil_energy_j_per_scattering_cycle=recoil_cycle_energy,
                reabsorption_energy_fraction=reabsorption_energy_fraction,
                include_condensate_core_energy=include_condensate_core_energy,
                cadence_status=cadence,
                states=tuple(states),
                condensate_depletion_frame=depletion_frame,
                first_excluded_frame=(
                    depletion_frame if depletion_frame is not None else pulse
                ),
                stop_reason=solved.reason,
            )
        state = solved.state
        if state.temperature_nk + 1e-12 < states[-1].temperature_nk:
            raise RuntimeError("positive recoil energy lowered the temperature")
        if (
            depletion_frame is None
            and state.condensate_fraction <= 1.0 - condensate_depletion_fraction
        ):
            depletion_frame = pulse
        if depletion_frame is not None:
            excluded = replace(
                state,
                accepted_frame=False,
                failure_reason="condensate_depletion_threshold",
            )
            states.append(excluded)
            if not continue_after_depletion_for_diagnostic:
                return ThermodynamicSequence(
                    initial_state_id=initial_state.state_id,
                    energy_coefficient=energy_coefficient,
                    pulse_scattering=pulse_scattering,
                    recoil_energy_j_per_scattering_cycle=recoil_cycle_energy,
                    reabsorption_energy_fraction=reabsorption_energy_fraction,
                    include_condensate_core_energy=include_condensate_core_energy,
                    cadence_status=cadence,
                    states=tuple(states),
                    condensate_depletion_frame=depletion_frame,
                    first_excluded_frame=depletion_frame,
                    stop_reason="condensate_depletion_threshold",
                )
        else:
            states.append(state)

    return ThermodynamicSequence(
        initial_state_id=initial_state.state_id,
        energy_coefficient=energy_coefficient,
        pulse_scattering=pulse_scattering,
        recoil_energy_j_per_scattering_cycle=recoil_cycle_energy,
        reabsorption_energy_fraction=reabsorption_energy_fraction,
        include_condensate_core_energy=include_condensate_core_energy,
        cadence_status=cadence,
        states=tuple(states),
        condensate_depletion_frame=depletion_frame,
        first_excluded_frame=depletion_frame,
        stop_reason=(
            "maximum_pulses_reached_after_depletion"
            if depletion_frame is not None
            else "maximum_pulses_reached"
        ),
    )


def simulate_thermodynamic_sequence(
    *,
    initial_state: OxfordInitialState,
    energy_coefficient: float,
    two_level_scattered_photons_per_atom_per_pulse: float,
    recoil_energy_j: float,
    maximum_pulses: int,
    cadence_name: str,
    cadence_period_s: float,
    contract: OxfordMultiframeContract,
    condensate_depletion_fraction: float = 0.30,
    reabsorption_energy_fraction: float = 0.0,
    include_condensate_core_energy: bool = False,
    continue_after_depletion_for_diagnostic: bool = False,
) -> ThermodynamicSequence:
    """Return one legacy Faraday trajectory with ``46/91`` applied once."""

    return _simulate_resolved_thermodynamic_sequence(
        initial_state=initial_state,
        energy_coefficient=energy_coefficient,
        pulse_scattering=_legacy_faraday_pulse_scattering(
            two_level_scattered_photons_per_atom_per_pulse
        ),
        recoil_energy_j=recoil_energy_j,
        maximum_pulses=maximum_pulses,
        cadence_name=cadence_name,
        cadence_period_s=cadence_period_s,
        contract=contract,
        condensate_depletion_fraction=condensate_depletion_fraction,
        reabsorption_energy_fraction=reabsorption_energy_fraction,
        include_condensate_core_energy=include_condensate_core_energy,
        continue_after_depletion_for_diagnostic=continue_after_depletion_for_diagnostic,
    )


def simulate_polarised_thermodynamic_sequence(
    *,
    imaging_method: Literal["pci", "dgi"],
    branch_scattering: BranchScatteringResult,
    initial_state: OxfordInitialState,
    energy_coefficient: float,
    recoil_energy_j: float,
    maximum_pulses: int,
    cadence_name: str,
    cadence_period_s: float,
    contract: OxfordMultiframeContract,
    condensate_depletion_fraction: float = 0.30,
    reabsorption_energy_fraction: float = 0.0,
    include_condensate_core_energy: bool = False,
    continue_after_depletion_for_diagnostic: bool = False,
) -> ThermodynamicSequence:
    """Return a PCI/DGI trajectory from the exact branch-summed disturbance."""

    pulse_scattering = thermodynamic_scattering_from_branch_result(
        imaging_method=imaging_method,
        result=branch_scattering,
    )
    return _simulate_resolved_thermodynamic_sequence(
        initial_state=initial_state,
        energy_coefficient=energy_coefficient,
        pulse_scattering=pulse_scattering,
        recoil_energy_j=recoil_energy_j,
        maximum_pulses=maximum_pulses,
        cadence_name=cadence_name,
        cadence_period_s=cadence_period_s,
        contract=contract,
        condensate_depletion_fraction=condensate_depletion_fraction,
        reabsorption_energy_fraction=reabsorption_energy_fraction,
        include_condensate_core_energy=include_condensate_core_energy,
        continue_after_depletion_for_diagnostic=continue_after_depletion_for_diagnostic,
    )


def condensate_core_sequence(
    sequence: ThermodynamicSequence,
    contract: OxfordMultiframeContract,
) -> tuple[DipolarThomasFermiState, ...]:
    """Pass only each updated ``N0`` into the polarised dipolar-TF core."""

    return tuple(
        condensate_core_state(state.condensate_atoms, contract)
        for state in sequence.states
    )


def thermodynamic_exposures(
    sequence: ThermodynamicSequence,
) -> tuple[ThermodynamicExposure, ...]:
    """Pair every simulated probe image with its post-pulse acceptance state.

    State zero is the cloud that forms image one.  The pulse associated with
    image ``q`` produces state ``q``; therefore a threshold-crossing pulse can
    still have a pre-pulse image but that exposure is excluded from the
    accepted sequence.  No state is extrapolated after the engine stops.
    """

    exposures: list[ThermodynamicExposure] = []
    for pre_state, post_state in zip(
        sequence.states[:-1], sequence.states[1:], strict=True
    ):
        expected = pre_state.frame_index + 1
        if post_state.frame_index != expected:
            raise RuntimeError("thermodynamic state indices are not consecutive")
        exposures.append(
            ThermodynamicExposure(
                exposure_index=post_state.frame_index,
                pre_state=pre_state,
                post_state=post_state,
                accepted_by_thermodynamics=post_state.accepted_frame,
                failure_reason=post_state.failure_reason,
            )
        )
    return tuple(exposures)


__all__ = [
    "CLASSICAL_HARMONIC_ENERGY_COEFFICIENT",
    "FARADAY_ROTATION_FACTOR",
    "FARADAY_SCATTERING_FACTOR",
    "IDEAL_BOSE_ENERGY_COEFFICIENT",
    "OxfordInitialState",
    "OxfordMultiframeContract",
    "StateSolveResult",
    "ThermodynamicPulseScattering",
    "ThermodynamicSequence",
    "ThermodynamicState",
    "ThermodynamicExposure",
    "anchored_thermal_atoms",
    "cadence_status",
    "closure_validation_residuals",
    "condensate_core_sequence",
    "condensate_core_state",
    "condensate_core_energy_j",
    "measured_scale_non_saturation",
    "nominal_critical_atoms",
    "oxford_multiframe_contract_from_configs",
    "simulate_polarised_thermodynamic_sequence",
    "simulate_thermodynamic_sequence",
    "solve_state_at_energy",
    "thermodynamic_scattering_from_branch_result",
    "thermodynamic_exposures",
]
