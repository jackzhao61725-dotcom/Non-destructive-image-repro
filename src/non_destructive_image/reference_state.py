"""Resolve one dissertation reference state from its existing source record."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, TypeAlias

import numpy as np

from .atomic_model import ThomasFermiState, build_thomas_fermi_state
from .dipolar_tf import DipolarThomasFermiState, build_dipolar_thomas_fermi_state


CONTACT_TF_CORE = "contact_only_thomas_fermi_condensate_core"
POLARISED_DIPOLAR_TF_CORE = "polarised_dipolar_thomas_fermi_condensate_core"
CondensateCoreState: TypeAlias = ThomasFermiState | DipolarThomasFermiState


@dataclass(frozen=True)
class ReferenceState:
    """One validated measured state and its source-linked interaction contract."""

    condition_id: str
    repetition_id: str
    condensate_atoms: float
    thermal_atoms: float
    temperature_nk: float
    trap_frequencies_hz: tuple[float, float, float]
    scattering_length_bohr: float
    scattering_length_status: str
    dipolar_length_bohr: float
    epsilon_dd: float
    magnetic_field_gauss: float
    dipole_orientation_deg: float
    dipole_axis: str
    dipole_axis_index: int
    theta_d_reference_axis: str
    dipole_rotation_plane: str
    imaging_axis: str
    combined_calibration_zip_member: str
    combined_calibration_sha256: str
    combined_calibration_source_csv_line: int
    source_csv_line: int
    dataset_doi: str
    dataset_zip_filename: str
    dataset_zip_sha256: str
    raw_repetitions_zip_member: str
    raw_repetitions_sha256: str
    temperature_source_units: str
    temperature_runtime_units: str

    @property
    def state_id(self) -> str:
        """Return ``repetition_id`` under the thermodynamic result-schema name."""

        return self.repetition_id

    @property
    def trapped_atoms(self) -> float:
        """Return ``N0 + Nth`` without separating the measured triplet."""

        return self.condensate_atoms + self.thermal_atoms

    @property
    def temperature_k(self) -> float:
        """Return the measured temperature converted from nK to K."""

        return self.temperature_nk * 1e-9

    @property
    def dipolar_tf_scattering_length_bohr(self) -> float:
        """Return ``a_dd / epsilon_dd`` for the calibrated dipolar-TF target."""

        return self.dipolar_length_bohr / self.epsilon_dd

    def condensate_runtime_mapping(
        self,
        optical_forward_model: str,
    ) -> dict[str, object]:
        """Return the source-linked runtime inputs for the selected core model."""

        common: dict[str, object] = {
            "model": optical_forward_model,
            "atom_number": self.condensate_atoms,
            "trap_frequencies_hz": list(self.trap_frequencies_hz),
            "temperature_k": self.temperature_k,
        }
        if optical_forward_model == CONTACT_TF_CORE:
            common.update(
                {
                    "scattering_length_bohr": self.scattering_length_bohr,
                    "scattering_length_status": self.scattering_length_status,
                }
            )
            return common
        if optical_forward_model == POLARISED_DIPOLAR_TF_CORE:
            common.update(
                {
                    "scattering_length_bohr": self.dipolar_tf_scattering_length_bohr,
                    "scattering_length_status": (
                        "derived runtime value a_dd/epsilon_dd; source paper display "
                        "value is retained separately"
                    ),
                    "source_reported_scattering_length_bohr": (
                        self.scattering_length_bohr
                    ),
                    "source_reported_scattering_length_status": (
                        self.scattering_length_status
                    ),
                    "dipolar_length_bohr": self.dipolar_length_bohr,
                    "epsilon_dd": self.epsilon_dd,
                    "dipole_axis": self.dipole_axis,
                    "dipole_axis_index": self.dipole_axis_index,
                    "fully_spin_polarised": True,
                }
            )
            return common
        raise ValueError(f"unsupported optical_forward_model {optical_forward_model!r}")


def _mapping(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _sequence(mapping: Mapping[str, object], key: str) -> Sequence[object]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence")
    return value


def _non_empty_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _finite_number(
    mapping: Mapping[str, object],
    key: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be a finite number")
    if positive and number <= 0.0:
        raise ValueError(f"{key} must be positive")
    if non_negative and number < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return number


def _sha256(mapping: Mapping[str, object], key: str) -> str:
    value = _non_empty_string(mapping, key).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{key} must be a 64-character hexadecimal SHA-256")
    return value


def _source_csv_line(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
        raise ValueError(f"{key} must be an integer after the header line")
    return value


def _positive_finite_tuple(
    values: Sequence[object],
    *,
    length: int,
    label: str,
) -> tuple[float, ...]:
    numbers: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must contain numeric values")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{label} must contain positive finite values")
        numbers.append(number)
    if len(numbers) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return tuple(numbers)


def select_reference_condition(
    initial_condition_config: Mapping[str, object],
    condition_id: str,
) -> Mapping[str, object]:
    """Return exactly one source condition selected by its stable repository ID."""

    if not isinstance(condition_id, str) or not condition_id:
        raise ValueError("condition_id must be a non-empty string")
    conditions = _sequence(initial_condition_config, "initial_conditions")
    matches = [
        condition
        for condition in conditions
        if isinstance(condition, Mapping) and condition.get("id") == condition_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one initial-condition record {condition_id!r}")
    return matches[0]


def reference_states_from_config(
    initial_condition_config: Mapping[str, object],
    *,
    condition_id: str,
) -> tuple[ReferenceState, ...]:
    """Return all validated 300 ms repetitions for one source condition."""

    condition = select_reference_condition(initial_condition_config, condition_id)
    thermodynamics = _mapping(condition, "multiframe_thermodynamics")
    repetitions = _sequence(thermodynamics, "repetitions_300_ms")

    trap_record = _mapping(thermodynamics, "trap_frequency_record")
    raw_trap = _sequence(trap_record, "frequencies_hz")
    trap = _positive_finite_tuple(
        raw_trap,
        length=3,
        label="trap frequencies in Hz",
    )

    scattering_length_bohr = _finite_number(
        condition,
        "scattering_length_bohr",
        positive=True,
    )
    scattering_length_status = _non_empty_string(
        condition,
        "scattering_length_status",
    )
    atomic_constants = _mapping(initial_condition_config, "atomic_constants")
    dipolar_length_bohr = _finite_number(
        atomic_constants,
        "dipolar_length_bohr",
        positive=True,
    )
    dipolar_contract = _mapping(condition, "polarised_dipolar_tf_contract")
    if dipolar_contract.get("fully_spin_polarised") is not True:
        raise ValueError("fully_spin_polarised must be true for the target contract")
    if _non_empty_string(dipolar_contract, "dipolar_length_source") != (
        "atomic_constants.dipolar_length_bohr"
    ):
        raise ValueError("dipolar_length_source must identify the atomic constant")
    if _non_empty_string(dipolar_contract, "epsilon_dd_source") != (
        "multiframe_thermodynamics.closure.combined_calibration.epsilon_dd"
    ):
        raise ValueError("epsilon_dd_source must identify the source calibration")
    _non_empty_string(dipolar_contract, "scattering_length_policy")

    dipole_orientation_deg = _finite_number(condition, "dipole_orientation_deg")
    if dipole_orientation_deg != 0.0:
        raise ValueError("the selected Oxford target requires theta_d = 0 degrees")
    theta_d_reference_axis = _non_empty_string(
        dipolar_contract,
        "theta_d_reference_axis",
    )
    dipole_rotation_plane = _non_empty_string(
        dipolar_contract,
        "dipole_rotation_plane",
    )
    dipole_axis = _non_empty_string(dipolar_contract, "dipole_axis")
    imaging_axis = _non_empty_string(dipolar_contract, "imaging_axis")
    dipole_axis_index = dipolar_contract.get("dipole_axis_index")
    if (
        theta_d_reference_axis != "y"
        or dipole_rotation_plane != "y-z"
        or dipole_axis != "y"
        or dipole_axis_index != 1
        or imaging_axis != "x"
    ):
        raise ValueError(
            "the source dipole geometry must map theta_d=0 to axis y in the "
            "y-z plane, and the project model must adopt imaging along x"
        )

    dataset_doi = _non_empty_string(thermodynamics, "dataset_doi")
    dataset_zip = _mapping(thermodynamics, "dataset_zip")
    raw_source = _mapping(thermodynamics, "raw_repetitions_source")
    closure = _mapping(thermodynamics, "closure")
    combined_calibration = _mapping(closure, "combined_calibration")
    epsilon_dd = _finite_number(combined_calibration, "epsilon_dd", positive=True)
    if epsilon_dd >= 1.0:
        raise ValueError("epsilon_dd must satisfy 0 < epsilon_dd < 1")
    magnetic_field_gauss = _finite_number(
        combined_calibration,
        "magnetic_field_gauss",
        positive=True,
    )
    combined_calibration_source_csv_line = _source_csv_line(
        combined_calibration,
        "source_csv_line",
    )
    temperature_source_units = _non_empty_string(
        raw_source,
        "temperature_source_units",
    )
    temperature_runtime_units = _non_empty_string(
        raw_source,
        "temperature_runtime_units",
    )
    if temperature_source_units != "K" or temperature_runtime_units != "nK":
        raise ValueError("Oxford temperature units must declare source K and runtime nK")

    common = {
        "condition_id": condition_id,
        "trap_frequencies_hz": (trap[0], trap[1], trap[2]),
        "scattering_length_bohr": scattering_length_bohr,
        "scattering_length_status": scattering_length_status,
        "dipolar_length_bohr": dipolar_length_bohr,
        "epsilon_dd": epsilon_dd,
        "magnetic_field_gauss": magnetic_field_gauss,
        "dipole_orientation_deg": dipole_orientation_deg,
        "dipole_axis": dipole_axis,
        "dipole_axis_index": int(dipole_axis_index),
        "theta_d_reference_axis": theta_d_reference_axis,
        "dipole_rotation_plane": dipole_rotation_plane,
        "imaging_axis": imaging_axis,
        "combined_calibration_zip_member": _non_empty_string(
            combined_calibration,
            "zip_member",
        ),
        "combined_calibration_sha256": _sha256(
            combined_calibration,
            "sha256",
        ),
        "combined_calibration_source_csv_line": (
            combined_calibration_source_csv_line
        ),
        "dataset_doi": dataset_doi,
        "dataset_zip_filename": _non_empty_string(dataset_zip, "filename"),
        "dataset_zip_sha256": _sha256(dataset_zip, "sha256"),
        "raw_repetitions_zip_member": _non_empty_string(raw_source, "zip_member"),
        "raw_repetitions_sha256": _sha256(raw_source, "sha256"),
        "temperature_source_units": temperature_source_units,
        "temperature_runtime_units": temperature_runtime_units,
    }

    states: list[ReferenceState] = []
    seen_ids: set[str] = set()
    for raw_state in repetitions:
        if not isinstance(raw_state, Mapping):
            raise ValueError("every Oxford repetition must be a mapping")
        repetition_id = _non_empty_string(raw_state, "id")
        if repetition_id in seen_ids:
            raise ValueError(f"duplicate Oxford repetition id {repetition_id!r}")
        seen_ids.add(repetition_id)
        source_csv_line = _source_csv_line(raw_state, "source_csv_line")
        condensate_atoms = _finite_number(
            raw_state,
            "condensate_atoms",
            non_negative=True,
        )
        thermal_atoms = _finite_number(
            raw_state,
            "thermal_atoms",
            non_negative=True,
        )
        trapped_atoms = condensate_atoms + thermal_atoms
        if not math.isfinite(trapped_atoms) or trapped_atoms <= 0.0:
            raise ValueError("N0 + Nth must be positive and finite")
        states.append(
            ReferenceState(
                repetition_id=repetition_id,
                condensate_atoms=condensate_atoms,
                thermal_atoms=thermal_atoms,
                temperature_nk=_finite_number(
                    raw_state,
                    "temperature_nk",
                    positive=True,
                ),
                source_csv_line=source_csv_line,
                **common,
            )
        )
    if not states:
        raise ValueError("repetitions_300_ms must contain at least one state")
    return tuple(states)


def reference_state_from_config(
    initial_condition_config: Mapping[str, object],
    *,
    condition_id: str,
    repetition_id: str,
) -> ReferenceState:
    """Select one measured repetition without copying or separating its triplet."""

    if not isinstance(repetition_id, str) or not repetition_id:
        raise ValueError("repetition_id must be a non-empty string")
    states = reference_states_from_config(
        initial_condition_config,
        condition_id=condition_id,
    )
    matches = [state for state in states if state.repetition_id == repetition_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one Oxford repetition {repetition_id!r} "
            f"within condition {condition_id!r}"
        )
    return matches[0]


def build_model_condensate_core(
    model_config: Mapping[str, object],
    *,
    condensate_atoms: float | None = None,
) -> CondensateCoreState:
    """Build the configured contact or polarised dipolar TF condensate core.

    Selector-based configs must first be resolved with
    :func:`resolve_model_reference`. ``condensate.model`` is required so a
    missing physical-model choice cannot silently fall back to contact TF. An
    explicit ``condensate_atoms`` changes only the core population and is used
    by repeated-exposure consumers.
    """

    constants = _mapping(model_config, "constants")
    atom = _mapping(model_config, "atom")
    condensate = _mapping(model_config, "condensate")
    model_value = condensate.get("model")
    if not isinstance(model_value, str) or not model_value:
        raise ValueError("condensate.model must be a non-empty string")

    raw_number = condensate.get("atom_number") if condensate_atoms is None else condensate_atoms
    if isinstance(raw_number, bool) or not isinstance(raw_number, (int, float)):
        raise ValueError("condensate atom number must be a positive finite number")
    number = float(raw_number)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError("condensate atom number must be a positive finite number")

    trap = _positive_finite_tuple(
        _sequence(condensate, "trap_frequencies_hz"),
        length=3,
        label="condensate trap frequencies in Hz",
    )
    bohr_radius = _finite_number(constants, "bohr_radius_m", positive=True)
    mass = _finite_number(atom, "mass_number", positive=True) * _finite_number(
        constants,
        "atomic_mass_unit",
        positive=True,
    )
    hbar = _finite_number(constants, "hbar", positive=True)
    boltzmann = _finite_number(constants, "boltzmann_constant", positive=True)
    scattering_length_bohr = _finite_number(
        condensate,
        "scattering_length_bohr",
        positive=True,
    )
    scattering_length_m = scattering_length_bohr * bohr_radius

    if model_value == CONTACT_TF_CORE:
        return build_thomas_fermi_state(
            number,
            scattering_length_m,
            trap,
            mass,
            hbar,
            boltzmann,
        )
    if model_value != POLARISED_DIPOLAR_TF_CORE:
        raise ValueError(f"unsupported condensate.model {model_value!r}")
    if condensate.get("fully_spin_polarised") is not True:
        raise ValueError("the dipolar TF core requires fully_spin_polarised=true")

    dipolar_length_bohr = _finite_number(
        condensate,
        "dipolar_length_bohr",
        positive=True,
    )
    epsilon_dd = _finite_number(condensate, "epsilon_dd", positive=True)
    if epsilon_dd >= 1.0:
        raise ValueError("the dipolar TF core requires 0 < epsilon_dd < 1")
    if not math.isclose(
        dipolar_length_bohr / scattering_length_bohr,
        epsilon_dd,
        rel_tol=2e-15,
        abs_tol=0.0,
    ):
        raise ValueError("runtime scattering length must equal a_dd/epsilon_dd")
    dipole_axis = condensate.get("dipole_axis_index")
    if isinstance(dipole_axis, bool) or not isinstance(dipole_axis, int):
        raise ValueError("dipole_axis_index must be one of 0, 1 or 2")

    return build_dipolar_thomas_fermi_state(
        atom_number=number,
        scattering_length_m=scattering_length_m,
        dipolar_length_m=dipolar_length_bohr * bohr_radius,
        trap_frequencies_hz=trap,
        dipole_axis=dipole_axis,
        atomic_mass_kg=mass,
        hbar_j_s=hbar,
        boltzmann_constant_j_k=boltzmann,
    )


def resolve_model_reference(
    model_config: Mapping[str, object],
    initial_condition_config: Mapping[str, object],
) -> dict[str, Any]:
    """Resolve one source selector into the runtime condensate mapping."""

    selector = model_config.get("reference_state_selector")
    if selector is None:
        return deepcopy(dict(model_config))
    if not isinstance(selector, Mapping):
        raise ValueError("reference_state_selector must be a mapping")
    if "condensate" in model_config:
        raise ValueError(
            "a selector-based model config must not contain a competing condensate mapping"
        )

    condition_id = _non_empty_string(selector, "condition_id")
    repetition_id = _non_empty_string(selector, "repetition_id")
    optical_forward_model = _non_empty_string(selector, "optical_forward_model")
    if optical_forward_model not in {CONTACT_TF_CORE, POLARISED_DIPOLAR_TF_CORE}:
        raise ValueError("the selector contains an unsupported optical forward model")
    thermal_halo_rendered = selector.get("thermal_halo_rendered")
    if thermal_halo_rendered is not False:
        raise ValueError("thermal_halo_rendered must be false for the active model")

    state = reference_state_from_config(
        initial_condition_config,
        condition_id=condition_id,
        repetition_id=repetition_id,
    )
    resolved = deepcopy(dict(model_config))
    resolved["condensate"] = state.condensate_runtime_mapping(optical_forward_model)
    resolved["resolved_reference_state"] = {
        "source_config_path": _non_empty_string(selector, "source_config_path"),
        "condition_id": state.condition_id,
        "repetition_id": state.repetition_id,
        "selection_basis": _non_empty_string(selector, "selection_basis"),
        "condensate_atoms": state.condensate_atoms,
        "thermal_atoms": state.thermal_atoms,
        "trapped_atoms": state.trapped_atoms,
        "temperature_nk": state.temperature_nk,
        "temperature_k": state.temperature_k,
        "trap_frequencies_hz": list(state.trap_frequencies_hz),
        "scattering_length_bohr": state.scattering_length_bohr,
        "scattering_length_status": state.scattering_length_status,
        "dipolar_length_bohr": state.dipolar_length_bohr,
        "epsilon_dd": state.epsilon_dd,
        "dipolar_tf_scattering_length_bohr": (
            state.dipolar_tf_scattering_length_bohr
        ),
        "runtime_scattering_length_bohr": (
            state.dipolar_tf_scattering_length_bohr
            if optical_forward_model == POLARISED_DIPOLAR_TF_CORE
            else state.scattering_length_bohr
        ),
        "magnetic_field_gauss": state.magnetic_field_gauss,
        "dipole_orientation_deg": state.dipole_orientation_deg,
        "dipole_axis": state.dipole_axis,
        "dipole_axis_index": state.dipole_axis_index,
        "theta_d_reference_axis": state.theta_d_reference_axis,
        "dipole_rotation_plane": state.dipole_rotation_plane,
        "imaging_axis": state.imaging_axis,
        "combined_calibration_zip_member": (
            state.combined_calibration_zip_member
        ),
        "combined_calibration_sha256": state.combined_calibration_sha256,
        "combined_calibration_source_csv_line": (
            state.combined_calibration_source_csv_line
        ),
        "source_csv_line": state.source_csv_line,
        "temperature_source_units": state.temperature_source_units,
        "temperature_runtime_units": state.temperature_runtime_units,
        "dataset_doi": state.dataset_doi,
        "dataset_zip_filename": state.dataset_zip_filename,
        "dataset_zip_sha256": state.dataset_zip_sha256,
        "raw_repetitions_zip_member": state.raw_repetitions_zip_member,
        "raw_repetitions_sha256": state.raw_repetitions_sha256,
        "optical_forward_model": optical_forward_model,
        "thermal_halo_rendered": thermal_halo_rendered,
    }
    return resolved


__all__ = [
    "CONTACT_TF_CORE",
    "POLARISED_DIPOLAR_TF_CORE",
    "CondensateCoreState",
    "ReferenceState",
    "build_model_condensate_core",
    "reference_state_from_config",
    "reference_states_from_config",
    "resolve_model_reference",
    "select_reference_condition",
]
