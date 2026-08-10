"""Shared deterministic builders for independently prepared orientation endpoints.

The module owns only the common physical construction used by the guarded
eligibility runner and a later orientation Stage B.  It does not own opening
gates, result lifecycles, stochastic acquisition, inference, or publication.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


_ORIENTATION_V1_ATOL = 1e-12
_ORIENTATION_V1_PROBE_AXIS = (1.0, 0.0, 0.0)
_ORIENTATION_V1_IDENTITIES = {
    "B_parallel_y": {
        "dipole_axis": "y",
        "dipole_axis_index": 1,
        "theta_d_deg": 0.0,
        "quantisation_axis": (0.0, 1.0, 0.0),
    },
    "B_parallel_z": {
        "dipole_axis": "z",
        "dipole_axis_index": 2,
        "theta_d_deg": 90.0,
        "quantisation_axis": (0.0, 0.0, 1.0),
    },
}


class OrientationEndpointScientificFailure(RuntimeError):
    """A physically unsupported endpoint that should be reported explicitly."""


def _unit_vector(value: object, *, label: str) -> tuple[float, float, float]:
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3,):
        raise ValueError(f"{label} must be a finite three-vector")
    if any(isinstance(component, (bool, np.bool_)) for component in raw.flat):
        raise ValueError(f"{label} must not contain boolean components")
    try:
        vector = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite three-vector") from exc
    if not np.isfinite(vector).all():
        raise ValueError(f"{label} must be a finite three-vector")
    if not np.isclose(
        np.linalg.norm(vector), 1.0, rtol=0.0, atol=_ORIENTATION_V1_ATOL
    ):
        raise ValueError(f"{label} must be a unit vector")
    return tuple(float(component) for component in vector)


def _finite_real(value: object, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{label} must be a finite real scalar")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be a finite real scalar")
    return number


def _positive(value: object, *, label: str) -> float:
    number = _finite_real(value, label=label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (Integral, np.integer)
    ):
        raise ValueError(f"{label} must be a positive integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _immutable_finite_nonnegative(
    values: object, *, shape: tuple[int, int], label: str
) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{label} density shape disagrees with its grid")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise OrientationEndpointScientificFailure(
            f"{label} is not a finite non-negative endpoint density"
        )
    array = np.array(array, dtype=float, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class OrientationEndpointSpec:
    """One of the two frozen ``+x``-probed orientation-v1 endpoints."""

    label: str
    dipole_axis: str
    dipole_axis_index: int
    theta_d_deg: float
    probe_axis: tuple[float, float, float]
    quantisation_axis: tuple[float, float, float]
    polarisation_axis: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or self.label not in _ORIENTATION_V1_IDENTITIES:
            raise ValueError(
                "endpoint label must be B_parallel_y or B_parallel_z for orientation v1"
            )
        identity = _ORIENTATION_V1_IDENTITIES[self.label]
        if (
            isinstance(self.dipole_axis_index, (bool, np.bool_))
            or not isinstance(self.dipole_axis_index, (int, np.integer))
            or self.dipole_axis != identity["dipole_axis"]
            or self.dipole_axis_index != identity["dipole_axis_index"]
        ):
            raise ValueError(
                "endpoint label, dipole-axis label and index disagree with the "
                "frozen orientation-v1 identity"
            )
        if isinstance(self.theta_d_deg, (bool, np.bool_)):
            raise ValueError("theta_d_deg must be finite")
        theta = float(self.theta_d_deg)
        if not np.isfinite(theta):
            raise ValueError("theta_d_deg must be finite")
        if not np.isclose(
            theta,
            float(identity["theta_d_deg"]),
            rtol=0.0,
            atol=_ORIENTATION_V1_ATOL,
        ):
            raise ValueError(
                "theta_d_deg disagrees with the frozen orientation-v1 identity"
            )
        probe = _unit_vector(self.probe_axis, label="probe_axis")
        if not np.allclose(
            probe,
            _ORIENTATION_V1_PROBE_AXIS,
            rtol=0.0,
            atol=_ORIENTATION_V1_ATOL,
        ):
            raise ValueError("orientation v1 requires probe_axis=+x")
        quantisation = _unit_vector(
            self.quantisation_axis, label="quantisation_axis"
        )
        if not np.allclose(
            quantisation,
            identity["quantisation_axis"],
            rtol=0.0,
            atol=_ORIENTATION_V1_ATOL,
        ):
            raise ValueError(
                "quantisation_axis disagrees with the frozen dipole-axis identity"
            )
        polarisation = _unit_vector(
            self.polarisation_axis, label="polarisation_axis"
        )
        expected_polarisation = np.cross(probe, quantisation)
        norm = float(np.linalg.norm(expected_polarisation))
        if norm <= 0.0:
            raise ValueError("probe and quantisation axes must be perpendicular")
        expected_polarisation /= norm
        if not (
            np.isclose(
                np.dot(probe, quantisation),
                0.0,
                rtol=0.0,
                atol=_ORIENTATION_V1_ATOL,
            )
            and np.allclose(
                polarisation,
                expected_polarisation,
                rtol=0.0,
                atol=_ORIENTATION_V1_ATOL,
            )
        ):
            raise ValueError("endpoint geometry must obey epsilon=k cross B")
        object.__setattr__(self, "theta_d_deg", theta)
        object.__setattr__(self, "probe_axis", probe)
        object.__setattr__(self, "quantisation_axis", quantisation)
        object.__setattr__(self, "polarisation_axis", polarisation)


@dataclass(frozen=True)
class OrientationEndpointBuildContract:
    """Source, camera and operator inputs shared by orientation consumers."""

    source_condition_id: str
    source_repetition_id: str
    detuning_hz: float
    field_of_view_m: float
    canonical_ngrid: int
    inverse_ngrid: int
    camera_pixel_size_m: float
    camera_output_shape: tuple[int, int]
    numerical_aperture: float
    wavelength_m: float
    photoelectrons_per_i0_pixel: float
    read_noise_electrons: float
    phase_plate_transmittance: float
    phase_plate_phase_rad: float
    independent_exposures_by_role: Mapping[str, int]

    def __post_init__(self) -> None:
        for label, value in (
            ("source_condition_id", self.source_condition_id),
            ("source_repetition_id", self.source_repetition_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be non-empty")
        positive_values = {
            label: _positive(value, label=label)
            for label, value in (
                ("field_of_view_m", self.field_of_view_m),
                ("camera_pixel_size_m", self.camera_pixel_size_m),
                ("numerical_aperture", self.numerical_aperture),
                ("wavelength_m", self.wavelength_m),
                ("photoelectrons_per_i0_pixel", self.photoelectrons_per_i0_pixel),
            )
        }
        detuning = _finite_real(self.detuning_hz, label="detuning_hz")
        read_noise = _finite_real(
            self.read_noise_electrons, label="read_noise_electrons"
        )
        if detuning == 0.0:
            raise ValueError("detuning_hz must be finite and non-zero")
        if read_noise < 0.0:
            raise ValueError("read_noise_electrons must be finite and non-negative")
        canonical_ngrid = _positive_integer(
            self.canonical_ngrid, label="canonical_ngrid"
        )
        inverse_ngrid = _positive_integer(self.inverse_ngrid, label="inverse_ngrid")
        try:
            raw_camera_shape = tuple(self.camera_output_shape)
        except TypeError as exc:
            raise ValueError(
                "camera_output_shape must contain two positive integers"
            ) from exc
        if len(raw_camera_shape) != 2:
            raise ValueError("camera_output_shape must contain two positive values")
        camera_shape = tuple(
            _positive_integer(value, label="camera_output_shape value")
            for value in raw_camera_shape
        )
        transmittance = _finite_real(
            self.phase_plate_transmittance, label="phase_plate_transmittance"
        )
        phase = _finite_real(self.phase_plate_phase_rad, label="phase_plate_phase_rad")
        if not 0.0 <= transmittance <= 1.0:
            raise ValueError("phase_plate_transmittance must lie in [0, 1]")
        if not isinstance(self.independent_exposures_by_role, Mapping):
            raise ValueError("independent_exposures_by_role must be a mapping")
        roles = dict(self.independent_exposures_by_role)
        expected_roles = {"atom", "bright_reference", "dark"}
        if set(roles) != expected_roles or any(
            _positive_integer(value, label=f"{role} exposure count") != 1
            for role, value in roles.items()
        ):
            raise ValueError("orientation PCI endpoints require one exposure per raw role")
        object.__setattr__(self, "field_of_view_m", positive_values["field_of_view_m"])
        object.__setattr__(
            self, "camera_pixel_size_m", positive_values["camera_pixel_size_m"]
        )
        object.__setattr__(self, "numerical_aperture", positive_values["numerical_aperture"])
        object.__setattr__(self, "wavelength_m", positive_values["wavelength_m"])
        object.__setattr__(
            self,
            "photoelectrons_per_i0_pixel",
            positive_values["photoelectrons_per_i0_pixel"],
        )
        object.__setattr__(self, "detuning_hz", detuning)
        object.__setattr__(self, "canonical_ngrid", canonical_ngrid)
        object.__setattr__(self, "inverse_ngrid", inverse_ngrid)
        object.__setattr__(self, "camera_output_shape", camera_shape)
        object.__setattr__(self, "read_noise_electrons", read_noise)
        object.__setattr__(self, "phase_plate_transmittance", transmittance)
        object.__setattr__(self, "phase_plate_phase_rad", phase)
        object.__setattr__(
            self,
            "independent_exposures_by_role",
            MappingProxyType(roles),
        )


@dataclass(frozen=True)
class OrientationEndpointFactories:
    """Dependency bundle enabling small synthetic tests without physical data."""

    reference_state_from_config: Callable[..., Any]
    build_dipolar_state: Callable[..., Any]
    thomas_fermi_profile_2d: Callable[..., Any]
    polarised_response_from_config: Callable[..., Any]
    complex_column_response: Callable[..., Any]
    build_grid: Callable[..., Any]
    detector_contract: Callable[..., Any]
    scalar_response_contract: Callable[..., Any]
    pci_transfer_contract: Callable[..., Any]
    pci_operator: Callable[..., Any]


@dataclass(frozen=True)
class OrientationEndpointProduct:
    """One independently instantiated physical endpoint and its two grids."""

    spec: OrientationEndpointSpec
    reference: Any
    state: Any
    response: Any
    scalar_response: Any
    transfer: Any
    canonical_grid: Any
    inverse_grid: Any
    canonical_density_m2: FloatArray
    inverse_density_m2: FloatArray
    canonical_operator: Any
    inverse_operator: Any


def maintained_orientation_endpoint_factories() -> OrientationEndpointFactories:
    """Resolve the maintained implementation used by guarded numerical runners."""

    from .atomic_response import (
        complex_column_response,
        polarised_optical_response_from_config,
    )
    from .dipolar_tf import build_dipolar_thomas_fermi_state
    from .profiles import thomas_fermi_profile_2d
    from .reference_state import reference_state_from_config
    from .reconstruction.contracts import DetectorContract
    from .reconstruction.resolution import build_uniform_physical_camera_grid
    from .reconstruction.scalar_measurements import (
        PCILinkedRawOperator,
        PCITransferContract,
        ScalarOpticalResponseContract,
    )

    return OrientationEndpointFactories(
        reference_state_from_config=reference_state_from_config,
        build_dipolar_state=build_dipolar_thomas_fermi_state,
        thomas_fermi_profile_2d=thomas_fermi_profile_2d,
        polarised_response_from_config=polarised_optical_response_from_config,
        complex_column_response=complex_column_response,
        build_grid=build_uniform_physical_camera_grid,
        detector_contract=DetectorContract,
        scalar_response_contract=ScalarOpticalResponseContract,
        pci_transfer_contract=PCITransferContract,
        pci_operator=PCILinkedRawOperator,
    )


def _grid(contract: OrientationEndpointBuildContract, ngrid: int, factory: Any) -> Any:
    return factory(
        ngrid=ngrid,
        field_of_view_m=contract.field_of_view_m,
        camera_pixel_size_m=contract.camera_pixel_size_m,
        camera_output_shape=contract.camera_output_shape,
        numerical_aperture=contract.numerical_aperture,
        wavelength_m=contract.wavelength_m,
    )


def build_orientation_endpoint(
    *,
    spec: OrientationEndpointSpec,
    contract: OrientationEndpointBuildContract,
    model_config: Mapping[str, Any],
    initial_condition_config: Mapping[str, Any],
    factories: OrientationEndpointFactories | None = None,
) -> OrientationEndpointProduct:
    """Build one static equilibrium endpoint with independent response/operators."""

    if not isinstance(spec, OrientationEndpointSpec):
        raise TypeError("spec must be an OrientationEndpointSpec")
    if not isinstance(contract, OrientationEndpointBuildContract):
        raise TypeError("contract must be an OrientationEndpointBuildContract")
    if not isinstance(model_config, Mapping) or not isinstance(
        initial_condition_config, Mapping
    ):
        raise TypeError("model and initial-condition configs must be mappings")
    resolved = factories or maintained_orientation_endpoint_factories()
    if not isinstance(resolved, OrientationEndpointFactories):
        raise TypeError("factories must be an OrientationEndpointFactories")

    reference = resolved.reference_state_from_config(
        initial_condition_config,
        condition_id=contract.source_condition_id,
        repetition_id=contract.source_repetition_id,
    )
    constants = model_config["constants"]
    atom = model_config["atom"]
    try:
        state = resolved.build_dipolar_state(
            atom_number=reference.condensate_atoms,
            scattering_length_m=(
                reference.dipolar_tf_scattering_length_bohr
                * float(constants["bohr_radius_m"])
            ),
            dipolar_length_m=(
                reference.dipolar_length_bohr * float(constants["bohr_radius_m"])
            ),
            trap_frequencies_hz=reference.trap_frequencies_hz,
            dipole_axis=spec.dipole_axis_index,
            atomic_mass_kg=(
                float(atom["mass_number"]) * float(constants["atomic_mass_unit"])
            ),
            hbar_j_s=float(constants["hbar"]),
            boltzmann_constant_j_k=float(constants["boltzmann_constant"]),
        )
    except RuntimeError as exc:
        raise OrientationEndpointScientificFailure(
            f"{spec.label} dipolar equilibrium endpoint is unsupported"
        ) from exc

    oriented = copy.deepcopy(dict(model_config))
    geometry = oriented["polarised_atomic_response"]["geometry"]
    geometry["probe_wavevector_unit_vector"] = list(spec.probe_axis)
    geometry["quantisation_axis_unit_vector"] = list(spec.quantisation_axis)
    geometry["selected_polarisation_unit_vector"] = list(spec.polarisation_axis)
    geometry["theta_d_deg"] = spec.theta_d_deg
    response = resolved.polarised_response_from_config(oriented)
    column_response = resolved.complex_column_response(
        np.ones((1, 1), dtype=float),
        contract.detuning_hz,
        float(atom["resonant_cross_section_m2"]),
        float(atom["natural_linewidth_rad_s"]),
        response,
    )
    scalar_response = resolved.scalar_response_contract(
        float(np.asarray(column_response.phase_map_rad)[0, 0]),
        float(np.asarray(column_response.optical_depth_map)[0, 0]),
    )
    transfer = resolved.pci_transfer_contract(
        contract.phase_plate_transmittance,
        contract.phase_plate_phase_rad,
    )
    canonical_grid = _grid(contract, contract.canonical_ngrid, resolved.build_grid)
    inverse_grid = _grid(contract, contract.inverse_ngrid, resolved.build_grid)

    def density_for(grid: Any, *, label: str) -> FloatArray:
        profile = resolved.thomas_fermi_profile_2d(
            grid.y_grid_m,
            grid.z_grid_m,
            float(state.radii_m[1]),
            float(state.radii_m[2]),
        )
        density = float(state.peak_column_density_m2[0]) * np.asarray(
            profile, dtype=float
        )
        result = _immutable_finite_nonnegative(
            density, shape=grid.y_grid_m.shape, label=f"{spec.label} {label}"
        )
        if float(np.sum(result)) <= 0.0:
            raise OrientationEndpointScientificFailure(
                f"{spec.label} {label} density has zero support"
            )
        return result

    canonical_density = density_for(canonical_grid, label="canonical")
    inverse_density = density_for(inverse_grid, label="inverse")
    detector = resolved.detector_contract(
        contract.photoelectrons_per_i0_pixel,
        contract.read_noise_electrons,
    )
    canonical_operator = resolved.pci_operator(
        grid=canonical_grid,
        detector=detector,
        response=scalar_response,
        transfer=transfer,
        independent_exposures_by_role=contract.independent_exposures_by_role,
    )
    inverse_operator = resolved.pci_operator(
        grid=inverse_grid,
        detector=resolved.detector_contract(
            contract.photoelectrons_per_i0_pixel,
            contract.read_noise_electrons,
        ),
        response=resolved.scalar_response_contract(
            scalar_response.phase_per_column_density_rad_m2,
            scalar_response.optical_depth_per_column_density_m2,
        ),
        transfer=resolved.pci_transfer_contract(
            contract.phase_plate_transmittance,
            contract.phase_plate_phase_rad,
        ),
        independent_exposures_by_role=contract.independent_exposures_by_role,
    )
    return OrientationEndpointProduct(
        spec=spec,
        reference=reference,
        state=state,
        response=response,
        scalar_response=scalar_response,
        transfer=transfer,
        canonical_grid=canonical_grid,
        inverse_grid=inverse_grid,
        canonical_density_m2=canonical_density,
        inverse_density_m2=inverse_density,
        canonical_operator=canonical_operator,
        inverse_operator=inverse_operator,
    )


def build_orientation_endpoint_pair(
    *,
    specs: tuple[OrientationEndpointSpec, OrientationEndpointSpec],
    contract: OrientationEndpointBuildContract,
    model_config: Mapping[str, Any],
    initial_condition_config: Mapping[str, Any],
    factories: OrientationEndpointFactories | None = None,
) -> tuple[OrientationEndpointProduct, OrientationEndpointProduct]:
    """Build two endpoints independently and reject shared mutable physics objects."""

    if len(specs) != 2 or specs[0].label == specs[1].label:
        raise ValueError("orientation endpoint pair requires two distinct labels")
    first = build_orientation_endpoint(
        spec=specs[0],
        contract=contract,
        model_config=model_config,
        initial_condition_config=initial_condition_config,
        factories=factories,
    )
    second = build_orientation_endpoint(
        spec=specs[1],
        contract=contract,
        model_config=model_config,
        initial_condition_config=initial_condition_config,
        factories=factories,
    )
    for name in (
        "state",
        "response",
        "scalar_response",
        "transfer",
        "canonical_grid",
        "inverse_grid",
        "canonical_density_m2",
        "inverse_density_m2",
        "canonical_operator",
        "inverse_operator",
    ):
        if getattr(first, name) is getattr(second, name):
            raise RuntimeError(f"orientation endpoints unexpectedly share {name}")
    return first, second


__all__ = [
    "OrientationEndpointBuildContract",
    "OrientationEndpointFactories",
    "OrientationEndpointProduct",
    "OrientationEndpointScientificFailure",
    "OrientationEndpointSpec",
    "build_orientation_endpoint",
    "build_orientation_endpoint_pair",
    "maintained_orientation_endpoint_factories",
]
