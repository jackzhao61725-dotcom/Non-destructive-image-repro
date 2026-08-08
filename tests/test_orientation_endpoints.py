"""Synthetic tests for the shared orientation-endpoint construction API."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from non_destructive_image import orientation_endpoints as endpoints


@dataclass(frozen=True)
class _ScalarResponse:
    phase_per_column_density_rad_m2: float
    optical_depth_per_column_density_m2: float

    @property
    def complex_exponent_per_column_density_m2(self) -> complex:
        return complex(
            -0.5 * self.optical_depth_per_column_density_m2,
            self.phase_per_column_density_rad_m2,
        )


@dataclass(frozen=True)
class _Transfer:
    phase_plate_transmittance: float
    phase_plate_phase_rad: float


def _specs() -> tuple[endpoints.OrientationEndpointSpec, endpoints.OrientationEndpointSpec]:
    return (
        endpoints.OrientationEndpointSpec(
            label="B_parallel_y",
            dipole_axis="y",
            dipole_axis_index=1,
            theta_d_deg=0.0,
            probe_axis=(1.0, 0.0, 0.0),
            quantisation_axis=(0.0, 1.0, 0.0),
            polarisation_axis=(0.0, 0.0, 1.0),
        ),
        endpoints.OrientationEndpointSpec(
            label="B_parallel_z",
            dipole_axis="z",
            dipole_axis_index=2,
            theta_d_deg=90.0,
            probe_axis=(1.0, 0.0, 0.0),
            quantisation_axis=(0.0, 0.0, 1.0),
            polarisation_axis=(0.0, -1.0, 0.0),
        ),
    )


def _spec_values(label: str) -> dict[str, Any]:
    if label == "B_parallel_y":
        return {
            "label": label,
            "dipole_axis": "y",
            "dipole_axis_index": 1,
            "theta_d_deg": 0.0,
            "probe_axis": (1.0, 0.0, 0.0),
            "quantisation_axis": (0.0, 1.0, 0.0),
            "polarisation_axis": (0.0, 0.0, 1.0),
        }
    if label == "B_parallel_z":
        return {
            "label": label,
            "dipole_axis": "z",
            "dipole_axis_index": 2,
            "theta_d_deg": 90.0,
            "probe_axis": (1.0, 0.0, 0.0),
            "quantisation_axis": (0.0, 0.0, 1.0),
            "polarisation_axis": (0.0, -1.0, 0.0),
        }
    raise ValueError("unsupported synthetic orientation label")


def _contract() -> endpoints.OrientationEndpointBuildContract:
    return endpoints.OrientationEndpointBuildContract(
        source_condition_id="synthetic_condition",
        source_repetition_id="synthetic_repetition",
        detuning_hz=1.0,
        field_of_view_m=10.0,
        canonical_ngrid=5,
        inverse_ngrid=3,
        camera_pixel_size_m=1.0,
        camera_output_shape=(3, 3),
        numerical_aperture=0.1,
        wavelength_m=1.0,
        photoelectrons_per_i0_pixel=100.0,
        read_noise_electrons=1.0,
        phase_plate_transmittance=0.9,
        phase_plate_phase_rad=0.5,
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 1,
            "dark": 1,
        },
    )


def _model() -> dict[str, Any]:
    return {
        "constants": {
            "bohr_radius_m": 1.0,
            "atomic_mass_unit": 1.0,
            "hbar": 1.0,
            "boltzmann_constant": 1.0,
        },
        "atom": {
            "mass_number": 1.0,
            "resonant_cross_section_m2": 1.0,
            "natural_linewidth_rad_s": 1.0,
        },
        "polarised_atomic_response": {
            "geometry": {
                "probe_wavevector_unit_vector": [1.0, 0.0, 0.0],
                "quantisation_axis_unit_vector": [0.0, 1.0, 0.0],
                "selected_polarisation_unit_vector": [0.0, 0.0, 1.0],
                "theta_d_deg": 0.0,
            }
        },
    }


def _factories(
    *,
    state_factory: Callable[..., Any] | None = None,
    profile_factory: Callable[..., Any] | None = None,
) -> tuple[endpoints.OrientationEndpointFactories, dict[str, list[Any]]]:
    calls: dict[str, list[Any]] = {"state_axes": [], "geometries": []}

    def reference_factory(
        _config: Any, *, condition_id: str, repetition_id: str
    ) -> Any:
        return SimpleNamespace(
            condition_id=condition_id,
            repetition_id=repetition_id,
            condensate_atoms=10.0,
            dipolar_tf_scattering_length_bohr=2.0,
            dipolar_length_bohr=1.0,
            trap_frequencies_hz=np.array([1.0, 2.0, 3.0]),
        )

    def default_state_factory(**kwargs: Any) -> Any:
        calls["state_axes"].append(kwargs["dipole_axis"])
        return SimpleNamespace(
            radii_m=np.array([3.0, 2.0, 1.0]),
            peak_column_density_m2=np.array([5.0]),
        )

    chosen_state_factory = state_factory or default_state_factory

    def response_factory(config: dict[str, Any]) -> Any:
        geometry = copy.deepcopy(config["polarised_atomic_response"]["geometry"])
        calls["geometries"].append(geometry)
        return SimpleNamespace(geometry=geometry)

    def column_response(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            phase_map_rad=np.array([[0.2]]),
            optical_depth_map=np.array([[0.1]]),
        )

    def grid_factory(*, ngrid: int, **_kwargs: Any) -> Any:
        axis = np.linspace(-1.0, 1.0, ngrid)
        y_grid, z_grid = np.meshgrid(axis, axis)
        return SimpleNamespace(
            y_grid_m=y_grid,
            z_grid_m=z_grid,
            pupil=np.ones_like(y_grid),
        )

    def default_profile(y: np.ndarray, _z: np.ndarray, *_radii: float) -> np.ndarray:
        return np.ones_like(y)

    chosen_profile_factory = profile_factory or default_profile
    factories = endpoints.OrientationEndpointFactories(
        reference_state_from_config=reference_factory,
        build_dipolar_state=chosen_state_factory,
        thomas_fermi_profile_2d=chosen_profile_factory,
        polarised_response_from_config=response_factory,
        complex_column_response=column_response,
        build_grid=grid_factory,
        detector_contract=lambda *args: SimpleNamespace(args=args),
        scalar_response_contract=_ScalarResponse,
        pci_transfer_contract=_Transfer,
        pci_operator=lambda **kwargs: SimpleNamespace(kwargs=kwargs),
    )
    return factories, calls


def test_pair_builds_independent_oriented_products_from_synthetic_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        endpoints,
        "maintained_orientation_endpoint_factories",
        lambda: (_ for _ in ()).throw(AssertionError("maintained physics imported")),
    )
    factories, calls = _factories()
    model = _model()
    original = copy.deepcopy(model)
    first, second = endpoints.build_orientation_endpoint_pair(
        specs=_specs(),
        contract=_contract(),
        model_config=model,
        initial_condition_config={"synthetic": True},
        factories=factories,
    )
    assert calls["state_axes"] == [1, 2]
    assert calls["geometries"][0]["quantisation_axis_unit_vector"] == [0.0, 1.0, 0.0]
    assert calls["geometries"][1]["quantisation_axis_unit_vector"] == [0.0, 0.0, 1.0]
    assert calls["geometries"][1]["selected_polarisation_unit_vector"] == [0.0, -1.0, 0.0]
    assert model == original
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
        assert getattr(first, name) is not getattr(second, name)
    assert first.canonical_density_m2.shape == (5, 5)
    assert first.inverse_density_m2.shape == (3, 3)
    assert not first.canonical_density_m2.flags.writeable
    assert not second.inverse_density_m2.flags.writeable
    assert not np.shares_memory(
        first.canonical_density_m2, second.canonical_density_m2
    )
    assert not np.shares_memory(first.inverse_density_m2, second.inverse_density_m2)


@pytest.mark.parametrize("spec", _specs(), ids=("B_parallel_y", "B_parallel_z"))
def test_frozen_orientation_v1_specs_are_accepted(
    spec: endpoints.OrientationEndpointSpec,
) -> None:
    expected = _spec_values(spec.label)
    assert spec.dipole_axis == expected["dipole_axis"]
    assert spec.dipole_axis_index == expected["dipole_axis_index"]
    assert spec.theta_d_deg == expected["theta_d_deg"]
    assert spec.probe_axis == expected["probe_axis"]
    assert spec.quantisation_axis == expected["quantisation_axis"]
    assert spec.polarisation_axis == expected["polarisation_axis"]


@pytest.mark.parametrize(
    ("base_label", "changes", "message"),
    [
        (
            "B_parallel_y",
            {"label": "B_parallel_z"},
            "frozen orientation-v1 identity",
        ),
        (
            "B_parallel_y",
            {"label": "B_parallel_x", "dipole_axis": "x", "dipole_axis_index": 0},
            "endpoint label",
        ),
        (
            "B_parallel_y",
            {"dipole_axis": "z", "dipole_axis_index": 2},
            "frozen orientation-v1 identity",
        ),
        (
            "B_parallel_y",
            {"theta_d_deg": 90.0},
            "theta_d_deg",
        ),
        (
            "B_parallel_y",
            {"dipole_axis_index": True},
            "frozen orientation-v1 identity",
        ),
        (
            "B_parallel_y",
            {"theta_d_deg": False},
            "theta_d_deg",
        ),
        (
            "B_parallel_y",
            {
                "probe_axis": (0.0, 0.0, 1.0),
                "polarisation_axis": (-1.0, 0.0, 0.0),
            },
            "probe_axis=\\+x",
        ),
        (
            "B_parallel_y",
            {
                "quantisation_axis": (0.0, 0.0, 1.0),
                "polarisation_axis": (0.0, -1.0, 0.0),
            },
            "quantisation_axis",
        ),
        (
            "B_parallel_y",
            {
                "quantisation_axis": (0.0, -1.0, 0.0),
                "polarisation_axis": (0.0, 0.0, -1.0),
            },
            "quantisation_axis",
        ),
        (
            "B_parallel_z",
            {
                "quantisation_axis": (0.0, 0.0, -1.0),
                "polarisation_axis": (0.0, 1.0, 0.0),
            },
            "quantisation_axis",
        ),
        (
            "B_parallel_z",
            {"polarisation_axis": (0.0, 1.0, 0.0)},
            "epsilon=k cross B",
        ),
        (
            "B_parallel_y",
            {"probe_axis": (2.0, 0.0, 0.0)},
            "probe_axis must be a unit vector",
        ),
        (
            "B_parallel_y",
            {"quantisation_axis": (0.0, 2.0, 0.0)},
            "quantisation_axis must be a unit vector",
        ),
        (
            "B_parallel_y",
            {"polarisation_axis": (0.0, 0.0, 2.0)},
            "polarisation_axis must be a unit vector",
        ),
        (
            "B_parallel_y",
            {"probe_axis": (True, False, False)},
            "probe_axis must not contain boolean components",
        ),
        (
            "B_parallel_y",
            {"quantisation_axis": (False, True, False)},
            "quantisation_axis must not contain boolean components",
        ),
        (
            "B_parallel_y",
            {"polarisation_axis": (False, False, True)},
            "polarisation_axis must not contain boolean components",
        ),
    ],
    ids=(
        "label-axis-mismatch",
        "x-orientation",
        "dipole-axis-mismatch",
        "theta-mismatch",
        "boolean-axis-index",
        "boolean-theta",
        "non-x-probe",
        "axis-versus-quantisation-mismatch",
        "negative-y",
        "negative-z",
        "signed-polarisation-drift",
        "probe-not-normalised",
        "quantisation-not-normalised",
        "polarisation-not-normalised",
        "boolean-probe-vector",
        "boolean-quantisation-vector",
        "boolean-polarisation-vector",
    ),
)
def test_frozen_orientation_v1_rejects_coordinate_or_identity_drift(
    base_label: str,
    changes: dict[str, Any],
    message: str,
) -> None:
    values = _spec_values(base_label)
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        endpoints.OrientationEndpointSpec(**values)


def test_build_contract_rejects_shared_or_invalid_raw_role_topology() -> None:
    values = _contract().__dict__ | {
        "independent_exposures_by_role": {
            "atom": 2,
            "bright_reference": 1,
            "dark": 1,
        }
    }
    with pytest.raises(ValueError, match="one exposure per raw role"):
        endpoints.OrientationEndpointBuildContract(**values)
    values = _contract().__dict__ | {"phase_plate_transmittance": 1.1}
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        endpoints.OrientationEndpointBuildContract(**values)


def test_build_contract_normalises_numpy_scalars_and_freezes_mutable_inputs() -> None:
    camera_shape = [np.int64(3), np.int32(3)]
    roles = {"atom": np.int64(1), "bright_reference": np.int32(1), "dark": 1}
    values = _contract().__dict__ | {
        "detuning_hz": np.float64(1.0),
        "field_of_view_m": np.float32(10.0),
        "canonical_ngrid": np.int64(5),
        "inverse_ngrid": np.int32(3),
        "camera_pixel_size_m": np.float64(1.0),
        "camera_output_shape": camera_shape,
        "numerical_aperture": np.float32(0.1),
        "wavelength_m": np.float64(1.0),
        "photoelectrons_per_i0_pixel": np.float32(100.0),
        "read_noise_electrons": np.float64(1.0),
        "phase_plate_transmittance": np.float32(0.9),
        "phase_plate_phase_rad": np.float64(0.5),
        "independent_exposures_by_role": roles,
    }
    contract = endpoints.OrientationEndpointBuildContract(**values)
    camera_shape[0] = 7
    roles["atom"] = 2

    assert contract.camera_output_shape == (3, 3)
    assert dict(contract.independent_exposures_by_role) == {
        "atom": 1,
        "bright_reference": 1,
        "dark": 1,
    }
    for field in ("canonical_ngrid", "inverse_ngrid"):
        assert type(getattr(contract, field)) is int
    for field in (
        "detuning_hz",
        "field_of_view_m",
        "camera_pixel_size_m",
        "numerical_aperture",
        "wavelength_m",
        "photoelectrons_per_i0_pixel",
        "read_noise_electrons",
        "phase_plate_transmittance",
        "phase_plate_phase_rad",
    ):
        assert type(getattr(contract, field)) is float
    with pytest.raises(TypeError):
        contract.independent_exposures_by_role["atom"] = 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("detuning_hz", True),
        ("field_of_view_m", np.bool_(True)),
        ("camera_pixel_size_m", "1.0"),
        ("numerical_aperture", "0.1"),
        ("wavelength_m", False),
        ("photoelectrons_per_i0_pixel", "100.0"),
        ("read_noise_electrons", np.bool_(False)),
        ("phase_plate_transmittance", "0.9"),
        ("phase_plate_phase_rad", True),
    ],
)
def test_build_contract_rejects_boolean_or_numeric_string_scalars(
    field: str, value: object
) -> None:
    values = _contract().__dict__ | {field: value}
    with pytest.raises(ValueError, match="finite real scalar"):
        endpoints.OrientationEndpointBuildContract(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_ngrid", "5"),
        ("inverse_ngrid", np.float64(3.0)),
        ("camera_output_shape", (True, 3)),
        ("camera_output_shape", (np.bool_(True), 3)),
    ],
)
def test_build_contract_rejects_non_integral_size_values(
    field: str, value: object
) -> None:
    values = _contract().__dict__ | {field: value}
    with pytest.raises(ValueError, match="positive integer"):
        endpoints.OrientationEndpointBuildContract(**values)


@pytest.mark.parametrize("value", (True, np.bool_(True), "1"))
def test_build_contract_rejects_non_integral_exposure_counts(value: object) -> None:
    roles = {"atom": value, "bright_reference": 1, "dark": 1}
    values = _contract().__dict__ | {"independent_exposures_by_role": roles}
    with pytest.raises(ValueError, match="positive integer"):
        endpoints.OrientationEndpointBuildContract(**values)


@pytest.mark.parametrize("field", ("canonical_ngrid", "inverse_ngrid"))
@pytest.mark.parametrize("value", (True, np.bool_(True)))
def test_build_contract_rejects_boolean_grid_sizes(
    field: str, value: bool | np.bool_
) -> None:
    values = _contract().__dict__ | {field: value}
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        endpoints.OrientationEndpointBuildContract(**values)


def test_unsupported_equilibrium_and_zero_density_are_scientific_failures() -> None:
    def unsupported_state(**_kwargs: Any) -> Any:
        raise RuntimeError("synthetic no stationary solution")

    factories, _calls = _factories(state_factory=unsupported_state)
    with pytest.raises(
        endpoints.OrientationEndpointScientificFailure,
        match="dipolar equilibrium endpoint is unsupported",
    ):
        endpoints.build_orientation_endpoint(
            spec=_specs()[0],
            contract=_contract(),
            model_config=_model(),
            initial_condition_config={},
            factories=factories,
        )

    factories, _calls = _factories(
        profile_factory=lambda y, _z, *_radii: np.zeros_like(y)
    )
    with pytest.raises(
        endpoints.OrientationEndpointScientificFailure, match="density has zero support"
    ):
        endpoints.build_orientation_endpoint(
            spec=_specs()[0],
            contract=_contract(),
            model_config=_model(),
            initial_condition_config={},
            factories=factories,
        )


def test_state_builder_value_error_remains_a_programming_or_config_error() -> None:
    def invalid_state_input(**_kwargs: Any) -> Any:
        raise ValueError("synthetic programming or configuration defect")

    factories, _calls = _factories(state_factory=invalid_state_input)
    with pytest.raises(ValueError, match="programming or configuration defect") as exc_info:
        endpoints.build_orientation_endpoint(
            spec=_specs()[0],
            contract=_contract(),
            model_config=_model(),
            initial_condition_config={},
            factories=factories,
        )
    assert not isinstance(exc_info.value, endpoints.OrientationEndpointScientificFailure)


def test_programming_shape_drift_is_not_relabelled_as_scientific_failure() -> None:
    factories, _calls = _factories(
        profile_factory=lambda _y, _z, *_radii: np.ones((2, 2))
    )
    with pytest.raises(ValueError, match="density shape disagrees with its grid"):
        endpoints.build_orientation_endpoint(
            spec=_specs()[0],
            contract=_contract(),
            model_config=_model(),
            initial_condition_config={},
            factories=factories,
        )
