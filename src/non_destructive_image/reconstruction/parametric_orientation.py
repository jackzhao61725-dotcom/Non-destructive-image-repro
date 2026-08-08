"""Low-dimensional independent-endpoint PCI inference for orientation studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .independent_endpoint_information import IndependentEndpointRawBlock
from .linked_scalar_fit import (
    LinkedScalarFitOptions,
    LinkedScalarFitResult,
    fit_linked_scalar_sequence,
)
from .free_radius_model import FreeRadiusCompactDensityModel
from .parameters import SmoothTFParameters, from_internal
from .scalar_measurements import PCILinkedRawOperator, PCINuisanceValues


FloatArray = NDArray[np.floating]
_ENDPOINT_LABELS = ("B_parallel_y", "B_parallel_z")
_FIELD_ORIENTATIONS = ("y", "z")
_EXPOSURES = {"atom": 1, "bright_reference": 1, "dark": 1}


def _immutable(values: ArrayLike) -> FloatArray:
    source = np.asarray(values, dtype=float)
    result = np.frombuffer(source.tobytes(order="C"), dtype=float)
    return result.reshape(source.shape)


@dataclass(frozen=True)
class ParametricOrientationProvenance:
    """Identity and factorisation boundary for the v2 orientation estimator."""

    contract_label: str
    endpoint_labels: tuple[str, str]
    field_orientations: tuple[str, str]
    imaging_axis: str
    independent_preparations: bool
    independent_raw_blocks: bool
    temporal_coupling_used: bool
    generator_reference_used: bool

    def __post_init__(self) -> None:
        if self.contract_label != "chapter_5_orientation_information_contract_v2":
            raise ValueError("parametric orientation contract label changed")
        if tuple(self.endpoint_labels) != _ENDPOINT_LABELS:
            raise ValueError("endpoint labels must follow canonical By/Bz order")
        if tuple(self.field_orientations) != _FIELD_ORIENTATIONS:
            raise ValueError("field orientations must follow canonical y/z order")
        if self.imaging_axis != "x":
            raise ValueError("orientation PCI imaging axis must be x")
        if self.independent_preparations is not True:
            raise ValueError("orientation endpoints must be independently prepared")
        if self.independent_raw_blocks is not True:
            raise ValueError("orientation endpoints must own independent raw blocks")
        if self.temporal_coupling_used is not False:
            raise ValueError("temporal coupling is forbidden")
        if self.generator_reference_used is not False:
            raise ValueError("generator references are forbidden during endpoint fitting")


@dataclass(frozen=True, eq=False)
class ParametricEndpointFitInput:
    """Endpoint-local raw data, model, starts, bounds and optical nuisances."""

    operator: PCILinkedRawOperator
    model: FreeRadiusCompactDensityModel
    raw_block: IndependentEndpointRawBlock
    start_ids: tuple[str, ...]
    initial_parameter_vectors: tuple[ArrayLike, ...]
    parameter_lower: ArrayLike
    parameter_upper: ArrayLike
    initial_nuisance: PCINuisanceValues
    nuisance_lower: ArrayLike
    nuisance_upper: ArrayLike
    options: LinkedScalarFitOptions

    def __post_init__(self) -> None:
        if not isinstance(self.operator, PCILinkedRawOperator):
            raise TypeError("parametric orientation fits require PCILinkedRawOperator")
        if not isinstance(self.model, FreeRadiusCompactDensityModel):
            raise TypeError("model must be FreeRadiusCompactDensityModel")
        if not isinstance(self.raw_block, IndependentEndpointRawBlock):
            raise TypeError("raw_block has the wrong type")
        if not isinstance(self.initial_nuisance, PCINuisanceValues):
            raise TypeError("initial_nuisance must be PCINuisanceValues")
        if not isinstance(self.options, LinkedScalarFitOptions):
            raise TypeError("options must be LinkedScalarFitOptions")
        if dict(self.operator.independent_exposures_by_role) != _EXPOSURES:
            raise ValueError("v2 requires one independent exposure per PCI raw role")
        if self.raw_block.observed_electrons[0].shape != self.operator.grid.camera_shape:
            raise ValueError("endpoint raw camera shape differs from its operator")
        if not np.array_equal(self.model.y_grid_m, self.operator.grid.y_grid_m) or not np.array_equal(
            self.model.z_grid_m,
            self.operator.grid.z_grid_m,
        ):
            raise ValueError("endpoint density model and operator grids differ")

        start_ids = tuple(str(value) for value in self.start_ids)
        if not start_ids or len(set(start_ids)) != len(start_ids):
            raise ValueError("start ids must be non-empty and unique")
        starts = tuple(_immutable(value) for value in self.initial_parameter_vectors)
        if len(starts) != len(start_ids) or any(
            value.shape != (self.model.parameter_count,) or np.any(~np.isfinite(value))
            for value in starts
        ):
            raise ValueError("one finite five-parameter vector is required per start")
        lower = _immutable(self.parameter_lower)
        upper = _immutable(self.parameter_upper)
        if (
            lower.shape != (self.model.parameter_count,)
            or upper.shape != lower.shape
            or np.any(~np.isfinite(lower))
            or np.any(~np.isfinite(upper))
            or np.any(upper <= lower)
            or any(np.any(value < lower) or np.any(value > upper) for value in starts)
        ):
            raise ValueError("parameter bounds or starts are invalid")
        nuisance_lower = _immutable(self.nuisance_lower)
        nuisance_upper = _immutable(self.nuisance_upper)
        nuisance_initial = np.asarray(
            (
                self.initial_nuisance.i0_photoelectrons_per_pixel,
                self.initial_nuisance.dark_electrons_per_pixel,
            ),
            dtype=float,
        )
        if (
            nuisance_lower.shape != (2,)
            or nuisance_upper.shape != (2,)
            or np.any(~np.isfinite(nuisance_lower))
            or np.any(~np.isfinite(nuisance_upper))
            or np.any(nuisance_lower < 0.0)
            or np.any(nuisance_upper <= nuisance_lower)
            or np.any(nuisance_initial < nuisance_lower)
            or np.any(nuisance_initial > nuisance_upper)
        ):
            raise ValueError("nuisance bounds or initial values are invalid")
        object.__setattr__(self, "start_ids", start_ids)
        object.__setattr__(self, "initial_parameter_vectors", starts)
        object.__setattr__(self, "parameter_lower", lower)
        object.__setattr__(self, "parameter_upper", upper)
        object.__setattr__(self, "nuisance_lower", nuisance_lower)
        object.__setattr__(self, "nuisance_upper", nuisance_upper)


@dataclass(frozen=True, eq=False)
class ParametricStartResult:
    """One retained optimiser terminal from a predeclared start."""

    start_id: str
    status: Literal["success", "fit_failure"]
    message: str
    weighted_chi_square: float | None
    fit_result: LinkedScalarFitResult | None

    def __post_init__(self) -> None:
        if not self.start_id:
            raise ValueError("start result id cannot be empty")
        if self.status not in ("success", "fit_failure"):
            raise ValueError("unknown start result status")
        if self.status == "success":
            if self.fit_result is None or self.weighted_chi_square is None:
                raise ValueError("successful start requires a fit and objective")
            if not np.isfinite(self.weighted_chi_square):
                raise ValueError("successful start objective must be finite")
        elif self.fit_result is None:
            if self.weighted_chi_square is not None:
                raise ValueError("exceptional fit failure cannot carry an objective")


@dataclass(frozen=True)
class ParametricEndpointObservables:
    """Model-conditional low-order quantities from one endpoint fit."""

    A: float
    centre_y_um: float
    centre_z_um: float
    sigma_y_um: float
    sigma_z_um: float
    aspect_ratio_y_over_z: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.A,
                self.centre_y_um,
                self.centre_z_um,
                self.sigma_y_um,
                self.sigma_z_um,
                self.aspect_ratio_y_over_z,
            ),
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("endpoint observables must be finite")
        if self.A <= 0.0 or min(self.sigma_y_um, self.sigma_z_um) <= 0.0:
            raise ValueError("endpoint amplitude and widths must be positive")
        if self.aspect_ratio_y_over_z <= 0.0:
            raise ValueError("endpoint aspect ratio must be positive")


@dataclass(frozen=True, eq=False)
class ParametricEndpointFit:
    """All starts and the selected model-conditional endpoint estimate."""

    endpoint_label: str
    field_orientation: str
    status: Literal["success", "fit_failure"]
    message: str
    start_results: tuple[ParametricStartResult, ...]
    selected_start_id: str | None
    selected_fit: LinkedScalarFitResult | None
    physical_parameters: SmoothTFParameters | None
    observables: ParametricEndpointObservables | None

    def __post_init__(self) -> None:
        if self.endpoint_label not in _ENDPOINT_LABELS:
            raise ValueError("unknown endpoint label")
        expected = _FIELD_ORIENTATIONS[_ENDPOINT_LABELS.index(self.endpoint_label)]
        if self.field_orientation != expected:
            raise ValueError("endpoint label and field orientation disagree")
        if not self.start_results:
            raise ValueError("endpoint fit must retain every declared start")
        if self.status == "success":
            if any(
                value is None
                for value in (
                    self.selected_start_id,
                    self.selected_fit,
                    self.physical_parameters,
                    self.observables,
                )
            ):
                raise ValueError("successful endpoint fit is incomplete")
        elif any(
            value is not None
            for value in (
                self.selected_start_id,
                self.selected_fit,
                self.physical_parameters,
                self.observables,
            )
        ):
            raise ValueError("failed endpoint fit cannot publish an estimate")


@dataclass(frozen=True)
class ParametricOrientationPairFit:
    """Two independent endpoint fits combined only as an ordered record."""

    endpoints: tuple[ParametricEndpointFit, ParametricEndpointFit]
    provenance: ParametricOrientationProvenance

    def __post_init__(self) -> None:
        if tuple(value.endpoint_label for value in self.endpoints) != _ENDPOINT_LABELS:
            raise ValueError("pair fit must follow canonical By/Bz order")


def parametric_observables(
    parameters: SmoothTFParameters,
    *,
    profile_exponent: float,
) -> ParametricEndpointObservables:
    """Convert one compact-profile fit to integrated response and rms widths."""

    exponent = float(profile_exponent)
    if not np.isfinite(exponent) or exponent < 1.0:
        raise ValueError("profile exponent must be finite and at least one")
    area_conversion = 1e-12
    integrated = (
        np.pi
        / (exponent + 1.0)
        * parameters.column_density_peak_m2
        * parameters.radius_y_um
        * parameters.radius_z_um
        * area_conversion
    )
    width_denominator = np.sqrt(2.0 * (exponent + 2.0))
    sigma_y = parameters.radius_y_um / width_denominator
    sigma_z = parameters.radius_z_um / width_denominator
    return ParametricEndpointObservables(
        A=float(integrated),
        centre_y_um=parameters.y0_um,
        centre_z_um=parameters.z0_um,
        sigma_y_um=float(sigma_y),
        sigma_z_um=float(sigma_z),
        aspect_ratio_y_over_z=float(sigma_y / sigma_z),
    )


def _same_measurement_design(
    left: PCILinkedRawOperator,
    right: PCILinkedRawOperator,
) -> bool:
    return (
        left.detector == right.detector
        and left.transfer == right.transfer
        and left.response == right.response
        and dict(left.independent_exposures_by_role)
        == dict(right.independent_exposures_by_role)
        and np.array_equal(left.grid.y_grid_m, right.grid.y_grid_m)
        and np.array_equal(left.grid.z_grid_m, right.grid.z_grid_m)
        and np.array_equal(left.grid.pupil, right.grid.pupil)
        and np.array_equal(left.grid.roi_mask, right.grid.roi_mask)
    )


def _validate_pair_inputs(
    inputs: tuple[ParametricEndpointFitInput, ParametricEndpointFitInput],
    provenance: ParametricOrientationProvenance,
) -> None:
    if not isinstance(provenance, ParametricOrientationProvenance):
        raise TypeError("provenance has the wrong type")
    if tuple(value.raw_block.endpoint_label for value in inputs) != _ENDPOINT_LABELS:
        raise ValueError("fit inputs must follow canonical By/Bz order")
    if inputs[0].operator is inputs[1].operator:
        raise ValueError("orientation endpoints require distinct operator instances")
    if inputs[0].operator.response is inputs[1].operator.response:
        raise ValueError("orientation endpoints require distinct response instances")
    if not _same_measurement_design(inputs[0].operator, inputs[1].operator):
        raise ValueError("orientation endpoint measurement designs differ")
    if inputs[0].model.profile_exponent != inputs[1].model.profile_exponent:
        raise ValueError("orientation endpoints must use the same profile exponent")
    if tuple(inputs[0].model.parameter_names) != tuple(inputs[1].model.parameter_names):
        raise ValueError("orientation endpoint parameterisations differ")
    owners = tuple(
        owner for value in inputs for owner in value.raw_block.role_owner_ids
    )
    if len(set(owners)) != 6:
        raise ValueError("all six orientation raw-role owner ids must be unique")


def _fit_one_endpoint(item: ParametricEndpointFitInput) -> ParametricEndpointFit:
    attempts: list[ParametricStartResult] = []
    for start_id, initial in zip(
        item.start_ids,
        item.initial_parameter_vectors,
        strict=True,
    ):
        try:
            result = fit_linked_scalar_sequence(
                item.operator,
                item.model,
                item.raw_block.as_linked_observation(),
                initial_density_coefficients=initial[None, :],
                density_parameter_lower=item.parameter_lower,
                density_coefficient_upper=item.parameter_upper,
                initial_nuisance=item.initial_nuisance,
                nuisance_lower=item.nuisance_lower,
                nuisance_upper=item.nuisance_upper,
                regularisation=None,
                options=item.options,
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            attempts.append(
                ParametricStartResult(
                    start_id=start_id,
                    status="fit_failure",
                    message=f"{type(exc).__name__}: {exc}",
                    weighted_chi_square=None,
                    fit_result=None,
                )
            )
            continue
        success = bool(
            result.diagnostics.success
            and np.isfinite(result.diagnostics.weighted_chi_square)
        )
        attempts.append(
            ParametricStartResult(
                start_id=start_id,
                status="success" if success else "fit_failure",
                message=result.diagnostics.message,
                weighted_chi_square=float(result.diagnostics.weighted_chi_square),
                fit_result=result,
            )
        )

    successful = [value for value in attempts if value.status == "success"]
    if not successful:
        return ParametricEndpointFit(
            endpoint_label=item.raw_block.endpoint_label,
            field_orientation=item.raw_block.field_orientation,
            status="fit_failure",
            message="no predeclared start produced a successful finite fit",
            start_results=tuple(attempts),
            selected_start_id=None,
            selected_fit=None,
            physical_parameters=None,
            observables=None,
        )
    selected = min(successful, key=lambda value: float(value.weighted_chi_square))
    if selected.fit_result is None:
        raise RuntimeError("successful selected start lost its fit result")
    parameters = from_internal(selected.fit_result.density_coefficients[0])
    observables = parametric_observables(
        parameters,
        profile_exponent=item.model.profile_exponent,
    )
    return ParametricEndpointFit(
        endpoint_label=item.raw_block.endpoint_label,
        field_orientation=item.raw_block.field_orientation,
        status="success",
        message=f"selected start {selected.start_id}",
        start_results=tuple(attempts),
        selected_start_id=selected.start_id,
        selected_fit=selected.fit_result,
        physical_parameters=parameters,
        observables=observables,
    )


def fit_independent_parametric_pci_endpoints(
    inputs: tuple[ParametricEndpointFitInput, ParametricEndpointFitInput],
    *,
    provenance: ParametricOrientationProvenance,
) -> ParametricOrientationPairFit:
    """Fit By and Bz separately, retaining every predeclared start terminal."""

    if len(inputs) != 2:
        raise ValueError("orientation fitting requires exactly two endpoints")
    _validate_pair_inputs(inputs, provenance)
    return ParametricOrientationPairFit(
        endpoints=(_fit_one_endpoint(inputs[0]), _fit_one_endpoint(inputs[1])),
        provenance=provenance,
    )


__all__ = [
    "ParametricEndpointFit",
    "ParametricEndpointFitInput",
    "ParametricEndpointObservables",
    "ParametricOrientationPairFit",
    "ParametricOrientationProvenance",
    "ParametricStartResult",
    "fit_independent_parametric_pci_endpoints",
    "parametric_observables",
]
