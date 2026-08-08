"""Information summaries for two independently prepared PCI endpoints.

This module deliberately does not construct a linked two-frame likelihood.
Each endpoint owns its atom, bright-reference and dark blocks, is fitted by a
separate one-frame :class:`PCILinkedRawOperator`, and retains separate optical
nuisance parameters.  Only low-order observable results are paired after both
fits have completed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .linked_scalar_fit import (
    LinkedRawObservation,
    LinkedScalarFitDiagnostics,
    LinkedScalarFitOptions,
    LinkedScalarFitResult,
    draw_linked_raw_observation,
    fit_linked_scalar_sequence,
)
from .object_models import NonnegativeBilinearDensityModel
from .observables import ObservableIntegrationSupport, extract_density_observables
from .regularisation import CurvatureRegularisation
from .scalar_measurements import (
    PCILinkedRawOperator,
    PCINuisanceValues,
)


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
EndpointStatus = Literal["success", "fit_failure"]
EstimateStatus = Literal["complete", "point_only", "unresolved"]

ENDPOINT_LABELS: tuple[str, str] = ("B_parallel_y", "B_parallel_z")
FIELD_ORIENTATIONS: tuple[str, str] = ("y", "z")
RAW_ROLE_NAMES: tuple[str, str, str] = (
    "atom_000",
    "bright_reference",
    "dark",
)
OBSERVABLE_NAMES: tuple[str, ...] = (
    "A",
    "y_c_um",
    "z_c_um",
    "sigma_y_um",
    "sigma_z_um",
)
DERIVED_OBSERVABLE_NAME = "aspect_ratio_y_over_z"
POSITIVE_OBSERVABLE_NAMES: tuple[str, ...] = (
    "A",
    "sigma_y_um",
    "sigma_z_um",
    DERIVED_OBSERVABLE_NAME,
)
SUPPORT_NAMES: tuple[str, str, str] = ("inner", "primary", "outer")
_V1_INDEPENDENT_EXPOSURES_BY_ROLE = {
    "atom": 1,
    "bright_reference": 1,
    "dark": 1,
}
OBSERVABLE_UNITS = {
    "A": "response_integral",
    "y_c_um": "um",
    "z_c_um": "um",
    "sigma_y_um": "um",
    "sigma_z_um": "um",
    DERIVED_OBSERVABLE_NAME: "1",
}


def _immutable(value: ArrayLike, *, dtype: type = float) -> NDArray:
    source = np.asarray(value, dtype=dtype)
    result = np.frombuffer(source.tobytes(order="C"), dtype=np.dtype(dtype))
    return result.reshape(source.shape)


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _array_identity(value: ArrayLike) -> dict[str, object]:
    """Return a platform-stable identity for one numerical contract array."""

    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        canonical = np.asarray(raw, dtype="<c16")
    elif np.issubdtype(raw.dtype, np.bool_):
        canonical = np.asarray(raw, dtype=np.uint8)
    else:
        canonical = np.asarray(raw, dtype="<f8")
    contiguous = np.ascontiguousarray(canonical)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _fit_input_sha256(item: IndependentEndpointFitInput) -> str:
    """Bind a fit to every numerical and physical input consumed downstream."""

    operator = item.operator
    grid = operator.grid
    model = item.model
    support = item.observable_support
    raw = item.raw_block
    regularisation = item.regularisation
    regularisation_identity = None
    if regularisation is not None:
        regularisation_identity = {
            "knot_y_um": _array_identity(regularisation.knot_y_um),
            "knot_z_um": _array_identity(regularisation.knot_z_um),
            "density_scale_m2": regularisation.density_scale_m2,
            "weight_um2": regularisation.weight_um2,
            "boundary_policy": regularisation.boundary_policy,
            "axis_weights": asdict(regularisation.axis_weights),
            "physical_density_matrix": _array_identity(
                regularisation.physical_density_matrix
            ),
        }
    payload = {
        "schema": "independent_endpoint_fit_input_v1",
        "raw_block": {
            "endpoint_label": raw.endpoint_label,
            "field_orientation": raw.field_orientation,
            "role_names": list(raw.role_names),
            "role_owner_ids": list(raw.role_owner_ids),
            "unit": raw.unit,
            "observed_electrons": [
                _array_identity(array) for array in raw.observed_electrons
            ],
        },
        "operator": {
            "class": type(operator).__qualname__,
            "grid": {
                "y_grid_m": _array_identity(grid.y_grid_m),
                "z_grid_m": _array_identity(grid.z_grid_m),
                "pupil": _array_identity(grid.pupil),
                "roi_mask": _array_identity(grid.roi_mask),
                "bin_size": grid.bin_size,
                "camera_pixel_size_m": grid.camera_pixel_size_m,
                "camera_output_shape": grid.camera_output_shape,
            },
            "detector": asdict(operator.detector),
            "response": asdict(operator.response),
            "transfer": asdict(operator.transfer),
            "independent_exposures_by_role": dict(
                operator.independent_exposures_by_role
            ),
            "jacobian_batch_size": operator.jacobian_batch_size,
        },
        "model": {
            "class": type(model).__qualname__,
            "y_grid_m": _array_identity(model.y_grid_m),
            "z_grid_m": _array_identity(model.z_grid_m),
            "knot_y_um": _array_identity(model.knot_y_um),
            "knot_z_um": _array_identity(model.knot_z_um),
            "coefficient_scale_m2": model.coefficient_scale_m2,
            "support_mask": _array_identity(model.support_mask),
        },
        "observable_support": {
            "y_grid_m": _array_identity(support.y_grid_m),
            "z_grid_m": _array_identity(support.z_grid_m),
            "support_mask": _array_identity(support.support_mask),
            "cell_area_m2": _array_identity(support.cell_area_m2),
        },
        "initial_density_coefficients": _array_identity(
            item.initial_density_coefficients
        ),
        "density_parameter_lower": _array_identity(item.density_parameter_lower),
        "density_coefficient_upper": _array_identity(
            item.density_coefficient_upper
        ),
        "initial_nuisance": asdict(item.initial_nuisance),
        "nuisance_lower": _array_identity(item.nuisance_lower),
        "nuisance_upper": _array_identity(item.nuisance_upper),
        "regularisation": regularisation_identity,
        "options": asdict(item.options or LinkedScalarFitOptions()),
    }
    serialised = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


@dataclass(frozen=True, eq=False)
class IndependentEndpointRawBlock:
    """One endpoint's three independently owned PCI raw-count roles."""

    endpoint_label: str
    field_orientation: str
    role_names: tuple[str, str, str]
    role_owner_ids: tuple[str, str, str]
    observed_electrons: tuple[ArrayLike, ArrayLike, ArrayLike]
    unit: str = "electrons"

    def __post_init__(self) -> None:
        label = _nonempty_text(self.endpoint_label, name="endpoint label")
        orientation = _nonempty_text(
            self.field_orientation,
            name="field orientation",
        )
        if label not in ENDPOINT_LABELS:
            raise ValueError("unknown independent-endpoint label")
        expected_orientation = FIELD_ORIENTATIONS[ENDPOINT_LABELS.index(label)]
        if orientation != expected_orientation:
            raise ValueError("endpoint label and field orientation disagree")
        if tuple(self.role_names) != RAW_ROLE_NAMES:
            raise ValueError("endpoint raw roles must use canonical PCI order")
        owners = tuple(
            _nonempty_text(value, name="raw-role owner id")
            for value in self.role_owner_ids
        )
        if len(set(owners)) != 3:
            raise ValueError("raw-role owner ids must be unique within one endpoint")
        arrays = tuple(_immutable(value) for value in self.observed_electrons)
        if len(arrays) != 3 or any(array.ndim != 2 for array in arrays):
            raise ValueError("endpoint raw roles must contain three 2D arrays")
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("endpoint raw-role arrays must share one camera shape")
        if any(np.any(~np.isfinite(array)) for array in arrays):
            raise ValueError("endpoint raw electron values must be finite")
        if self.unit != "electrons":
            raise ValueError("endpoint raw-count unit must be electrons")
        object.__setattr__(self, "endpoint_label", label)
        object.__setattr__(self, "field_orientation", orientation)
        object.__setattr__(self, "role_names", tuple(self.role_names))
        object.__setattr__(self, "role_owner_ids", owners)
        object.__setattr__(self, "observed_electrons", arrays)

    def as_linked_observation(self) -> LinkedRawObservation:
        """Return the existing one-frame raw observation without relabelling roles."""

        return LinkedRawObservation(self.role_names, self.observed_electrons)


@dataclass(frozen=True)
class IndependentEndpointPairProvenance:
    """Frozen identity and independence contract for the orientation pair."""

    contract_label: str
    endpoint_labels: tuple[str, str]
    field_orientations: tuple[str, str]
    imaging_axis: str
    raw_count_unit: str
    density_unit: str
    independent_preparations: bool
    independent_raw_blocks: bool
    temporal_coupling_used: bool
    cross_orientation_amplitude_calibration: bool

    def __post_init__(self) -> None:
        if self.contract_label != "chapter_5_orientation_information_contract_v1":
            raise ValueError("independent-endpoint contract label changed")
        if tuple(self.endpoint_labels) != ENDPOINT_LABELS:
            raise ValueError("endpoint labels must follow the canonical By/Bz order")
        if tuple(self.field_orientations) != FIELD_ORIENTATIONS:
            raise ValueError("field orientations must follow the canonical y/z order")
        if self.imaging_axis != "x":
            raise ValueError("independent-endpoint PCI imaging axis must be x")
        if self.raw_count_unit != "electrons":
            raise ValueError("raw-count unit changed")
        if self.density_unit != "m^-2":
            raise ValueError("column-density unit changed")
        if self.independent_preparations is not True:
            raise ValueError("orientation endpoints must be independently prepared")
        if self.independent_raw_blocks is not True:
            raise ValueError("orientation endpoints must own independent raw blocks")
        if self.temporal_coupling_used is not False:
            raise ValueError("temporal coupling is forbidden for independent endpoints")
        if type(self.cross_orientation_amplitude_calibration) is not bool:
            raise TypeError("cross-orientation amplitude calibration flag must be bool")
        if self.cross_orientation_amplitude_calibration:
            raise ValueError(
                "orientation v1 has no cross-orientation amplitude calibration"
            )


@dataclass(frozen=True)
class IndependentEndpointFitInput:
    """All endpoint-local inputs to one existing single-frame PCI fit."""

    operator: PCILinkedRawOperator
    model: NonnegativeBilinearDensityModel
    raw_block: IndependentEndpointRawBlock
    observable_support: ObservableIntegrationSupport
    initial_density_coefficients: ArrayLike
    density_parameter_lower: float | ArrayLike
    density_coefficient_upper: float | ArrayLike
    initial_nuisance: PCINuisanceValues
    nuisance_lower: ArrayLike
    nuisance_upper: ArrayLike
    regularisation: CurvatureRegularisation | None
    options: LinkedScalarFitOptions | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operator, PCILinkedRawOperator):
            raise TypeError("independent endpoints require PCILinkedRawOperator")
        if not isinstance(self.model, NonnegativeBilinearDensityModel):
            raise TypeError("independent endpoints require the frozen bilinear model")
        if not isinstance(self.raw_block, IndependentEndpointRawBlock):
            raise TypeError("raw_block has the wrong type")
        if not isinstance(self.observable_support, ObservableIntegrationSupport):
            raise TypeError("observable_support has the wrong type")
        if not isinstance(self.initial_nuisance, PCINuisanceValues):
            raise TypeError("initial_nuisance must be PCINuisanceValues")
        if self.options is not None and not isinstance(self.options, LinkedScalarFitOptions):
            raise TypeError("options must be LinkedScalarFitOptions or None")
        if self.regularisation is not None and not isinstance(
            self.regularisation,
            CurvatureRegularisation,
        ):
            raise TypeError("regularisation has the wrong type")
        if (
            dict(self.operator.independent_exposures_by_role)
            != _V1_INDEPENDENT_EXPOSURES_BY_ROLE
        ):
            raise ValueError(
                "orientation v1 requires exactly one independent exposure per raw role"
            )
        initial = np.asarray(self.initial_density_coefficients, dtype=float)
        if initial.shape != (self.model.parameter_count,) or np.any(~np.isfinite(initial)):
            raise ValueError("initial density coefficients have the wrong shape or values")
        density_lower = np.asarray(self.density_parameter_lower, dtype=float)
        density_upper = np.asarray(self.density_coefficient_upper, dtype=float)
        if density_lower.ndim == 0:
            density_lower = np.full(self.model.parameter_count, float(density_lower))
        if density_upper.ndim == 0:
            density_upper = np.full(self.model.parameter_count, float(density_upper))
        if (
            density_lower.shape != (self.model.parameter_count,)
            or density_upper.shape != (self.model.parameter_count,)
            or np.any(~np.isfinite(density_lower))
            or np.any(~np.isfinite(density_upper))
            or np.any(density_upper <= density_lower)
            or np.any(initial < density_lower)
            or np.any(initial > density_upper)
        ):
            raise ValueError("density bounds or initial coefficients are invalid")
        nuisance_lower = np.asarray(self.nuisance_lower, dtype=float)
        nuisance_upper = np.asarray(self.nuisance_upper, dtype=float)
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
        if self.raw_block.observed_electrons[0].shape != self.operator.grid.camera_shape:
            raise ValueError("endpoint raw camera shape differs from its operator")
        object.__setattr__(self, "initial_density_coefficients", _immutable(initial))
        object.__setattr__(self, "density_parameter_lower", _immutable(density_lower))
        object.__setattr__(self, "density_coefficient_upper", _immutable(density_upper))
        object.__setattr__(self, "nuisance_lower", _immutable(nuisance_lower))
        object.__setattr__(self, "nuisance_upper", _immutable(nuisance_upper))


@dataclass(frozen=True, eq=False)
class EndpointObservableVector:
    """Five axis observables from one successfully fitted endpoint."""

    names: tuple[str, ...]
    units: tuple[str, ...]
    values: FloatArray
    supported_mask: BoolArray

    def __post_init__(self) -> None:
        if tuple(self.names) != OBSERVABLE_NAMES:
            raise ValueError("endpoint observable order changed")
        expected_units = tuple(OBSERVABLE_UNITS[name] for name in OBSERVABLE_NAMES)
        if tuple(self.units) != expected_units:
            raise ValueError("endpoint observable units changed")
        values = np.asarray(self.values, dtype=float)
        supported = np.asarray(self.supported_mask, dtype=bool)
        if values.shape != (len(OBSERVABLE_NAMES),) or supported.shape != values.shape:
            raise ValueError("endpoint observable values have the wrong shape")
        if not np.array_equal(supported, np.isfinite(values)):
            raise ValueError("endpoint support mask must identify finite values exactly")
        object.__setattr__(self, "values", _immutable(values))
        object.__setattr__(self, "supported_mask", _immutable(supported, dtype=bool))


@dataclass(frozen=True)
class IndependentEndpointFit:
    """Outcome of one endpoint-local fit; failure does not erase its peer."""

    endpoint_label: str
    field_orientation: str
    role_owner_ids: tuple[str, str, str]
    fit_input_sha256: str
    status: EndpointStatus
    message: str
    fit_result: LinkedScalarFitResult | None
    observables: EndpointObservableVector | None

    def __post_init__(self) -> None:
        if self.endpoint_label not in ENDPOINT_LABELS:
            raise ValueError("unknown endpoint fit label")
        expected = FIELD_ORIENTATIONS[ENDPOINT_LABELS.index(self.endpoint_label)]
        if self.field_orientation != expected:
            raise ValueError("endpoint fit label and orientation disagree")
        if len(set(self.role_owner_ids)) != 3:
            raise ValueError("endpoint fit owner ids must be unique")
        digest = self.fit_input_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("endpoint fit input identity must be a SHA-256 digest")
        _nonempty_text(self.message, name="endpoint fit message")
        if self.status == "success":
            if self.fit_result is None or self.observables is None:
                raise ValueError("successful endpoint fit requires result and observables")
            if not self.fit_result.diagnostics.success:
                raise ValueError("successful endpoint status disagrees with diagnostics")
        elif self.status == "fit_failure":
            if self.observables is not None:
                raise ValueError("failed endpoint fit cannot report observables")
        else:
            raise ValueError("unknown endpoint fit status")

    @property
    def nuisance_owner_ids(self) -> tuple[str, str]:
        """Owners of the bright/dark data that constrain this fit's nuisances."""

        return self.role_owner_ids[1], self.role_owner_ids[2]


@dataclass(frozen=True)
class IndependentEndpointPairFit:
    """Two factorised endpoint fits with no joint Jacobian or nuisance block."""

    endpoints: tuple[IndependentEndpointFit, IndependentEndpointFit]
    provenance: IndependentEndpointPairProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, IndependentEndpointPairProvenance):
            raise TypeError("independent endpoint provenance has the wrong type")
        if any(not isinstance(item, IndependentEndpointFit) for item in self.endpoints):
            raise TypeError("independent endpoint pair contains the wrong fit type")
        if tuple(item.endpoint_label for item in self.endpoints) != ENDPOINT_LABELS:
            raise ValueError("endpoint fits must follow the canonical By/Bz order")
        owners = tuple(
            owner for endpoint in self.endpoints for owner in endpoint.role_owner_ids
        )
        if len(set(owners)) != 6:
            raise ValueError("all six orientation-pair raw-role owner ids must be unique")
        nuisance_owners = tuple(
            owner for endpoint in self.endpoints for owner in endpoint.nuisance_owner_ids
        )
        if len(set(nuisance_owners)) != 4:
            raise ValueError("bright/dark nuisance ownership cannot cross endpoints")

    @property
    def fit_success_mask(self) -> tuple[bool, bool]:
        return (
            self.endpoints[0].status == "success",
            self.endpoints[1].status == "success",
        )


@dataclass(frozen=True, eq=False)
class IndependentEndpointPointFitRecord:
    """Serializable numerical state for one successful endpoint point fit."""

    endpoint_label: str
    field_orientation: str
    role_owner_ids: tuple[str, str, str]
    fit_input_sha256: str
    density_coefficients: ArrayLike
    nuisance_names: tuple[str, ...]
    nuisance_values: ArrayLike
    diagnostics: LinkedScalarFitDiagnostics

    def __post_init__(self) -> None:
        if self.endpoint_label not in ENDPOINT_LABELS:
            raise ValueError("unknown point-fit snapshot endpoint")
        endpoint_index = ENDPOINT_LABELS.index(self.endpoint_label)
        if self.field_orientation != FIELD_ORIENTATIONS[endpoint_index]:
            raise ValueError("point-fit snapshot endpoint and orientation disagree")
        owners = tuple(
            _nonempty_text(value, name="point-fit snapshot role owner")
            for value in self.role_owner_ids
        )
        if len(owners) != 3 or len(set(owners)) != 3:
            raise ValueError("point-fit snapshot requires three unique role owners")
        digest = self.fit_input_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("point-fit snapshot input identity must be SHA-256")
        coefficients = np.asarray(self.density_coefficients, dtype=float)
        nuisance = np.asarray(self.nuisance_values, dtype=float)
        names = tuple(self.nuisance_names)
        if (
            coefficients.ndim != 1
            or coefficients.size == 0
            or np.any(~np.isfinite(coefficients))
        ):
            raise ValueError("snapshot density coefficients must be a finite vector")
        if (
            len(names) == 0
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
            or nuisance.shape != (len(names),)
            or np.any(~np.isfinite(nuisance))
        ):
            raise ValueError("snapshot nuisance state is invalid")
        diagnostics = self.diagnostics
        if not isinstance(diagnostics, LinkedScalarFitDiagnostics):
            raise TypeError("snapshot diagnostics have the wrong type")
        if diagnostics.success is not True:
            raise ValueError("a successful point-fit snapshot requires success diagnostics")
        _nonempty_text(diagnostics.message, name="point-fit diagnostic message")
        residual = np.asarray(diagnostics.whitened_residual_vector, dtype=float)
        if residual.ndim != 1 or residual.size == 0 or np.any(~np.isfinite(residual)):
            raise ValueError("snapshot whitened residual must be a finite vector")
        finite_nonnegative = (
            diagnostics.weighted_chi_square,
            diagnostics.reduced_chi_square,
            diagnostics.regularisation_objective,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in finite_nonnegative):
            raise ValueError("snapshot diagnostic objectives must be finite and non-negative")
        if (
            diagnostics.degrees_of_freedom <= 0
            or diagnostics.data_jacobian_rank < 0
            or diagnostics.nfev <= 0
            or diagnostics.irls_iterations <= 0
            or diagnostics.active_lower_density_coefficients < 0
            or diagnostics.active_upper_density_coefficients < 0
            or diagnostics.active_nuisance_bounds < 0
        ):
            raise ValueError("snapshot diagnostic counts are invalid")
        condition = float(diagnostics.data_jacobian_condition)
        if np.isnan(condition) or condition < 0.0:
            raise ValueError("snapshot Jacobian condition is invalid")
        object.__setattr__(self, "role_owner_ids", owners)
        object.__setattr__(self, "density_coefficients", _immutable(coefficients))
        object.__setattr__(self, "nuisance_names", names)
        object.__setattr__(self, "nuisance_values", _immutable(nuisance))


@dataclass(frozen=True)
class IndependentEndpointPointFitSnapshot:
    """Process-independent snapshot of a successful factorised endpoint pair."""

    endpoints: tuple[
        IndependentEndpointPointFitRecord,
        IndependentEndpointPointFitRecord,
    ]
    provenance: IndependentEndpointPairProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, IndependentEndpointPairProvenance):
            raise TypeError("point-fit snapshot provenance has the wrong type")
        if any(
            not isinstance(endpoint, IndependentEndpointPointFitRecord)
            for endpoint in self.endpoints
        ):
            raise TypeError("point-fit snapshot contains the wrong endpoint type")
        if tuple(endpoint.endpoint_label for endpoint in self.endpoints) != ENDPOINT_LABELS:
            raise ValueError("point-fit snapshot must follow canonical By/Bz order")
        owners = tuple(
            owner for endpoint in self.endpoints for owner in endpoint.role_owner_ids
        )
        if len(set(owners)) != 6:
            raise ValueError("point-fit snapshot requires six unique role owners")
        if self.endpoints[0].fit_input_sha256 == self.endpoints[1].fit_input_sha256:
            raise ValueError("point-fit snapshot endpoint inputs must be distinct")


def snapshot_successful_independent_endpoint_pair_fit(
    fit: IndependentEndpointPairFit,
) -> IndependentEndpointPointFitSnapshot:
    """Capture only the state needed to restore a successful point fit safely."""

    if not isinstance(fit, IndependentEndpointPairFit):
        raise TypeError("fit has the wrong type")
    if fit.fit_success_mask != (True, True):
        raise ValueError("point-fit snapshots require two successful endpoints")
    records: list[IndependentEndpointPointFitRecord] = []
    for endpoint in fit.endpoints:
        result = endpoint.fit_result
        if result is None:
            raise RuntimeError("successful endpoint fit lost its numerical result")
        if result.density_coefficients.shape[0] != 1:
            raise ValueError("orientation point-fit snapshots require one density frame")
        if endpoint.message != result.diagnostics.message:
            raise ValueError("endpoint message and fit diagnostics disagree")
        records.append(
            IndependentEndpointPointFitRecord(
                endpoint_label=endpoint.endpoint_label,
                field_orientation=endpoint.field_orientation,
                role_owner_ids=endpoint.role_owner_ids,
                fit_input_sha256=endpoint.fit_input_sha256,
                density_coefficients=result.density_coefficients[0],
                nuisance_names=result.nuisance_names,
                nuisance_values=result.nuisance_values,
                diagnostics=result.diagnostics,
            )
        )
    return IndependentEndpointPointFitSnapshot(
        endpoints=(records[0], records[1]),
        provenance=fit.provenance,
    )


def _restored_diagnostics_agree(
    item: IndependentEndpointFitInput,
    record: IndependentEndpointPointFitRecord,
    *,
    prediction_vector: FloatArray,
    conditional_variance: FloatArray,
    prediction_jacobian: FloatArray,
) -> None:
    """Reject a snapshot whose numerical state is inconsistent with its inputs."""

    observed = np.concatenate(
        [
            np.asarray(array, dtype=float)[item.operator.grid.roi_mask]
            for array in item.raw_block.observed_electrons
        ]
    )
    if observed.shape != prediction_vector.shape or conditional_variance.shape != observed.shape:
        raise ValueError("restored point-fit raw and prediction shapes disagree")
    whitened = (observed - prediction_vector) / np.sqrt(conditional_variance)
    stored = np.asarray(record.diagnostics.whitened_residual_vector, dtype=float)
    if stored.shape != whitened.shape or not np.allclose(
        stored,
        whitened,
        rtol=5.0e-12,
        atol=5.0e-12,
    ):
        raise ValueError("point-fit snapshot residual identity changed")
    chi_square = float(whitened @ whitened)
    diagnostics = record.diagnostics
    if not np.isclose(
        diagnostics.weighted_chi_square,
        chi_square,
        rtol=5.0e-12,
        atol=5.0e-12,
    ):
        raise ValueError("point-fit snapshot chi-square identity changed")
    jacobian = np.asarray(prediction_jacobian, dtype=float)
    if (
        jacobian.ndim != 2
        or jacobian.shape[0] != prediction_vector.size
        or np.any(~np.isfinite(jacobian))
    ):
        raise ValueError("restored point-fit Jacobian is invalid")
    weighted_jacobian = jacobian / np.sqrt(conditional_variance)[:, None]
    singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
    threshold = (
        np.finfo(float).eps * max(weighted_jacobian.shape) * singular_values[0]
        if singular_values.size
        else float("inf")
    )
    expected_rank = int(np.count_nonzero(singular_values > threshold))
    expected_condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size
        and expected_rank == weighted_jacobian.shape[1]
        and singular_values[-1] > threshold
        else float("inf")
    )
    if diagnostics.data_jacobian_rank != expected_rank:
        raise ValueError("point-fit snapshot Jacobian rank identity changed")
    stored_condition = float(diagnostics.data_jacobian_condition)
    condition_matches = (
        np.isinf(stored_condition) and np.isinf(expected_condition)
    ) or (
        np.isfinite(stored_condition)
        and np.isfinite(expected_condition)
        and np.isclose(
            stored_condition,
            expected_condition,
            rtol=5.0e-12,
            atol=5.0e-12,
        )
    )
    if not condition_matches:
        raise ValueError("point-fit snapshot Jacobian condition identity changed")
    expected_dof = max(prediction_vector.size - expected_rank, 1)
    if diagnostics.degrees_of_freedom != expected_dof or not np.isclose(
        diagnostics.reduced_chi_square,
        chi_square / expected_dof,
        rtol=5.0e-12,
        atol=5.0e-12,
    ):
        raise ValueError("point-fit snapshot degrees-of-freedom identity changed")
    coefficients = np.asarray(record.density_coefficients, dtype=float)
    density_tolerance = 1.0e-6 * np.maximum(
        item.density_coefficient_upper - item.density_parameter_lower,
        1.0,
    )
    lower_active = int(
        np.count_nonzero(
            coefficients - item.density_parameter_lower <= density_tolerance
        )
    )
    upper_active = int(
        np.count_nonzero(
            item.density_coefficient_upper - coefficients <= density_tolerance
        )
    )
    nuisance = np.asarray(record.nuisance_values, dtype=float)
    nuisance_tolerance = 1.0e-6 * np.maximum(item.nuisance_upper, 1.0)
    nuisance_active = int(
        np.count_nonzero(
            (nuisance - item.nuisance_lower <= nuisance_tolerance)
            | (item.nuisance_upper - nuisance <= nuisance_tolerance)
        )
    )
    if (
        diagnostics.active_lower_density_coefficients != lower_active
        or diagnostics.active_upper_density_coefficients != upper_active
        or diagnostics.active_nuisance_bounds != nuisance_active
    ):
        raise ValueError("point-fit snapshot active-bound identity changed")
    if item.regularisation is None:
        regularisation_objective = 0.0
    else:
        residual = item.regularisation.residual_from_coefficients(
            coefficients,
            coefficient_scale_m2=item.model.coefficient_scale_m2,
        )
        regularisation_objective = 0.5 * float(residual @ residual)
    if not np.isclose(
        diagnostics.regularisation_objective,
        regularisation_objective,
        rtol=5.0e-12,
        atol=5.0e-12,
    ):
        raise ValueError("point-fit snapshot regularisation identity changed")


def restore_successful_independent_endpoint_pair_fit(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    snapshot: IndependentEndpointPointFitSnapshot,
) -> IndependentEndpointPairFit:
    """Restore a point fit after binding and recomputing every derived array."""

    if not isinstance(snapshot, IndependentEndpointPointFitSnapshot):
        raise TypeError("snapshot has the wrong type")
    _validate_input_pair(inputs, snapshot.provenance)
    restored: list[IndependentEndpointFit] = []
    for item, record in zip(inputs, snapshot.endpoints, strict=True):
        if (
            item.raw_block.endpoint_label != record.endpoint_label
            or item.raw_block.field_orientation != record.field_orientation
            or item.raw_block.role_owner_ids != record.role_owner_ids
        ):
            raise ValueError("point-fit snapshot endpoint ownership changed")
        current_identity = _fit_input_sha256(item)
        if current_identity != record.fit_input_sha256:
            raise ValueError("point-fit snapshot input identity changed")
        coefficients = np.asarray(record.density_coefficients, dtype=float)
        nuisance_values = np.asarray(record.nuisance_values, dtype=float)
        if coefficients.shape != (item.model.parameter_count,):
            raise ValueError("point-fit snapshot coefficient shape changed")
        if np.any(coefficients < item.density_parameter_lower) or np.any(
            coefficients > item.density_coefficient_upper
        ):
            raise ValueError("point-fit snapshot coefficients violate current bounds")
        expected_nuisance_names = tuple(item.operator.nuisance_names)
        if record.nuisance_names != expected_nuisance_names:
            raise ValueError("point-fit snapshot nuisance names changed")
        if nuisance_values.shape != (2,) or np.any(
            nuisance_values < item.nuisance_lower
        ) or np.any(nuisance_values > item.nuisance_upper):
            raise ValueError("point-fit snapshot nuisances violate current bounds")
        nuisance = PCINuisanceValues(*nuisance_values)
        prediction = item.operator.expected_linked_sequence_and_jacobian_model(
            item.model,
            [coefficients],
            nuisance,
        )
        conditional_variance = np.concatenate(
            [
                np.asarray(array, dtype=float)[item.operator.grid.roi_mask]
                for array in prediction.conditional_variance_electrons2
            ]
        )
        _restored_diagnostics_agree(
            item,
            record,
            prediction_vector=np.asarray(prediction.prediction_vector, dtype=float),
            conditional_variance=conditional_variance,
            prediction_jacobian=np.asarray(prediction.jacobian, dtype=float),
        )
        density = item.model.column_density(coefficients)
        result = LinkedScalarFitResult(
            density_coefficients=coefficients[None, :],
            column_density_m2=(density,),
            nuisance_names=record.nuisance_names,
            nuisance_values=nuisance_values,
            prediction=prediction,
            diagnostics=record.diagnostics,
        )
        restored.append(
            IndependentEndpointFit(
                endpoint_label=record.endpoint_label,
                field_orientation=record.field_orientation,
                role_owner_ids=record.role_owner_ids,
                fit_input_sha256=record.fit_input_sha256,
                status="success",
                message=record.diagnostics.message,
                fit_result=result,
                observables=_observable_vector(density, item.observable_support),
            )
        )
    return IndependentEndpointPairFit(
        endpoints=(restored[0], restored[1]),
        provenance=snapshot.provenance,
    )


@dataclass(frozen=True, eq=False)
class IndependentEndpointBootstrap:
    """Endpoint-aligned conditional samples from independently refitted draws.

    ``samples`` has shape ``(draw, endpoint, observable)``.  A failed endpoint
    occupies only its own all-NaN slice; the peer endpoint remains available.
    """

    point_fit: IndependentEndpointPairFit
    fit_success_mask: BoolArray
    samples: FloatArray
    supported_mask: BoolArray

    def __post_init__(self) -> None:
        if not isinstance(self.point_fit, IndependentEndpointPairFit):
            raise TypeError("point_fit has the wrong type")
        success = np.asarray(self.fit_success_mask, dtype=bool)
        samples = np.asarray(self.samples, dtype=float)
        supported = np.asarray(self.supported_mask, dtype=bool)
        if success.ndim != 2 or success.shape[1] != 2 or success.shape[0] < 1:
            raise ValueError("bootstrap fit-success mask must have shape (draw, 2)")
        expected_shape = (success.shape[0], 2, len(OBSERVABLE_NAMES))
        if samples.shape != expected_shape or supported.shape != expected_shape:
            raise ValueError("independent-endpoint bootstrap arrays have the wrong shape")
        if not np.array_equal(supported, np.isfinite(samples)):
            raise ValueError("bootstrap support mask must identify finite values exactly")
        if np.any(supported & ~success[:, :, None]):
            raise ValueError("failed endpoint refits cannot retain supported observables")
        object.__setattr__(self, "fit_success_mask", _immutable(success, dtype=bool))
        object.__setattr__(self, "samples", _immutable(samples))
        object.__setattr__(self, "supported_mask", _immutable(supported, dtype=bool))


@dataclass(frozen=True, eq=False)
class IndependentEndpointSupportPostprocessing:
    """One pair fit evaluated on the frozen inner, primary and outer supports."""

    support_names: tuple[str, str, str]
    endpoint_fit_success_mask: BoolArray
    values: FloatArray
    supported_mask: BoolArray

    def __post_init__(self) -> None:
        if tuple(self.support_names) != SUPPORT_NAMES:
            raise ValueError("support names must be exactly inner, primary and outer")
        success = np.asarray(self.endpoint_fit_success_mask, dtype=bool)
        values = np.asarray(self.values, dtype=float)
        supported = np.asarray(self.supported_mask, dtype=bool)
        expected_shape = (len(SUPPORT_NAMES), 2, len(OBSERVABLE_NAMES))
        if success.shape != (2,):
            raise ValueError("endpoint fit-success mask must have shape (2,)")
        if values.shape != expected_shape or supported.shape != expected_shape:
            raise ValueError("multi-support observable arrays have the wrong shape")
        if not np.array_equal(supported, np.isfinite(values)):
            raise ValueError("multi-support mask must identify finite values exactly")
        if np.any(supported & ~success[None, :, None]):
            raise ValueError("failed endpoint fits cannot report support observables")
        object.__setattr__(
            self,
            "endpoint_fit_success_mask",
            _immutable(success, dtype=bool),
        )
        object.__setattr__(self, "values", _immutable(values))
        object.__setattr__(self, "supported_mask", _immutable(supported, dtype=bool))

    def support_index(self, support_name: str) -> int:
        """Return the canonical array index of a named support."""

        if support_name not in self.support_names:
            raise ValueError("unknown independent-endpoint support name")
        return self.support_names.index(support_name)


@dataclass(frozen=True)
class IndependentEndpointBootstrapDraw:
    """One conditional raw draw and its endpoint-local refit outcomes."""

    draw_id: int
    raw_blocks: tuple[IndependentEndpointRawBlock, IndependentEndpointRawBlock]
    fit: IndependentEndpointPairFit
    postprocessed: IndependentEndpointSupportPostprocessing
    fit_messages: tuple[str, str]

    def __post_init__(self) -> None:
        if type(self.draw_id) is not int or self.draw_id < 0:
            raise ValueError("bootstrap draw_id must be a non-negative integer")
        if tuple(block.endpoint_label for block in self.raw_blocks) != ENDPOINT_LABELS:
            raise ValueError("bootstrap raw blocks changed endpoint order")
        owners = tuple(owner for block in self.raw_blocks for owner in block.role_owner_ids)
        if len(set(owners)) != 6:
            raise ValueError("bootstrap raw-block owners must remain endpoint-local")
        if not isinstance(self.fit, IndependentEndpointPairFit):
            raise TypeError("bootstrap draw fit has the wrong type")
        if tuple(block.role_owner_ids for block in self.raw_blocks) != tuple(
            endpoint.role_owner_ids for endpoint in self.fit.endpoints
        ):
            raise ValueError("bootstrap raw ownership differs from the refit")
        if not isinstance(
            self.postprocessed,
            IndependentEndpointSupportPostprocessing,
        ):
            raise TypeError("bootstrap draw postprocessing has the wrong type")
        if tuple(self.postprocessed.endpoint_fit_success_mask) != self.fit.fit_success_mask:
            raise ValueError("bootstrap fit and postprocessing success masks differ")
        messages = tuple(
            _nonempty_text(message, name="bootstrap endpoint fit message")
            for message in self.fit_messages
        )
        if messages != tuple(endpoint.message for endpoint in self.fit.endpoints):
            raise ValueError("bootstrap fit messages changed endpoint identity")
        object.__setattr__(self, "fit_messages", messages)


@dataclass(frozen=True)
class ConditionalEndpointEstimate:
    """Point and optional complete conditional interval for one quantity."""

    observable_name: str
    quantity: str
    unit: str
    estimate: float | None
    status: EstimateStatus
    lower: float | None
    upper: float | None
    requested_draws: int
    supported_draws: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.observable_name not in (*OBSERVABLE_NAMES, DERIVED_OBSERVABLE_NAME):
            raise ValueError("unknown independent-endpoint observable")
        expected_unit = (
            "1" if self.quantity == "ratio_Bz_over_By" else OBSERVABLE_UNITS[self.observable_name]
        )
        if self.unit != expected_unit:
            raise ValueError("conditional endpoint estimate unit changed")
        if (
            type(self.requested_draws) is not int
            or type(self.supported_draws) is not int
            or self.requested_draws < 1
            or not 0 <= self.supported_draws <= self.requested_draws
        ):
            raise ValueError("conditional endpoint draw counts are invalid")
        if self.status == "complete":
            values = (self.estimate, self.lower, self.upper)
            if any(value is None or not np.isfinite(value) for value in values):
                raise ValueError("complete estimate requires finite point and interval")
            assert self.lower is not None and self.upper is not None
            if (
                self.requested_draws < 2
                or self.supported_draws != self.requested_draws
                or self.lower > self.upper
                or self.reasons
            ):
                raise ValueError("complete estimate interval or reasons are invalid")
        elif self.status == "point_only":
            if self.estimate is None or not np.isfinite(self.estimate):
                raise ValueError("point-only estimate requires a finite point")
            if self.lower is not None or self.upper is not None or not self.reasons:
                raise ValueError("point-only estimate must suppress bounds with reasons")
        elif self.status == "unresolved":
            if any(value is not None for value in (self.estimate, self.lower, self.upper)):
                raise ValueError("unresolved estimate cannot report values")
            if not self.reasons:
                raise ValueError("unresolved estimate requires reasons")
        else:
            raise ValueError("unknown estimate status")


@dataclass(frozen=True)
class IndependentObservablePairSummary:
    """By, Bz, difference and optional ratio for one observable."""

    observable_name: str
    by: ConditionalEndpointEstimate
    bz: ConditionalEndpointEstimate
    delta_b: ConditionalEndpointEstimate
    ratio_bz_over_by: ConditionalEndpointEstimate | None

    def __post_init__(self) -> None:
        if self.observable_name not in (*OBSERVABLE_NAMES, DERIVED_OBSERVABLE_NAME):
            raise ValueError("unknown independent-endpoint summary observable")
        records = (self.by, self.bz, self.delta_b)
        quantities = (*ENDPOINT_LABELS, "delta_Bz_minus_By")
        if any(
            record.observable_name != self.observable_name
            or record.quantity != quantity
            for record, quantity in zip(records, quantities, strict=True)
        ):
            raise ValueError("independent-endpoint summary identity changed")
        ratio_expected = self.observable_name in POSITIVE_OBSERVABLE_NAMES
        if ratio_expected != (self.ratio_bz_over_by is not None):
            raise ValueError("independent-endpoint ratio membership changed")
        if self.ratio_bz_over_by is not None and (
            self.ratio_bz_over_by.observable_name != self.observable_name
            or self.ratio_bz_over_by.quantity != "ratio_Bz_over_By"
        ):
            raise ValueError("independent-endpoint ratio identity changed")


@dataclass(frozen=True)
class IndependentEndpointInformationSummary:
    """Observable-first result for the independently prepared By/Bz pair."""

    observables: tuple[IndependentObservablePairSummary, ...]
    provenance: IndependentEndpointPairProvenance

    def __post_init__(self) -> None:
        expected_names = (*OBSERVABLE_NAMES, DERIVED_OBSERVABLE_NAME)
        if tuple(item.observable_name for item in self.observables) != expected_names:
            raise ValueError("independent-endpoint information order changed")
        if not isinstance(self.provenance, IndependentEndpointPairProvenance):
            raise TypeError("independent-endpoint information provenance has wrong type")


def _same_grid(left: PCILinkedRawOperator, right: PCILinkedRawOperator) -> bool:
    left_grid = left.grid
    right_grid = right.grid
    return (
        np.array_equal(left_grid.y_grid_m, right_grid.y_grid_m)
        and np.array_equal(left_grid.z_grid_m, right_grid.z_grid_m)
        and np.array_equal(left_grid.pupil, right_grid.pupil)
        and np.array_equal(left_grid.roi_mask, right_grid.roi_mask)
        and left_grid.bin_size == right_grid.bin_size
        and left_grid.camera_pixel_size_m == right_grid.camera_pixel_size_m
        and left_grid.camera_output_shape == right_grid.camera_output_shape
    )


def _same_basis(
    left: NonnegativeBilinearDensityModel,
    right: NonnegativeBilinearDensityModel,
) -> bool:
    return (
        np.array_equal(left.y_grid_m, right.y_grid_m)
        and np.array_equal(left.z_grid_m, right.z_grid_m)
        and np.array_equal(left.knot_y_um, right.knot_y_um)
        and np.array_equal(left.knot_z_um, right.knot_z_um)
        and left.coefficient_scale_m2 == right.coefficient_scale_m2
        and np.array_equal(left.support_mask, right.support_mask)
    )


def _same_measurement_design(
    left: PCILinkedRawOperator,
    right: PCILinkedRawOperator,
) -> bool:
    """Return parity of non-response detector and PCI acquisition settings."""

    return (
        left.detector == right.detector
        and left.transfer == right.transfer
        and dict(left.independent_exposures_by_role)
        == dict(right.independent_exposures_by_role)
    )


def _validate_input_pair(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    provenance: IndependentEndpointPairProvenance,
) -> None:
    if not isinstance(provenance, IndependentEndpointPairProvenance):
        raise TypeError("provenance has the wrong type")
    if tuple(item.raw_block.endpoint_label for item in inputs) != ENDPOINT_LABELS:
        raise ValueError("fit inputs must follow the canonical By/Bz order")
    if inputs[0].operator is inputs[1].operator:
        raise ValueError("orientation endpoints require distinct operator instances")
    if inputs[0].operator.response is inputs[1].operator.response:
        raise ValueError("orientation endpoints require distinct response instances")
    if inputs[0].operator.response != inputs[1].operator.response:
        raise ValueError(
            "orientation v1 requires the ideal rotation-covariant scalar response"
        )
    owners = tuple(owner for item in inputs for owner in item.raw_block.role_owner_ids)
    if len(set(owners)) != 6:
        raise ValueError("all six orientation-pair raw-role owner ids must be unique")
    if not _same_grid(inputs[0].operator, inputs[1].operator):
        raise ValueError("orientation endpoint reconstruction grids differ")
    if not _same_measurement_design(inputs[0].operator, inputs[1].operator):
        raise ValueError(
            "orientation endpoint detector or PCI acquisition settings differ"
        )
    if not _same_basis(inputs[0].model, inputs[1].model):
        raise ValueError("orientation endpoint density bases or model supports differ")
    if not inputs[0].observable_support.is_identical_to(inputs[1].observable_support):
        raise ValueError("orientation endpoint observable supports differ")
    for item in inputs:
        if (
            dict(item.operator.independent_exposures_by_role)
            != _V1_INDEPENDENT_EXPOSURES_BY_ROLE
        ):
            raise ValueError(
                "orientation v1 requires exactly one independent exposure per raw role"
            )
        if item.raw_block.observed_electrons[0].shape != item.operator.grid.camera_shape:
            raise ValueError("endpoint raw camera shape differs from its operator")
        if (
            not np.array_equal(item.model.y_grid_m, item.operator.grid.y_grid_m)
            or not np.array_equal(item.model.z_grid_m, item.operator.grid.z_grid_m)
        ):
            raise ValueError("endpoint density model and operator grids differ")
        support = item.observable_support
        if (
            not np.array_equal(support.y_grid_m, item.model.y_grid_m)
            or not np.array_equal(support.z_grid_m, item.model.z_grid_m)
            or np.any(support.support_mask & ~item.model.support_mask)
        ):
            raise ValueError("endpoint observable support differs from the model contract")


def _validate_fit_input_binding(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    fit: IndependentEndpointPairFit,
) -> None:
    """Reject post-fit substitution of any raw, operator or fit-contract input."""

    if tuple(endpoint.endpoint_label for endpoint in fit.endpoints) != ENDPOINT_LABELS:
        raise ValueError("fit input binding changed endpoint order")
    for item, endpoint in zip(inputs, fit.endpoints, strict=True):
        if endpoint.fit_input_sha256 != _fit_input_sha256(item):
            raise ValueError(
                f"{endpoint.endpoint_label} fit input identity differs from the fit"
            )


def _observable_vector(
    density_m2: ArrayLike,
    support: ObservableIntegrationSupport,
) -> EndpointObservableVector:
    summary = extract_density_observables(density_m2, support)
    values = np.asarray(
        [
            summary.integrated_response,
            np.nan if summary.centroid_y_m is None else summary.centroid_y_m * 1e6,
            np.nan if summary.centroid_z_m is None else summary.centroid_z_m * 1e6,
            (
                np.nan
                if summary.covariance_m2 is None
                else np.sqrt(summary.covariance_m2[0, 0]) * 1e6
            ),
            (
                np.nan
                if summary.covariance_m2 is None
                else np.sqrt(summary.covariance_m2[1, 1]) * 1e6
            ),
        ],
        dtype=float,
    )
    return EndpointObservableVector(
        names=OBSERVABLE_NAMES,
        units=tuple(OBSERVABLE_UNITS[name] for name in OBSERVABLE_NAMES),
        values=values,
        supported_mask=np.isfinite(values),
    )


def fit_independent_pci_endpoints(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    *,
    provenance: IndependentEndpointPairProvenance,
) -> IndependentEndpointPairFit:
    """Fit By and Bz independently and combine only their observable records."""

    if len(inputs) != 2:
        raise ValueError("independent endpoint fitting requires exactly By and Bz")
    _validate_input_pair(inputs, provenance)
    outcomes: list[IndependentEndpointFit] = []
    for item in inputs:
        fit_input_sha256 = _fit_input_sha256(item)
        try:
            result = fit_linked_scalar_sequence(
                item.operator,
                item.model,
                item.raw_block.as_linked_observation(),
                initial_density_coefficients=item.initial_density_coefficients[None, :],
                density_parameter_lower=item.density_parameter_lower,
                density_coefficient_upper=item.density_coefficient_upper,
                initial_nuisance=item.initial_nuisance,
                nuisance_lower=item.nuisance_lower,
                nuisance_upper=item.nuisance_upper,
                regularisation=item.regularisation,
                options=item.options,
            )
        except (FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            outcomes.append(
                IndependentEndpointFit(
                    endpoint_label=item.raw_block.endpoint_label,
                    field_orientation=item.raw_block.field_orientation,
                    role_owner_ids=item.raw_block.role_owner_ids,
                    fit_input_sha256=fit_input_sha256,
                    status="fit_failure",
                    message=f"{type(exc).__name__}: {exc}",
                    fit_result=None,
                    observables=None,
                )
            )
            continue
        success = bool(result.diagnostics.success)
        outcomes.append(
            IndependentEndpointFit(
                endpoint_label=item.raw_block.endpoint_label,
                field_orientation=item.raw_block.field_orientation,
                role_owner_ids=item.raw_block.role_owner_ids,
                fit_input_sha256=fit_input_sha256,
                status="success" if success else "fit_failure",
                message=result.diagnostics.message,
                fit_result=result,
                observables=(
                    _observable_vector(
                        result.column_density_m2[0],
                        item.observable_support,
                    )
                    if success
                    else None
                ),
            )
        )
    return IndependentEndpointPairFit(
        endpoints=(outcomes[0], outcomes[1]),
        provenance=provenance,
    )


def postprocess_independent_endpoint_supports(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    fit: IndependentEndpointPairFit,
    supports: Mapping[str, ObservableIntegrationSupport],
) -> IndependentEndpointSupportPostprocessing:
    """Evaluate one pair fit on three supports without refitting either endpoint."""

    _validate_input_pair(inputs, fit.provenance)
    _validate_fit_input_binding(inputs, fit)
    if tuple(item.raw_block.role_owner_ids for item in inputs) != tuple(
        endpoint.role_owner_ids for endpoint in fit.endpoints
    ):
        raise ValueError("support postprocessing input ownership differs from the fit")
    if tuple(supports) != SUPPORT_NAMES:
        raise ValueError("supports must be ordered as inner, primary and outer")
    values = np.full(
        (len(SUPPORT_NAMES), 2, len(OBSERVABLE_NAMES)),
        np.nan,
        dtype=float,
    )
    supported = np.zeros(values.shape, dtype=bool)
    for support_index, support_name in enumerate(SUPPORT_NAMES):
        support = supports[support_name]
        if not isinstance(support, ObservableIntegrationSupport):
            raise TypeError(f"{support_name} support has the wrong type")
        for endpoint_index, (item, endpoint) in enumerate(
            zip(inputs, fit.endpoints, strict=True)
        ):
            if (
                not np.array_equal(support.y_grid_m, item.model.y_grid_m)
                or not np.array_equal(support.z_grid_m, item.model.z_grid_m)
                or np.any(support.support_mask & ~item.model.support_mask)
            ):
                raise ValueError(
                    f"{support_name} support differs from the endpoint model contract"
                )
            if endpoint.status != "success":
                continue
            if endpoint.fit_result is None:
                raise RuntimeError("successful endpoint fit lost its numerical result")
            vector = _observable_vector(
                endpoint.fit_result.column_density_m2[0],
                support,
            )
            values[support_index, endpoint_index] = vector.values
            supported[support_index, endpoint_index] = vector.supported_mask
    return IndependentEndpointSupportPostprocessing(
        support_names=SUPPORT_NAMES,
        endpoint_fit_success_mask=np.asarray(fit.fit_success_mask, dtype=bool),
        values=values,
        supported_mask=supported,
    )


def draw_and_refit_independent_endpoint_bootstrap(
    inputs: tuple[IndependentEndpointFitInput, IndependentEndpointFitInput],
    point_fit: IndependentEndpointPairFit,
    supports: Mapping[str, ObservableIntegrationSupport],
    *,
    draw_id: int,
    endpoint_rngs: tuple[np.random.Generator, np.random.Generator],
) -> IndependentEndpointBootstrapDraw:
    """Draw and refit one conditional By/Bz pair with independent RNG streams.

    Each endpoint draws its own atom, bright-reference and dark arrays from its
    own point-fit prediction.  A failed endpoint refit is retained once and is
    never redrawn; its peer endpoint and raw block remain intact.
    """

    if type(draw_id) is not int or draw_id < 0:
        raise ValueError("bootstrap draw_id must be a non-negative integer")
    if (
        len(endpoint_rngs) != 2
        or any(not isinstance(rng, np.random.Generator) for rng in endpoint_rngs)
    ):
        raise TypeError("endpoint_rngs must contain two numpy.random.Generator objects")
    if endpoint_rngs[0] is endpoint_rngs[1]:
        raise ValueError("orientation endpoints require distinct RNG objects")
    _validate_input_pair(inputs, point_fit.provenance)
    _validate_fit_input_binding(inputs, point_fit)
    if point_fit.fit_success_mask != (True, True):
        raise ValueError("conditional drawing requires two successful point fits")
    postprocess_independent_endpoint_supports(inputs, point_fit, supports)

    drawn_inputs: list[IndependentEndpointFitInput] = []
    raw_blocks: list[IndependentEndpointRawBlock] = []
    for endpoint_index, (item, endpoint, rng) in enumerate(
        zip(inputs, point_fit.endpoints, endpoint_rngs, strict=True)
    ):
        result = endpoint.fit_result
        if result is None:
            raise RuntimeError("successful point fit lost its numerical result")
        raw_observation = draw_linked_raw_observation(
            item.operator,
            result.prediction,
            rng,
        )
        owner_ids = tuple(
            f"{owner}:conditional_draw:{draw_id}"
            for owner in item.raw_block.role_owner_ids
        )
        raw_block = IndependentEndpointRawBlock(
            endpoint_label=ENDPOINT_LABELS[endpoint_index],
            field_orientation=FIELD_ORIENTATIONS[endpoint_index],
            role_names=RAW_ROLE_NAMES,
            role_owner_ids=owner_ids,
            observed_electrons=raw_observation.observed_electrons,
            unit=item.raw_block.unit,
        )
        raw_blocks.append(raw_block)
        drawn_inputs.append(
            replace(
                item,
                raw_block=raw_block,
                initial_density_coefficients=result.density_coefficients[0],
                initial_nuisance=PCINuisanceValues(*result.nuisance_values),
            )
        )
    fit = fit_independent_pci_endpoints(
        (drawn_inputs[0], drawn_inputs[1]),
        provenance=point_fit.provenance,
    )
    postprocessed = postprocess_independent_endpoint_supports(
        (drawn_inputs[0], drawn_inputs[1]),
        fit,
        supports,
    )
    return IndependentEndpointBootstrapDraw(
        draw_id=draw_id,
        raw_blocks=(raw_blocks[0], raw_blocks[1]),
        fit=fit,
        postprocessed=postprocessed,
        fit_messages=(fit.endpoints[0].message, fit.endpoints[1].message),
    )


def assemble_independent_endpoint_bootstrap(
    point_fit: IndependentEndpointPairFit,
    draws: tuple[IndependentEndpointBootstrapDraw, ...],
    *,
    support_name: str = "primary",
) -> IndependentEndpointBootstrap:
    """Assemble aligned endpoint samples without dropping or replacing failures."""

    if not draws:
        raise ValueError("at least one independent-endpoint bootstrap draw is required")
    draw_ids = tuple(draw.draw_id for draw in draws)
    if draw_ids != tuple(range(len(draws))):
        raise ValueError(
            "bootstrap draws must retain every requested row in zero-based order"
        )
    samples: list[FloatArray] = []
    success: list[BoolArray] = []
    for draw in draws:
        if draw.fit.provenance != point_fit.provenance:
            raise ValueError("bootstrap draw provenance differs from the point fit")
        support_index = draw.postprocessed.support_index(support_name)
        samples.append(draw.postprocessed.values[support_index])
        success.append(draw.postprocessed.endpoint_fit_success_mask)
    sample_array = np.stack(samples)
    return IndependentEndpointBootstrap(
        point_fit=point_fit,
        fit_success_mask=np.stack(success),
        samples=sample_array,
        supported_mask=np.isfinite(sample_array),
    )


def _conditional_estimate(
    *,
    observable_name: str,
    quantity: str,
    unit: str,
    point_value: float,
    point_supported: bool,
    sample_values: FloatArray,
    sample_supported: BoolArray,
    confidence_level: float,
    extra_reason: str | None = None,
) -> ConditionalEndpointEstimate:
    estimate = float(point_value) if point_supported and np.isfinite(point_value) else None
    requested_draws = int(sample_supported.size)
    supported_draws = int(np.count_nonzero(sample_supported))
    reasons: list[str] = []
    if estimate is None:
        reasons.append("point_estimate_not_supported")
    if not np.all(sample_supported):
        reasons.append("incomplete_endpoint_refits_or_observable_support")
    if sample_values.size < 2:
        reasons.append("fewer_than_two_requested_draws")
    if extra_reason is not None:
        reasons.append(extra_reason)
    if estimate is not None and not reasons:
        alpha = 0.5 * (1.0 - confidence_level)
        lower, upper = np.quantile(sample_values, [alpha, 1.0 - alpha])
        return ConditionalEndpointEstimate(
            observable_name=observable_name,
            quantity=quantity,
            unit=unit,
            estimate=estimate,
            status="complete",
            lower=float(lower),
            upper=float(upper),
            requested_draws=requested_draws,
            supported_draws=supported_draws,
            reasons=(),
        )
    return ConditionalEndpointEstimate(
        observable_name=observable_name,
        quantity=quantity,
        unit=unit,
        estimate=estimate,
        status="point_only" if estimate is not None else "unresolved",
        lower=None,
        upper=None,
        requested_draws=requested_draws,
        supported_draws=supported_draws,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _point_matrix(point_fit: IndependentEndpointPairFit) -> tuple[FloatArray, BoolArray]:
    values = np.full((2, len(OBSERVABLE_NAMES)), np.nan, dtype=float)
    supported = np.zeros(values.shape, dtype=bool)
    for index, endpoint in enumerate(point_fit.endpoints):
        if endpoint.observables is not None:
            values[index] = endpoint.observables.values
            supported[index] = endpoint.observables.supported_mask
    return values, supported


def _summarise_observable(
    *,
    name: str,
    unit: str,
    point_values: FloatArray,
    point_supported: BoolArray,
    sample_values: FloatArray,
    sample_supported: BoolArray,
    fit_success: BoolArray,
    confidence_level: float,
) -> IndependentObservablePairSummary:
    endpoint_records = tuple(
        _conditional_estimate(
            observable_name=name,
            quantity=ENDPOINT_LABELS[index],
            unit=unit,
            point_value=float(point_values[index]),
            point_supported=bool(point_supported[index]),
            sample_values=sample_values[:, index],
            sample_supported=sample_supported[:, index] & fit_success[:, index],
            confidence_level=confidence_level,
        )
        for index in range(2)
    )
    paired_supported = (
        np.all(sample_supported, axis=1) & np.all(fit_success, axis=1)
    )
    delta = _conditional_estimate(
        observable_name=name,
        quantity="delta_Bz_minus_By",
        unit=unit,
        point_value=float(point_values[1] - point_values[0]),
        point_supported=bool(np.all(point_supported)),
        sample_values=sample_values[:, 1] - sample_values[:, 0],
        sample_supported=paired_supported,
        confidence_level=confidence_level,
    )
    ratio: ConditionalEndpointEstimate | None = None
    if name in POSITIVE_OBSERVABLE_NAMES:
        positive_point = bool(np.all(point_supported) and np.all(point_values > 0.0))
        positive_samples = np.all(sample_values > 0.0, axis=1)
        ratio_reason = (
            "cross_orientation_amplitude_calibration_not_supplied"
            if name == "A"
            else None
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_samples = sample_values[:, 1] / sample_values[:, 0]
            ratio_point = point_values[1] / point_values[0]
        ratio = _conditional_estimate(
            observable_name=name,
            quantity="ratio_Bz_over_By",
            unit="1",
            point_value=float(ratio_point),
            point_supported=positive_point and ratio_reason is None,
            sample_values=ratio_samples,
            sample_supported=paired_supported & positive_samples,
            confidence_level=confidence_level,
            extra_reason=ratio_reason,
        )
    return IndependentObservablePairSummary(
        observable_name=name,
        by=endpoint_records[0],
        bz=endpoint_records[1],
        delta_b=delta,
        ratio_bz_over_by=ratio,
    )


def summarise_independent_endpoint_information(
    bootstrap: IndependentEndpointBootstrap,
    *,
    confidence_level: float,
) -> IndependentEndpointInformationSummary:
    """Summarise endpoint-local information and paired orientation contrasts."""

    if not isinstance(bootstrap, IndependentEndpointBootstrap):
        raise TypeError("bootstrap has the wrong type")
    level = float(confidence_level)
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    point_values, point_supported = _point_matrix(bootstrap.point_fit)
    summaries: list[IndependentObservablePairSummary] = []
    for observable_index, name in enumerate(OBSERVABLE_NAMES):
        summaries.append(
            _summarise_observable(
                name=name,
                unit=OBSERVABLE_UNITS[name],
                point_values=point_values[:, observable_index],
                point_supported=point_supported[:, observable_index],
                sample_values=bootstrap.samples[:, :, observable_index],
                sample_supported=bootstrap.supported_mask[:, :, observable_index],
                fit_success=bootstrap.fit_success_mask,
                confidence_level=level,
            )
        )

    sigma_y_index = OBSERVABLE_NAMES.index("sigma_y_um")
    sigma_z_index = OBSERVABLE_NAMES.index("sigma_z_um")
    aspect_point_supported = (
        point_supported[:, sigma_y_index]
        & point_supported[:, sigma_z_index]
        & (point_values[:, sigma_y_index] > 0.0)
        & (point_values[:, sigma_z_index] > 0.0)
    )
    aspect_sample_supported = (
        bootstrap.supported_mask[:, :, sigma_y_index]
        & bootstrap.supported_mask[:, :, sigma_z_index]
        & (bootstrap.samples[:, :, sigma_y_index] > 0.0)
        & (bootstrap.samples[:, :, sigma_z_index] > 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        aspect_points = point_values[:, sigma_y_index] / point_values[:, sigma_z_index]
        aspect_samples = (
            bootstrap.samples[:, :, sigma_y_index]
            / bootstrap.samples[:, :, sigma_z_index]
        )
    summaries.append(
        _summarise_observable(
            name=DERIVED_OBSERVABLE_NAME,
            unit="1",
            point_values=aspect_points,
            point_supported=aspect_point_supported,
            sample_values=aspect_samples,
            sample_supported=aspect_sample_supported,
            fit_success=bootstrap.fit_success_mask,
            confidence_level=level,
        )
    )
    return IndependentEndpointInformationSummary(
        observables=tuple(summaries),
        provenance=bootstrap.point_fit.provenance,
    )


__all__ = [
    "ENDPOINT_LABELS",
    "OBSERVABLE_NAMES",
    "RAW_ROLE_NAMES",
    "SUPPORT_NAMES",
    "IndependentEndpointBootstrap",
    "IndependentEndpointBootstrapDraw",
    "IndependentEndpointFitInput",
    "IndependentEndpointInformationSummary",
    "IndependentEndpointPairFit",
    "IndependentEndpointPairProvenance",
    "IndependentEndpointPointFitRecord",
    "IndependentEndpointPointFitSnapshot",
    "IndependentEndpointRawBlock",
    "IndependentEndpointSupportPostprocessing",
    "assemble_independent_endpoint_bootstrap",
    "draw_and_refit_independent_endpoint_bootstrap",
    "fit_independent_pci_endpoints",
    "postprocess_independent_endpoint_supports",
    "restore_successful_independent_endpoint_pair_fit",
    "snapshot_successful_independent_endpoint_pair_fit",
    "summarise_independent_endpoint_information",
]
