"""Raw camera-acquisition diagnostics for RAI, PCI and DGI screening."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating]


def _immutable_array(values: np.ndarray) -> NDArray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _real_array(values: ArrayLike, label: str) -> NDArray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{label} must be real numeric data, not boolean")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{label} must be real numeric data")
    return np.asarray(array, dtype=float)


def _finite_array(values: ArrayLike, label: str) -> FloatArray:
    array = _real_array(values, label)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{label} must be a non-empty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return _immutable_array(np.asarray(array, dtype=float))


def _nonnegative_expected(values: ArrayLike, label: str) -> FloatArray:
    array = _finite_array(values, label)
    if np.any(array < 0.0):
        raise ValueError(f"{label} cannot contain negative expected electron counts")
    return array


def _finite_nonnegative(value: float, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a real scalar, not boolean")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _finite_positive(value: float, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be a real scalar, not boolean")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _positive_integer(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _nonempty_text(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be text")
    result = value.strip()
    if not result:
        raise ValueError(f"{label} cannot be empty")
    return result


def _seed_tuple(seed_components: tuple[int, ...]) -> tuple[int, ...]:
    components: list[int] = []
    for component in seed_components:
        if not isinstance(component, int) or isinstance(component, bool):
            raise TypeError("seed components must be integers")
        if component < 0:
            raise ValueError("seed components cannot be negative")
        components.append(component)
    if not components:
        raise ValueError("seed_components cannot be empty")
    return tuple(components)


def _matching_frames(*frames: RawElectronFrame) -> tuple[int, int]:
    if not frames:
        raise ValueError("at least one raw electron frame is required")
    if not all(isinstance(frame, RawElectronFrame) for frame in frames):
        raise TypeError("diagnostic inputs must be RawElectronFrame instances")
    shapes = {frame.expected_electrons.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError("raw electron frames must share one registered shape")
    cameras = {frame.camera_contract_id for frame in frames}
    samplings = {frame.sampling_contract_id for frame in frames}
    if len(cameras) != 1:
        raise ValueError("raw electron frames must share one camera_contract_id")
    if len(samplings) != 1:
        raise ValueError("raw electron frames must share one sampling_contract_id")
    return frames[0].expected_electrons.shape


@dataclass(frozen=True, eq=False)
class RawElectronFrame:
    """One raw or independently averaged camera channel in electron units.

    ``expected_electrons`` and ``observed_electrons`` are per-exposure means.
    If ``independent_exposures`` is greater than one, the observed array is the
    arithmetic mean of that many identically distributed camera readouts and
    ``variance_electrons2`` includes the corresponding ``1/n`` reduction.
    Records use identity equality so incomplete metadata can never masquerade
    as equality of numerical camera payloads.
    """

    role: str
    expected_electrons: FloatArray = field(repr=False, compare=False)
    observed_electrons: FloatArray = field(repr=False, compare=False)
    read_noise_electrons_rms: float
    camera_contract_id: str
    sampling_contract_id: str
    bit_generator: str
    rng_provenance: Literal[
        "seed_components_replayable",
        "caller_owned_rng_not_self_contained",
        "deterministic_fixture",
    ]
    independent_exposures: int = 1
    seed_components: tuple[int, ...] = ()
    unit: Literal["electron"] = "electron"

    def __post_init__(self) -> None:
        role = _nonempty_text(self.role, "raw electron frame role")
        expected = _nonnegative_expected(self.expected_electrons, "expected_electrons")
        observed = _finite_array(self.observed_electrons, "observed_electrons")
        if observed.shape != expected.shape:
            raise ValueError("expected and observed electron arrays must share a shape")
        read_noise = _finite_nonnegative(
            self.read_noise_electrons_rms,
            "read_noise_electrons_rms",
        )
        exposure_count = _positive_integer(
            self.independent_exposures,
            "independent_exposures",
        )
        if self.unit != "electron":
            raise ValueError("RawElectronFrame unit must be 'electron'")
        allowed_rng_provenance = {
            "seed_components_replayable",
            "caller_owned_rng_not_self_contained",
            "deterministic_fixture",
        }
        if self.rng_provenance not in allowed_rng_provenance:
            raise ValueError("unsupported RawElectronFrame rng_provenance")
        components = () if not self.seed_components else _seed_tuple(self.seed_components)
        if self.rng_provenance == "seed_components_replayable" and not components:
            raise ValueError("replayable raw frames require seed_components")
        if self.rng_provenance != "seed_components_replayable" and components:
            raise ValueError("seed_components require replayable rng_provenance")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "expected_electrons", expected)
        object.__setattr__(self, "observed_electrons", observed)
        object.__setattr__(self, "read_noise_electrons_rms", read_noise)
        object.__setattr__(
            self,
            "camera_contract_id",
            _nonempty_text(self.camera_contract_id, "camera_contract_id"),
        )
        object.__setattr__(
            self,
            "sampling_contract_id",
            _nonempty_text(self.sampling_contract_id, "sampling_contract_id"),
        )
        object.__setattr__(
            self,
            "bit_generator",
            _nonempty_text(self.bit_generator, "bit_generator"),
        )
        object.__setattr__(self, "independent_exposures", exposure_count)
        object.__setattr__(self, "seed_components", components)

    @property
    def variance_electrons2(self) -> FloatArray:
        """Return conditional per-pixel variance of the stored mean frame."""

        variance = (
            self.expected_electrons + self.read_noise_electrons_rms**2
        ) / self.independent_exposures
        return _immutable_array(np.asarray(variance, dtype=float))


@dataclass(frozen=True, eq=False)
class ProcessedDiagnostic:
    """A processed diagnostic with delta-method variance and validity mask.

    This object is for display, screening and initialisation. It does not replace
    the linked raw camera frames in a quantitative likelihood.
    """

    quantity_name: str
    unit: Literal["electron", "1"]
    expected_value: FloatArray = field(repr=False, compare=False)
    observed_value: FloatArray = field(repr=False, compare=False)
    variance: FloatArray = field(repr=False, compare=False)
    expected_valid_mask: NDArray[np.bool_] = field(repr=False, compare=False)
    observed_valid_mask: NDArray[np.bool_] = field(repr=False, compare=False)
    jacobian: Mapping[str, FloatArray] = field(repr=False, compare=False)
    shared_nuisance_jacobian: Mapping[str, FloatArray] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    shared_nuisance_variance: Mapping[str, float] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        quantity = _nonempty_text(self.quantity_name, "diagnostic quantity_name")
        if self.unit not in {"electron", "1"}:
            raise ValueError("diagnostic unit must be 'electron' or '1'")
        expected = _real_array(self.expected_value, "diagnostic expected_value")
        observed = _real_array(self.observed_value, "diagnostic observed_value")
        variance = _real_array(self.variance, "diagnostic variance")
        expected_valid_input = np.asarray(self.expected_valid_mask)
        observed_valid_input = np.asarray(self.observed_valid_mask)
        if not np.issubdtype(expected_valid_input.dtype, np.bool_) or not np.issubdtype(
            observed_valid_input.dtype, np.bool_
        ):
            raise TypeError("diagnostic validity masks must contain boolean values")
        expected_valid = np.asarray(expected_valid_input, dtype=bool)
        observed_valid = np.asarray(observed_valid_input, dtype=bool)
        if expected.ndim != 2 or expected.size == 0:
            raise ValueError("diagnostic values must be non-empty two-dimensional arrays")
        if observed.shape != expected.shape or variance.shape != expected.shape:
            raise ValueError("diagnostic value and variance arrays must share a shape")
        if expected_valid.shape != expected.shape or observed_valid.shape != expected.shape:
            raise ValueError("diagnostic validity masks must share the value shape")
        if not np.isfinite(expected[expected_valid]).all():
            raise ValueError("diagnostic expected values must be finite where valid")
        if not np.isnan(expected[~expected_valid]).all():
            raise ValueError("diagnostic expected values must be NaN where unsupported")
        if np.any(variance[expected_valid] < 0.0) or not np.isfinite(
            variance[expected_valid]
        ).all():
            raise ValueError("diagnostic variance must be finite and non-negative where valid")
        if not np.isnan(variance[~expected_valid]).all():
            raise ValueError("diagnostic variance must be NaN where unsupported")
        if not np.isfinite(observed[observed_valid]).all():
            raise ValueError("diagnostic observed values must be finite where valid")
        if not np.isnan(observed[~observed_valid]).all():
            raise ValueError("diagnostic observed values must be NaN where unsupported")

        expected_frozen = _immutable_array(np.asarray(expected, dtype=float))
        observed_frozen = _immutable_array(np.asarray(observed, dtype=float))
        variance_frozen = _immutable_array(np.asarray(variance, dtype=float))
        expected_valid_frozen = _immutable_array(np.asarray(expected_valid, dtype=bool))
        observed_valid_frozen = _immutable_array(np.asarray(observed_valid, dtype=bool))
        jacobian: dict[str, FloatArray] = {}
        for name, values in self.jacobian.items():
            label = str(name).strip()
            if not label or label in jacobian:
                raise ValueError("diagnostic Jacobian channel names must be unique and non-empty")
            array = _real_array(values, f"diagnostic Jacobian {label}")
            if array.shape != expected.shape:
                raise ValueError("every diagnostic Jacobian array must share the value shape")
            if not np.isfinite(array[expected_valid]).all():
                raise ValueError("diagnostic Jacobian must be finite where valid")
            if not np.isnan(array[~expected_valid]).all():
                raise ValueError("diagnostic Jacobian must be NaN where unsupported")
            jacobian[label] = _immutable_array(np.asarray(array, dtype=float))
        if not jacobian:
            raise ValueError("diagnostic Jacobian cannot be empty")
        raw_shared_jacobian: dict[str, ArrayLike] = {}
        for name, values in self.shared_nuisance_jacobian.items():
            label = _nonempty_text(name, "shared nuisance name")
            if label in raw_shared_jacobian:
                raise ValueError("shared nuisance names must be unique after normalisation")
            raw_shared_jacobian[label] = values
        raw_shared_variance: dict[str, float] = {}
        for name, value in self.shared_nuisance_variance.items():
            label = _nonempty_text(name, "shared nuisance name")
            if label in raw_shared_variance:
                raise ValueError("shared nuisance names must be unique after normalisation")
            raw_shared_variance[label] = value
        if set(raw_shared_jacobian) != set(raw_shared_variance):
            raise ValueError("shared nuisance Jacobian and variance names must match")
        shared_jacobian: dict[str, FloatArray] = {}
        shared_variance: dict[str, float] = {}
        for label, values in raw_shared_jacobian.items():
            array = _real_array(values, f"shared nuisance Jacobian {label}")
            if array.shape != expected.shape:
                raise ValueError("shared nuisance Jacobians must share the value shape")
            if not np.isfinite(array[expected_valid]).all() or not np.isnan(
                array[~expected_valid]
            ).all():
                raise ValueError("shared nuisance Jacobian validity does not match diagnostic")
            shared_jacobian[label] = _immutable_array(np.asarray(array, dtype=float))
            shared_variance[label] = _finite_nonnegative(
                raw_shared_variance[label],
                f"shared nuisance variance {label}",
            )
        object.__setattr__(self, "quantity_name", quantity)
        object.__setattr__(self, "expected_value", expected_frozen)
        object.__setattr__(self, "observed_value", observed_frozen)
        object.__setattr__(self, "variance", variance_frozen)
        object.__setattr__(self, "expected_valid_mask", expected_valid_frozen)
        object.__setattr__(self, "observed_valid_mask", observed_valid_frozen)
        object.__setattr__(self, "jacobian", MappingProxyType(jacobian))
        object.__setattr__(
            self,
            "shared_nuisance_jacobian",
            MappingProxyType(shared_jacobian),
        )
        object.__setattr__(
            self,
            "shared_nuisance_variance",
            MappingProxyType(shared_variance),
        )

    @property
    def valid_mask(self) -> NDArray[np.bool_]:
        """Return pixels valid in both the expected and observed diagnostics."""

        return _immutable_array(self.expected_valid_mask & self.observed_valid_mask)


def simulate_raw_electron_frame(
    expected_electrons: ArrayLike,
    *,
    read_noise_electrons_rms: float,
    rng: np.random.Generator | None = None,
    role: str,
    camera_contract_id: str,
    sampling_contract_id: str,
    independent_exposures: int = 1,
    seed_components: tuple[int, ...] | None = None,
) -> RawElectronFrame:
    """Draw a raw Poisson--Gaussian frame or iid frame mean.

    Expected means are validated rather than clipped. Negative observed values
    remain valid because Gaussian read noise is applied after photoelectron
    generation. ``seed_components`` creates a self-contained replay record;
    caller-owned ``rng`` is composable but explicitly not self-contained and
    must not be used alone as complete stochastic provenance.
    """

    expected = _nonnegative_expected(expected_electrons, "expected_electrons")
    read_noise = _finite_nonnegative(
        read_noise_electrons_rms,
        "read_noise_electrons_rms",
    )
    exposure_count = _positive_integer(independent_exposures, "independent_exposures")
    if (rng is None) == (seed_components is None):
        raise ValueError("provide exactly one of rng or seed_components")
    stored_seed: tuple[int, ...] = ()
    rng_provenance: Literal[
        "seed_components_replayable",
        "caller_owned_rng_not_self_contained",
    ]
    if seed_components is not None:
        stored_seed = _seed_tuple(seed_components)
        rng = np.random.default_rng(np.random.SeedSequence(stored_seed))
        rng_provenance = "seed_components_replayable"
    elif not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    else:
        rng_provenance = "caller_owned_rng_not_self_contained"
    assert rng is not None
    observed = (
        rng.poisson(expected * exposure_count) / exposure_count
        + rng.normal(0.0, read_noise / np.sqrt(exposure_count), expected.shape)
    )
    return RawElectronFrame(
        role=role,
        expected_electrons=expected,
        observed_electrons=observed,
        read_noise_electrons_rms=read_noise,
        camera_contract_id=camera_contract_id,
        sampling_contract_id=sampling_contract_id,
        bit_generator=type(rng.bit_generator).__name__,
        rng_provenance=rng_provenance,
        independent_exposures=exposure_count,
        seed_components=stored_seed,
    )


def deterministic_raw_electron_frame(
    expected_electrons: ArrayLike,
    *,
    read_noise_electrons_rms: float,
    role: str,
    camera_contract_id: str,
    sampling_contract_id: str,
    independent_exposures: int = 1,
) -> RawElectronFrame:
    """Return an expectation-only raw frame with the physical variance intact.

    The observed payload is set equal to the expected electron count.  This is
    useful for deterministic design scans that require the same Poisson--read
    variance and downstream acquisition algebra as a noisy draw, without
    pretending that a stochastic camera realisation was generated.
    """

    expected = _nonnegative_expected(expected_electrons, "expected_electrons")
    return RawElectronFrame(
        role=role,
        expected_electrons=expected,
        observed_electrons=expected,
        read_noise_electrons_rms=read_noise_electrons_rms,
        camera_contract_id=camera_contract_id,
        sampling_contract_id=sampling_contract_id,
        bit_generator="not_applicable",
        rng_provenance="deterministic_fixture",
        independent_exposures=independent_exposures,
    )


def simulate_intensity_frame(
    intensity_over_i0: ArrayLike,
    *,
    photoelectrons_per_i0_pixel: float,
    read_noise_electrons_rms: float,
    rng: np.random.Generator | None = None,
    role: str,
    camera_contract_id: str,
    sampling_contract_id: str,
    independent_exposures: int = 1,
    seed_components: tuple[int, ...] | None = None,
) -> RawElectronFrame:
    """Convert a non-negative ``I/I0`` image to one noisy electron frame."""

    intensity = _nonnegative_expected(intensity_over_i0, "intensity_over_i0")
    scale = _finite_positive(
        photoelectrons_per_i0_pixel,
        "photoelectrons_per_i0_pixel",
    )
    return simulate_raw_electron_frame(
        intensity * scale,
        read_noise_electrons_rms=read_noise_electrons_rms,
        rng=rng,
        role=role,
        camera_contract_id=camera_contract_id,
        sampling_contract_id=sampling_contract_id,
        independent_exposures=independent_exposures,
        seed_components=seed_components,
    )


def paired_difference_diagnostic(
    atom_frame: RawElectronFrame,
    reference_frame: RawElectronFrame,
    *,
    shared_dark_frame: RawElectronFrame | None = None,
    quantity_name: str = "atom_minus_reference_electrons",
) -> ProcessedDiagnostic:
    """Return an acquisition-aware atom-minus-reference diagnostic.

    A single shared dark estimate cancels exactly from
    ``(atom-dark)-(reference-dark)``. It is still validated when supplied so the
    raw acquisition record cannot silently mix shapes or camera units.
    """

    frames = (atom_frame, reference_frame)
    if shared_dark_frame is not None:
        frames += (shared_dark_frame,)
    shape = _matching_frames(*frames)
    expected = atom_frame.expected_electrons - reference_frame.expected_electrons
    observed = atom_frame.observed_electrons - reference_frame.observed_electrons
    variance = atom_frame.variance_electrons2 + reference_frame.variance_electrons2
    valid = np.ones(shape, dtype=bool)
    ones = np.ones(shape, dtype=float)
    return ProcessedDiagnostic(
        quantity_name=quantity_name,
        unit="electron",
        expected_value=expected,
        observed_value=observed,
        variance=variance,
        expected_valid_mask=valid,
        observed_valid_mask=valid,
        jacobian={"atom": ones, "reference": -ones},
    )


def rai_transmission_diagnostic(
    atom_frame: RawElectronFrame,
    reference_frame: RawElectronFrame,
    dark_frame: RawElectronFrame,
) -> ProcessedDiagnostic:
    """Return RAI transmission ``(A-D)/(R-D)`` and its delta variance."""

    shape = _matching_frames(atom_frame, reference_frame, dark_frame)
    expected_a = atom_frame.expected_electrons - dark_frame.expected_electrons
    expected_r = reference_frame.expected_electrons - dark_frame.expected_electrons
    observed_a = atom_frame.observed_electrons - dark_frame.observed_electrons
    observed_r = reference_frame.observed_electrons - dark_frame.observed_electrons
    expected_valid = (expected_a >= 0.0) & (expected_r > 0.0)
    observed_valid = observed_r > 0.0

    expected_value = np.full(shape, np.nan, dtype=float)
    observed_value = np.full(shape, np.nan, dtype=float)
    variance = np.full(shape, np.nan, dtype=float)
    j_atom = np.full(shape, np.nan, dtype=float)
    j_reference = np.full(shape, np.nan, dtype=float)
    j_dark = np.full(shape, np.nan, dtype=float)
    expected_value[expected_valid] = (
        expected_a[expected_valid] / expected_r[expected_valid]
    )
    observed_value[observed_valid] = (
        observed_a[observed_valid] / observed_r[observed_valid]
    )
    j_atom[expected_valid] = 1.0 / expected_r[expected_valid]
    j_reference[expected_valid] = (
        -expected_value[expected_valid] / expected_r[expected_valid]
    )
    j_dark[expected_valid] = (
        (expected_value[expected_valid] - 1.0) / expected_r[expected_valid]
    )
    variance[expected_valid] = (
        j_atom[expected_valid] ** 2 * atom_frame.variance_electrons2[expected_valid]
        + j_reference[expected_valid] ** 2
        * reference_frame.variance_electrons2[expected_valid]
        + j_dark[expected_valid] ** 2 * dark_frame.variance_electrons2[expected_valid]
    )
    return ProcessedDiagnostic(
        quantity_name="rai_transmission",
        unit="1",
        expected_value=expected_value,
        observed_value=observed_value,
        variance=variance,
        expected_valid_mask=expected_valid,
        observed_valid_mask=observed_valid,
        jacobian={"atom": j_atom, "reference": j_reference, "dark": j_dark},
    )


def rai_optical_density_diagnostic(
    atom_frame: RawElectronFrame,
    reference_frame: RawElectronFrame,
    dark_frame: RawElectronFrame,
) -> ProcessedDiagnostic:
    """Return ``-log((A-D)/(R-D))`` without repairing invalid ratios."""

    shape = _matching_frames(atom_frame, reference_frame, dark_frame)
    expected_a = atom_frame.expected_electrons - dark_frame.expected_electrons
    expected_r = reference_frame.expected_electrons - dark_frame.expected_electrons
    observed_a = atom_frame.observed_electrons - dark_frame.observed_electrons
    observed_r = reference_frame.observed_electrons - dark_frame.observed_electrons
    expected_valid = (expected_a > 0.0) & (expected_r > 0.0)
    observed_valid = (observed_a > 0.0) & (observed_r > 0.0)
    expected_value = np.full(shape, np.nan, dtype=float)
    observed_value = np.full(shape, np.nan, dtype=float)
    variance = np.full(shape, np.nan, dtype=float)
    j_atom = np.full(shape, np.nan, dtype=float)
    j_reference = np.full(shape, np.nan, dtype=float)
    j_dark = np.full(shape, np.nan, dtype=float)
    expected_value[expected_valid] = -np.log(
        expected_a[expected_valid] / expected_r[expected_valid]
    )
    observed_value[observed_valid] = -np.log(
        observed_a[observed_valid] / observed_r[observed_valid]
    )
    j_atom[expected_valid] = -1.0 / expected_a[expected_valid]
    j_reference[expected_valid] = 1.0 / expected_r[expected_valid]
    j_dark[expected_valid] = (
        1.0 / expected_a[expected_valid] - 1.0 / expected_r[expected_valid]
    )
    variance[expected_valid] = (
        j_atom[expected_valid] ** 2 * atom_frame.variance_electrons2[expected_valid]
        + j_reference[expected_valid] ** 2
        * reference_frame.variance_electrons2[expected_valid]
        + j_dark[expected_valid] ** 2 * dark_frame.variance_electrons2[expected_valid]
    )
    return ProcessedDiagnostic(
        quantity_name="rai_optical_density",
        unit="1",
        expected_value=expected_value,
        observed_value=observed_value,
        variance=variance,
        expected_valid_mask=expected_valid,
        observed_valid_mask=observed_valid,
        jacobian={"atom": j_atom, "reference": j_reference, "dark": j_dark},
    )


def pci_contrast_diagnostic(
    atom_frame: RawElectronFrame,
    bright_reference_frame: RawElectronFrame,
    dark_frame: RawElectronFrame,
) -> ProcessedDiagnostic:
    """Return signed PCI contrast ``(P-D)/(B-D)-1`` and its variance."""

    shape = _matching_frames(atom_frame, bright_reference_frame, dark_frame)
    expected_p = atom_frame.expected_electrons - dark_frame.expected_electrons
    expected_b = (
        bright_reference_frame.expected_electrons - dark_frame.expected_electrons
    )
    observed_p = atom_frame.observed_electrons - dark_frame.observed_electrons
    observed_b = (
        bright_reference_frame.observed_electrons - dark_frame.observed_electrons
    )
    expected_valid = (expected_p >= 0.0) & (expected_b > 0.0)
    observed_valid = observed_b > 0.0
    expected_value = np.full(shape, np.nan, dtype=float)
    observed_value = np.full(shape, np.nan, dtype=float)
    variance = np.full(shape, np.nan, dtype=float)
    j_atom = np.full(shape, np.nan, dtype=float)
    j_reference = np.full(shape, np.nan, dtype=float)
    j_dark = np.full(shape, np.nan, dtype=float)
    q_expected = np.full(shape, np.nan, dtype=float)
    q_expected[expected_valid] = expected_p[expected_valid] / expected_b[expected_valid]
    expected_value[expected_valid] = q_expected[expected_valid] - 1.0
    observed_value[observed_valid] = (
        observed_p[observed_valid] / observed_b[observed_valid] - 1.0
    )
    j_atom[expected_valid] = 1.0 / expected_b[expected_valid]
    j_reference[expected_valid] = -q_expected[expected_valid] / expected_b[expected_valid]
    j_dark[expected_valid] = expected_value[expected_valid] / expected_b[expected_valid]
    variance[expected_valid] = (
        j_atom[expected_valid] ** 2 * atom_frame.variance_electrons2[expected_valid]
        + j_reference[expected_valid] ** 2
        * bright_reference_frame.variance_electrons2[expected_valid]
        + j_dark[expected_valid] ** 2 * dark_frame.variance_electrons2[expected_valid]
    )
    return ProcessedDiagnostic(
        quantity_name="pci_normalised_contrast",
        unit="1",
        expected_value=expected_value,
        observed_value=observed_value,
        variance=variance,
        expected_valid_mask=expected_valid,
        observed_valid_mask=observed_valid,
        jacobian={"atom": j_atom, "bright_reference": j_reference, "dark": j_dark},
    )


def dgi_signal_diagnostic(
    atom_stop_frame: RawElectronFrame,
    leakage_stop_frame: RawElectronFrame,
    stop_dark_frame: RawElectronFrame,
    open_reference_frame: RawElectronFrame,
    open_dark_frame: RawElectronFrame,
    *,
    open_to_stop_scale: float = 1.0,
    open_to_stop_scale_variance: float = 0.0,
) -> ProcessedDiagnostic:
    """Return DGI ``(A-L)/(kappa*(B-D_B))`` and its delta variance."""

    shape = _matching_frames(
        atom_stop_frame,
        leakage_stop_frame,
        stop_dark_frame,
        open_reference_frame,
        open_dark_frame,
    )
    scale = _finite_positive(open_to_stop_scale, "open_to_stop_scale")
    scale_variance = _finite_nonnegative(
        open_to_stop_scale_variance,
        "open_to_stop_scale_variance",
    )
    expected_open = (
        open_reference_frame.expected_electrons - open_dark_frame.expected_electrons
    )
    observed_open = (
        open_reference_frame.observed_electrons - open_dark_frame.observed_electrons
    )
    # The shared stop-dark exposure cancels algebraically.  Subtract the two
    # stop-port frames directly to avoid catastrophic cancellation when their
    # common dark offset is large.
    expected_numerator = (
        atom_stop_frame.expected_electrons
        - leakage_stop_frame.expected_electrons
    )
    observed_numerator = (
        atom_stop_frame.observed_electrons
        - leakage_stop_frame.observed_electrons
    )
    expected_denominator = scale * expected_open
    observed_denominator = scale * observed_open
    expected_valid = expected_denominator > 0.0
    observed_valid = observed_denominator > 0.0
    expected_value = np.full(shape, np.nan, dtype=float)
    observed_value = np.full(shape, np.nan, dtype=float)
    variance = np.full(shape, np.nan, dtype=float)
    jacobians = {
        name: np.full(shape, np.nan, dtype=float)
        for name in (
            "atom_stop",
            "leakage_stop",
            "stop_dark",
            "open_reference",
            "open_dark",
            "open_to_stop_scale",
        )
    }
    expected_value[expected_valid] = (
        expected_numerator[expected_valid] / expected_denominator[expected_valid]
    )
    observed_value[observed_valid] = (
        observed_numerator[observed_valid] / observed_denominator[observed_valid]
    )
    jacobians["atom_stop"][expected_valid] = 1.0 / expected_denominator[expected_valid]
    jacobians["leakage_stop"][expected_valid] = -1.0 / expected_denominator[expected_valid]
    jacobians["stop_dark"][expected_valid] = 0.0
    jacobians["open_reference"][expected_valid] = (
        -expected_value[expected_valid] / expected_open[expected_valid]
    )
    jacobians["open_dark"][expected_valid] = (
        expected_value[expected_valid] / expected_open[expected_valid]
    )
    jacobians["open_to_stop_scale"][expected_valid] = (
        -expected_value[expected_valid] / scale
    )
    variance[expected_valid] = (
        jacobians["atom_stop"][expected_valid] ** 2
        * atom_stop_frame.variance_electrons2[expected_valid]
        + jacobians["leakage_stop"][expected_valid] ** 2
        * leakage_stop_frame.variance_electrons2[expected_valid]
        + jacobians["open_reference"][expected_valid] ** 2
        * open_reference_frame.variance_electrons2[expected_valid]
        + jacobians["open_dark"][expected_valid] ** 2
        * open_dark_frame.variance_electrons2[expected_valid]
    )
    return ProcessedDiagnostic(
        quantity_name="dgi_open_normalised_atom_minus_leakage",
        unit="1",
        expected_value=expected_value,
        observed_value=observed_value,
        variance=variance,
        expected_valid_mask=expected_valid,
        observed_valid_mask=observed_valid,
        jacobian=jacobians,
        shared_nuisance_jacobian={
            "open_to_stop_scale": jacobians["open_to_stop_scale"]
        },
        shared_nuisance_variance={"open_to_stop_scale": scale_variance},
    )


def conditional_block_snr(
    diagnostic: ProcessedDiagnostic,
    support_mask: ArrayLike,
) -> float:
    """Return block SNR using conditional independent-pixel variance.

    Shared calibration terms must be supplied through a full covariance at the
    sequence-analysis stage; this helper deliberately does not hide them.
    """

    if not isinstance(diagnostic, ProcessedDiagnostic):
        raise TypeError("diagnostic must be a ProcessedDiagnostic")
    support_array = np.asarray(support_mask)
    if not np.issubdtype(support_array.dtype, np.bool_):
        raise TypeError("support_mask must contain boolean values")
    support = np.asarray(support_array, dtype=bool)
    if support.shape != diagnostic.expected_value.shape:
        raise ValueError("support_mask must share the diagnostic image shape")
    if np.any(support & ~diagnostic.expected_valid_mask):
        raise ValueError("support_mask contains unsupported expected pixels")
    active = support
    if not np.any(active):
        raise ValueError("support_mask must select at least one pixel")
    denominator2 = float(np.sum(diagnostic.variance[active]))
    if not np.isfinite(denominator2) or denominator2 <= 0.0:
        raise ValueError("selected diagnostic variance must be finite and positive")
    signal = float(np.sum(diagnostic.expected_value[active]))
    return abs(signal) / np.sqrt(denominator2)


def marginal_block_snr(
    diagnostic: ProcessedDiagnostic,
    support_mask: ArrayLike,
) -> float:
    """Return block SNR including declared shared scalar nuisances.

    Each shared nuisance contributes the low-rank block variance
    ``sigma_q^2 * (sum_i d s_i / d q)^2``.  Raw-frame pixel noise remains in
    ``diagnostic.variance``.
    """

    conditional = conditional_block_snr(diagnostic, support_mask)
    if not diagnostic.shared_nuisance_variance:
        return conditional
    support = np.asarray(support_mask, dtype=bool)
    signal = float(np.sum(diagnostic.expected_value[support]))
    denominator2 = float(np.sum(diagnostic.variance[support]))
    for name, nuisance_variance in diagnostic.shared_nuisance_variance.items():
        derivative_sum = float(
            np.sum(diagnostic.shared_nuisance_jacobian[name][support])
        )
        denominator2 += nuisance_variance * derivative_sum**2
    if not np.isfinite(denominator2) or denominator2 <= 0.0:
        raise ValueError("selected marginal diagnostic variance must be finite and positive")
    return abs(signal) / np.sqrt(denominator2)


def delta_method_covariance(
    jacobian: ArrayLike,
    raw_covariance: ArrayLike,
    *,
    symmetry_tolerance: float = 1e-12,
) -> FloatArray:
    """Propagate a finite covariance matrix as ``J Sigma J.T``."""

    derivative = _real_array(jacobian, "jacobian")
    covariance = _real_array(raw_covariance, "raw_covariance")
    if derivative.ndim != 2 or derivative.size == 0:
        raise ValueError("jacobian must be a non-empty two-dimensional matrix")
    if covariance.shape != (derivative.shape[1], derivative.shape[1]):
        raise ValueError("raw_covariance shape must match the Jacobian input dimension")
    if not np.isfinite(derivative).all() or not np.isfinite(covariance).all():
        raise ValueError("jacobian and raw_covariance must be finite")
    tolerance = _finite_nonnegative(symmetry_tolerance, "symmetry_tolerance")
    entry_scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=0.0,
        atol=tolerance * entry_scale,
    ):
        raise ValueError("raw_covariance must be symmetric")
    symmetric_covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -tolerance * scale:
        raise ValueError("raw_covariance must be positive semidefinite")
    # A tiny negative eigenvalue inside the declared tolerance is numerical
    # round-off, not physical negative variance.  Project that mode to zero so
    # downstream diagonal variances cannot become negative.
    covariance_psd = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    propagated = derivative @ covariance_psd @ derivative.T
    propagated = 0.5 * (propagated + propagated.T)
    propagated_eigenvalues, propagated_eigenvectors = np.linalg.eigh(propagated)
    propagated = (
        propagated_eigenvectors * np.maximum(propagated_eigenvalues, 0.0)
    ) @ propagated_eigenvectors.T
    propagated = 0.5 * (propagated + propagated.T)
    return _immutable_array(np.asarray(propagated, dtype=float))
