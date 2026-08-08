"""Polarisation-resolved optical response for the active PCI/DGI model.

The maintained scalar helpers in :mod:`non_destructive_image.light_atom` are
the unit-strength two-level baseline.  This module applies the selected
polarisation and Clebsch--Gordan branch weights exactly once, while keeping
coherent phase and non-negative spontaneous scattering as separate consumers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .light_atom import (
    dimensionless_detuning,
    intensity_at_atoms,
    residual_optical_depth,
    scalar_phase_shift,
)


LinearEigenmode = Literal["parallel", "perpendicular"]


def _finite_scalar(value: object, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be a finite scalar")
    return scalar


def _unit_vector(value: object, name: str) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite three-vector")
    if not np.isclose(np.linalg.norm(vector), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be a unit vector")
    return tuple(float(component) for component in vector)


@dataclass(frozen=True)
class OpticalBranch:
    """One spherical optical branch relative to the quantisation axis."""

    label: str
    q: int
    intensity_fraction: float
    relative_line_strength: float
    closure_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("branch label must be non-empty")
        if type(self.q) is not int or self.q not in (-1, 0, 1):
            raise ValueError("branch q must be one of -1, 0 or +1")
        intensity_fraction = _finite_scalar(
            self.intensity_fraction,
            f"{self.label} intensity_fraction",
        )
        relative_line_strength = _finite_scalar(
            self.relative_line_strength,
            f"{self.label} relative_line_strength",
        )
        if intensity_fraction < 0.0:
            raise ValueError("branch intensity fractions must be non-negative")
        if relative_line_strength <= 0.0:
            raise ValueError("branch relative line strengths must be positive")
        if not isinstance(self.closure_status, str) or not self.closure_status.strip():
            raise ValueError("branch closure_status must be non-empty")
        object.__setattr__(self, "intensity_fraction", intensity_fraction)
        object.__setattr__(self, "relative_line_strength", relative_line_strength)

    @property
    def weighted_line_strength(self) -> float:
        """Return ``polarisation fraction * relative line strength``."""

        return self.intensity_fraction * self.relative_line_strength


@dataclass(frozen=True)
class PolarisedOpticalResponse:
    """Selected isolated-line response in the transverse Jones subspace."""

    species: str
    transition_label: str
    ground_j: float
    ground_m_j: float
    excited_j: float
    rank0_factor: float
    rank1_helicity_factor: float
    rank2_factor: float
    probe_axis: tuple[float, float, float]
    quantisation_axis: tuple[float, float, float]
    polarisation_axis: tuple[float, float, float]
    selected_eigenmode: LinearEigenmode
    branches: tuple[OpticalBranch, ...]
    model_status: str

    def __post_init__(self) -> None:
        if not isinstance(self.species, str) or not self.species.strip():
            raise ValueError("species must be non-empty")
        if not isinstance(self.transition_label, str) or not self.transition_label.strip():
            raise ValueError("transition_label must be non-empty")
        if not isinstance(self.model_status, str) or not self.model_status.strip():
            raise ValueError("model_status must be non-empty")

        ground_j = _finite_scalar(self.ground_j, "ground_j")
        ground_m_j = _finite_scalar(self.ground_m_j, "ground_m_j")
        excited_j = _finite_scalar(self.excited_j, "excited_j")
        if ground_j < 0.0 or excited_j < 0.0 or abs(ground_m_j) > ground_j:
            raise ValueError("J and m_J do not define a physical angular-momentum state")
        if any(
            not np.isclose(2.0 * value, round(2.0 * value), rtol=0.0, atol=1e-12)
            for value in (ground_j, ground_m_j, excited_j)
        ):
            raise ValueError("J and m_J must be integer or half-integer quantum numbers")
        if not np.isclose(ground_m_j, -ground_j, rtol=0.0, atol=1e-12):
            raise ValueError("the selected response requires the lowest stretched m_J=-J state")
        if not np.isclose(excited_j, ground_j + 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("the isolated response requires a J-to-J+1 transition")

        rank0 = _finite_scalar(self.rank0_factor, "rank0_factor")
        rank1 = _finite_scalar(self.rank1_helicity_factor, "rank1_helicity_factor")
        rank2 = _finite_scalar(self.rank2_factor, "rank2_factor")
        probe = _unit_vector(self.probe_axis, "probe_axis")
        quantisation = _unit_vector(self.quantisation_axis, "quantisation_axis")
        polarisation = _unit_vector(self.polarisation_axis, "polarisation_axis")

        if self.selected_eigenmode not in ("parallel", "perpendicular"):
            raise ValueError("selected_eigenmode must be 'parallel' or 'perpendicular'")
        if not np.isclose(np.dot(probe, quantisation), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("the selected model requires probe_axis perpendicular to B")
        if not np.isclose(np.dot(probe, polarisation), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("polarisation_axis must be transverse to the probe")
        polarisation_dot_b = abs(float(np.dot(polarisation, quantisation)))
        expected_dot = 1.0 if self.selected_eigenmode == "parallel" else 0.0
        if not np.isclose(polarisation_dot_b, expected_dot, rtol=0.0, atol=1e-12):
            raise ValueError("polarisation_axis does not match the selected eigenmode")

        branches = tuple(self.branches)
        if len(branches) != 3 or {branch.q for branch in branches} != {-1, 0, 1}:
            raise ValueError("the isolated J-to-J+1 model requires unique q=-1,0,+1 branches")
        if len({branch.label for branch in branches}) != len(branches):
            raise ValueError("branch labels must be unique")
        if not np.isclose(
            sum(branch.intensity_fraction for branch in branches),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("branch intensity fractions must sum to one")

        by_q = {branch.q: branch for branch in branches}
        expected_line_strengths = {
            -1: 1.0,
            0: 1.0 / (ground_j + 1.0),
            1: 1.0 / ((ground_j + 1.0) * (2.0 * ground_j + 1.0)),
        }
        if any(
            not np.isclose(
                by_q[q].relative_line_strength,
                expected_line_strengths[q],
                rtol=0.0,
                atol=1e-12,
            )
            for q in (-1, 0, 1)
        ):
            raise ValueError("branch strengths disagree with the stretched-state Clebsch--Gordan factors")
        expected_fractions = (
            {-1: 0.0, 0: 1.0, 1: 0.0}
            if self.selected_eigenmode == "parallel"
            else {-1: 0.5, 0: 0.0, 1: 0.5}
        )
        if any(
            not np.isclose(
                by_q[q].intensity_fraction,
                expected_fractions[q],
                rtol=0.0,
                atol=1e-12,
            )
            for q in (-1, 0, 1)
        ):
            raise ValueError("branch fractions do not represent the selected linear eigenmode")

        parallel_factor = rank0 + rank2
        perpendicular_factor = rank0 - rank2 / 2.0
        branch_parallel = by_q[0].relative_line_strength
        branch_perpendicular = (
            by_q[-1].relative_line_strength + by_q[1].relative_line_strength
        ) / 2.0
        branch_helicity = (
            by_q[1].relative_line_strength - by_q[-1].relative_line_strength
        ) / 2.0
        for derived, branch_value, name in (
            (parallel_factor, branch_parallel, "parallel rank/branch response"),
            (perpendicular_factor, branch_perpendicular, "perpendicular rank/branch response"),
            (rank1, branch_helicity, "rank-1 helicity/branch response"),
        ):
            if not np.isclose(derived, branch_value, rtol=0.0, atol=1e-12):
                raise ValueError(f"inconsistent {name}")

        selected_factor = (
            parallel_factor
            if self.selected_eigenmode == "parallel"
            else perpendicular_factor
        )
        if selected_factor <= 0.0 or not np.isclose(
            selected_factor,
            sum(branch.weighted_line_strength for branch in branches),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("selected rank and branch response factors disagree")

        object.__setattr__(self, "ground_j", ground_j)
        object.__setattr__(self, "ground_m_j", ground_m_j)
        object.__setattr__(self, "excited_j", excited_j)
        object.__setattr__(self, "rank0_factor", rank0)
        object.__setattr__(self, "rank1_helicity_factor", rank1)
        object.__setattr__(self, "rank2_factor", rank2)
        object.__setattr__(self, "probe_axis", probe)
        object.__setattr__(self, "quantisation_axis", quantisation)
        object.__setattr__(self, "polarisation_axis", polarisation)
        object.__setattr__(self, "branches", branches)

    @property
    def branch_weights(self) -> tuple[float, ...]:
        """Return non-negative branch weights in stored branch order."""

        return tuple(branch.weighted_line_strength for branch in self.branches)

    @property
    def effective_line_strength(self) -> float:
        """Return the selected common-detuning response factor."""

        return float(sum(self.branch_weights))

    @property
    def parallel_line_strength(self) -> float:
        """Return the projected rank-0 plus rank-2 parallel eigenvalue."""

        return self.rank0_factor + self.rank2_factor

    @property
    def perpendicular_line_strength(self) -> float:
        """Return the projected rank-0 minus half-rank-2 eigenvalue."""

        return self.rank0_factor - self.rank2_factor / 2.0

    @property
    def circular_response_difference(self) -> float:
        """Return ``q=+1 minus q=-1`` in the stated helicity convention."""

        return 2.0 * self.rank1_helicity_factor


@dataclass(frozen=True)
class ColumnOpticalResponse:
    """Branch-resolved phase, extinction and complex object transmission."""

    branch_labels: tuple[str, ...]
    branch_phase_maps_rad: NDArray[np.floating]
    branch_optical_depth_maps: NDArray[np.floating]
    phase_map_rad: NDArray[np.floating]
    optical_depth_map: NDArray[np.floating]
    object_field: NDArray[np.complexfloating]


@dataclass(frozen=True)
class BranchScatteringResult:
    """Replayable common-detuning scattering result before any consumer."""

    response: PolarisedOpticalResponse
    detuning_hz: float
    probe_power_mw: float
    pulse_duration_s: float
    saturation_intensity_w_m2: float
    natural_linewidth_rad_s: float
    probe_diameter_m: float
    use_peak_intensity: bool
    branch_labels: tuple[str, ...]
    dimensionless_detuning: float
    incident_saturation_parameter: float
    branch_saturation_parameters: tuple[float, ...]
    total_saturation_parameter: float
    photons_per_atom_by_branch: tuple[float, ...]
    total_photons_per_atom: float

    def __post_init__(self) -> None:
        if not isinstance(self.response, PolarisedOpticalResponse):
            raise ValueError("response must be a PolarisedOpticalResponse")
        if type(self.use_peak_intensity) is not bool:
            raise ValueError("use_peak_intensity must be boolean")
        detuning = _finite_scalar(self.detuning_hz, "detuning_hz")
        power = _finite_scalar(self.probe_power_mw, "probe_power_mw")
        duration = _finite_scalar(self.pulse_duration_s, "pulse_duration_s")
        saturation_intensity = _finite_scalar(
            self.saturation_intensity_w_m2,
            "saturation_intensity_w_m2",
        )
        linewidth = _finite_scalar(
            self.natural_linewidth_rad_s,
            "natural_linewidth_rad_s",
        )
        diameter = _finite_scalar(self.probe_diameter_m, "probe_diameter_m")
        if power < 0.0 or duration < 0.0:
            raise ValueError("probe power and pulse duration must be non-negative")
        if saturation_intensity <= 0.0 or linewidth <= 0.0 or diameter <= 0.0:
            raise ValueError(
                "saturation intensity, linewidth and probe diameter must be positive"
            )

        labels = tuple(self.branch_labels)
        expected_labels = tuple(branch.label for branch in self.response.branches)
        if labels != expected_labels:
            raise ValueError("branch labels disagree with the bound optical response")
        branch_saturation = tuple(float(value) for value in self.branch_saturation_parameters)
        branch_photons = tuple(float(value) for value in self.photons_per_atom_by_branch)
        if len(branch_saturation) != len(labels) or len(branch_photons) != len(labels):
            raise ValueError("branch arrays must match the bound optical response")
        if not np.all(np.isfinite(branch_saturation)) or any(
            value < 0.0 for value in branch_saturation
        ):
            raise ValueError("branch saturation parameters must be finite and non-negative")
        if not np.all(np.isfinite(branch_photons)) or any(
            value < 0.0 for value in branch_photons
        ):
            raise ValueError("branch scattered photons must be finite and non-negative")

        delta = _finite_scalar(self.dimensionless_detuning, "dimensionless_detuning")
        incident = _finite_scalar(
            self.incident_saturation_parameter,
            "incident_saturation_parameter",
        )
        total_saturation = _finite_scalar(
            self.total_saturation_parameter,
            "total_saturation_parameter",
        )
        total_photons = _finite_scalar(
            self.total_photons_per_atom,
            "total_photons_per_atom",
        )
        if incident < 0.0 or total_saturation < 0.0 or total_photons < 0.0:
            raise ValueError("scattering and saturation values must be non-negative")

        expected_delta = dimensionless_detuning(detuning, linewidth)
        expected_incident = intensity_at_atoms(
            power,
            diameter,
            use_peak_intensity=self.use_peak_intensity,
        ) / saturation_intensity
        expected_branch_saturation = tuple(
            expected_incident * weight for weight in self.response.branch_weights
        )
        expected_total_saturation = float(sum(expected_branch_saturation))
        common_rate_factor = (linewidth / 2.0) * duration / (
            1.0 + expected_total_saturation + expected_delta**2
        )
        expected_branch_photons = tuple(
            float(common_rate_factor * value)
            for value in expected_branch_saturation
        )
        expected_total_photons = float(sum(expected_branch_photons))
        scalar_pairs = (
            (delta, expected_delta, "dimensionless detuning"),
            (incident, expected_incident, "incident saturation"),
            (total_saturation, expected_total_saturation, "total saturation"),
            (total_photons, expected_total_photons, "total scattered photons"),
        )
        for actual, expected, name in scalar_pairs:
            if not np.isclose(actual, expected, rtol=1e-14, atol=0.0):
                raise ValueError(f"{name} disagrees with the replayed branch formula")
        if not np.allclose(
            branch_saturation,
            expected_branch_saturation,
            rtol=1e-14,
            atol=0.0,
        ):
            raise ValueError("branch saturation disagrees with the replayed response")
        if not np.allclose(
            branch_photons,
            expected_branch_photons,
            rtol=1e-14,
            atol=0.0,
        ):
            raise ValueError("branch photons disagree with the shared-denominator formula")

        object.__setattr__(self, "detuning_hz", detuning)
        object.__setattr__(self, "probe_power_mw", power)
        object.__setattr__(self, "pulse_duration_s", duration)
        object.__setattr__(self, "saturation_intensity_w_m2", saturation_intensity)
        object.__setattr__(self, "natural_linewidth_rad_s", linewidth)
        object.__setattr__(self, "probe_diameter_m", diameter)
        object.__setattr__(self, "branch_labels", labels)
        object.__setattr__(self, "dimensionless_detuning", delta)
        object.__setattr__(self, "incident_saturation_parameter", incident)
        object.__setattr__(self, "branch_saturation_parameters", branch_saturation)
        object.__setattr__(self, "total_saturation_parameter", total_saturation)
        object.__setattr__(self, "photons_per_atom_by_branch", branch_photons)
        object.__setattr__(self, "total_photons_per_atom", total_photons)


def polarised_optical_response_from_config(
    config: Mapping[str, Any],
) -> PolarisedOpticalResponse:
    """Resolve and validate the active ``polarised_atomic_response`` block."""

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    try:
        response = config["polarised_atomic_response"]
        transition = response["transition"]
        geometry = response["geometry"]
        rank = response["rank_factors"]
        branch_records = response["selected_branches"]
    except (KeyError, TypeError) as exc:
        raise ValueError("config is missing the polarised atomic-response contract") from exc
    if not isinstance(response, Mapping) or not isinstance(branch_records, list):
        raise ValueError("polarised atomic-response records have invalid types")

    try:
        branches = tuple(
            OpticalBranch(
                label=record["label"],
                q=record["q"],
                intensity_fraction=record["intensity_fraction"],
                relative_line_strength=record["relative_line_strength"],
                closure_status=record["closure_status"],
            )
            for record in branch_records
        )
        return PolarisedOpticalResponse(
            species=transition["species"],
            transition_label=transition["label"],
            ground_j=transition["ground_J"],
            ground_m_j=transition["ground_m_J"],
            excited_j=transition["excited_J"],
            rank0_factor=rank["rank0_factor"],
            rank1_helicity_factor=rank["rank1_helicity_factor"],
            rank2_factor=rank["rank2_factor"],
            probe_axis=geometry["probe_wavevector_unit_vector"],
            quantisation_axis=geometry["quantisation_axis_unit_vector"],
            polarisation_axis=geometry["selected_polarisation_unit_vector"],
            selected_eigenmode=geometry["selected_linear_eigenmode"],
            branches=branches,
            model_status=response["model_status"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("polarised atomic-response fields are incomplete") from exc


def _column_density_array(column_density_m2: ArrayLike) -> NDArray[np.floating]:
    raw = np.asarray(column_density_m2)
    if np.iscomplexobj(raw):
        raise ValueError("column_density_m2 must be real")
    density = np.asarray(raw, dtype=float)
    if density.ndim != 2 or density.size == 0:
        raise ValueError("column_density_m2 must be a non-empty two-dimensional array")
    if not np.isfinite(density).all():
        raise ValueError("column_density_m2 must be finite")
    if np.any(density < 0.0):
        raise ValueError("column_density_m2 must be non-negative")
    return density


def _branch_detunings(
    response: PolarisedOpticalResponse,
    detuning_hz: float,
    detuning_by_branch_hz: Mapping[str, float] | None,
) -> tuple[float, ...]:
    nominal = _finite_scalar(detuning_hz, "detuning_hz")
    if detuning_by_branch_hz is None:
        return (nominal,) * len(response.branches)
    if not isinstance(detuning_by_branch_hz, Mapping):
        raise ValueError("detuning_by_branch_hz must be a mapping")
    labels = {branch.label for branch in response.branches}
    if set(detuning_by_branch_hz) != labels:
        raise ValueError("detuning_by_branch_hz keys must match the branch labels")
    return tuple(
        _finite_scalar(detuning_by_branch_hz[branch.label], f"{branch.label} detuning")
        for branch in response.branches
    )


def complex_column_response(
    column_density_m2: ArrayLike,
    detuning_hz: float,
    resonant_cross_section_m2: float,
    natural_linewidth_rad_s: float,
    response: PolarisedOpticalResponse,
    *,
    detuning_by_branch_hz: Mapping[str, float] | None = None,
) -> ColumnOpticalResponse:
    """Return the selected branch-resolved complex column transmission.

    ``detuning_hz`` is the probe-minus-transition frequency in cycles per
    second.  ``natural_linewidth_rad_s`` is the angular spontaneous-decay rate.
    Supplying absolute probe-minus-branch-transition values through
    ``detuning_by_branch_hz`` activates the exact coherent branch sum; otherwise
    the approved common-detuning approximation is used.
    """

    if not isinstance(response, PolarisedOpticalResponse):
        raise ValueError("response must be a PolarisedOpticalResponse")
    density = _column_density_array(column_density_m2)
    cross_section = _finite_scalar(
        resonant_cross_section_m2,
        "resonant_cross_section_m2",
    )
    linewidth = _finite_scalar(
        natural_linewidth_rad_s,
        "natural_linewidth_rad_s",
    )
    if cross_section <= 0.0 or linewidth <= 0.0:
        raise ValueError("cross section and natural linewidth must be positive")
    branch_detunings = _branch_detunings(
        response,
        detuning_hz,
        detuning_by_branch_hz,
    )

    branch_phase_maps = []
    branch_optical_depth_maps = []
    for branch, branch_detuning in zip(response.branches, branch_detunings, strict=True):
        weight = branch.weighted_line_strength
        branch_phase_maps.append(
            weight
            * scalar_phase_shift(
                branch_detuning,
                density,
                cross_section,
                linewidth,
            )
        )
        branch_optical_depth_maps.append(
            weight
            * residual_optical_depth(
                branch_detuning,
                density,
                cross_section,
                linewidth,
            )
        )

    phase_by_branch = np.stack(branch_phase_maps, axis=0)
    optical_depth_by_branch = np.stack(branch_optical_depth_maps, axis=0)
    phase = np.sum(phase_by_branch, axis=0)
    optical_depth = np.sum(optical_depth_by_branch, axis=0)
    object_field = np.exp(-optical_depth / 2.0 + 1j * phase)
    return ColumnOpticalResponse(
        branch_labels=tuple(branch.label for branch in response.branches),
        branch_phase_maps_rad=phase_by_branch,
        branch_optical_depth_maps=optical_depth_by_branch,
        phase_map_rad=phase,
        optical_depth_map=optical_depth,
        object_field=object_field,
    )


def branch_summed_scattered_photons_per_atom(
    detuning_hz: float,
    probe_power_mw: float,
    pulse_duration_s: float,
    saturation_intensity_w_m2: float,
    natural_linewidth_rad_s: float,
    probe_diameter_m: float,
    response: PolarisedOpticalResponse,
    *,
    use_peak_intensity: bool = True,
) -> BranchScatteringResult:
    """Return common-detuning scattering with one shared saturation denominator.

    Branch numerators are constructed from the incident saturation, selected
    polarisation fractions and relative line strengths.  Their sum enters the
    denominator once; this is deliberately not ``effective factor *`` the old
    saturated two-level total.
    """

    if not isinstance(response, PolarisedOpticalResponse):
        raise ValueError("response must be a PolarisedOpticalResponse")
    detuning = _finite_scalar(detuning_hz, "detuning_hz")
    power = _finite_scalar(probe_power_mw, "probe_power_mw")
    duration = _finite_scalar(pulse_duration_s, "pulse_duration_s")
    saturation_intensity = _finite_scalar(
        saturation_intensity_w_m2,
        "saturation_intensity_w_m2",
    )
    linewidth = _finite_scalar(
        natural_linewidth_rad_s,
        "natural_linewidth_rad_s",
    )
    diameter = _finite_scalar(probe_diameter_m, "probe_diameter_m")
    if power < 0.0 or duration < 0.0:
        raise ValueError("probe power and pulse duration must be non-negative")
    if saturation_intensity <= 0.0 or linewidth <= 0.0 or diameter <= 0.0:
        raise ValueError("saturation intensity, linewidth and probe diameter must be positive")

    incident_saturation = intensity_at_atoms(
        power,
        diameter,
        use_peak_intensity=use_peak_intensity,
    ) / saturation_intensity
    branch_saturation = tuple(
        incident_saturation * weight for weight in response.branch_weights
    )
    total_saturation = float(sum(branch_saturation))
    delta = dimensionless_detuning(detuning, linewidth)
    common_rate_factor = (linewidth / 2.0) * duration / (
        1.0 + total_saturation + delta**2
    )
    branch_photons = tuple(
        float(common_rate_factor * saturation) for saturation in branch_saturation
    )
    return BranchScatteringResult(
        response=response,
        detuning_hz=detuning,
        probe_power_mw=power,
        pulse_duration_s=duration,
        saturation_intensity_w_m2=saturation_intensity,
        natural_linewidth_rad_s=linewidth,
        probe_diameter_m=diameter,
        use_peak_intensity=use_peak_intensity,
        branch_labels=tuple(branch.label for branch in response.branches),
        dimensionless_detuning=float(delta),
        incident_saturation_parameter=float(incident_saturation),
        branch_saturation_parameters=branch_saturation,
        total_saturation_parameter=total_saturation,
        photons_per_atom_by_branch=branch_photons,
        total_photons_per_atom=float(sum(branch_photons)),
    )


__all__ = [
    "BranchScatteringResult",
    "ColumnOpticalResponse",
    "OpticalBranch",
    "PolarisedOpticalResponse",
    "branch_summed_scattered_photons_per_atom",
    "complex_column_response",
    "polarised_optical_response_from_config",
]
