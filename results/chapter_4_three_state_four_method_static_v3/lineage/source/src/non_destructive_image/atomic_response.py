"""Polarisation-resolved scalar and Jones responses for the imaging model.

The maintained scalar helpers in :mod:`non_destructive_image.light_atom` are
the unit-strength two-level baseline.  This module applies Clebsch--Gordan
branch strengths exactly once, while keeping coherent fields and non-negative
spontaneous scattering as separate consumers.
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
class ParallelJonesOpticalResponse:
    """Jones response for a linearly polarised probe propagating along ``B``.

    The incident linear field is an equal-amplitude superposition of the two
    circular eigenmodes.  Their *full* Clebsch--Gordan line strengths belong in
    the circular transmissions.  The equal input amplitudes enter the basis
    transformation and the one-half power fractions enter scattering; neither
    fraction is multiplied into an eigen-transmission exponent.
    """

    species: str
    transition_label: str
    ground_j: float
    ground_m_j: float
    excited_j: float
    probe_axis: tuple[float, float, float]
    quantisation_axis: tuple[float, float, float]
    input_polarisation_axis: tuple[float, float, float]
    faraday_polarisation_axis: tuple[float, float, float]
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

        probe = _unit_vector(self.probe_axis, "probe_axis")
        quantisation = _unit_vector(self.quantisation_axis, "quantisation_axis")
        input_polarisation = _unit_vector(
            self.input_polarisation_axis,
            "input_polarisation_axis",
        )
        faraday_polarisation = _unit_vector(
            self.faraday_polarisation_axis,
            "faraday_polarisation_axis",
        )
        if not np.isclose(np.dot(probe, quantisation), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("the Jones response requires probe_axis parallel to +B")
        if not np.isclose(np.dot(probe, input_polarisation), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("input_polarisation_axis must be transverse to the probe")
        expected_faraday = np.cross(input_polarisation, probe)
        if not np.allclose(
            faraday_polarisation,
            expected_faraday,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "faraday_polarisation_axis must equal input_polarisation_axis cross probe_axis"
            )

        branches = tuple(self.branches)
        if len(branches) != 3 or {branch.q for branch in branches} != {-1, 0, 1}:
            raise ValueError("the isolated J-to-J+1 model requires unique q=-1,0,+1 branches")
        if len({branch.label for branch in branches}) != len(branches):
            raise ValueError("branch labels must be unique")
        by_q = {branch.q: branch for branch in branches}
        expected_line_strengths = {
            -1: 1.0,
            0: 1.0 / (ground_j + 1.0),
            1: 1.0 / ((ground_j + 1.0) * (2.0 * ground_j + 1.0)),
        }
        expected_fractions = {-1: 0.5, 0: 0.0, 1: 0.5}
        if any(
            not np.isclose(
                by_q[q].relative_line_strength,
                expected_line_strengths[q],
                rtol=0.0,
                atol=1e-12,
            )
            for q in (-1, 0, 1)
        ):
            raise ValueError("branch strengths disagree with the stretched-state factors")
        if any(
            not np.isclose(
                by_q[q].intensity_fraction,
                expected_fractions[q],
                rtol=0.0,
                atol=1e-12,
            )
            for q in (-1, 0, 1)
        ):
            raise ValueError("a linear input must place equal power in q=-1 and q=+1")

        object.__setattr__(self, "ground_j", ground_j)
        object.__setattr__(self, "ground_m_j", ground_m_j)
        object.__setattr__(self, "excited_j", excited_j)
        object.__setattr__(self, "probe_axis", probe)
        object.__setattr__(self, "quantisation_axis", quantisation)
        object.__setattr__(self, "input_polarisation_axis", input_polarisation)
        object.__setattr__(self, "faraday_polarisation_axis", faraday_polarisation)
        object.__setattr__(self, "branches", branches)

    @property
    def branch_weights(self) -> tuple[float, ...]:
        """Return non-negative scattering weights in stored branch order."""

        return tuple(branch.weighted_line_strength for branch in self.branches)

    @property
    def effective_line_strength(self) -> float:
        """Return the total common-detuning scattering strength."""

        return float(sum(self.branch_weights))

    @property
    def common_phase_factor(self) -> float:
        """Return half the sum of the two circular line strengths."""

        by_q = {branch.q: branch for branch in self.branches}
        return float(
            (by_q[-1].relative_line_strength + by_q[1].relative_line_strength) / 2.0
        )

    @property
    def faraday_rotation_factor(self) -> float:
        """Return ``(q=+1 minus q=-1)/2`` in the stated basis convention."""

        by_q = {branch.q: branch for branch in self.branches}
        return float(
            (by_q[1].relative_line_strength - by_q[-1].relative_line_strength) / 2.0
        )


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
class JonesColumnOpticalResponse:
    """Circular eigen-transmissions and their two linear output components."""

    branch_labels: tuple[str, str]
    branch_phase_maps_rad: NDArray[np.floating]
    branch_optical_depth_maps: NDArray[np.floating]
    circular_transmission_fields: NDArray[np.complexfloating]
    common_phase_map_rad: NDArray[np.floating]
    faraday_rotation_map_rad: NDArray[np.floating]
    common_optical_depth_map: NDArray[np.floating]
    co_polarised_field: NDArray[np.complexfloating]
    faraday_orthogonal_field: NDArray[np.complexfloating]
    total_intensity_fraction: NDArray[np.floating]

    def __post_init__(self) -> None:
        labels = tuple(self.branch_labels)
        if len(labels) != 2 or len(set(labels)) != 2:
            raise ValueError("branch_labels must contain the two circular eigenmodes")
        phase = np.asarray(self.branch_phase_maps_rad, dtype=float)
        optical_depth = np.asarray(self.branch_optical_depth_maps, dtype=float)
        circular = np.asarray(self.circular_transmission_fields, dtype=complex)
        if phase.ndim != 3 or phase.shape[0] != 2:
            raise ValueError("branch phase maps must have shape (2, *image_shape)")
        if optical_depth.shape != phase.shape or circular.shape != phase.shape:
            raise ValueError("circular branch arrays must have identical shapes")
        if not np.isfinite(phase).all():
            raise ValueError("branch phase maps must be finite")
        if not np.isfinite(optical_depth).all() or np.any(optical_depth < 0.0):
            raise ValueError("branch optical-depth maps must be finite and non-negative")
        if not np.isfinite(circular).all():
            raise ValueError("circular transmission fields must be finite")

        image_shape = phase.shape[1:]
        real_maps = {
            "common_phase_map_rad": np.asarray(self.common_phase_map_rad, dtype=float),
            "faraday_rotation_map_rad": np.asarray(
                self.faraday_rotation_map_rad,
                dtype=float,
            ),
            "common_optical_depth_map": np.asarray(
                self.common_optical_depth_map,
                dtype=float,
            ),
            "total_intensity_fraction": np.asarray(
                self.total_intensity_fraction,
                dtype=float,
            ),
        }
        complex_maps = {
            "co_polarised_field": np.asarray(self.co_polarised_field, dtype=complex),
            "faraday_orthogonal_field": np.asarray(
                self.faraday_orthogonal_field,
                dtype=complex,
            ),
        }
        for name, array in {**real_maps, **complex_maps}.items():
            if array.shape != image_shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite and match the image shape")
        if np.any(real_maps["common_optical_depth_map"] < 0.0):
            raise ValueError("common_optical_depth_map must be non-negative")
        if np.any(real_maps["total_intensity_fraction"] < 0.0):
            raise ValueError("total_intensity_fraction must be non-negative")

        expected_circular = np.exp(-optical_depth / 2.0 + 1j * phase)
        expected_co = (expected_circular[0] + expected_circular[1]) / 2.0
        expected_cross = 0.5j * (expected_circular[0] - expected_circular[1])
        expected_intensity = np.abs(expected_co) ** 2 + np.abs(expected_cross) ** 2
        replay_checks = (
            (circular, expected_circular, "circular transmission fields"),
            (real_maps["common_phase_map_rad"], np.mean(phase, axis=0), "common phase"),
            (
                real_maps["faraday_rotation_map_rad"],
                (phase[1] - phase[0]) / 2.0,
                "Faraday rotation",
            ),
            (
                real_maps["common_optical_depth_map"],
                np.mean(optical_depth, axis=0),
                "common optical depth",
            ),
            (complex_maps["co_polarised_field"], expected_co, "co-polarised field"),
            (
                complex_maps["faraday_orthogonal_field"],
                expected_cross,
                "Faraday-orthogonal field",
            ),
            (
                real_maps["total_intensity_fraction"],
                expected_intensity,
                "total transmitted intensity",
            ),
        )
        for actual, expected, name in replay_checks:
            if not np.allclose(actual, expected, rtol=2e-14, atol=2e-15):
                raise ValueError(f"{name} disagrees with the circular-field reconstruction")

        eigenmode_average = np.mean(np.abs(expected_circular) ** 2, axis=0)
        if not np.allclose(expected_intensity, eigenmode_average, rtol=2e-14, atol=2e-15):
            raise ValueError("linear and circular bases do not conserve transmitted intensity")

        object.__setattr__(self, "branch_labels", labels)
        object.__setattr__(self, "branch_phase_maps_rad", phase)
        object.__setattr__(self, "branch_optical_depth_maps", optical_depth)
        object.__setattr__(self, "circular_transmission_fields", circular)
        for name, array in {**real_maps, **complex_maps}.items():
            object.__setattr__(self, name, array)


@dataclass(frozen=True)
class BranchScatteringResult:
    """Replayable common-detuning scattering result before any consumer."""

    response: PolarisedOpticalResponse | ParallelJonesOpticalResponse
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
        if not isinstance(
            self.response,
            (PolarisedOpticalResponse, ParallelJonesOpticalResponse),
        ):
            raise ValueError("response must be a supported polarised optical response")
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


def parallel_jones_optical_response_from_config(
    config: Mapping[str, Any],
) -> ParallelJonesOpticalResponse:
    """Resolve the three-state parallel-geometry Jones-response contract."""

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    try:
        three_state = config["geometries"]["three_state_equilibrium"]
        response = three_state["jones_atomic_response"]
        transition = response["transition"]
        geometry = response["geometry"]
        branch_records = response["branches"]
    except (KeyError, TypeError) as exc:
        raise ValueError("config is missing the three-state Jones-response contract") from exc
    if not isinstance(response, Mapping) or not isinstance(branch_records, list):
        raise ValueError("three-state Jones-response records have invalid types")

    try:
        branches = tuple(
            OpticalBranch(
                label=record["label"],
                q=record["q"],
                intensity_fraction=record["incident_power_fraction"],
                relative_line_strength=record["relative_line_strength"],
                closure_status=record["closure_status"],
            )
            for record in branch_records
        )
        return ParallelJonesOpticalResponse(
            species=transition["species"],
            transition_label=transition["label"],
            ground_j=transition["ground_J"],
            ground_m_j=transition["ground_m_J"],
            excited_j=transition["excited_J"],
            probe_axis=geometry["probe_wavevector_unit_vector"],
            quantisation_axis=geometry["quantisation_axis_unit_vector"],
            input_polarisation_axis=geometry[
                "input_linear_polarisation_unit_vector"
            ],
            faraday_polarisation_axis=geometry[
                "faraday_orthogonal_axis_unit_vector"
            ],
            branches=branches,
            model_status=response["model_status"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("three-state Jones-response fields are incomplete") from exc


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


def parallel_jones_column_response(
    column_density_m2: ArrayLike,
    detuning_hz: float,
    resonant_cross_section_m2: float,
    natural_linewidth_rad_s: float,
    response: ParallelJonesOpticalResponse,
    *,
    detuning_by_branch_hz: Mapping[str, float] | None = None,
) -> JonesColumnOpticalResponse:
    """Return the exact two-component field for ``k`` parallel to ``B``.

    The stored circular phase and optical-depth maps use the full q=-1 and
    q=+1 line strengths.  The equal incident amplitudes are combined only when
    the circular eigenfields are projected back onto the input-linear and
    Faraday-orthogonal axes.
    """

    if not isinstance(response, ParallelJonesOpticalResponse):
        raise ValueError("response must be a ParallelJonesOpticalResponse")
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

    active_branches = tuple(branch for branch in response.branches if branch.q in (-1, 1))
    if tuple(branch.q for branch in active_branches) != (-1, 1):
        raise ValueError("circular branches must be stored in q=-1,q=+1 order")
    nominal_detuning = _finite_scalar(detuning_hz, "detuning_hz")
    if detuning_by_branch_hz is None:
        branch_detunings = (nominal_detuning, nominal_detuning)
    else:
        if not isinstance(detuning_by_branch_hz, Mapping):
            raise ValueError("detuning_by_branch_hz must be a mapping")
        active_labels = {branch.label for branch in active_branches}
        if set(detuning_by_branch_hz) != active_labels:
            raise ValueError(
                "detuning_by_branch_hz keys must match the two circular branch labels"
            )
        branch_detunings = tuple(
            _finite_scalar(
                detuning_by_branch_hz[branch.label],
                f"{branch.label} detuning",
            )
            for branch in active_branches
        )

    phase_maps = []
    optical_depth_maps = []
    for branch, branch_detuning in zip(
        active_branches,
        branch_detunings,
        strict=True,
    ):
        phase_maps.append(
            branch.relative_line_strength
            * scalar_phase_shift(
                branch_detuning,
                density,
                cross_section,
                linewidth,
            )
        )
        optical_depth_maps.append(
            branch.relative_line_strength
            * residual_optical_depth(
                branch_detuning,
                density,
                cross_section,
                linewidth,
            )
        )

    phase_by_branch = np.stack(phase_maps, axis=0)
    optical_depth_by_branch = np.stack(optical_depth_maps, axis=0)
    circular_fields = np.exp(
        -optical_depth_by_branch / 2.0 + 1j * phase_by_branch
    )
    co_polarised = (circular_fields[0] + circular_fields[1]) / 2.0
    faraday_orthogonal = 0.5j * (circular_fields[0] - circular_fields[1])
    total_intensity = np.abs(co_polarised) ** 2 + np.abs(faraday_orthogonal) ** 2
    return JonesColumnOpticalResponse(
        branch_labels=tuple(branch.label for branch in active_branches),
        branch_phase_maps_rad=phase_by_branch,
        branch_optical_depth_maps=optical_depth_by_branch,
        circular_transmission_fields=circular_fields,
        common_phase_map_rad=np.mean(phase_by_branch, axis=0),
        faraday_rotation_map_rad=(phase_by_branch[1] - phase_by_branch[0]) / 2.0,
        common_optical_depth_map=np.mean(optical_depth_by_branch, axis=0),
        co_polarised_field=co_polarised,
        faraday_orthogonal_field=faraday_orthogonal,
        total_intensity_fraction=total_intensity,
    )


def branch_summed_scattered_photons_per_atom(
    detuning_hz: float,
    probe_power_mw: float,
    pulse_duration_s: float,
    saturation_intensity_w_m2: float,
    natural_linewidth_rad_s: float,
    probe_diameter_m: float,
    response: PolarisedOpticalResponse | ParallelJonesOpticalResponse,
    *,
    use_peak_intensity: bool = True,
) -> BranchScatteringResult:
    """Return common-detuning scattering with one shared saturation denominator.

    Branch numerators are constructed from the incident saturation, selected
    polarisation fractions and relative line strengths.  Their sum enters the
    denominator once; this is deliberately not ``effective factor *`` the old
    saturated two-level total.
    """

    if not isinstance(
        response,
        (PolarisedOpticalResponse, ParallelJonesOpticalResponse),
    ):
        raise ValueError("response must be a supported polarised optical response")
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
    "JonesColumnOpticalResponse",
    "OpticalBranch",
    "ParallelJonesOpticalResponse",
    "PolarisedOpticalResponse",
    "branch_summed_scattered_photons_per_atom",
    "complex_column_response",
    "parallel_jones_column_response",
    "parallel_jones_optical_response_from_config",
    "polarised_optical_response_from_config",
]
