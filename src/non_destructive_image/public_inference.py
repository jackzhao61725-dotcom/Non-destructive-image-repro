"""Small inference helpers used by the bounded public example.

This module contains only the two operations demonstrated by
``scripts/run_public_example.py``: a DPFI fit of a smooth Thomas--Fermi BEC
and extraction of visible spacing and valley contrast from a prescribed
three-Gaussian profile.  The protected ensemble runners and their scratch
lifecycle are intentionally not part of this interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq, least_squares

from .four_method_jones_imaging import simulate_matched_four_method_jones_images
from .target_multiframe_noise_acquisition import DPFI_ROLES


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _flatten_roles(roles: Mapping[str, np.ndarray]) -> np.ndarray:
    _require(tuple(roles) == DPFI_ROLES, "DPFI raw-frame order changed")
    return np.concatenate(
        [np.asarray(roles[role], dtype=float).ravel() for role in DPFI_ROLES]
    )


@dataclass
class PublicBECFitContext:
    """Inputs needed to predict and fit one DPFI acquisition of a smooth BEC."""

    physical_model: Mapping[str, Any]
    solver: Mapping[str, Any]
    n0: float
    ry0_m: float
    rz0_m: float
    y_m: np.ndarray
    z_m: np.ndarray
    object_pixel_m: float
    atomic_config: Mapping[str, Any]
    orientation_config: Mapping[str, Any]
    optical_transfer: Any
    camera_y_m: np.ndarray
    camera_z_m: np.ndarray
    camera_pixel_m: float
    detuning_hz: float
    dgi_stop_optical_depth: float
    count_scale: float
    read_noise_electrons_rms: float
    _optical_cache: dict[bytes, tuple[np.ndarray, Any]] = field(default_factory=dict)
    optical_evaluations: int = 0

    @property
    def physical_truth(self) -> np.ndarray:
        """Return ``log(eta), log(rho_y), y0`` for the initial BEC."""

        return np.zeros(3, dtype=float)

    def column_density(self, physical: Sequence[float]) -> np.ndarray:
        """Construct a normalised Thomas--Fermi trial column density."""

        values = np.asarray(physical, dtype=float)
        _require(values.shape == (3,) and np.isfinite(values).all(), "invalid BEC trial")
        eta = math.exp(float(values[0]))
        rho_y = math.exp(float(values[1]))
        y0_m = float(values[2]) * 1e-6
        atom_number = eta * self.n0
        ry_m = rho_y * self.ry0_m
        rz_m = self.rz0_m * eta**0.2
        y, z = np.meshgrid(self.y_m, self.z_m)
        support = np.maximum(
            1.0 - ((y - y0_m) / ry_m) ** 2 - (z / rz_m) ** 2,
            0.0,
        )
        density = (
            5.0
            * atom_number
            / (2.0 * np.pi * ry_m * rz_m)
            * support**1.5
        )
        represented = float(np.sum(density) * self.object_pixel_m**2)
        _require(
            np.isfinite(represented) and represented > 0.0,
            "BEC trial misses the object grid",
        )
        density *= atom_number / represented
        _require(
            np.isfinite(density).all() and np.all(density >= 0.0),
            "invalid BEC trial density",
        )
        return np.asarray(density, dtype=float)

    def optical(self, physical: Sequence[float]) -> tuple[np.ndarray, Any]:
        """Propagate one trial density through the four-readout imaging model."""

        values = np.asarray(physical, dtype=np.float64)
        key = values.tobytes(order="C")
        if key not in self._optical_cache:
            if len(self._optical_cache) >= 64:
                self._optical_cache.clear()
            density = self.column_density(values)
            image = simulate_matched_four_method_jones_images(
                density,
                self.y_m,
                self.z_m,
                model_config=self.atomic_config,
                orientation_config=self.orientation_config,
                optical_transfer=self.optical_transfer,
                detuning_hz=self.detuning_hz,
                camera_pixel_size_m=self.camera_pixel_m,
                phase_plate_transmittance=0.95,
                phase_plate_phase_rad=np.pi / 2.0,
                dgi_stop_optical_depth=self.dgi_stop_optical_depth,
            )
            _require(
                np.array_equal(np.asarray(image.camera_y_m), self.camera_y_m),
                "camera y axis changed",
            )
            _require(
                np.array_equal(np.asarray(image.camera_z_m), self.camera_z_m),
                "camera z axis changed",
            )
            self.optical_evaluations += 1
            self._optical_cache[key] = (density, image)
        return self._optical_cache[key]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return physical and camera-calibration bounds for the DPFI fit."""

        physical = self.physical_model
        lower = np.asarray(
            [
                math.log(float(physical["eta_bounds"][0])),
                math.log(float(physical["rho_y_bounds"][0])),
                float(physical["y0_um_bounds"][0]),
                0.5,
                0.5,
                0.0,
                0.0,
            ],
            dtype=float,
        )
        upper = np.asarray(
            [
                math.log(float(physical["eta_bounds"][1])),
                math.log(float(physical["rho_y_bounds"][1])),
                float(physical["y0_um_bounds"][1]),
                1.5,
                1.5,
                20.0,
                20.0,
            ],
            dtype=float,
        )
        return lower, upper

    def truth_vector(self, physical: Sequence[float] | None = None) -> np.ndarray:
        """Add the ideal H/V gains and dark levels to a physical BEC vector."""

        state = self.physical_truth if physical is None else np.asarray(physical, dtype=float)
        return np.concatenate([state, [1.0, 1.0, 0.0, 0.0]])

    def expected_roles(self, vector: Sequence[float]) -> dict[str, np.ndarray]:
        """Return expected electron counts in the six DPFI raw frames."""

        values = np.asarray(vector, dtype=float)
        _require(values.shape == (7,) and np.isfinite(values).all(), "invalid fit vector")
        _, image = self.optical(values[:3])
        h_gain, v_gain, dark_h, dark_v = (float(value) for value in values[3:])
        shape = (self.camera_z_m.size, self.camera_y_m.size)
        one = np.ones(shape, dtype=float)
        roles = {
            "atom_h": self.count_scale
            * h_gain
            * np.asarray(image.dpfi_h_camera_intensity_over_i0)
            + dark_h,
            "atom_v": self.count_scale
            * v_gain
            * np.asarray(image.dpfi_v_camera_intensity_over_i0)
            + dark_v,
            "blank_h": self.count_scale * h_gain * 0.5 * one + dark_h,
            "blank_v": self.count_scale * v_gain * 0.5 * one + dark_v,
            "dark_h": dark_h * one,
            "dark_v": dark_v * one,
        }
        _require(tuple(roles) == DPFI_ROLES, "DPFI raw-frame order changed")
        return {name: np.asarray(value, dtype=float) for name, value in roles.items()}


def _initial_camera_parameters(
    context: PublicBECFitContext,
    observed: Mapping[str, np.ndarray],
) -> np.ndarray:
    lower, upper = context.bounds()
    dark_h = max(float(np.mean(observed["dark_h"])), 0.0)
    dark_v = max(float(np.mean(observed["dark_v"])), 0.0)
    h_gain = (float(np.mean(observed["blank_h"])) - dark_h) / (
        0.5 * context.count_scale
    )
    v_gain = (float(np.mean(observed["blank_v"])) - dark_v) / (
        0.5 * context.count_scale
    )
    raw = np.nan_to_num(
        np.asarray([h_gain, v_gain, dark_h, dark_v], dtype=float),
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    margin = 1e-10 * (upper[3:] - lower[3:])
    return np.clip(raw, lower[3:] + margin, upper[3:] - margin)


def fit_public_dpfi(
    context: PublicBECFitContext,
    observed: Mapping[str, np.ndarray],
    physical_start: Sequence[float],
) -> dict[str, Any]:
    """Fit one DPFI acquisition with two frozen-variance least-squares passes."""

    observed_vector = _flatten_roles(observed)
    lower, upper = context.bounds()
    current = np.concatenate(
        [
            np.asarray(physical_start, dtype=float),
            _initial_camera_parameters(context, observed),
        ]
    )
    solver = context.solver
    evaluations: list[int] = []
    final = None
    for _ in range(int(solver["irls_frozen_variance_passes"])):
        initial_expected = _flatten_roles(context.expected_roles(current))
        _require(
            np.isfinite(initial_expected).all() and np.all(initial_expected >= 0.0),
            "invalid expected DPFI counts",
        )
        sigma = np.sqrt(initial_expected + context.read_noise_electrons_rms**2)

        def residual(values: np.ndarray) -> np.ndarray:
            prediction = _flatten_roles(context.expected_roles(values))
            return (prediction - observed_vector) / sigma

        final = least_squares(
            residual,
            current,
            bounds=(lower, upper),
            method=str(solver["method"]),
            loss=str(solver["loss"]),
            x_scale=str(solver["x_scale"]),
            max_nfev=int(solver["maximum_function_evaluations_per_pass"]),
            xtol=float(solver["xtol"]),
            ftol=float(solver["ftol"]),
            gtol=float(solver["gtol"]),
            tr_solver=str(solver["trust_region_solver"]),
            jac="2-point",
        )
        evaluations.append(int(final.nfev))
        current = np.asarray(final.x, dtype=float)

    assert final is not None
    expected = _flatten_roles(context.expected_roles(current))
    sigma = np.sqrt(expected + context.read_noise_electrons_rms**2)
    whitened = (expected - observed_vector) / sigma
    finite = bool(
        np.isfinite(current).all()
        and np.isfinite(expected).all()
        and np.all(expected >= 0.0)
        and np.isfinite(whitened).all()
    )
    return {
        "converged": bool(final.success and finite),
        "termination_message": str(final.message).replace("\n", " "),
        "nfev_total": int(sum(evaluations)),
        "reduced_raw_objective": float(np.mean(whitened**2)) if finite else None,
        "eta_hat": math.exp(float(current[0])) if finite else None,
        "rho_y_hat": math.exp(float(current[1])) if finite else None,
        "y0_hat_um": float(current[2]) if finite else None,
    }


def three_peak_observables(
    component_spacing_um: float,
    component_width_um: float,
    *,
    centre_um: float = 0.0,
    weights: Sequence[float] = (0.82, 1.0, 0.82),
) -> dict[str, Any]:
    """Measure visible peak spacing and valley-to-peak ratio of three Gaussians."""

    spacing = float(component_spacing_um)
    width = float(component_width_um)
    centre = float(centre_um)
    weight = np.asarray(weights, dtype=float)
    _require(spacing > 0.0 and width > 0.0, "spacing and width must be positive")
    _require(weight.shape == (3,) and np.all(weight > 0.0), "three positive weights required")
    centres = np.asarray((centre - spacing, centre, centre + spacing), dtype=float)

    def line(position: float | np.ndarray) -> np.ndarray:
        points = np.asarray(position, dtype=float)[..., None]
        return np.sum(weight * np.exp(-0.5 * ((points - centres) / width) ** 2), axis=-1)

    def derivative(position: float | np.ndarray) -> np.ndarray:
        points = np.asarray(position, dtype=float)[..., None]
        offsets = points - centres
        return np.sum(
            -weight * offsets / width**2 * np.exp(-0.5 * (offsets / width) ** 2),
            axis=-1,
        )

    def curvature(position: float) -> float:
        offsets = position - centres
        return float(
            np.sum(
                weight
                * (offsets**2 / width**4 - 1.0 / width**2)
                * np.exp(-0.5 * (offsets / width) ** 2)
            )
        )

    half_range = max(6.0, 2.0 * spacing + 3.0 * width)
    scan = np.linspace(centre - half_range, centre + half_range, 4001)
    slopes = derivative(scan)
    roots: list[float] = []
    for left, right, f_left, f_right in zip(
        scan[:-1], scan[1:], slopes[:-1], slopes[1:], strict=True
    ):
        if f_left == 0.0:
            roots.append(float(left))
        elif f_left * f_right < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda value: float(derivative(value)),
                        float(left),
                        float(right),
                        xtol=1e-13,
                        rtol=1e-13,
                    )
                )
            )
    roots = sorted({round(value, 12) for value in roots})
    peaks = [value for value in roots if curvature(value) < 0.0]
    valleys = [value for value in roots if curvature(value) > 0.0]
    valid = bool(
        len(peaks) == 3
        and len(valleys) == 2
        and all(peaks[index] < valleys[index] < peaks[index + 1] for index in range(2))
    )
    if not valid:
        return {
            "topology_valid": False,
            "peak_count": len(peaks),
            "valley_count": len(valleys),
            "d_peak_um": None,
            "nu_vp": None,
        }
    peak_levels = np.asarray(line(np.asarray(peaks)), dtype=float)
    valley_levels = np.asarray(line(np.asarray(valleys)), dtype=float)
    ratios = [
        valley_levels[index]
        / (0.5 * (peak_levels[index] + peak_levels[index + 1]))
        for index in range(2)
    ]
    return {
        "topology_valid": True,
        "peak_count": 3,
        "valley_count": 2,
        "d_peak_um": float(np.mean(np.diff(peaks))),
        "nu_vp": float(np.mean(ratios)),
    }
