"""Camera-noise propagation for the two ideal Faraday readouts.

The optical fields and camera-pixel integration are handled upstream.  This
module begins with expected electron counts in the already integrated camera
ports.  DFFI uses an ideal crossed port and an open-path reference.  DPFI uses
two matched bright ports; the two ports are kept separate through camera
sampling and are combined only after their independent Poisson and read-noise
variances have been assigned.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating]


def _array(values: ArrayLike, label: str, *, nonnegative: bool = False) -> FloatArray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.size == 0:
        raise ValueError(f"{label} must be a non-empty two-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must contain only finite values")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{label} cannot contain negative expected counts")
    return result


def _read_noise(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("read noise must be finite and non-negative")
    return result


def _freeze(values: ArrayLike) -> FloatArray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FaradayFrameNoiseModel:
    """Expected processed signal and its pixelwise delta-method variance."""

    method: str
    expected_signal: FloatArray
    blank_signal: FloatArray
    mu: FloatArray
    variance: FloatArray
    jacobians: Mapping[str, FloatArray]
    raw_means: Mapping[str, FloatArray]
    raw_variances: Mapping[str, FloatArray]

    @property
    def image_snr(self) -> float:
        valid = self.variance > 0.0
        if not np.any(valid):
            return 0.0
        return float(np.sqrt(np.sum(self.mu[valid] ** 2 / self.variance[valid])))


def _model(
    method: str,
    expected: FloatArray,
    blank: FloatArray,
    jacobians: Mapping[str, FloatArray],
    raw_means: Mapping[str, FloatArray],
    read_noise_electrons_rms: float,
) -> FaradayFrameNoiseModel:
    read_noise = _read_noise(read_noise_electrons_rms)
    raw_variances = {
        name: np.asarray(values + read_noise**2, dtype=float)
        for name, values in raw_means.items()
    }
    variance = sum(
        np.asarray(jacobians[name], dtype=float) ** 2 * raw_variances[name]
        for name in raw_means
    )
    frozen_jacobians = MappingProxyType(
        {name: _freeze(values) for name, values in jacobians.items()}
    )
    frozen_means = MappingProxyType(
        {name: _freeze(values) for name, values in raw_means.items()}
    )
    frozen_variances = MappingProxyType(
        {name: _freeze(values) for name, values in raw_variances.items()}
    )
    return FaradayFrameNoiseModel(
        method=method,
        expected_signal=_freeze(expected),
        blank_signal=_freeze(blank),
        mu=_freeze(expected - blank),
        variance=_freeze(variance),
        jacobians=frozen_jacobians,
        raw_means=frozen_means,
        raw_variances=frozen_variances,
    )


def process_dffi_counts(
    crossed_atom: ArrayLike,
    crossed_blank: ArrayLike,
    open_reference: ArrayLike,
    open_dark: ArrayLike,
) -> FloatArray:
    """Form the common-I0 DFFI signal from camera-count arrays."""

    atom = _array(crossed_atom, "crossed_atom")
    blank = _array(crossed_blank, "crossed_blank")
    bright = _array(open_reference, "open_reference")
    dark = _array(open_dark, "open_dark")
    if any(value.shape != atom.shape for value in (blank, bright, dark)):
        raise ValueError("DFFI raw count arrays must share one shape")
    denominator = bright - dark
    if np.any(denominator <= 0.0):
        raise ValueError("DFFI open-reference denominator must be positive")
    return np.asarray((atom - blank) / denominator, dtype=float)


def dffi_frame_noise_model(
    crossed_atom: ArrayLike,
    crossed_blank: ArrayLike,
    crossed_dark: ArrayLike,
    open_reference: ArrayLike,
    open_dark: ArrayLike,
    *,
    read_noise_electrons_rms: float,
) -> FaradayFrameNoiseModel:
    """Propagate noise for an ideal crossed-analyser DFFI acquisition.

    The crossed-path dark frame is retained in the raw-role inventory, as in
    the DGI stopped path, but cancels when the same frame is subtracted from
    the atom and blank crossed-port frames.
    """

    atom, blank, crossed_off, bright, open_off = (
        _array(values, label, nonnegative=True)
        for values, label in (
            (crossed_atom, "crossed_atom"),
            (crossed_blank, "crossed_blank"),
            (crossed_dark, "crossed_dark"),
            (open_reference, "open_reference"),
            (open_dark, "open_dark"),
        )
    )
    if any(value.shape != atom.shape for value in (blank, crossed_off, bright, open_off)):
        raise ValueError("DFFI raw means must share one shape")
    open_level = bright - open_off
    if np.any(open_level <= 0.0):
        raise ValueError("DFFI open-reference denominator must be positive")
    expected = (atom - blank) / open_level
    jacobians = {
        "crossed_atom": 1.0 / open_level,
        "crossed_blank": -1.0 / open_level,
        "crossed_dark": np.zeros(atom.shape, dtype=float),
        "open_reference": -expected / open_level,
        "open_dark": expected / open_level,
    }
    raw_means = {
        "crossed_atom": atom,
        "crossed_blank": blank,
        "crossed_dark": crossed_off,
        "open_reference": bright,
        "open_dark": open_off,
    }
    return _model(
        "dffi",
        expected,
        np.zeros(atom.shape, dtype=float),
        jacobians,
        raw_means,
        read_noise_electrons_rms,
    )


def process_dpfi_counts(
    atom_h: ArrayLike,
    atom_v: ArrayLike,
    blank_h: ArrayLike,
    blank_v: ArrayLike,
    dark_h: ArrayLike,
    dark_v: ArrayLike,
) -> FloatArray:
    """Form the common-I0 DPFI port difference after separate port sampling."""

    h, v, h0, v0, dh, dv = (
        _array(values, label)
        for values, label in (
            (atom_h, "atom_h"),
            (atom_v, "atom_v"),
            (blank_h, "blank_h"),
            (blank_v, "blank_v"),
            (dark_h, "dark_h"),
            (dark_v, "dark_v"),
        )
    )
    if any(value.shape != h.shape for value in (v, h0, v0, dh, dv)):
        raise ValueError("DPFI port arrays must share one registered shape")
    numerator = (h - dh) - (v - dv)
    denominator = (h0 - dh) + (v0 - dv)
    if np.any(denominator <= 0.0):
        raise ValueError("DPFI blank-port denominator must be positive")
    return np.asarray(numerator / denominator, dtype=float)


def dpfi_frame_noise_model(
    atom_h: ArrayLike,
    atom_v: ArrayLike,
    blank_h: ArrayLike,
    blank_v: ArrayLike,
    dark_h: ArrayLike,
    dark_v: ArrayLike,
    *,
    read_noise_electrons_rms: float,
) -> FaradayFrameNoiseModel:
    """Propagate independent port shot noise and read noise through DPFI."""

    h, v, h0, v0, dh, dv = (
        _array(values, label, nonnegative=True)
        for values, label in (
            (atom_h, "atom_h"),
            (atom_v, "atom_v"),
            (blank_h, "blank_h"),
            (blank_v, "blank_v"),
            (dark_h, "dark_h"),
            (dark_v, "dark_v"),
        )
    )
    if any(value.shape != h.shape for value in (v, h0, v0, dh, dv)):
        raise ValueError("DPFI raw means must share one registered shape")
    numerator = (h - dh) - (v - dv)
    denominator = (h0 - dh) + (v0 - dv)
    if np.any(denominator <= 0.0):
        raise ValueError("DPFI blank-port denominator must be positive")
    expected = numerator / denominator
    jacobians = {
        "atom_h": 1.0 / denominator,
        "atom_v": -1.0 / denominator,
        "blank_h": -expected / denominator,
        "blank_v": -expected / denominator,
        "dark_h": (expected - 1.0) / denominator,
        "dark_v": (expected + 1.0) / denominator,
    }
    raw_means = {
        "atom_h": h,
        "atom_v": v,
        "blank_h": h0,
        "blank_v": v0,
        "dark_h": dh,
        "dark_v": dv,
    }
    return _model(
        "dpfi",
        expected,
        np.zeros(h.shape, dtype=float),
        jacobians,
        raw_means,
        read_noise_electrons_rms,
    )


__all__ = [
    "FaradayFrameNoiseModel",
    "dffi_frame_noise_model",
    "dpfi_frame_noise_model",
    "process_dffi_counts",
    "process_dpfi_counts",
]
