"""Shared-calibration DGI/DPFI acquisition for target multiframe ensembles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating]
METHODS = ("dgi", "dpfi")
METHOD_IDS = {"dgi": 1, "dpfi": 2}
DGI_ROLES = (
    "atom_stop",
    "leakage_stop",
    "stop_dark",
    "open_reference",
    "open_dark",
)
DPFI_ROLES = (
    "atom_h",
    "atom_v",
    "blank_h",
    "blank_v",
    "dark_h",
    "dark_v",
)
ROLE_IDS = {
    "dgi": {role: index for index, role in enumerate(DGI_ROLES, start=1)},
    "dpfi": {role: index for index, role in enumerate(DPFI_ROLES, start=1)},
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    _require(result >= minimum, f"{name} must be at least {minimum}")
    return result


def _strict_frame_indices(values: Sequence[int]) -> tuple[int, ...]:
    frames = tuple(
        _strict_integer(value, "image_q", minimum=1) for value in values
    )
    _require(bool(frames), "image_q set is empty")
    _require(len(set(frames)) == len(frames), "image_q is duplicated")
    _require(frames == tuple(range(1, max(frames) + 1)), "image_q must be consecutive from one")
    return frames


def _validated_role_arrays(
    roles: Mapping[str, ArrayLike], expected_keys: set[str]
) -> dict[str, FloatArray]:
    _require(set(roles) == expected_keys, "raw-role inventory changed")
    arrays: dict[str, FloatArray] = {}
    shape: tuple[int, int] | None = None
    for key in sorted(expected_keys):
        values = np.asarray(roles[key], dtype=float)
        _require(
            values.ndim == 2
            and values.size > 0
            and np.isfinite(values).all(),
            f"invalid raw-role array: {key}",
        )
        if shape is None:
            shape = values.shape
        _require(values.shape == shape, f"raw-role shape changed: {key}")
        arrays[key] = values
    return arrays


def _frame_inventory(expected_roles: Mapping[str, Mapping[str, ArrayLike]]) -> tuple[int, ...]:
    _require(set(expected_roles) == set(METHODS), "method inventory changed")
    dgi_atom_keys = tuple(
        key for key in expected_roles["dgi"] if key.startswith("atom_stop_q")
    )
    _require(bool(dgi_atom_keys), "DGI atom-frame inventory is empty")
    frames = _strict_frame_indices(
        tuple(sorted(role_and_q(key)[1] for key in dgi_atom_keys))
    )
    dgi_keys = {f"atom_stop_q{q}" for q in frames} | {
        "leakage_stop",
        "stop_dark",
        "open_reference",
        "open_dark",
    }
    dpfi_keys = (
        {f"atom_h_q{q}" for q in frames}
        | {f"atom_v_q{q}" for q in frames}
        | {"blank_h", "blank_v", "dark_h", "dark_v"}
    )
    _validated_role_arrays(expected_roles["dgi"], dgi_keys)
    _validated_role_arrays(expected_roles["dpfi"], dpfi_keys)
    return frames


def random_identity_count(
    *, duration_count: int, draw_count: int, frame_count: int
) -> int:
    """Return the number of raw-frame RNG identities in the frozen topology."""

    duration_count = _strict_integer(duration_count, "duration_count", minimum=1)
    draw_count = _strict_integer(draw_count, "draw_count", minimum=1)
    frame_count = _strict_integer(frame_count, "frame_count", minimum=1)
    roles_per_sequence = (frame_count + 4) + (2 * frame_count + 4)
    return duration_count * draw_count * roles_per_sequence


def seed_tuple(
    prefix: Sequence[int],
    *,
    duration_us: int,
    method: str,
    draw_id: int,
    role: str,
    image_q: int,
) -> tuple[int, ...]:
    """Build one stable raw-frame seed identity."""

    _require(method in METHODS, "unknown readout method")
    _require(role in ROLE_IDS[method], "role does not belong to method")
    duration_us = _strict_integer(duration_us, "duration_us", minimum=1)
    draw_id = _strict_integer(draw_id, "draw_id", minimum=0)
    image_q = _strict_integer(image_q, "image_q", minimum=0)
    if role.startswith("atom_"):
        _require(image_q > 0, "atom roles require image_q > 0")
    else:
        _require(image_q == 0, "shared calibration roles require image_q = 0")
    _require(bool(prefix), "seed prefix is empty")
    values = tuple(
        _strict_integer(value, "seed prefix value", minimum=0) for value in prefix
    )
    return (
        *values,
        duration_us,
        METHOD_IDS[method],
        draw_id,
        ROLE_IDS[method][role],
        image_q,
    )


def sample_raw_frame(
    expected_electrons: ArrayLike,
    *,
    read_noise_electrons_rms: float,
    seed: Sequence[int],
) -> FloatArray:
    """Apply the declared Poisson-plus-Gaussian detector model once."""

    expected = np.asarray(expected_electrons, dtype=float)
    read_noise = float(read_noise_electrons_rms)
    _require(
        expected.ndim == 2
        and expected.size > 0
        and np.isfinite(expected).all()
        and np.all(expected >= 0.0),
        "expected camera electrons must be a finite non-negative 2D array",
    )
    _require(np.isfinite(read_noise) and read_noise >= 0.0, "invalid read noise")
    _require(bool(seed), "seed is empty")
    validated_seed = tuple(
        _strict_integer(value, "seed value", minimum=0) for value in seed
    )
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(validated_seed)))
    return rng.poisson(expected).astype(float) + rng.normal(
        0.0, read_noise, expected.shape
    )


def expected_sequence_roles(
    *,
    dgi_camera_intensity_over_i0: Mapping[int, ArrayLike],
    dpfi_h_camera_intensity_over_i0: Mapping[int, ArrayLike],
    dpfi_v_camera_intensity_over_i0: Mapping[int, ArrayLike],
    photoelectrons_per_i0_pixel: float,
    dgi_blank_leakage_intensity: float = 1.0e-4,
) -> dict[str, dict[str, FloatArray]]:
    """Construct all expected raw roles for one duration and one sequence."""

    q_values = _strict_frame_indices(
        tuple(sorted(dgi_camera_intensity_over_i0))
    )
    _require(
        q_values == _strict_frame_indices(tuple(sorted(dpfi_h_camera_intensity_over_i0)))
        == _strict_frame_indices(tuple(sorted(dpfi_v_camera_intensity_over_i0))),
        "DGI and DPFI frame sets differ",
    )
    count_scale = float(photoelectrons_per_i0_pixel)
    leakage = float(dgi_blank_leakage_intensity)
    _require(np.isfinite(count_scale) and count_scale > 0.0, "invalid count scale")
    _require(np.isfinite(leakage) and 0.0 <= leakage < 1.0, "invalid DGI leakage")

    arrays: dict[str, dict[str, FloatArray]] = {"dgi": {}, "dpfi": {}}
    shape: tuple[int, int] | None = None
    for q in q_values:
        dgi = np.asarray(dgi_camera_intensity_over_i0[q], dtype=float)
        h_port = np.asarray(dpfi_h_camera_intensity_over_i0[q], dtype=float)
        v_port = np.asarray(dpfi_v_camera_intensity_over_i0[q], dtype=float)
        _require(
            dgi.ndim == 2
            and dgi.size > 0
            and dgi.shape == h_port.shape == v_port.shape
            and np.isfinite(dgi).all()
            and np.isfinite(h_port).all()
            and np.isfinite(v_port).all()
            and np.all(dgi >= 0.0)
            and np.all(h_port >= 0.0)
            and np.all(v_port >= 0.0),
            "invalid expected camera intensity",
        )
        if shape is None:
            shape = dgi.shape
        _require(dgi.shape == shape, "camera shape changes with image_q")
        arrays["dgi"][f"atom_stop_q{q}"] = count_scale * dgi
        arrays["dpfi"][f"atom_h_q{q}"] = count_scale * h_port
        arrays["dpfi"][f"atom_v_q{q}"] = count_scale * v_port

    assert shape is not None
    one = np.ones(shape, dtype=float)
    zero = np.zeros(shape, dtype=float)
    arrays["dgi"].update(
        {
            "leakage_stop": count_scale * leakage * one,
            "stop_dark": zero.copy(),
            "open_reference": count_scale * one,
            "open_dark": zero.copy(),
        }
    )
    arrays["dpfi"].update(
        {
            "blank_h": count_scale * 0.5 * one,
            "blank_v": count_scale * 0.5 * one,
            "dark_h": zero.copy(),
            "dark_v": zero.copy(),
        }
    )
    return arrays


def role_and_q(key: str) -> tuple[str, int]:
    """Separate an expected-role key into its role and image index."""

    if "_q" not in key:
        return key, 0
    role, q_text = key.rsplit("_q", maxsplit=1)
    _require(q_text.isdigit(), "invalid image_q suffix")
    return role, _strict_integer(int(q_text), "image_q", minimum=1)


def sample_sequence(
    expected_roles: Mapping[str, Mapping[str, ArrayLike]],
    *,
    prefix: Sequence[int],
    duration_us: int,
    draw_id: int,
    read_noise_electrons_rms: float,
) -> tuple[dict[str, dict[str, FloatArray]], list[dict[str, Any]]]:
    """Sample every atom and shared calibration role exactly once."""

    frames = _frame_inventory(expected_roles)
    sampled: dict[str, dict[str, FloatArray]] = {"dgi": {}, "dpfi": {}}
    ledger: list[dict[str, Any]] = []
    for method in METHODS:
        for key, expected in expected_roles[method].items():
            role, image_q = role_and_q(key)
            seed = seed_tuple(
                prefix,
                duration_us=duration_us,
                method=method,
                draw_id=draw_id,
                role=role,
                image_q=image_q,
            )
            _require(
                (image_q in frames if role.startswith("atom_") else image_q == 0),
                "raw-role frame identity changed",
            )
            observed = sample_raw_frame(
                expected,
                read_noise_electrons_rms=read_noise_electrons_rms,
                seed=seed,
            )
            sampled[method][key] = observed
            ledger.append(
                {
                    "duration_us": int(duration_us),
                    "method": method,
                    "draw_id": int(draw_id),
                    "role": role,
                    "image_q": image_q,
                    "seed": ":".join(str(value) for value in seed),
                }
            )
    _require(
        len({row["seed"] for row in ledger}) == len(ledger),
        "raw-frame seed identity duplicated within sequence",
    )
    return sampled, ledger


def process_dgi_sequence(
    sampled_roles: Mapping[str, ArrayLike], image_q: Sequence[int]
) -> dict[int, FloatArray]:
    """Process DGI frames with one shared calibration set."""

    frames = _strict_frame_indices(image_q)
    expected_keys = {f"atom_stop_q{q}" for q in frames} | {
        "leakage_stop",
        "stop_dark",
        "open_reference",
        "open_dark",
    }
    arrays = _validated_role_arrays(sampled_roles, expected_keys)
    reference = arrays["open_reference"]
    open_dark = arrays["open_dark"]
    denominator = reference - open_dark
    _require(np.isfinite(denominator).all() and np.all(denominator > 0.0), "DGI denominator failed")
    leakage = arrays["leakage_stop"]
    processed = {
        q: (arrays[f"atom_stop_q{q}"] - leakage) / denominator for q in frames
    }
    _require(all(np.isfinite(values).all() for values in processed.values()), "DGI processed signal is non-finite")
    return processed


def process_dpfi_sequence(
    sampled_roles: Mapping[str, ArrayLike], image_q: Sequence[int]
) -> dict[int, FloatArray]:
    """Process DPFI after independently sampling both bright ports."""

    frames = _strict_frame_indices(image_q)
    expected_keys = (
        {f"atom_h_q{q}" for q in frames}
        | {f"atom_v_q{q}" for q in frames}
        | {"blank_h", "blank_v", "dark_h", "dark_v"}
    )
    arrays = _validated_role_arrays(sampled_roles, expected_keys)
    dark_h = arrays["dark_h"]
    dark_v = arrays["dark_v"]
    denominator = (
        arrays["blank_h"] - dark_h + arrays["blank_v"] - dark_v
    )
    _require(np.isfinite(denominator).all() and np.all(denominator > 0.0), "DPFI denominator failed")
    processed: dict[int, FloatArray] = {}
    for q in frames:
        h_port = arrays[f"atom_h_q{q}"] - dark_h
        v_port = arrays[f"atom_v_q{q}"] - dark_v
        processed[q] = (h_port - v_port) / denominator
    _require(all(np.isfinite(values).all() for values in processed.values()), "DPFI processed signal is non-finite")
    return processed


def axial_profile(signal: ArrayLike) -> FloatArray:
    """Sum the camera signal along z while retaining the camera-y samples."""

    values = np.asarray(signal, dtype=float)
    _require(
        values.ndim == 2 and values.size > 0 and np.isfinite(values).all(),
        "invalid camera signal",
    )
    profile = np.sum(values, axis=0)
    _require(np.isfinite(profile).all(), "axial profile is non-finite")
    return profile


__all__ = [
    "DGI_ROLES",
    "DPFI_ROLES",
    "METHODS",
    "axial_profile",
    "expected_sequence_roles",
    "process_dgi_sequence",
    "process_dpfi_sequence",
    "random_identity_count",
    "role_and_q",
    "sample_raw_frame",
    "sample_sequence",
    "seed_tuple",
]
