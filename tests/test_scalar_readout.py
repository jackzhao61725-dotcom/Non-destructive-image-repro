from __future__ import annotations

import numpy as np
import pytest

from non_destructive_image import (
    simulate_dgi_image,
    simulate_dgi_object_field,
    simulate_fourier_image,
    simulate_pci_image,
    simulate_pci_object_field,
    simulate_selected_scalar_readout,
)


def _complex_object() -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, 16)
    grid_y, grid_z = np.meshgrid(axis, axis, indexing="ij")
    profile = np.maximum(0.0, 1.0 - grid_y**2 / 0.8**2 - grid_z**2 / 0.6**2)
    phase = 0.2 * profile
    optical_depth = 0.04 * profile
    object_field = np.exp(-optical_depth / 2.0 + 1j * phase)
    pupil = ((grid_y**2 + grid_z**2) <= 0.9**2).astype(float)
    return object_field, pupil


def test_complex_object_wrappers_match_shared_fourier_propagation() -> None:
    object_field, pupil = _complex_object()
    pci_reference = 0.95 * np.exp(1j * np.pi / 2.0)
    dgi_reference = 10.0 ** (-4.0 / 2.0)

    np.testing.assert_array_equal(
        simulate_pci_object_field(object_field, pupil),
        simulate_fourier_image(object_field, pupil, pci_reference),
    )
    np.testing.assert_array_equal(
        simulate_dgi_object_field(object_field, pupil),
        simulate_fourier_image(object_field, pupil, dgi_reference),
    )


def test_phase_only_complex_entry_is_exactly_backward_compatible() -> None:
    object_field, pupil = _complex_object()
    phase = np.angle(object_field)
    phase_only = np.exp(1j * phase)

    np.testing.assert_array_equal(
        simulate_pci_object_field(phase_only, pupil),
        simulate_pci_image(phase, pupil),
    )
    np.testing.assert_array_equal(
        simulate_dgi_object_field(phase_only, pupil),
        simulate_dgi_image(phase, pupil),
    )


def test_atom_free_backgrounds_follow_declared_field_conventions() -> None:
    object_field = np.ones((8, 8), dtype=complex)
    pupil = np.ones((8, 8))

    np.testing.assert_array_equal(
        simulate_pci_object_field(object_field, pupil),
        np.full((8, 8), 0.95**2),
    )
    np.testing.assert_array_equal(
        simulate_dgi_object_field(object_field, pupil),
        np.full((8, 8), 10.0**-4),
    )


def test_absorption_is_retained_in_complex_scattered_field() -> None:
    object_field, pupil = _complex_object()
    result = simulate_pci_object_field(
        object_field,
        np.ones_like(pupil),
        return_intermediates=True,
    )

    np.testing.assert_array_equal(result["object_field"], object_field)
    np.testing.assert_array_equal(result["scattered_field"], object_field - 1.0)
    assert np.any(np.abs(result["object_field"]) < 1.0)


def test_readout_selector_dispatches_only_pci_and_dgi() -> None:
    object_field, pupil = _complex_object()

    np.testing.assert_array_equal(
        simulate_selected_scalar_readout("pci", object_field, pupil),
        simulate_pci_object_field(object_field, pupil),
    )
    np.testing.assert_array_equal(
        simulate_selected_scalar_readout("dgi", object_field, pupil),
        simulate_dgi_object_field(object_field, pupil),
    )
    with pytest.raises(ValueError, match="exactly"):
        simulate_selected_scalar_readout("PCI", object_field, pupil)


@pytest.mark.parametrize(
    ("object_field", "pupil", "message"),
    [
        (np.ones(4), np.ones(4), "two-dimensional"),
        (np.ones((2, 2)), np.ones((3, 3)), "same two-dimensional shape"),
        (np.array([[np.nan + 0.0j]]), np.ones((1, 1)), "finite"),
        (np.ones((1, 1)), np.array([[np.inf]]), "finite"),
    ],
)
def test_complex_object_entries_reject_invalid_arrays(
    object_field: np.ndarray,
    pupil: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_pci_object_field(object_field, pupil)
    with pytest.raises(ValueError, match=message):
        simulate_dgi_object_field(object_field, pupil)


@pytest.mark.parametrize("transmittance", [-0.1, 1.1, np.nan])
def test_pci_rejects_invalid_field_amplitude_transmission(transmittance: float) -> None:
    with pytest.raises(ValueError, match="field amplitude"):
        simulate_pci_object_field(
            np.ones((2, 2)),
            np.ones((2, 2)),
            phase_plate_transmittance=transmittance,
        )


@pytest.mark.parametrize("optical_depth", [-0.1, np.inf, np.nan])
def test_dgi_rejects_invalid_base10_intensity_optical_depth(
    optical_depth: float,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        simulate_dgi_object_field(
            np.ones((2, 2)),
            np.ones((2, 2)),
            stop_optical_depth=optical_depth,
        )


@pytest.mark.parametrize(
    ("phase", "pupil", "message"),
    [
        (np.ones(4), np.ones(4), "two-dimensional"),
        (np.ones((2, 2)), np.ones((3, 3)), "same two-dimensional shape"),
        (np.array([[np.nan]]), np.ones((1, 1)), "finite"),
        (np.array([[1.0j]]), np.ones((1, 1)), "real"),
    ],
)
def test_phase_only_entries_apply_the_same_array_contract(
    phase: np.ndarray,
    pupil: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        simulate_pci_image(phase, pupil)
    with pytest.raises(ValueError, match=message):
        simulate_dgi_image(phase, pupil)


def test_fourier_image_rejects_non_scalar_reference_field() -> None:
    with pytest.raises(ValueError, match="scalar field amplitude"):
        simulate_fourier_image(
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
        )
