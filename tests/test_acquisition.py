from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from isolated_non_destructive_image import (  # noqa: E402
    load_isolated_non_destructive_image_module,
)


_ACQUISITION = load_isolated_non_destructive_image_module(
    "acquisition", namespace="_ndi_corrected_acquisition_scientific_tests_v1"
)
ProcessedDiagnostic = _ACQUISITION.ProcessedDiagnostic
RawElectronFrame = _ACQUISITION.RawElectronFrame
conditional_block_snr = _ACQUISITION.conditional_block_snr
deterministic_raw_electron_frame = _ACQUISITION.deterministic_raw_electron_frame
delta_method_covariance = _ACQUISITION.delta_method_covariance
dgi_signal_diagnostic = _ACQUISITION.dgi_signal_diagnostic
marginal_block_snr = _ACQUISITION.marginal_block_snr
paired_difference_diagnostic = _ACQUISITION.paired_difference_diagnostic
pci_contrast_diagnostic = _ACQUISITION.pci_contrast_diagnostic
rai_optical_density_diagnostic = _ACQUISITION.rai_optical_density_diagnostic
rai_transmission_diagnostic = _ACQUISITION.rai_transmission_diagnostic
simulate_intensity_frame = _ACQUISITION.simulate_intensity_frame
simulate_raw_electron_frame = _ACQUISITION.simulate_raw_electron_frame


def _frame(
    expected: float | np.ndarray,
    *,
    observed: float | np.ndarray | None = None,
    role: str,
    read_noise: float = 1.4,
    exposures: int = 1,
) -> RawElectronFrame:
    expected_array = np.broadcast_to(np.asarray(expected, dtype=float), (2, 2)).copy()
    if observed is None:
        observed_array = expected_array.copy()
    else:
        observed_array = np.broadcast_to(np.asarray(observed, dtype=float), (2, 2)).copy()
    return RawElectronFrame(
        role=role,
        expected_electrons=expected_array,
        observed_electrons=observed_array,
        read_noise_electrons_rms=read_noise,
        camera_contract_id="orca_fusion_fast_v1",
        sampling_contract_id="m10_0p650um_153x153",
        bit_generator="deterministic_fixture",
        rng_provenance="deterministic_fixture",
        independent_exposures=exposures,
    )


def test_raw_frame_mean_draw_matches_exact_poisson_gaussian_recipe() -> None:
    expected = np.array([[0.0, 1.5], [7.0, 30.0]])
    seed_components = (20260724, 0, 0)
    exposures = 5
    read_noise = 1.4
    manual_rng = np.random.default_rng(np.random.SeedSequence(seed_components))
    manual = (
        manual_rng.poisson(expected * exposures) / exposures
        + manual_rng.normal(0.0, read_noise / np.sqrt(exposures), expected.shape)
    )
    result = simulate_raw_electron_frame(
        expected,
        read_noise_electrons_rms=read_noise,
        role="atom",
        camera_contract_id="orca_fusion_fast_v1",
        sampling_contract_id="m10_0p650um_153x153",
        independent_exposures=exposures,
        seed_components=seed_components,
    )

    np.testing.assert_array_equal(result.observed_electrons, manual)
    np.testing.assert_allclose(
        result.variance_electrons2,
        (expected + read_noise**2) / exposures,
    )
    assert result.seed_components == seed_components
    assert result.bit_generator == "PCG64"
    assert result.rng_provenance == "seed_components_replayable"
    assert not result.expected_electrons.flags.writeable
    assert not result.observed_electrons.flags.writeable


def test_intensity_frame_converts_i0_scale_once_and_replays_seed() -> None:
    intensity = np.array([[0.1, 0.5], [1.0, 1.2]])
    kwargs = dict(
        photoelectrons_per_i0_pixel=240.0,
        read_noise_electrons_rms=1.4,
        role="pci_atom",
        camera_contract_id="orca_fusion_fast_v1",
        sampling_contract_id="m10_0p650um_153x153",
        seed_components=(7, 0, 2, 0),
    )
    first = simulate_intensity_frame(intensity, **kwargs)
    second = simulate_intensity_frame(intensity, **kwargs)
    third = simulate_intensity_frame(
        intensity,
        **{**kwargs, "seed_components": (7, 0, 2, 1)},
    )

    np.testing.assert_allclose(first.expected_electrons, intensity * 240.0)
    np.testing.assert_array_equal(first.observed_electrons, second.observed_electrons)
    assert not np.array_equal(first.observed_electrons, third.observed_electrons)


def test_deterministic_raw_frame_preserves_expected_camera_variance() -> None:
    expected = np.array([[0.0, 3.0], [8.0, 21.0]])
    frame = deterministic_raw_electron_frame(
        expected,
        read_noise_electrons_rms=1.4,
        role="expected_pci_atom",
        camera_contract_id="orca_fusion_fast_v1",
        sampling_contract_id="m10_0p650um_153x153",
        independent_exposures=2,
    )

    np.testing.assert_array_equal(frame.expected_electrons, expected)
    np.testing.assert_array_equal(frame.observed_electrons, expected)
    np.testing.assert_allclose(frame.variance_electrons2, (expected + 1.4**2) / 2.0)
    assert frame.rng_provenance == "deterministic_fixture"
    assert frame.bit_generator == "not_applicable"
    assert frame.seed_components == ()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"expected_electrons": [[-1.0]]}, "negative expected"),
        ({"read_noise_electrons_rms": -0.1}, "read_noise"),
        ({"independent_exposures": 0}, "independent_exposures"),
        ({"independent_exposures": 1.5}, "integer"),
    ],
)
def test_raw_frame_rejects_invalid_contract(kwargs: dict, match: str) -> None:
    parameters = {
        "expected_electrons": [[1.0]],
        "read_noise_electrons_rms": 1.4,
        "rng": np.random.default_rng(1),
        "role": "test",
        "camera_contract_id": "orca_fusion_fast_v1",
        "sampling_contract_id": "m10_0p650um_153x153",
        "independent_exposures": 1,
    }
    parameters.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=match):
        simulate_raw_electron_frame(**parameters)


def test_raw_frame_requires_exactly_one_replay_source() -> None:
    common = dict(
        expected_electrons=[[1.0]],
        read_noise_electrons_rms=1.4,
        role="test",
        camera_contract_id="orca_fusion_fast_v1",
        sampling_contract_id="m10_0p650um_153x153",
    )
    with pytest.raises(ValueError, match="exactly one"):
        simulate_raw_electron_frame(**common)
    with pytest.raises(ValueError, match="exactly one"):
        simulate_raw_electron_frame(
            **common,
            rng=np.random.default_rng(1),
            seed_components=(1,),
        )

    caller_owned = simulate_raw_electron_frame(
        **common,
        rng=np.random.default_rng(1),
    )
    assert caller_owned.rng_provenance == "caller_owned_rng_not_self_contained"
    assert caller_owned.seed_components == ()


def test_frame_arrays_cannot_be_made_writeable_and_contracts_must_match() -> None:
    frame = _frame(1.0, role="test")
    with pytest.raises(ValueError):
        frame.expected_electrons.setflags(write=True)
    with pytest.raises(ValueError):
        frame.observed_electrons.setflags(write=True)

    mismatched = RawElectronFrame(
        role="other_camera",
        expected_electrons=np.ones((2, 2)),
        observed_electrons=np.ones((2, 2)),
        read_noise_electrons_rms=1.4,
        camera_contract_id="other_camera",
        sampling_contract_id="m10_0p650um_153x153",
        bit_generator="fixture",
        rng_provenance="deterministic_fixture",
    )
    with pytest.raises(ValueError, match="camera_contract_id"):
        paired_difference_diagnostic(frame, mismatched)


def test_acquisition_records_use_identity_not_incomplete_value_equality() -> None:
    first = _frame(1.0, role="same_metadata")
    second = _frame(2.0, role="same_metadata")
    assert first is not second
    assert first != second

    first_diagnostic = paired_difference_diagnostic(
        first,
        _frame(0.0, role="reference"),
    )
    second_diagnostic = paired_difference_diagnostic(
        second,
        _frame(0.0, role="reference"),
    )
    assert first_diagnostic is not second_diagnostic
    assert first_diagnostic != second_diagnostic


def test_diagnostic_masks_require_boolean_dtype() -> None:
    shape = (1, 1)
    with pytest.raises(TypeError, match="boolean"):
        ProcessedDiagnostic(
            quantity_name="bad_mask",
            unit="1",
            expected_value=np.zeros(shape),
            observed_value=np.zeros(shape),
            variance=np.ones(shape),
            expected_valid_mask=np.array([[np.nan]]),
            observed_valid_mask=np.ones(shape, dtype=bool),
            jacobian={"raw": np.ones(shape)},
        )


def test_shared_nuisance_names_are_normalised_without_silent_collision() -> None:
    shape = (1, 1)
    diagnostic = ProcessedDiagnostic(
        quantity_name="shared",
        unit="1",
        expected_value=np.ones(shape),
        observed_value=np.ones(shape),
        variance=np.ones(shape),
        expected_valid_mask=np.ones(shape, dtype=bool),
        observed_valid_mask=np.ones(shape, dtype=bool),
        jacobian={"raw": np.ones(shape)},
        shared_nuisance_jacobian={" scale ": np.ones(shape)},
        shared_nuisance_variance={"scale": 0.25},
    )
    assert set(diagnostic.shared_nuisance_jacobian) == {"scale"}
    assert set(diagnostic.shared_nuisance_variance) == {"scale"}

    with pytest.raises(ValueError, match="unique after normalisation"):
        ProcessedDiagnostic(
            quantity_name="duplicate_shared",
            unit="1",
            expected_value=np.ones(shape),
            observed_value=np.ones(shape),
            variance=np.ones(shape),
            expected_valid_mask=np.ones(shape, dtype=bool),
            observed_valid_mask=np.ones(shape, dtype=bool),
            jacobian={"raw": np.ones(shape)},
            shared_nuisance_jacobian={
                "scale": np.ones(shape),
                " scale ": np.ones(shape),
            },
            shared_nuisance_variance={"scale": 0.25},
        )


def test_paired_difference_includes_reference_noise_and_shared_dark_cancels() -> None:
    atom = _frame(12.0, observed=13.0, role="atom", read_noise=1.0)
    reference = _frame(9.0, observed=8.0, role="reference", read_noise=2.0)
    dark_a = _frame(4.0, observed=100.0, role="shared_dark", read_noise=5.0)
    dark_b = _frame(4.0, observed=-100.0, role="shared_dark", read_noise=5.0)

    first = paired_difference_diagnostic(atom, reference, shared_dark_frame=dark_a)
    second = paired_difference_diagnostic(atom, reference, shared_dark_frame=dark_b)

    np.testing.assert_allclose(first.expected_value, 3.0)
    np.testing.assert_allclose(first.observed_value, 5.0)
    np.testing.assert_allclose(first.variance, (12.0 + 1.0) + (9.0 + 4.0))
    np.testing.assert_array_equal(first.observed_value, second.observed_value)
    support = np.ones((2, 2), dtype=bool)
    assert conditional_block_snr(first, support) == pytest.approx(
        abs(4 * 3.0) / np.sqrt(4 * 26.0)
    )


def test_rai_od_jacobian_and_variance_match_closed_form() -> None:
    atom = _frame(82.0, role="atom", read_noise=1.0)
    reference = _frame(102.0, role="reference", read_noise=1.0, exposures=2)
    dark = _frame(2.0, role="dark", read_noise=1.0, exposures=4)
    diagnostic = rai_optical_density_diagnostic(atom, reference, dark)
    a = 80.0
    r = 100.0
    j = np.array([-1.0 / a, 1.0 / r, 1.0 / a - 1.0 / r])
    raw_variance = np.array([
        82.0 + 1.0,
        (102.0 + 1.0) / 2,
        (2.0 + 1.0) / 4,
    ])

    np.testing.assert_allclose(diagnostic.expected_value, -np.log(0.8))
    np.testing.assert_allclose(diagnostic.jacobian["atom"], j[0])
    np.testing.assert_allclose(diagnostic.jacobian["reference"], j[1])
    np.testing.assert_allclose(diagnostic.jacobian["dark"], j[2])
    np.testing.assert_allclose(diagnostic.variance, np.sum(j**2 * raw_variance))
    assert diagnostic.unit == "1"


def test_rai_transmission_matches_ratio_and_delta_variance() -> None:
    atom = _frame(82.0, role="atom", read_noise=1.0)
    reference = _frame(102.0, role="reference", read_noise=1.0, exposures=2)
    dark = _frame(2.0, role="dark", read_noise=1.0, exposures=4)
    diagnostic = rai_transmission_diagnostic(atom, reference, dark)
    j = np.array([1.0 / 100.0, -0.8 / 100.0, (0.8 - 1.0) / 100.0])
    raw_variance = np.array(
        [82.0 + 1.0, (102.0 + 1.0) / 2.0, (2.0 + 1.0) / 4.0]
    )

    np.testing.assert_allclose(diagnostic.expected_value, 0.8)
    np.testing.assert_allclose(diagnostic.jacobian["atom"], j[0])
    np.testing.assert_allclose(diagnostic.jacobian["reference"], j[1])
    np.testing.assert_allclose(diagnostic.jacobian["dark"], j[2])
    np.testing.assert_allclose(diagnostic.variance, np.sum(j**2 * raw_variance))

    optical_depth = rai_optical_density_diagnostic(atom, reference, dark)
    np.testing.assert_allclose(
        optical_depth.jacobian["atom"],
        -diagnostic.jacobian["atom"] / diagnostic.expected_value,
    )
    np.testing.assert_allclose(
        optical_depth.jacobian["reference"],
        -diagnostic.jacobian["reference"] / diagnostic.expected_value,
    )
    np.testing.assert_allclose(
        optical_depth.jacobian["dark"],
        -diagnostic.jacobian["dark"] / diagnostic.expected_value,
    )
    np.testing.assert_allclose(
        optical_depth.variance,
        diagnostic.variance / diagnostic.expected_value**2,
    )


def test_expected_dark_corrected_numerator_must_be_nonnegative() -> None:
    atom = _frame(1.0, observed=4.0, role="atom")
    reference = _frame(5.0, observed=5.0, role="reference")
    dark = _frame(2.0, observed=1.0, role="dark")

    transmission = rai_transmission_diagnostic(atom, reference, dark)
    pci = pci_contrast_diagnostic(atom, reference, dark)
    assert not transmission.expected_valid_mask.any()
    assert not pci.expected_valid_mask.any()
    assert transmission.observed_valid_mask.all()
    assert pci.observed_valid_mask.all()
    assert np.isnan(transmission.expected_value).all()
    assert np.isnan(pci.expected_value).all()
    assert np.isfinite(transmission.observed_value).all()
    assert np.isfinite(pci.observed_value).all()

    zero_atom = _frame(2.0, observed=2.0, role="zero_atom")
    assert rai_transmission_diagnostic(
        zero_atom, reference, dark
    ).expected_valid_mask.all()
    assert pci_contrast_diagnostic(
        zero_atom, reference, dark
    ).expected_valid_mask.all()


def test_rai_invalid_processed_ratio_is_unsupported_without_clipping() -> None:
    atom = _frame(10.0, observed=0.0, role="atom")
    reference = _frame(20.0, observed=20.0, role="reference")
    dark = _frame(1.0, observed=1.0, role="dark")
    diagnostic = rai_optical_density_diagnostic(atom, reference, dark)

    assert not diagnostic.valid_mask.any()
    assert np.isnan(diagnostic.observed_value).all()
    np.testing.assert_allclose(diagnostic.expected_value, -np.log(9.0 / 19.0))


def test_pci_signed_contrast_jacobian_and_shared_dark_limit() -> None:
    atom = _frame(122.0, role="pci_atom", read_noise=1.0)
    reference = _frame(102.0, role="pci_reference", read_noise=1.0)
    dark = _frame(2.0, role="dark", read_noise=1.0)
    diagnostic = pci_contrast_diagnostic(atom, reference, dark)
    q = 1.2
    contrast = 0.2

    np.testing.assert_allclose(diagnostic.expected_value, contrast)
    np.testing.assert_allclose(diagnostic.jacobian["atom"], 1.0 / 100.0)
    np.testing.assert_allclose(diagnostic.jacobian["bright_reference"], -q / 100.0)
    np.testing.assert_allclose(diagnostic.jacobian["dark"], contrast / 100.0)
    np.testing.assert_allclose(
        diagnostic.variance,
        (1.0 / 100.0) ** 2 * 123.0
        + (q / 100.0) ** 2 * 103.0
        + (contrast / 100.0) ** 2 * 3.0,
    )

    atom_free = pci_contrast_diagnostic(reference, reference, dark)
    np.testing.assert_allclose(atom_free.expected_value, 0.0)
    np.testing.assert_allclose(atom_free.jacobian["dark"], 0.0)


def test_dgi_uses_open_reference_and_stop_dark_cancels() -> None:
    atom = _frame(15.0, observed=16.0, role="dgi_atom", read_noise=1.0)
    leakage = _frame(5.0, observed=4.0, role="dgi_leakage", read_noise=1.0)
    stop_dark = _frame(2.0, observed=200.0, role="stop_dark", read_noise=4.0)
    open_reference = _frame(102.0, observed=102.0, role="open_reference", read_noise=1.0)
    open_dark = _frame(2.0, observed=2.0, role="open_dark", read_noise=1.0)
    diagnostic = dgi_signal_diagnostic(
        atom,
        leakage,
        stop_dark,
        open_reference,
        open_dark,
        open_to_stop_scale=0.5,
        open_to_stop_scale_variance=0.01,
    )

    np.testing.assert_allclose(diagnostic.expected_value, 10.0 / 50.0)
    np.testing.assert_allclose(diagnostic.observed_value, 12.0 / 50.0)
    np.testing.assert_allclose(diagnostic.jacobian["atom_stop"], 1.0 / 50.0)
    np.testing.assert_allclose(diagnostic.jacobian["leakage_stop"], -1.0 / 50.0)
    np.testing.assert_allclose(diagnostic.jacobian["stop_dark"], 0.0)
    np.testing.assert_allclose(diagnostic.jacobian["open_reference"], -0.2 / 100.0)
    np.testing.assert_allclose(diagnostic.jacobian["open_dark"], 0.2 / 100.0)
    np.testing.assert_allclose(diagnostic.jacobian["open_to_stop_scale"], -0.2 / 0.5)
    expected_local_variance = (
        (16.0 + 6.0) / 50.0**2
        + (103.0 + 3.0) * (0.2 / 100.0) ** 2
    )
    np.testing.assert_allclose(diagnostic.variance, expected_local_variance)
    assert diagnostic.shared_nuisance_variance["open_to_stop_scale"] == 0.01
    np.testing.assert_allclose(
        diagnostic.shared_nuisance_jacobian["open_to_stop_scale"],
        -0.4,
    )

    support = np.ones((2, 2), dtype=bool)
    conditional = abs(4.0 * 0.2) / np.sqrt(4.0 * expected_local_variance)
    marginal_variance = 4.0 * expected_local_variance + 0.01 * (4.0 * -0.4) ** 2
    assert conditional_block_snr(diagnostic, support) == pytest.approx(conditional)
    assert marginal_block_snr(diagnostic, support) == pytest.approx(
        abs(4.0 * 0.2) / np.sqrt(marginal_variance)
    )
    assert marginal_block_snr(diagnostic, support) < conditional

    alternate_stop_dark = _frame(
        9000.0,
        observed=-7000.0,
        role="stop_dark",
        read_noise=800.0,
    )
    alternate = dgi_signal_diagnostic(
        atom,
        leakage,
        alternate_stop_dark,
        open_reference,
        open_dark,
        open_to_stop_scale=0.5,
        open_to_stop_scale_variance=0.01,
    )
    np.testing.assert_array_equal(alternate.expected_value, diagnostic.expected_value)
    np.testing.assert_array_equal(alternate.observed_value, diagnostic.observed_value)
    np.testing.assert_array_equal(alternate.variance, diagnostic.variance)


def test_expected_and_observed_validity_are_independent() -> None:
    atom = _frame(3.0, observed=4.0, role="atom")
    reference = _frame(1.0, observed=20.0, role="reference")
    dark = _frame(2.0, observed=1.0, role="dark")
    diagnostic = pci_contrast_diagnostic(atom, reference, dark)

    assert not diagnostic.expected_valid_mask.any()
    assert diagnostic.observed_valid_mask.all()
    assert np.isnan(diagnostic.expected_value).all()
    assert np.isfinite(diagnostic.observed_value).all()
    with pytest.raises(ValueError, match="unsupported expected"):
        conditional_block_snr(diagnostic, np.ones((2, 2), dtype=bool))


def test_method_jacobians_match_finite_differences() -> None:
    step = 1e-5

    def central_difference(function, vector: np.ndarray, index: int) -> float:
        upper = vector.copy()
        lower = vector.copy()
        upper[index] += step
        lower[index] -= step
        return (function(upper) - function(lower)) / (2.0 * step)

    rai_values = np.array([82.0, 102.0, 2.0])
    rai_function = lambda x: -np.log((x[0] - x[2]) / (x[1] - x[2]))
    rai = rai_optical_density_diagnostic(
        _frame(82.0, role="atom"),
        _frame(102.0, role="reference"),
        _frame(2.0, role="dark"),
    )
    rai_j = [rai.jacobian[name][0, 0] for name in ("atom", "reference", "dark")]
    np.testing.assert_allclose(
        rai_j,
        [central_difference(rai_function, rai_values, index) for index in range(3)],
        rtol=1e-7,
        atol=1e-10,
    )

    pci_values = np.array([122.0, 102.0, 2.0])
    pci_function = lambda x: (x[0] - x[2]) / (x[1] - x[2]) - 1.0
    pci = pci_contrast_diagnostic(
        _frame(122.0, role="atom"),
        _frame(102.0, role="reference"),
        _frame(2.0, role="dark"),
    )
    pci_j = [
        pci.jacobian[name][0, 0]
        for name in ("atom", "bright_reference", "dark")
    ]
    np.testing.assert_allclose(
        pci_j,
        [central_difference(pci_function, pci_values, index) for index in range(3)],
        rtol=1e-7,
        atol=1e-10,
    )

    dgi_values = np.array([15.0, 5.0, 2.0, 102.0, 2.0, 0.5])
    dgi_function = lambda x: (
        ((x[0] - x[2]) - (x[1] - x[2])) / (x[5] * (x[3] - x[4]))
    )
    dgi = dgi_signal_diagnostic(
        _frame(15.0, role="atom"),
        _frame(5.0, role="leakage"),
        _frame(2.0, role="stop_dark"),
        _frame(102.0, role="open_reference"),
        _frame(2.0, role="open_dark"),
        open_to_stop_scale=0.5,
    )
    dgi_j = [
        dgi.jacobian[name][0, 0]
        for name in (
            "atom_stop",
            "leakage_stop",
            "stop_dark",
            "open_reference",
            "open_dark",
            "open_to_stop_scale",
        )
    ]
    np.testing.assert_allclose(
        dgi_j,
        [central_difference(dgi_function, dgi_values, index) for index in range(6)],
        rtol=1e-7,
        atol=1e-10,
    )


def test_delta_method_retains_shared_reference_cross_frame_covariance() -> None:
    b = 100.0
    q1, q2 = 1.2, 0.8
    c1, c2 = q1 - 1.0, q2 - 1.0
    jacobian = np.array(
        [
            [1.0 / b, 0.0, -q1 / b, c1 / b],
            [0.0, 1.0 / b, -q2 / b, c2 / b],
        ]
    )
    variances = np.array([80.0, 90.0, 103.0, 3.0])
    propagated = delta_method_covariance(jacobian, np.diag(variances))
    expected_off_diagonal = (q1 * q2 * variances[2] + c1 * c2 * variances[3]) / b**2

    assert propagated[0, 1] == pytest.approx(expected_off_diagonal)
    assert propagated[1, 0] == pytest.approx(expected_off_diagonal)
    assert propagated[0, 1] > 0.0
    assert np.linalg.eigvalsh(propagated).min() >= -1e-14


def test_delta_method_rejects_nonsymmetric_or_indefinite_covariance() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        delta_method_covariance([[1.0, 0.0]], [[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        delta_method_covariance([[1.0, 0.0]], [[1.0, 2.0], [2.0, 1.0]])


def test_delta_method_projects_only_tolerance_scale_negative_roundoff() -> None:
    propagated = delta_method_covariance(
        np.eye(2),
        np.diag([1.0, -5e-13]),
        symmetry_tolerance=1e-12,
    )

    np.testing.assert_allclose(propagated, np.diag([1.0, 0.0]), atol=1e-15)
    assert np.linalg.eigvalsh(propagated).min() >= 0.0
    with pytest.raises(ValueError):
        propagated.setflags(write=True)


@pytest.mark.parametrize("bad", [[[True]], [[1.0 + 1.0j]]])
def test_raw_frame_rejects_boolean_or_complex_expected_data(bad: list) -> None:
    with pytest.raises(TypeError, match="real numeric"):
        simulate_raw_electron_frame(
            bad,
            read_noise_electrons_rms=1.4,
            role="test",
            camera_contract_id="orca_fusion_fast_v1",
            sampling_contract_id="m10_0p650um_153x153",
            seed_components=(1,),
        )
