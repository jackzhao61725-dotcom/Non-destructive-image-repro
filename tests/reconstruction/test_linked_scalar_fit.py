from __future__ import annotations

import importlib

import numpy as np
import pytest

from non_destructive_image.reconstruction.contracts import (
    DetectorContract,
    ReconstructionGrid,
)
from non_destructive_image.reconstruction.linked_scalar_fit import (
    LinkedRawObservation,
    LinkedScalarFitOptions,
    draw_linked_raw_observation,
    estimate_linked_nuisance_from_references,
    fit_linked_scalar_sequence,
)
from non_destructive_image.reconstruction.object_models import (
    NonnegativeBilinearDensityModel,
)
from non_destructive_image.reconstruction.scalar_measurements import (
    DGILinkedRawOperator,
    DGINuisanceValues,
    DGITransferContract,
    PCILinkedRawOperator,
    PCINuisanceValues,
    PCITransferContract,
    ScalarOpticalResponseContract,
)


def _grid_model() -> tuple[ReconstructionGrid, NonnegativeBilinearDensityModel]:
    axis_m = (np.arange(12, dtype=float) - 6.0) * 0.5e-6
    y_grid_m, z_grid_m = np.meshgrid(axis_m, axis_m)
    grid = ReconstructionGrid.from_arrays(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        pupil=np.ones((12, 12), dtype=complex),
        bin_size=2,
        roi_mask=np.ones((6, 6), dtype=bool),
    )
    model = NonnegativeBilinearDensityModel.from_grid(
        y_grid_m=y_grid_m,
        z_grid_m=z_grid_m,
        knot_y_um=[-2.0, 0.0, 2.0],
        knot_z_um=[-2.0, 0.0, 2.0],
        coefficient_scale_m2=1e14,
    )
    return grid, model


def _response() -> ScalarOpticalResponseContract:
    return ScalarOpticalResponseContract(1.9e-16, 3.8e-18)


def test_linked_scalar_fit_options_freeze_lsmr_controls() -> None:
    options = LinkedScalarFitOptions(trust_region_solver="lsmr")
    assert options.method == "trf"
    assert options.loss == "linear"
    assert options.x_scale == "jac"
    assert options.lsmr_atol == pytest.approx(1e-6)
    assert options.lsmr_btol == pytest.approx(1e-6)
    assert options.lsmr_conlim == pytest.approx(1e8)
    assert options.lsmr_maxiter is None
    assert options.lsmr_regularize is True

    with pytest.raises(ValueError, match="LSMR"):
        LinkedScalarFitOptions(lsmr_atol=0.0)
    with pytest.raises(ValueError, match="positive integer"):
        LinkedScalarFitOptions(lsmr_maxiter=True)
    with pytest.raises(TypeError, match="must be bool"):
        LinkedScalarFitOptions(lsmr_regularize=1)  # type: ignore[arg-type]


def test_joint_fit_passes_custom_controls_to_scipy_lsmr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid, model = _grid_model()
    operator = PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(140.0, 0.7),
        response=_response(),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 1,
            "dark": 1,
        },
        jacobian_batch_size=3,
    )
    true_coefficients = np.asarray(
        [0.08, 0.15, 0.05, 0.25, 0.9, 0.2, 0.04, 0.12, 0.06],
        dtype=float,
    )
    truth_nuisance = PCINuisanceValues(140.0, 0.4)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        [true_coefficients],
        truth_nuisance,
    )

    trf_module = importlib.import_module("scipy.optimize._lsq.trf")
    scipy_lsmr = trf_module.lsmr
    observed_calls: list[dict[str, object]] = []

    def lsmr_spy(*args: object, **kwargs: object) -> object:
        observed_calls.append(dict(kwargs))
        return scipy_lsmr(*args, **kwargs)

    monkeypatch.setattr(trf_module, "lsmr", lsmr_spy)
    fit_linked_scalar_sequence(
        operator,
        model,
        LinkedRawObservation(prediction.role_names, prediction.expected_electrons),
        initial_density_coefficients=0.8 * true_coefficients[None, :],
        density_coefficient_upper=2.0,
        initial_nuisance=PCINuisanceValues(130.0, 0.2),
        nuisance_lower=[80.0, 0.0],
        nuisance_upper=[200.0, 5.0],
        regularisation=None,
        options=LinkedScalarFitOptions(
            trust_region_solver="lsmr",
            irls_iterations=1,
            max_nfev=3,
            lsmr_atol=2.5e-9,
            lsmr_btol=7.5e-9,
            lsmr_conlim=4.2e6,
            lsmr_maxiter=17,
            lsmr_regularize=False,
        ),
    )

    assert observed_calls
    for call in observed_calls:
        assert call["atol"] == pytest.approx(2.5e-9)
        assert call["btol"] == pytest.approx(7.5e-9)
        assert call["conlim"] == pytest.approx(4.2e6)
        assert call["maxiter"] == 17
        assert "regularize" not in call


@pytest.mark.parametrize("method", ["PCI", "DGI"])
def test_joint_linked_raw_fit_recovers_noiseless_density_and_nuisance(
    method: str,
) -> None:
    grid, model = _grid_model()
    true_coefficients = np.asarray(
        [0.08, 0.15, 0.05, 0.25, 0.9, 0.2, 0.04, 0.12, 0.06],
        dtype=float,
    )
    if method == "PCI":
        operator = PCILinkedRawOperator(
            grid=grid,
            detector=DetectorContract(140.0, 0.7),
            response=_response(),
            transfer=PCITransferContract(0.95, np.pi / 2.0),
            independent_exposures_by_role={
                "atom": 1,
                "bright_reference": 1,
                "dark": 1,
            },
            jacobian_batch_size=3,
        )
        truth_nuisance = PCINuisanceValues(140.0, 0.4)
        initial_nuisance = PCINuisanceValues(130.0, 0.2)
        nuisance_lower = [80.0, 0.0]
        nuisance_upper = [200.0, 5.0]
    else:
        operator = DGILinkedRawOperator(
            grid=grid,
            detector=DetectorContract(140.0, 0.7),
            response=_response(),
            transfer=DGITransferContract(2.0),
            independent_exposures_by_role={
                "atom_stop": 1,
                "leakage_stop": 1,
                "stop_dark": 1,
                "open_reference": 1,
                "open_dark": 1,
            },
            jacobian_batch_size=3,
        )
        truth_nuisance = DGINuisanceValues(140.0, 0.4, 0.2, 0.9)
        initial_nuisance = DGINuisanceValues(130.0, 0.2, 0.1, 0.85)
        nuisance_lower = [80.0, 0.0, 0.0, 0.5]
        nuisance_upper = [200.0, 5.0, 5.0, 1.5]
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        [true_coefficients],
        truth_nuisance,
    )
    observation = LinkedRawObservation(
        prediction.role_names,
        prediction.expected_electrons,
    )
    result = fit_linked_scalar_sequence(
        operator,
        model,
        observation,
        initial_density_coefficients=0.8 * true_coefficients[None, :],
        density_coefficient_upper=2.0,
        initial_nuisance=initial_nuisance,
        nuisance_lower=nuisance_lower,
        nuisance_upper=nuisance_upper,
        regularisation=None,
        options=LinkedScalarFitOptions(
            irls_iterations=1,
            max_nfev=80,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        ),
    )

    assert result.diagnostics.success is True
    np.testing.assert_allclose(
        result.density_coefficients[0],
        true_coefficients,
        rtol=2e-5,
        atol=2e-7,
    )
    if method == "PCI":
        expected_nuisance = [140.0, 0.4]
    else:
        expected_nuisance = [140.0, 0.4, 0.2, 0.9]
    np.testing.assert_allclose(
        result.nuisance_values,
        expected_nuisance,
        rtol=2e-6,
        atol=2e-7,
    )
    assert result.diagnostics.weighted_chi_square < 1e-10
    assert result.prediction.role_names == observation.role_names


def test_raw_draw_is_seed_reproducible_and_keeps_shared_roles_once() -> None:
    grid, model = _grid_model()
    operator = PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(100.0, 0.7),
        response=_response(),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 2,
            "dark": 3,
        },
    )
    coefficients = np.full(model.parameter_count, 0.2)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        [coefficients, 0.8 * coefficients],
        PCINuisanceValues(100.0, 0.2),
    )
    left = draw_linked_raw_observation(
        operator,
        prediction,
        np.random.default_rng(27),
    )
    right = draw_linked_raw_observation(
        operator,
        prediction,
        np.random.default_rng(27),
    )

    assert left.role_names == (
        "atom_000",
        "atom_001",
        "bright_reference",
        "dark",
    )
    for left_role, right_role in zip(
        left.observed_electrons,
        right.observed_electrons,
        strict=True,
    ):
        np.testing.assert_array_equal(left_role, right_role)

    initial = estimate_linked_nuisance_from_references(operator, left)
    assert isinstance(initial, PCINuisanceValues)
    assert initial.i0_photoelectrons_per_pixel > 0.0
    assert initial.dark_electrons_per_pixel >= 0.0


def test_dgi_reference_initializer_uses_only_shared_roles() -> None:
    grid, model = _grid_model()
    operator = DGILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(100.0, 0.7),
        response=_response(),
        transfer=DGITransferContract(2.0),
        independent_exposures_by_role={
            "atom_stop": 1,
            "leakage_stop": 1,
            "stop_dark": 1,
            "open_reference": 1,
            "open_dark": 1,
        },
    )
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        [np.full(model.parameter_count, 0.2)],
        DGINuisanceValues(100.0, 0.4, 0.2, 0.9),
    )
    initial = estimate_linked_nuisance_from_references(
        operator,
        LinkedRawObservation(prediction.role_names, prediction.expected_electrons),
    )
    assert isinstance(initial, DGINuisanceValues)
    np.testing.assert_allclose(
        [
            initial.i0_photoelectrons_per_pixel,
            initial.stop_dark_electrons_per_pixel,
            initial.open_dark_electrons_per_pixel,
            initial.open_to_stop_scale,
        ],
        [100.0, 0.4, 0.2, 0.9],
        rtol=5e-15,
        atol=5e-15,
    )


def test_linked_fit_rejects_role_order_and_nuisance_type_mismatch() -> None:
    grid, model = _grid_model()
    operator = PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(100.0, 0.7),
        response=_response(),
        transfer=PCITransferContract(0.95, np.pi / 2.0),
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 1,
            "dark": 1,
        },
    )
    coefficients = np.full(model.parameter_count, 0.2)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        [coefficients],
        PCINuisanceValues(100.0, 0.2),
    )
    reversed_observation = LinkedRawObservation(
        tuple(reversed(prediction.role_names)),
        tuple(reversed(prediction.expected_electrons)),
    )
    with pytest.raises(ValueError, match="role order"):
        fit_linked_scalar_sequence(
            operator,
            model,
            reversed_observation,
            initial_density_coefficients=coefficients[None, :],
            density_coefficient_upper=2.0,
            initial_nuisance=PCINuisanceValues(100.0, 0.2),
            nuisance_lower=[50.0, 0.0],
            nuisance_upper=[150.0, 5.0],
            regularisation=None,
        )
    with pytest.raises(TypeError, match="does not match"):
        fit_linked_scalar_sequence(
            operator,
            model,
            LinkedRawObservation(
                prediction.role_names,
                prediction.expected_electrons,
            ),
            initial_density_coefficients=coefficients[None, :],
            density_coefficient_upper=2.0,
            initial_nuisance=DGINuisanceValues(100.0, 0.2, 0.1, 1.0),
            nuisance_lower=[50.0, 0.0],
            nuisance_upper=[150.0, 5.0],
            regularisation=None,
        )
