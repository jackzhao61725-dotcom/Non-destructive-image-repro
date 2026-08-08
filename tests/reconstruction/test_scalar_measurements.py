from __future__ import annotations

import numpy as np
import pytest

from non_destructive_image import reconstruction
from non_destructive_image.imaging import (
    simulate_dgi_object_field,
    simulate_pci_object_field,
)
from non_destructive_image.reconstruction.contracts import (
    DetectorContract,
    ReconstructionGrid,
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


def _grid_and_model() -> tuple[ReconstructionGrid, NonnegativeBilinearDensityModel]:
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
        coefficient_scale_m2=1.0e14,
    )
    return grid, model


def _response() -> ScalarOpticalResponseContract:
    return ScalarOpticalResponseContract(
        phase_per_column_density_rad_m2=1.9e-16,
        optical_depth_per_column_density_m2=3.8e-18,
    )


def _parameters(model: NonnegativeBilinearDensityModel, scale: float) -> np.ndarray:
    values = np.linspace(0.15, 0.95, model.parameter_count)
    return scale * values


def test_linked_pci_roles_match_shared_forward_image_and_appear_once() -> None:
    grid, model = _grid_and_model()
    transfer = PCITransferContract(0.95, np.pi / 2.0)
    operator = PCILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(120.0, 0.7),
        response=_response(),
        transfer=transfer,
        independent_exposures_by_role={
            "atom": 1,
            "bright_reference": 2,
            "dark": 3,
        },
        jacobian_batch_size=3,
    )
    vectors = [_parameters(model, 0.8), _parameters(model, 1.0)]
    nuisance = PCINuisanceValues(120.0, 0.4)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        vectors,
        nuisance,
    )

    assert prediction.role_names == (
        "atom_000",
        "atom_001",
        "bright_reference",
        "dark",
    )
    assert prediction.role_frame_indices == (0, 1, None, None)
    assert prediction.shared_role_names == ("bright_reference", "dark")
    assert prediction.prediction_vector.size == 4 * grid.roi_pixel_count
    assert prediction.jacobian.shape == (
        4 * grid.roi_pixel_count,
        2 * model.parameter_count + 2,
    )
    density = model.column_density(vectors[0])
    object_field = np.exp(
        _response().complex_exponent_per_column_density_m2 * density
    )
    expected_image = grid.camera_average(
        simulate_pci_object_field(
            object_field,
            grid.pupil,
            transfer.phase_plate_transmittance,
            transfer.phase_plate_phase_rad,
        )
    )
    np.testing.assert_allclose(
        prediction.expected_electrons[0],
        nuisance.i0_photoelectrons_per_pixel * expected_image
        + nuisance.dark_electrons_per_pixel,
        rtol=2e-15,
        atol=1e-13,
    )

    array_prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        np.stack(vectors),
        nuisance,
    )
    np.testing.assert_allclose(
        array_prediction.prediction_vector,
        prediction.prediction_vector,
    )
    np.testing.assert_allclose(array_prediction.jacobian, prediction.jacobian)
    np.testing.assert_allclose(
        prediction.conditional_variance_electrons2[2],
        (prediction.expected_electrons[2] + 0.7**2) / 2.0,
    )
    np.testing.assert_allclose(
        prediction.conditional_variance_electrons2[3],
        (prediction.expected_electrons[3] + 0.7**2) / 3.0,
    )


def test_linked_dgi_roles_match_shared_forward_image_and_appear_once() -> None:
    grid, model = _grid_and_model()
    transfer = DGITransferContract(4.0)
    operator = DGILinkedRawOperator(
        grid=grid,
        detector=DetectorContract(160.0, 0.7),
        response=_response(),
        transfer=transfer,
        independent_exposures_by_role={
            "atom_stop": 1,
            "leakage_stop": 2,
            "stop_dark": 3,
            "open_reference": 4,
            "open_dark": 5,
        },
        jacobian_batch_size=4,
    )
    vectors = [_parameters(model, 0.7), _parameters(model, 1.1)]
    nuisance = DGINuisanceValues(
        160.0,
        stop_dark_electrons_per_pixel=0.5,
        open_dark_electrons_per_pixel=0.25,
        open_to_stop_scale=0.9,
    )
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        vectors,
        nuisance,
    )

    assert prediction.role_names == (
        "atom_stop_000",
        "atom_stop_001",
        "leakage_stop",
        "stop_dark",
        "open_reference",
        "open_dark",
    )
    assert prediction.role_frame_indices == (0, 1, None, None, None, None)
    assert prediction.prediction_vector.size == 6 * grid.roi_pixel_count
    assert prediction.jacobian.shape == (
        6 * grid.roi_pixel_count,
        2 * model.parameter_count + 4,
    )
    density = model.column_density(vectors[1])
    object_field = np.exp(
        _response().complex_exponent_per_column_density_m2 * density
    )
    expected_image = grid.camera_average(
        simulate_dgi_object_field(
            object_field,
            grid.pupil,
            transfer.stop_optical_depth,
        )
    )
    np.testing.assert_allclose(
        prediction.expected_electrons[1],
        nuisance.i0_photoelectrons_per_pixel
        * nuisance.open_to_stop_scale
        * expected_image
        + nuisance.stop_dark_electrons_per_pixel,
        rtol=2e-15,
        atol=1e-13,
    )
    assert np.unique(prediction.expected_electrons[2]).size == 1
    assert np.unique(prediction.expected_electrons[3]).size == 1


@pytest.mark.parametrize("method", ["pci", "dgi"])
def test_linked_raw_density_and_nuisance_jacobian_matches_central_difference(
    method: str,
) -> None:
    grid, model = _grid_and_model()
    vector = _parameters(model, 0.9)
    if method == "pci":
        operator = PCILinkedRawOperator(
            grid=grid,
            detector=DetectorContract(130.0, 0.7),
            response=_response(),
            transfer=PCITransferContract(0.95, np.pi / 2.0),
            independent_exposures_by_role={
                "atom": 1,
                "bright_reference": 1,
                "dark": 1,
            },
            jacobian_batch_size=3,
        )
        nuisance = PCINuisanceValues(130.0, 0.3)
        nuisance_plus = PCINuisanceValues(130.0 + 1e-3, 0.3)
        nuisance_minus = PCINuisanceValues(130.0 - 1e-3, 0.3)
    else:
        operator = DGILinkedRawOperator(
            grid=grid,
            detector=DetectorContract(130.0, 0.7),
            response=_response(),
            transfer=DGITransferContract(4.0),
            independent_exposures_by_role={
                "atom_stop": 1,
                "leakage_stop": 1,
                "stop_dark": 1,
                "open_reference": 1,
                "open_dark": 1,
            },
            jacobian_batch_size=3,
        )
        nuisance = DGINuisanceValues(130.0, 0.3, 0.2, 0.9)
        nuisance_plus = DGINuisanceValues(130.0 + 1e-3, 0.3, 0.2, 0.9)
        nuisance_minus = DGINuisanceValues(130.0 - 1e-3, 0.3, 0.2, 0.9)
    prediction = operator.expected_linked_sequence_and_jacobian_model(
        model,
        [vector],
        nuisance,
    )

    density_step = 1e-5
    plus_vector = vector.copy()
    minus_vector = vector.copy()
    plus_vector[4] += density_step
    minus_vector[4] -= density_step
    finite_density = (
        operator.expected_linked_sequence_and_jacobian_model(
            model,
            [plus_vector],
            nuisance,
        ).prediction_vector
        - operator.expected_linked_sequence_and_jacobian_model(
            model,
            [minus_vector],
            nuisance,
        ).prediction_vector
    ) / (2.0 * density_step)
    np.testing.assert_allclose(
        prediction.jacobian[:, 4],
        finite_density,
        rtol=5e-6,
        atol=2e-7,
    )

    nuisance_step = 1e-3
    finite_i0 = (
        operator.expected_linked_sequence_and_jacobian_model(
            model,
            [vector],
            nuisance_plus,
        ).prediction_vector
        - operator.expected_linked_sequence_and_jacobian_model(
            model,
            [vector],
            nuisance_minus,
        ).prediction_vector
    ) / (2.0 * nuisance_step)
    np.testing.assert_allclose(
        prediction.jacobian[:, model.parameter_count],
        finite_i0,
        rtol=2e-10,
        atol=2e-10,
    )


def test_linked_raw_contract_rejects_invalid_inputs() -> None:
    grid, _ = _grid_and_model()
    with pytest.raises(ValueError, match="roles do not match"):
        PCILinkedRawOperator(
            grid=grid,
            detector=DetectorContract(100.0, 0.7),
            response=_response(),
            transfer=PCITransferContract(0.95, np.pi / 2.0),
            independent_exposures_by_role={"atom": 1},
        )
    with pytest.raises(ValueError, match="non-negative"):
        PCINuisanceValues(100.0, -0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PCITransferContract(1.1, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        DGITransferContract(-1.0)


def test_linked_raw_contract_is_exported_from_reconstruction_package() -> None:
    assert reconstruction.PCILinkedRawOperator is PCILinkedRawOperator
    assert reconstruction.DGILinkedRawOperator is DGILinkedRawOperator
    assert (
        reconstruction.ScalarOpticalResponseContract
        is ScalarOpticalResponseContract
    )
