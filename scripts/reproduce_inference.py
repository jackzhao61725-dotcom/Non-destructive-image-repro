"""Run fixed-seed PCI endpoint fits without exposing generator truth to the fit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from non_destructive_image.reconstruction.free_radius_model import (
    FreeRadiusCompactDensityModel,
)
from non_destructive_image.reconstruction.independent_endpoint_information import (
    IndependentEndpointRawBlock,
    RAW_ROLE_NAMES,
)
from non_destructive_image.reconstruction.linked_scalar_fit import (
    LinkedScalarFitOptions,
    estimate_linked_nuisance_from_references,
)
from non_destructive_image.reconstruction.parameters import SmoothTFParameters, to_internal
from non_destructive_image.reconstruction.parametric_orientation import (
    ParametricEndpointFitInput,
    ParametricOrientationProvenance,
    fit_independent_parametric_pci_endpoints,
)
from non_destructive_image.reconstruction.scalar_measurements import PCINuisanceValues

try:  # Package import in tests; direct import when executed as a file.
    from scripts._common import (
        REPOSITORY_ROOT,
        endpoint_products,
        input_identity,
        load_configs,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI acceptance
    from _common import (
        REPOSITORY_ROOT,
        endpoint_products,
        input_identity,
        load_configs,
        write_json,
    )


ENDPOINT_LABELS = ("B_parallel_y", "B_parallel_z")
FIELD_ORIENTATIONS = ("y", "z")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=REPOSITORY_ROOT / "configs"
    )
    parser.add_argument(
        "--output", type=Path, default=REPOSITORY_ROOT / "outputs" / "inference.json"
    )
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _seeded_rng(
    reproduction: Mapping[str, Any],
    *,
    fluence_index: int,
    draw_id: int,
    endpoint_index: int,
    role_index: int,
) -> np.random.Generator:
    seed = np.random.SeedSequence(
        int(reproduction["stochastic"]["master_seed_uint64"]),
        spawn_key=(fluence_index, draw_id, endpoint_index, role_index),
    )
    return np.random.Generator(np.random.PCG64DXSM(seed))


def _raw_block(
    reproduction: Mapping[str, Any],
    *,
    expected: Sequence[np.ndarray],
    read_noise_electrons: float,
    fluence_index: int,
    draw_id: int,
    endpoint_index: int,
) -> IndependentEndpointRawBlock:
    observed: list[np.ndarray] = []
    owners: list[str] = []
    for role_index, (role_name, mean) in enumerate(
        zip(RAW_ROLE_NAMES, expected, strict=True)
    ):
        rng = _seeded_rng(
            reproduction,
            fluence_index=fluence_index,
            draw_id=draw_id,
            endpoint_index=endpoint_index,
            role_index=role_index,
        )
        values = rng.poisson(mean) + rng.normal(0.0, read_noise_electrons, mean.shape)
        observed.append(np.asarray(values, dtype=float))
        owners.append(
            f"draw{draw_id:03d}:{ENDPOINT_LABELS[endpoint_index]}:{role_name}"
        )
    return IndependentEndpointRawBlock(
        endpoint_label=ENDPOINT_LABELS[endpoint_index],
        field_orientation=FIELD_ORIENTATIONS[endpoint_index],
        role_names=RAW_ROLE_NAMES,
        role_owner_ids=tuple(owners),
        observed_electrons=tuple(observed),
    )


def _physical_parameters(record: Mapping[str, Any]) -> SmoothTFParameters:
    return SmoothTFParameters(
        column_density_peak_m2=float(record["column_density_peak_m2"]),
        y0_um=float(record["centre_y_um"]),
        z0_um=float(record["centre_z_um"]),
        radius_y_um=float(record["radius_y_um"]),
        radius_z_um=float(record["radius_z_um"]),
    )


def _parameter_contract(
    reproduction: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    estimator = reproduction["estimator"]
    starts = estimator["starts"]
    start_ids = tuple(str(record["id"]) for record in starts)
    vectors = tuple(to_internal(_physical_parameters(record)) for record in starts)
    bounds = estimator["bounds"]
    lower = to_internal(
        SmoothTFParameters(
            float(bounds["column_density_peak_m2"][0]),
            float(bounds["centre_y_um"][0]),
            float(bounds["centre_z_um"][0]),
            float(bounds["radius_y_um"][0]),
            float(bounds["radius_z_um"][0]),
        )
    )
    upper = to_internal(
        SmoothTFParameters(
            float(bounds["column_density_peak_m2"][1]),
            float(bounds["centre_y_um"][1]),
            float(bounds["centre_z_um"][1]),
            float(bounds["radius_y_um"][1]),
            float(bounds["radius_z_um"][1]),
        )
    )
    return start_ids, vectors, lower, upper


def _options(reproduction: Mapping[str, Any]) -> LinkedScalarFitOptions:
    solver = reproduction["estimator"]["solver"]
    return LinkedScalarFitOptions(
        method=str(solver["method"]),
        loss=str(solver["loss"]),
        x_scale=str(solver["x_scale"]),
        irls_iterations=int(solver["irls_iterations"]),
        max_nfev=int(solver["maximum_function_evaluations"]),
        xtol=float(solver["xtol"]),
        ftol=float(solver["ftol"]),
        gtol=float(solver["gtol"]),
        trust_region_solver=str(solver["trust_region_solver"]),
    )


def _fit_inputs(
    reproduction: Mapping[str, Any],
    *,
    products: Sequence[Any],
    blocks: Sequence[IndependentEndpointRawBlock],
) -> tuple[ParametricEndpointFitInput, ParametricEndpointFitInput]:
    start_ids, starts, lower, upper = _parameter_contract(reproduction)
    count_scale = float(
        reproduction["acquisition"]["photoelectrons_per_i0_pixel_at_300_mw_us"]
    )
    dark = float(reproduction["acquisition"]["dark_expected_electrons_per_pixel"])
    declared = PCINuisanceValues(count_scale, dark)
    nuisance_bounds = reproduction["estimator"]["nuisance_bounds"]
    nuisance_lower = np.asarray(
        [
            count_scale
            * float(nuisance_bounds["incident_intensity_relative_to_declared"][0]),
            float(nuisance_bounds["dark_electrons_per_pixel"][0]),
        ],
        dtype=float,
    )
    nuisance_upper = np.asarray(
        [
            count_scale
            * float(nuisance_bounds["incident_intensity_relative_to_declared"][1]),
            float(nuisance_bounds["dark_electrons_per_pixel"][1]),
        ],
        dtype=float,
    )
    inputs: list[ParametricEndpointFitInput] = []
    for product, block in zip(products, blocks, strict=True):
        estimated = estimate_linked_nuisance_from_references(
            product.inverse_operator, block.as_linked_observation()
        )
        if not isinstance(estimated, PCINuisanceValues):
            raise TypeError("PCI nuisance estimator returned the wrong type")
        margin = 1e-8 * np.maximum(nuisance_upper - nuisance_lower, 1.0)
        nuisance = PCINuisanceValues(
            float(
                np.clip(
                    estimated.i0_photoelectrons_per_pixel,
                    nuisance_lower[0] + margin[0],
                    nuisance_upper[0] - margin[0],
                )
            ),
            float(
                np.clip(
                    estimated.dark_electrons_per_pixel,
                    nuisance_lower[1] + margin[1],
                    nuisance_upper[1] - margin[1],
                )
            ),
        )
        if declared.i0_photoelectrons_per_pixel <= 0.0:
            raise ValueError("declared incident count must be positive")
        inputs.append(
            ParametricEndpointFitInput(
                operator=product.inverse_operator,
                model=FreeRadiusCompactDensityModel.from_grid(
                    y_grid_m=product.inverse_grid.y_grid_m,
                    z_grid_m=product.inverse_grid.z_grid_m,
                    profile_exponent=float(reproduction["estimator"]["profile_exponent"]),
                ),
                raw_block=block,
                start_ids=start_ids,
                initial_parameter_vectors=starts,
                parameter_lower=lower,
                parameter_upper=upper,
                initial_nuisance=nuisance,
                nuisance_lower=nuisance_lower,
                nuisance_upper=nuisance_upper,
                options=_options(reproduction),
            )
        )
    return inputs[0], inputs[1]


def _provenance() -> ParametricOrientationProvenance:
    return ParametricOrientationProvenance(
        contract_label="chapter_5_orientation_information_contract_v2",
        endpoint_labels=ENDPOINT_LABELS,
        field_orientations=FIELD_ORIENTATIONS,
        imaging_axis="x",
        independent_preparations=True,
        independent_raw_blocks=True,
        temporal_coupling_used=False,
        generator_reference_used=False,
    )


def _truth(product: Any) -> dict[str, float]:
    radii_um = np.asarray(product.state.radii_m, dtype=float) * 1e6
    return {
        "sigma_y_um": float(radii_um[1] / np.sqrt(7.0)),
        "sigma_z_um": float(radii_um[2] / np.sqrt(7.0)),
    }


def _endpoint_record(endpoint: Any, truth: Mapping[str, float]) -> dict[str, Any]:
    observables = None if endpoint.observables is None else asdict(endpoint.observables)
    parameters = (
        None if endpoint.physical_parameters is None else endpoint.physical_parameters.as_dict()
    )
    errors = None
    if observables is not None:
        errors = {
            name: {
                "signed_um": float(observables[name]) - float(truth[name]),
                "fractional_percent": 100.0
                * (float(observables[name]) / float(truth[name]) - 1.0),
            }
            for name in ("sigma_y_um", "sigma_z_um")
        }
    return {
        "endpoint_label": endpoint.endpoint_label,
        "status": endpoint.status,
        "message": endpoint.message,
        "selected_start_id": endpoint.selected_start_id,
        "physical_parameters": parameters,
        "observables": observables,
        "truth_consumed_after_fit": dict(truth),
        "errors": errors,
        "start_terminals": [
            {
                "start_id": item.start_id,
                "status": item.status,
                "weighted_chi_square": item.weighted_chi_square,
                "message": item.message,
                "function_evaluations": (
                    None
                    if item.fit_result is None
                    else item.fit_result.diagnostics.nfev
                ),
            }
            for item in endpoint.start_results
        ],
    }


def reproduce(config_dir: Path, *, draws: int) -> dict[str, Any]:
    """Run the requested fixed-seed point-fit pairs and return a JSON payload."""

    if isinstance(draws, bool) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    model, reference, reproduction = load_configs(config_dir)
    products = endpoint_products(model, reference, reproduction)
    nuisance = PCINuisanceValues(
        float(reproduction["acquisition"]["photoelectrons_per_i0_pixel_at_300_mw_us"]),
        float(reproduction["acquisition"]["dark_expected_electrons_per_pixel"]),
    )
    expected_by_endpoint = []
    for product in products:
        names, expected = product.canonical_operator.expected_linked_sequence_from_density_maps(
            [product.canonical_density_m2], nuisance
        )
        if names != RAW_ROLE_NAMES:
            raise RuntimeError("PCI raw-role order changed")
        expected_by_endpoint.append(expected)
    fluences = tuple(float(value) for value in reproduction["probe"]["fluence_scan_mw_us"])
    reference_fluence = float(reproduction["probe"]["reference_fluence_mw_us"])
    fluence_index = fluences.index(reference_fluence)
    truths = tuple(_truth(product) for product in products)
    draw_rows: list[dict[str, Any]] = []
    for draw_id in range(draws):
        blocks = tuple(
            _raw_block(
                reproduction,
                expected=expected_by_endpoint[index],
                read_noise_electrons=products[index].canonical_operator.read_noise_electrons,
                fluence_index=fluence_index,
                draw_id=draw_id,
                endpoint_index=index,
            )
            for index in range(2)
        )
        inputs = _fit_inputs(reproduction, products=products, blocks=blocks)
        fit = fit_independent_parametric_pci_endpoints(
            inputs, provenance=_provenance()
        )
        endpoints = [
            _endpoint_record(endpoint, truths[index])
            for index, endpoint in enumerate(fit.endpoints)
        ]
        contrasts: dict[str, Any] = {}
        for name in ("sigma_y_um", "sigma_z_um"):
            true_value = truths[1][name] - truths[0][name]
            if any(item["observables"] is None for item in endpoints):
                estimated = None
            else:
                estimated = (
                    float(endpoints[1]["observables"][name])
                    - float(endpoints[0]["observables"][name])
                )
            contrasts[name] = {
                "truth_um": true_value,
                "estimate_um": estimated,
                "sign_correct": (
                    None if estimated is None else bool(np.sign(estimated) == np.sign(true_value))
                ),
            }
        draw_rows.append(
            {"draw_id": draw_id, "endpoints": endpoints, "contrasts": contrasts}
        )
    return {
        "schema": "equilibrium_imaging_pci_point_fit_reproduction_v1",
        "status": "fixed_seed_point_fits_not_sampling_coverage",
        "identity": input_identity(config_dir),
        "source_commit": model["source"]["commit"],
        "fluence_mw_us": reference_fluence,
        "draw_count": draws,
        "seed_policy": reproduction["stochastic"],
        "generator_truth_used_by_fit": False,
        "draws": draw_rows,
        "limitations": [
            "point fits do not reproduce the dissertation's 64-draw interval",
            "fit uncertainty is conditional on the projected Thomas-Fermi family",
            "endpoints are independent equilibrium states",
        ],
    }


def main() -> None:
    args = _arguments()
    if args.draws <= 0:
        raise SystemExit("--draws must be positive")
    if args.validate_only:
        model, reference, reproduction = load_configs(args.config_dir)
        endpoint_products(model, reference, reproduction)
        _parameter_contract(reproduction)
        _options(reproduction)
        print("configuration and inference contract validated")
        return
    payload = reproduce(args.config_dir, draws=args.draws)
    write_json(args.output, payload)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
