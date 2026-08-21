import numpy as np

from scripts.run_public_example import run_example


def test_public_example_recomputes_static_route_and_one_fit() -> None:
    result = run_example()

    assert result["status"] == "public_example_complete"
    assert [row["state"] for row in result["static_three_state_route"]] == [
        "smooth_bec",
        "connected_modulated",
        "separated_droplets",
    ]
    fit = result["one_bec_fit"]
    assert fit["method"] == "DPFI"
    assert np.isclose(fit["estimate"]["eta"], 1.0169854776784302, atol=1e-9)
    assert np.isclose(fit["estimate"]["rho_y"], 0.9990153746851956, atol=1e-9)
    assert np.isclose(fit["estimate"]["y0_um"], 0.13129908705261648, atol=1e-9)
    assert len(result["repeated_bec_route"]) == 4
