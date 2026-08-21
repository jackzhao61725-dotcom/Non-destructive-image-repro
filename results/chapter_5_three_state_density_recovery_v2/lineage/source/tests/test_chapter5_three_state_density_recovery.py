from __future__ import annotations

import numpy as np

from non_destructive_image.chapter5_three_state_density_recovery import summarise


def test_summary_uses_all_finite_selected_fits() -> None:
    rows = [
        {
            "method": "dpfi",
            "state": "connected_modulated",
            "display_label": "SSP",
            "duration_us": "25",
            "draw_id": str(draw),
            "eta_hat": str(value),
            "d_peak_hat_um": str(2.7 + value / 100.0),
            "d_peak_truth_um": "2.75",
            "nu_vp_hat": str(0.6 + value / 100.0),
            "nu_vp_truth": "0.65",
        }
        for draw, value in enumerate((0.8, 0.9, 1.0, 1.1, 1.2))
    ]
    config = {
        "observables": [
            {"name": "eta", "column": "eta_hat", "truth": 1.0, "unit": "ratio"},
            {"name": "d_peak", "column": "d_peak_hat_um", "truth_column": "d_peak_truth_um", "unit": "um"},
            {"name": "nu_vp", "column": "nu_vp_hat", "truth_column": "nu_vp_truth", "unit": "ratio"},
        ]
    }
    summary = summarise(rows, config)
    assert len(summary) == 3
    eta = next(row for row in summary if row["observable"] == "eta")
    assert eta["draw_count"] == 5
    assert eta["median"] == 1.0
    assert np.isclose(eta["q16"], np.quantile([0.8, 0.9, 1.0, 1.1, 1.2], 0.16))
    assert eta["truth"] == 1.0
