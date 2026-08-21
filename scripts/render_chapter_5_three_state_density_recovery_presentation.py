"""Render the morphology-only presentation of admitted Section 5.3 summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _render(rows: list[dict[str, str]], output: Path) -> None:
    methods = ("dpfi", "dgi")
    states = (("connected_modulated", "SSP"), ("separated_droplets", "ID"))
    observables = (
        ("d_peak", r"$\overline{d}_{\mathrm{pk}}\;(\mu\mathrm{m})$"),
        ("nu_vp", r"$\nu_{\mathrm{vp}}$"),
    )
    durations = (25, 50, 100, 200, 400)
    styles = {
        "dpfi": {"label": "DPFI", "color": "#326C9B", "marker": "s"},
        "dgi": {"label": "DGI", "color": "#D17A22", "marker": "o"},
    }
    lookup = {
        (row["method"], row["state"], row["observable"], int(row["duration_us"])): row
        for row in rows
    }
    expected = {
        (method, state, observable, duration)
        for method in methods
        for state, _ in states
        for observable, _ in observables
        for duration in durations
    }
    if set(lookup) != expected:
        missing = sorted(expected - set(lookup))
        extra = sorted(set(lookup) - expected)
        raise ValueError(f"unexpected recovery inventory: missing={missing}, extra={extra}")
    for row in rows:
        values = [float(row[key]) for key in ("truth", "median", "q16", "q84")]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite summary row: {row}")
        if int(row["draw_count"]) != 64 or not values[2] <= values[1] <= values[3]:
            raise ValueError(f"invalid summary row: {row}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.25), sharex=True)
    positions = np.arange(len(durations), dtype=float)
    for column, (state, title) in enumerate(states):
        axes[0, column].set_title(title, pad=8, fontweight="bold")
        for row_index, (observable, ylabel) in enumerate(observables):
            axis = axes[row_index, column]
            truth = float(lookup[("dpfi", state, observable, durations[0])]["truth"])
            axis.axhline(truth, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
            for method in methods:
                cells = [lookup[(method, state, observable, duration)] for duration in durations]
                medians = np.asarray([float(cell["median"]) for cell in cells])
                q16 = np.asarray([float(cell["q16"]) for cell in cells])
                q84 = np.asarray([float(cell["q84"]) for cell in cells])
                style = styles[method]
                axis.fill_between(
                    positions,
                    q16,
                    q84,
                    color=style["color"],
                    alpha=0.16 if method == "dpfi" else 0.13,
                    linewidth=0,
                    zorder=2,
                )
                axis.plot(
                    positions,
                    medians,
                    color=style["color"],
                    marker=style["marker"],
                    markersize=4.8,
                    markerfacecolor="white" if method == "dgi" else style["color"],
                    markeredgewidth=1.0,
                    linewidth=1.25,
                    label=style["label"],
                    zorder=3,
                )
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_xlim(-0.25, len(durations) - 0.75)
            if row_index == 1:
                axis.set_xticks(positions, [str(value) for value in durations])
                axis.set_xlabel(r"Pulse duration $\tau\;(\mu\mathrm{s})$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    truth_handle = plt.Line2D([0], [0], color="#333333", linewidth=1.0, linestyle="--")
    figure.legend(
        handles + [truth_handle],
        labels + ["Input"],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.997),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
    )
    figure.subplots_adjust(left=0.105, right=0.985, bottom=0.10, top=0.91, hspace=0.25, wspace=0.28)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"Creator": "Non-destructive-image", "CreationDate": None, "ModDate": None}
    figure.savefig(output / "figure_5_5_three_state_density_recovery.pdf", bbox_inches="tight", metadata=metadata)
    figure.savefig(output / "figure_5_5_three_state_density_recovery.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/chapter_5_three_state_density_recovery_presentation_v1.json",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = ROOT / config["input"]["path"]
    actual_hash = _sha256(source)
    if actual_hash != config["input"]["sha256"]:
        raise ValueError(f"input hash mismatch: {actual_hash}")
    rows = [
        row
        for row in _load_rows(source)
        if row["observable"] in set(config["observables"])
    ]
    output = ROOT / config["output_directory"]
    _render(rows, output)
    (output / "presentation.json").write_text(
        json.dumps(
            {
                "label": config["label"],
                "input_sha256": actual_hash,
                "row_count": len(rows),
                "observables": config["observables"],
                "new_random_draws": 0,
                "new_optical_propagations": 0,
                "new_fits": 0,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
