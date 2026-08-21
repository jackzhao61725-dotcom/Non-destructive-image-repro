"""Render first-image recovery against the first-pulse condensate reduction."""

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


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _render(config: dict[str, object], recovery_rows: list[dict[str, str]], endpoint_rows: list[dict[str, str]]) -> Path:
    durations = tuple(int(value) for value in config["durations_us"])
    recovery = {
        (int(row["duration_us"]), row["observable"]): float(row["q84"]) - float(row["q16"])
        for row in recovery_rows
        if int(row["image_q"]) == 1 and row["observable"] in {"eta", "rho_y"}
    }
    expected = {(duration, observable) for duration in durations for observable in ("eta", "rho_y")}
    if set(recovery) != expected:
        raise ValueError(f"unexpected q=1 recovery inventory: {sorted(set(recovery) ^ expected)}")
    if not all(math.isfinite(value) and value > 0 for value in recovery.values()):
        raise ValueError("recovery spans must be finite and positive")

    losses: dict[int, float] = {}
    for duration in durations:
        values = {
            float(row["truth_eta"])
            for row in endpoint_rows
            if int(row["duration_us"]) == duration and int(row["image_q"]) == 2
        }
        if len(values) != 1:
            raise ValueError(f"duration {duration} has {len(values)} q=2 truth values")
        losses[duration] = 100.0 * (1.0 - values.pop())

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.15, 4.25))
    styles = {
        "eta": {"label": r"$\Delta_{68}(\eta_1)$", "color": "#0B78B5", "marker": "o", "linestyle": "-"},
        "rho_y": {"label": r"$\Delta_{68}(\rho_{y,1})$", "color": "#D55E00", "marker": "s", "linestyle": "--"},
    }
    x = np.asarray([losses[duration] for duration in durations])
    for observable in ("eta", "rho_y"):
        style = styles[observable]
        y = np.asarray([recovery[(duration, observable)] for duration in durations])
        axis.plot(
            x,
            y,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.45,
            markersize=5.1,
            markerfacecolor="white",
            markeredgewidth=1.1,
        )
    axis.set_xlabel("Reduction in condensed population after the first pulse (%)")
    axis.set_ylabel(r"$\Delta_{68}(O)$")
    axis.set_xlim(0, 5)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#D7DCE0", linewidth=0.65)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper right", ncol=2, handlelength=2.0)

    top = axis.twiny()
    top.set_xlim(axis.get_xlim())
    top.set_xticks(x, [str(duration) for duration in durations])
    top.set_xlabel(r"Pulse duration $\tau\;(\mu\mathrm{s})$")
    top.spines["right"].set_visible(False)

    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.80)
    output = ROOT / str(config["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"Creator": "Non-destructive-image", "CreationDate": None, "ModDate": None}
    figure.savefig(output / "figure_5_3_dpfi_recovery_spread_vs_first_image_loss.pdf", bbox_inches="tight", metadata=metadata)
    figure.savefig(output / "figure_5_3_dpfi_recovery_spread_vs_first_image_loss.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/chapter_5_bec_multiframe_recovery_loss_presentation_v1.json",
    )
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    loaded: dict[str, tuple[Path, list[dict[str, str]]]] = {}
    for name, spec in config["inputs"].items():
        path = ROOT / spec["path"]
        actual = _sha256(path)
        if actual != spec["sha256"]:
            raise ValueError(f"{name} hash mismatch: {actual}")
        loaded[name] = (path, _rows(path))
    output = _render(config, loaded["recovery"][1], loaded["endpoints"][1])
    print(output)


if __name__ == "__main__":
    main()
