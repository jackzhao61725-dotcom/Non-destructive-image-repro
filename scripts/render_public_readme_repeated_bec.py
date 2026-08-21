"""Render the README preview of admitted repeated-BEC recovery results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "chapter_5_bec_multiframe_eta_ry_dpfi_v1"
MANIFEST_SHA256 = "1946317254a08052d992260a45740f44de5f0d5359b1d4e706c55375f0017759"
TABLE = SOURCE / "presentation" / "figure_5_2_plotted_values.csv"
DEFAULT_OUTPUT = ROOT / "output" / "public" / "readme_repeated_bec.png"
DURATIONS = (25, 200, 400)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows() -> list[dict[str, str]]:
    manifest_path = SOURCE / "artifact_manifest.json"
    if _sha256(manifest_path) != MANIFEST_SHA256:
        raise ValueError("repeated-BEC evidence identity changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = TABLE.relative_to(SOURCE).as_posix()
    record = next(
        (item for item in manifest["artifacts"] if item["path"] == relative),
        None,
    )
    if record is None:
        raise ValueError("repeated-BEC plotted-values table is not manifested")
    if TABLE.stat().st_size != int(record["bytes"]) or _sha256(TABLE) != record["sha256"]:
        raise ValueError("repeated-BEC plotted-values table identity changed")
    with TABLE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if int(row["duration_us"]) in DURATIONS]
    if len(selected) != len(DURATIONS) * 15 * 2:
        raise ValueError("unexpected repeated-BEC README inventory")
    return selected


def render(output: Path = DEFAULT_OUTPUT) -> Path:
    """Render a compact two-panel PNG from the admitted plotted values."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    rows = _load_rows()
    lookup = {
        (int(row["duration_us"]), int(row["image_q"]), row["observable"]): row
        for row in rows
    }
    colours = {25: "#0072B2", 200: "#D55E00", 400: "#009E73"}
    markers = {25: "o", 200: "s", 400: "D"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    images = np.arange(1, 16, dtype=int)
    for axis, observable, label in (
        (axes[0], "eta", r"Remaining condensate fraction, $\eta_q$"),
        (axes[1], "rho_y", r"Axial-radius ratio, $\rho_{y,q}$"),
    ):
        for duration in DURATIONS:
            series = [lookup[(duration, int(image), observable)] for image in images]
            truth = np.asarray([float(row["truth"]) for row in series])
            median = np.asarray([float(row["median"]) for row in series])
            q16 = np.asarray([float(row["q16"]) for row in series])
            q84 = np.asarray([float(row["q84"]) for row in series])
            colour = colours[duration]
            axis.plot(images, truth, color=colour, linewidth=1.7)
            axis.fill_between(images, q16, q84, color=colour, alpha=0.15, linewidth=0)
            axis.plot(
                images,
                median,
                linestyle="none",
                marker=markers[duration],
                markersize=4.4,
                markerfacecolor="white",
                markeredgecolor=colour,
                markeredgewidth=1.2,
                label=rf"${duration}\,\mu\mathrm{{s}}$",
            )
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#D7DCE2", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    axes[1].set_xlabel(r"Image number, $q$")
    axes[1].set_xticks((1, 3, 5, 7, 9, 11, 13, 15))
    figure.suptitle("DPFI recovery across repeated exposures", fontsize=12.0, weight="bold")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    print(render(arguments.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
