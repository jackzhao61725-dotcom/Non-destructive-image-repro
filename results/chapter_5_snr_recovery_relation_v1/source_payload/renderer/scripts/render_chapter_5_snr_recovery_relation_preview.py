"""Render the Section 5.4 SNR--recovery relation from existing summaries only.

This presentation-only script performs no random draw, optical propagation,
equilibrium solve or nonlinear fit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dissertation_figure_style import BLUE, GREY, INK, LIGHT_GREY, ORANGE, dissertation_style


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_inputs(config: dict[str, object]) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for name, identity in config["inputs"].items():
        path = ROOT / identity["path"]
        if _sha256(path) != identity["sha256"]:
            raise ValueError(f"input identity changed: {path}")
        output[name] = _read_csv(path)
    return output


def _build_joined_rows(
    config: dict[str, object], inputs: dict[str, list[dict[str, str]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    bec_durations = tuple(int(value) for value in config["bec_durations_us"])
    bec_snr = {
        int(row["pulse_duration_us"]): float(row["whole_image_template_snr"])
        for row in inputs["bec_snr"]
        if int(row["image_q"]) == 1 and row["method"] == "dpfi"
    }
    for row in inputs["bec_recovery"]:
        duration = int(row["duration_us"])
        if int(row["image_q"]) != 1 or duration not in bec_durations:
            continue
        if row["observable"] not in {"eta", "rho_y"}:
            continue
        rows.append(
            {
                "family": "BEC",
                "state": "BEC",
                "method": "dpfi",
                "observable": row["observable"],
                "unit": "ratio",
                "duration_us": duration,
                "whole_image_snr": bec_snr[duration],
                "q16": float(row["q16"]),
                "q84": float(row["q84"]),
                "recovery_width": float(row["q84"]) - float(row["q16"]),
                "draw_count": int(row["predeclared_sequence_count"]),
            }
        )

    three_durations = tuple(int(value) for value in config["three_state_durations_us"])
    three_snr: dict[tuple[str, str, int], float] = {}
    for method in ("dpfi", "dgi"):
        for row in inputs[f"{method}_three_state_snr"]:
            key = (method, row["state"], int(row["duration_us"]))
            if key in three_snr:
                raise ValueError(f"duplicate three-state SNR key: {key}")
            three_snr[key] = float(row["whole_image_snr"])
    for row in inputs["three_state_recovery"]:
        duration = int(row["duration_us"])
        if duration not in three_durations:
            continue
        method = row["method"]
        state = row["state"]
        rows.append(
            {
                "family": "three_state",
                "state": row["display_label"],
                "method": method,
                "observable": row["observable"],
                "unit": row["unit"],
                "duration_us": duration,
                "whole_image_snr": three_snr[(method, state, duration)],
                "q16": float(row["q16"]),
                "q84": float(row["q84"]),
                "recovery_width": float(row["q84"]) - float(row["q16"]),
                "draw_count": int(row["draw_count"]),
            }
        )

    expected = 2 * len(bec_durations) + 2 * 2 * 3 * len(three_durations)
    keys = {
        (
            row["family"],
            row["state"],
            row["method"],
            row["observable"],
            row["duration_us"],
        )
        for row in rows
    }
    if len(rows) != expected or len(keys) != expected:
        raise ValueError(f"joined inventory changed: {len(rows)} rows, expected {expected}")
    for row in rows:
        values = (row["whole_image_snr"], row["q16"], row["q84"], row["recovery_width"])
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite joined value: {row}")
        if float(row["whole_image_snr"]) <= 0 or float(row["recovery_width"]) <= 0:
            raise ValueError(f"non-positive plotted value: {row}")
        if int(row["draw_count"]) != 64:
            raise ValueError(f"draw inventory changed: {row}")
    return sorted(
        rows,
        key=lambda row: (
            str(row["family"]),
            str(row["observable"]),
            str(row["method"]),
            str(row["state"]),
            int(row["duration_us"]),
        ),
    )


def _series(
    rows: list[dict[str, object]],
    *,
    family: str,
    observable: str,
    method: str | None = None,
    state: str | None = None,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if row["family"] == family
        and row["observable"] == observable
        and (method is None or row["method"] == method)
        and (state is None or row["state"] == state)
    ]
    return sorted(selected, key=lambda row: int(row["duration_us"]))


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, alpha=0.9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=4.0, width=0.9)


def _plot_relation(rows: list[dict[str, object]], output: Path) -> None:
    method_style = {
        "dpfi": {"label": "DPFI", "color": BLUE, "linestyle": "-"},
        "dgi": {"label": "DGI", "color": ORANGE, "linestyle": "--"},
    }
    state_style = {
        "SSP": {"marker": "s", "label": "SSP"},
        "ID": {"marker": "o", "label": "ID"},
    }
    with dissertation_style(svg_hashsalt="chapter-5-snr-recovery-relation-v1"):
        figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.75), sharex=True)
        axis_a, axis_b, axis_c, axis_d = axes.flat

        bec_styles = {
            "eta": {"label": r"$\eta_1$", "marker": "s", "linestyle": "-", "fill": BLUE},
            "rho_y": {
                "label": r"$\rho_{y,1}$",
                "marker": "o",
                "linestyle": "--",
                "fill": "white",
            },
        }
        for observable, style in bec_styles.items():
            cells = _series(rows, family="BEC", observable=observable)
            axis_a.plot(
                [float(cell["whole_image_snr"]) for cell in cells],
                [float(cell["recovery_width"]) for cell in cells],
                color=BLUE,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=style["fill"],
                markeredgecolor=BLUE,
                markeredgewidth=1.1,
                markersize=5.3,
                linewidth=1.55,
                label=style["label"],
            )
        axis_a.set_title(r"(a) BEC: DPFI", loc="left", fontweight="bold", pad=7)
        axis_a.set_ylabel(r"$\Delta_{68}(O)$")
        axis_a.legend(frameon=False, loc="upper right", handlelength=2.2)

        panel_specs = (
            (axis_b, "eta", r"(b) SSP/ID population", r"$\Delta_{68}(\eta_s)$"),
            (
                axis_c,
                "d_peak",
                r"(c) SSP/ID peak separation",
                r"$\Delta_{68}(\overline{d}_{\mathrm{pk}})\;(\mu\mathrm{m})$",
            ),
            (axis_d, "nu_vp", r"(d) SSP/ID valley ratio", r"$\Delta_{68}(\nu_{\mathrm{vp}})$"),
        )
        for axis, observable, title, ylabel in panel_specs:
            for method in ("dpfi", "dgi"):
                for state in ("SSP", "ID"):
                    cells = _series(
                        rows,
                        family="three_state",
                        observable=observable,
                        method=method,
                        state=state,
                    )
                    axis.plot(
                        [float(cell["whole_image_snr"]) for cell in cells],
                        [float(cell["recovery_width"]) for cell in cells],
                        color=method_style[method]["color"],
                        linestyle=method_style[method]["linestyle"],
                        marker=state_style[state]["marker"],
                        markerfacecolor=(
                            method_style[method]["color"] if method == "dpfi" else "white"
                        ),
                        markeredgecolor=method_style[method]["color"],
                        markeredgewidth=1.1,
                        markersize=5.1,
                        linewidth=1.45,
                    )
            axis.set_title(title, loc="left", fontweight="bold", pad=7)
            axis.set_ylabel(ylabel)

        for axis in axes.flat:
            _style_axis(axis)
            axis.set_xlim(20, 123)
        axis_c.set_xlabel(r"$\mathrm{SNR}_{\mathrm{image}}$")
        axis_d.set_xlabel(r"$\mathrm{SNR}_{\mathrm{image}}$")

        method_handles = [
            Line2D(
                [0],
                [0],
                color=method_style[method]["color"],
                linestyle=method_style[method]["linestyle"],
                linewidth=1.6,
                label=method_style[method]["label"],
            )
            for method in ("dpfi", "dgi")
        ]
        state_handles = [
            Line2D(
                [0],
                [0],
                color=INK,
                marker=state_style[state]["marker"],
                markerfacecolor="white",
                linestyle="none",
                markersize=5.4,
                label=state,
            )
            for state in ("SSP", "ID")
        ]
        figure.legend(
            method_handles + state_handles,
            [handle.get_label() for handle in method_handles + state_handles],
            loc="upper left",
            bbox_to_anchor=(0.10, 0.995),
            ncol=4,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.3,
        )
        figure.text(
            0.985,
            0.972,
            r"Along each curve: $\tau=25\rightarrow400\,\mu\mathrm{s}$",
            ha="right",
            va="top",
            color=GREY,
            fontsize=11.5,
        )
        figure.subplots_adjust(left=0.115, right=0.985, bottom=0.105, top=0.905, hspace=0.34, wspace=0.30)

        output.mkdir(parents=True, exist_ok=False)
        metadata = {"Creator": "Non-destructive-image", "CreationDate": None, "ModDate": None}
        figure.savefig(output / "figure_5_6_snr_vs_recovery_width.pdf", bbox_inches="tight", metadata=metadata)
        figure.savefig(
            output / "figure_5_6_snr_vs_recovery_width.png",
            dpi=240,
            bbox_inches="tight",
            metadata={"Software": "Non-destructive-image"},
        )
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/chapter_5_snr_recovery_relation_preview_v1.json",
    )
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    inputs = _load_inputs(config)
    rows = _build_joined_rows(config, inputs)
    output = ROOT / config["output_directory"]
    _plot_relation(rows, output)
    _write_csv(output / "joined_snr_recovery_widths.csv", rows)
    summary = {
        "schema_version": 1,
        "label": config["label"],
        "status": "scratch_preview_not_retained",
        "joined_rows": len(rows),
        "new_random_draws": 0,
        "new_optical_propagations": 0,
        "new_equilibrium_solves": 0,
        "new_nonlinear_fits": 0,
        "inputs": config["inputs"],
        "claim_boundary": config["claim_boundary"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
