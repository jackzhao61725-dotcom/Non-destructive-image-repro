"""Render the Chapter 3 phase-wrapping concept figure.

The figure is deliberately local and phase-only: it maps object-plane column
density to the four analytic readout responses derived in Section 3.3.  It does not
include spatial propagation, pixel integration, detector noise or an inverse
estimator; those belong to Chapter 4 and later chapters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from scripts.dissertation_figure_style import (
    BLUE,
    GREY,
    INK,
    LIGHT_GREY,
    ORANGE,
    dissertation_style,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ATOMIC_CONFIG = REPOSITORY_ROOT / "configs" / "dissertation_v3_orca_fusion.json"
READOUT_CONFIG = (
    REPOSITORY_ROOT / "configs" / "target_three_state_four_method_noiseless_v4.json"
)
PROFILE_CONFIG = (
    REPOSITORY_ROOT / "configs" / "three_state_target_trap_profiles_v4.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / ".scratch" / "chapter_3_phase_wrap"

FARADAY_GREEN = "#008C67"
STATE_STYLES = {
    "smooth_bec": {"label": r"$\mathrm{BEC}_{0}$", "colour": BLUE, "marker": "o"},
    "connected_modulated": {"label": "SSP", "colour": ORANGE, "marker": "s"},
    "separated_droplets": {"label": "ID", "colour": FARADAY_GREEN, "marker": "^"},
}


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _require_finite_positive(value: float, label: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive, got {value!r}")
    return value


def _load_parameters() -> dict[str, object]:
    atomic = _read_json(ATOMIC_CONFIG)
    readout = _read_json(READOUT_CONFIG)
    profiles = _read_json(PROFILE_CONFIG)

    atom = atomic.get("atom")
    physical = readout.get("physical_contract")
    readout_contract = readout.get("readout_contract")
    generated_profiles = profiles.get("generated_truth_on_contract_grid")
    if not isinstance(atom, dict):
        raise RuntimeError("atomic config has no atom object")
    if not isinstance(physical, dict):
        raise RuntimeError("readout config has no physical_contract object")
    if not isinstance(readout_contract, dict):
        raise RuntimeError("readout config has no readout_contract object")
    if not isinstance(generated_profiles, list):
        raise RuntimeError("profile config has no generated target table")

    sigma0_m2 = _require_finite_positive(
        float(atom["resonant_cross_section_m2"]), "resonant cross-section"
    )
    linewidth_rad_s = _require_finite_positive(
        float(atom["natural_linewidth_rad_s"]), "natural linewidth"
    )
    detuning_hz = _require_finite_positive(
        float(physical["detuning_hz"]), "detuning"
    )
    linewidth_hz = linewidth_rad_s / (2.0 * np.pi)
    dimensionless_detuning = 2.0 * detuning_hz / linewidth_hz

    # Dominant-branch, phase-only reduction used in the Chapter 3 discussion:
    # phi_c = +phi_2L/2 and theta_F = -phi_2L/2.
    alpha_rad_m2 = (
        sigma0_m2
        * dimensionless_detuning
        / (4.0 * (1.0 + dimensionless_detuning**2))
    )
    alpha_rad_um2 = alpha_rad_m2 * 1.0e12

    phase_plate_amplitude = float(readout_contract["pci_phase_plate_amplitude"])
    stop_optical_depth = float(readout_contract["dgi_stop_optical_depth"])
    if not 0.0 <= phase_plate_amplitude <= 1.0:
        raise RuntimeError("PCI phase-plate amplitude is outside [0, 1]")
    if not np.isfinite(stop_optical_depth) or stop_optical_depth < 0.0:
        raise RuntimeError("DGI stop optical depth must be finite and non-negative")
    stop_field_amplitude = 10.0 ** (-stop_optical_depth / 2.0)

    peak_densities: dict[str, float] = {}
    for record in generated_profiles:
        if not isinstance(record, dict):
            raise RuntimeError("generated profile record is not an object")
        state_id = str(record["state_id"])
        if state_id not in STATE_STYLES:
            continue
        peak_densities[state_id] = _require_finite_positive(
            float(record["peak_column_density_um2"]),
            f"{state_id} peak column density",
        )
    if set(peak_densities) != set(STATE_STYLES):
        raise RuntimeError("the three target peak column densities were not found")

    return {
        "alpha_rad_um2": alpha_rad_um2,
        "phase_plate_amplitude": phase_plate_amplitude,
        "stop_field_amplitude": stop_field_amplitude,
        "peak_densities_um2": peak_densities,
    }


def _add_phase_axis(axis: plt.Axes, alpha_rad_um2: float, label: str) -> None:
    density_scale = 1.0e3

    def density_to_phase(density_thousands: np.ndarray) -> np.ndarray:
        return alpha_rad_um2 * density_scale * density_thousands

    def phase_to_density(phase: np.ndarray) -> np.ndarray:
        return phase / (alpha_rad_um2 * density_scale)

    phase_axis = axis.secondary_xaxis(
        "top", functions=(density_to_phase, phase_to_density)
    )
    phase_axis.set_xlabel(label, labelpad=7.0)
    phase_axis.set_xticks(
        [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0, np.pi],
        [r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"],
    )
    phase_axis.tick_params(axis="x", pad=2.0)


def _style_panel(
    axis: plt.Axes,
    *,
    title: str,
    ylabel: str,
) -> None:
    axis.set_title(title, loc="left", fontweight="bold", pad=8.0)
    axis.set_ylabel(ylabel)
    axis.axhline(0.0, color=GREY, linewidth=0.8, zorder=0)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.7, alpha=0.55)
    axis.tick_params(direction="out", length=4.0)
    axis.spines["right"].set_visible(False)


def render(
    output_dir: Path,
    *,
    stem: str = "figure_3_phase_wrap_preview",
    formats: tuple[str, ...] = ("png", "pdf", "svg"),
) -> tuple[Path, ...]:
    parameters = _load_parameters()
    alpha = float(parameters["alpha_rad_um2"])
    t_p = float(parameters["phase_plate_amplitude"])
    t_s = float(parameters["stop_field_amplitude"])
    peak_densities = dict(parameters["peak_densities_um2"])

    density_at_pi_um2 = np.pi / alpha
    density_um2 = np.linspace(0.0, density_at_pi_um2, 1601)
    density_thousands = density_um2 / 1.0e3
    phase = alpha * density_um2
    theta_f = -phase

    pci = 2.0 * np.sin(phase) ** 2 + t_p * np.sin(2.0 * phase)
    dgi = 2.0 * (1.0 - t_s) * np.sin(phase) ** 2
    dpfi = np.sin(2.0 * theta_f)
    dffi = np.sin(theta_f) ** 2

    output_dir.mkdir(parents=True, exist_ok=True)
    if not stem or Path(stem).name != stem:
        raise RuntimeError("output stem must be one non-empty filename stem")
    supported_formats = {"png", "pdf", "svg"}
    if not formats or not set(formats).issubset(supported_formats):
        raise RuntimeError("formats must be a non-empty subset of png, pdf and svg")
    output_paths = {suffix: output_dir / f"{stem}.{suffix}" for suffix in formats}

    with dissertation_style(svg_hashsalt="chapter-3-phase-wrap-v1"):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11.8, 6.7),
            sharex=True,
            constrained_layout=False,
        )
        (axis_pci, axis_dpfi), (axis_dgi, axis_dffi) = axes
        x_limit = density_at_pi_um2 / 1.0e3

        axis_pci.plot(density_thousands, pci, color=BLUE, linewidth=2.5)
        axis_dpfi.plot(density_thousands, dpfi, color=FARADAY_GREEN, linewidth=2.5)
        axis_dgi.plot(density_thousands, dgi, color=BLUE, linewidth=2.5)
        axis_dffi.plot(density_thousands, dffi, color=FARADAY_GREEN, linewidth=2.5)

        _style_panel(
            axis_pci,
            title="(a) PCI",
            ylabel=r"$\Delta I_{\mathrm{PCI}}/I_0$",
        )
        _style_panel(
            axis_dpfi,
            title="(b) DPFI",
            ylabel=r"$S_{\mathrm{DPFI}}$",
        )
        _style_panel(
            axis_dgi,
            title="(c) DGI",
            ylabel=r"$\Delta I_{\mathrm{DGI}}/I_0$",
        )
        _style_panel(
            axis_dffi,
            title="(d) DFFI",
            ylabel=r"$I_{\mathrm{DFFI}}/I_0$",
        )

        axis_pci.set_ylim(-0.48, 2.62)
        axis_dpfi.set_ylim(-1.20, 1.20)
        axis_dgi.set_ylim(-0.14, 2.18)
        axis_dffi.set_ylim(-0.07, 1.10)
        response_by_axis = {
            axis_pci: pci,
            axis_dpfi: dpfi,
            axis_dgi: dgi,
            axis_dffi: dffi,
        }
        for axis in axes.flat:
            axis.set_xlim(0.0, x_limit)
            axis.set_xticks([0.0, 4.0, 8.0, 12.0, 16.0])
            for state_id, density in peak_densities.items():
                style = STATE_STYLES[state_id]
                density_thousands_state = density / 1.0e3
                axis.axvline(
                    density_thousands_state,
                    color=str(style["colour"]),
                    linewidth=1.65,
                    linestyle=(0, (3, 3)),
                    alpha=0.90,
                    zorder=1,
                )
                response = float(
                    np.interp(density_thousands_state, density_thousands, response_by_axis[axis])
                )
                axis.scatter(
                    [density_thousands_state],
                    [response],
                    marker=str(style["marker"]),
                    s=54,
                    facecolor="white",
                    edgecolor=str(style["colour"]),
                    linewidth=1.8,
                    zorder=6,
                )

        axis_dgi.set_xlabel(
            r"Local column density $\widetilde n$ ($10^3\,\mu\mathrm{m}^{-2}$)"
        )
        axis_dffi.set_xlabel(
            r"Local column density $\widetilde n$ ($10^3\,\mu\mathrm{m}^{-2}$)"
        )
        _add_phase_axis(axis_pci, alpha, r"Common phase $\phi_{\mathrm{c}}$")
        _add_phase_axis(axis_dpfi, alpha, r"Rotation magnitude $|\theta_F|$")

        state_handles: list[mlines.Line2D] = []
        for state_id in ("smooth_bec", "connected_modulated", "separated_droplets"):
            style = STATE_STYLES[state_id]
            density = peak_densities[state_id] / 1.0e3
            state_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=str(style["colour"]),
                    linestyle=(0, (3, 3)),
                    marker=str(style["marker"]),
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    markersize=6.5,
                    linewidth=1.15,
                    label=f"{style['label']} peak  {density:.2f}",
                )
            )
        fig.legend(
            handles=state_handles,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
            title=r"Peak column densities ($10^3\,\mu\mathrm{m}^{-2}$)",
            handlelength=2.6,
            columnspacing=2.2,
        )

        fig.subplots_adjust(
            left=0.095,
            right=0.985,
            top=0.83,
            bottom=0.19,
            wspace=0.20,
            hspace=0.30,
        )
        for suffix, path in output_paths.items():
            save_options = {"dpi": 240} if suffix == "png" else {}
            fig.savefig(path, bbox_inches="tight", **save_options)
        plt.close(fig)

    return tuple(output_paths[suffix] for suffix in formats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the PNG, PDF and SVG previews.",
    )
    parser.add_argument(
        "--stem",
        default="figure_3_phase_wrap_preview",
        help="Filename stem shared by the rendered outputs.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
        help="One or more output formats.",
    )
    args = parser.parse_args()
    paths = render(
        args.output_dir.resolve(),
        stem=args.stem,
        formats=tuple(args.formats),
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
