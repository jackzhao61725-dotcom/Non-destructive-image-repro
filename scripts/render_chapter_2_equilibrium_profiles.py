"""Render the three target-scale analytic profiles defined in Chapter 2."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".scratch" / "target_three_state_profiles_v4" / "figure_2_1_equilibrium_density_profiles_candidate_v8.pdf"
CONFIG = ROOT / "configs" / "three_state_target_trap_profiles_v4.json"

Y_LIMIT_UM = 16.0
Z_LIMIT_UM = 2.0
DENSITY_SCALE_MAX = 10.0


def load_profiles() -> list[dict[str, object]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["label"] != "three_state_target_trap_profiles_v4":
        raise RuntimeError("unexpected three-state profile contract")
    profiles = list(config["profiles"])
    if [profile["state_id"] for profile in profiles] != [
        "smooth_bec",
        "connected_modulated",
        "separated_droplets",
    ]:
        raise RuntimeError("three-state profile order changed")
    return profiles


def smooth_profile(
    y_um: np.ndarray,
    z_um: np.ndarray,
    definition: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    atom_number = float(definition["atom_number"])
    radius_y_um = float(definition["radius_y_um"])
    radius_z_um = float(definition["radius_z_um"])
    amplitude = 5.0 * atom_number / (2.0 * np.pi * radius_y_um * radius_z_um)
    support = np.maximum(
        0.0,
        1.0 - (y_um / radius_y_um) ** 2 - (z_um / radius_z_um) ** 2,
    )
    density = amplitude * support**1.5

    axial_support = np.maximum(0.0, 1.0 - (y_um[0] / radius_y_um) ** 2)
    line_density = amplitude * radius_z_um * 3.0 * np.pi / 8.0 * axial_support**2
    return density, line_density


def modulated_profile(
    y_um: np.ndarray,
    z_um: np.ndarray,
    definition: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    atom_number = float(definition["atom_number"])
    centres_um = np.asarray(definition["component_centres_y_um"], dtype=float)
    weights = np.asarray(definition["component_weights"], dtype=float)
    sigma_y_um = float(definition["component_sigma_y_um"])
    sigma_z_um = float(definition["component_sigma_z_um"])
    amplitude = atom_number / (
        2.0 * np.pi * sigma_y_um * sigma_z_um * np.sum(weights)
    )

    axial_sum = np.zeros_like(y_um)
    for centre_um, weight in zip(centres_um, weights, strict=True):
        axial_sum += weight * np.exp(-0.5 * ((y_um - centre_um) / sigma_y_um) ** 2)

    transverse = np.exp(-0.5 * (z_um / sigma_z_um) ** 2)
    density = amplitude * axial_sum * transverse
    line_density = amplitude * np.sqrt(2.0 * np.pi) * sigma_z_um * axial_sum[0]
    return density, line_density


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"candidate already exists: {OUTPUT}")
    definitions = load_profiles()
    y_axis_um = np.linspace(-Y_LIMIT_UM, Y_LIMIT_UM, 841)
    z_axis_um = np.linspace(-Z_LIMIT_UM, Z_LIMIT_UM, 361)
    y_um, z_um = np.meshgrid(y_axis_um, z_axis_um)

    profiles = [
        smooth_profile(y_um, z_um, definitions[0]),
        modulated_profile(y_um, z_um, definitions[1]),
        modulated_profile(y_um, z_um, definitions[2]),
    ]
    titles = [r"$\mathrm{BEC}_0$", "SSP", "ID"]

    for definition, (_, line_density) in zip(definitions, profiles, strict=True):
        recovered_atoms = np.trapezoid(line_density, y_axis_um)
        if not np.isclose(
            recovered_atoms,
            float(definition["atom_number"]),
            rtol=2.0e-4,
        ):
            raise RuntimeError(f"Profile normalisation failed: {recovered_atoms}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12.2,
            "axes.titlesize": 12.5,
            "axes.labelsize": 12.2,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "mathtext.fontset": "stix",
        }
    )

    figure = plt.figure(figsize=(7.25, 4.8), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.055),
        height_ratios=(0.88, 0.90),
    )
    map_axes = [figure.add_subplot(grid[0, column]) for column in range(3)]
    line_axes = [figure.add_subplot(grid[1, column], sharex=map_axes[column]) for column in range(3)]
    colour_axis = figure.add_subplot(grid[0, 3])

    generated_peak = max(float(np.max(density)) for density, _ in profiles) / 1.0e3
    if generated_peak > DENSITY_SCALE_MAX:
        raise RuntimeError("rounded density scale no longer contains all profiles")
    for index, ((density, line_density), title) in enumerate(zip(profiles, titles, strict=True)):
        map_axis = map_axes[index]
        image = map_axis.imshow(
            density / 1.0e3,
            origin="lower",
            extent=(-Y_LIMIT_UM, Y_LIMIT_UM, -Z_LIMIT_UM, Z_LIMIT_UM),
            cmap="magma",
            vmin=0.0,
            vmax=DENSITY_SCALE_MAX,
            interpolation="none",
            aspect=3.0,
            rasterized=True,
        )
        map_axis.set_title(title, pad=5)
        map_axis.set_xlim(-Y_LIMIT_UM, Y_LIMIT_UM)
        map_axis.set_ylim(-Z_LIMIT_UM, Z_LIMIT_UM)
        map_axis.set_yticks((-1.0, 0.0, 1.0))
        map_axis.tick_params(axis="y", labelsize=10.0, pad=2)
        map_axis.tick_params(labelbottom=False)
        if index == 0:
            map_axis.set_ylabel(r"$z\;(\mu\mathrm{m})$")
        else:
            map_axis.tick_params(labelleft=False)

        line_axis = line_axes[index]
        central_cut = density[np.argmin(np.abs(z_axis_um)), :] / 1.0e3
        line_axis.plot(y_axis_um, central_cut, color="black", linewidth=1.5)
        line_axis.set_xlim(-Y_LIMIT_UM, Y_LIMIT_UM)
        line_axis.set_ylim(-0.3, 10.3)
        line_axis.set_xlabel(r"$y\;(\mu\mathrm{m})$")
        line_axis.set_yticks((0.0, 5.0, 10.0))
        line_axis.spines[["top", "right"]].set_visible(False)
        if index == 0:
            line_axis.set_ylabel(
                r"$\widetilde n(y,0)$" "\n" r"$(10^3\,\mu\mathrm{m}^{-2})$"
            )
        else:
            line_axis.tick_params(labelleft=False)

    figure.text(
        0.015,
        0.945,
        r"$\odot\;\mathbf{B}\parallel\mathbf{k}\parallel+\hat{\mathbf{x}}$",
        ha="left",
        va="center",
        fontsize=10.2,
    )

    colour_bar = figure.colorbar(image, cax=colour_axis)
    colour_bar.set_ticks((0.0, 5.0, 10.0))
    colour_bar.set_label(
        r"$\widetilde n(y,z)\;(10^3\,\mu\mathrm{m}^{-2})$"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT,
        bbox_inches="tight",
        metadata={"Title": "BEC, SSP and ID column densities"},
    )
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
