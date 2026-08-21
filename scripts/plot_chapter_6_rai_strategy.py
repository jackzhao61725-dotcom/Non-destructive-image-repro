"""Create the Chapter 6 RAI-anchored measurement-sequence schematic."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dissertation" / "figures" / "figure_6_1_rai_anchored_strategy.pdf"

INK = "#202428"
MUTED = "#5F666D"
BLUE = "#2C78A8"
BLUE_PALE = "#DCEAF3"
ORANGE = "#C77919"
ORANGE_PALE = "#F6E6CF"
GREY = "#E8EAEC"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def box(ax, xy, width, height, label, face, edge=INK, fontsize=9.2):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, color=MUTED, style="-|>", lw=1.15, mutation=10):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=4,
        )
    )


fig, ax = plt.subplots(figsize=(6.5, 3.35))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.text(0.03, 0.945, "Matched measurement sequences", weight="bold", fontsize=10.5, va="top")

box(ax, (0.035, 0.59), 0.18, 0.17, "Matched\npreparation", GREY)
box(ax, (0.035, 0.19), 0.18, 0.17, "Matched\npreparation", GREY)
ax.plot([0.02, 0.02], [0.275, 0.675], ls=(0, (2, 2)), lw=1.0, color=MUTED)

box(ax, (0.31, 0.595), 0.13, 0.16, "RAI", ORANGE_PALE, edge=ORANGE, fontsize=10)
arrow(ax, (0.215, 0.675), (0.31, 0.675))
box(ax, (0.61, 0.585), 0.30, 0.18, "Reference\ndistribution", ORANGE_PALE, edge=ORANGE)
arrow(ax, (0.44, 0.675), (0.61, 0.675), color=ORANGE)

box(ax, (0.275, 0.195), 0.19, 0.16, "DPFI or DGI", BLUE_PALE, edge=BLUE, fontsize=10)
arrow(ax, (0.215, 0.275), (0.275, 0.275))
ax.text(0.525, 0.275, r"$\cdots$", ha="center", va="center", fontsize=17, color=MUTED)
arrow(ax, (0.465, 0.275), (0.49, 0.275), color=BLUE)
arrow(ax, (0.56, 0.275), (0.60, 0.275), color=BLUE)
box(ax, (0.60, 0.195), 0.13, 0.16, "final\nRAI", ORANGE_PALE, edge=ORANGE)
box(ax, (0.80, 0.185), 0.18, 0.18, "Post-probe\ndistribution", ORANGE_PALE, edge=ORANGE)
arrow(ax, (0.73, 0.275), (0.80, 0.275), color=ORANGE)

ax.add_patch(Rectangle((0.01, 0.08), 0.98, 0.82, facecolor="none", edgecolor="#AEB4B8", lw=0.6))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
