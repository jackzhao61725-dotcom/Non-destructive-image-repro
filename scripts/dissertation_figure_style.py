"""Shared visual constants for dissertation-only presentation renderers."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt


INK = "#252A31"
GREY = "#66717D"
LIGHT_GREY = "#D7DCE2"
BLUE = "#0072B2"
ORANGE = "#D55E00"
WHITE = "#FFFFFF"


@contextmanager
def dissertation_style(*, svg_hashsalt: str) -> Iterator[None]:
    """Apply the common serif dissertation style within a local context."""

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": (
                "STIX Two Text",
                "STIXGeneral",
                "Times New Roman",
                "DejaVu Serif",
            ),
            "mathtext.fontset": "stix",
            "font.size": 12.2,
            "axes.labelsize": 12.2,
            "axes.titlesize": 12.5,
            "legend.fontsize": 12.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": svg_hashsalt,
        }
    ):
        yield


def add_panel_label(axis: mpl.axes.Axes, label: str, *, colour: str = INK) -> None:
    """Place one consistent panel label just inside the upper-left corner."""

    axis.text(
        0.02,
        0.98,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=colour,
    )
