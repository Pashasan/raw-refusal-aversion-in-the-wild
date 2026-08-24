"""Publication plot styling used across all paper figures."""

from __future__ import annotations

import matplotlib.pyplot as plt

# Blue palette used in every figure.
LIGHT_BLUE = "#aec7e8"
DARK_BLUE = "#1f77b4"
ORANGE = "#ff7f0e"


def set_paper_style(width: str = "double", dpi: int = 300) -> tuple[float, float]:
    """Apply the paper's matplotlib rcParams and return a (w, h) size hint.

    Parameters
    ----------
    width : 'single' or 'double'
        Column width. 'single' ≈ 3.5 inches; 'double' ≈ 6.75 inches.
    dpi : int
        Figure dpi used at save time.

    Returns
    -------
    (width_inches, height_inches) : default figure size hint.
    """
    size = (6.75, 2.5) if width == "double" else (3.5, 2.5)
    plt.rcParams.update({
        "figure.figsize": size,
        "figure.dpi": dpi,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return size
