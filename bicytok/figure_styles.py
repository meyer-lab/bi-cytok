import matplotlib.pyplot as plt

FONT_FAMILY = "serif"
FONT_NAME = "Times New Roman"  # matches manuscript mainfont in _quarto.yml

FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 10
FONT_SIZE_ANNOT = 10  # in-plot data annotations (heatmap cell labels, point/bar labels)

LINE_WIDTH = 1.5
LINE_WIDTH_AGGREGATE = (
    2.5  # emphasizes summary/aggregate overlays (e.g. "All off-target")
)
GRID_ALPHA = 0.3
GRID_LINESTYLE = "--"
LEGEND_FRAMEALPHA = 0.8

COLORS = {
    "target": "#CF4D6F",  # as currently used in raw_1D-hist.qmd
    "off_target": "#2774AE",  # anchor color
    "aggregate": "darkred",  # summary overlay across a group (e.g. "All off-target")
    "improved": "#2774AE",  # 2D metric >= 1D metric (pairing helps); anchor color
    "declined": "#C0392B",  # 2D metric < 1D metric (pairing hurts)
    "Treg": "#28aae2",
    "CD8 T": "#fbb040",
    "NK": "#f69792",
}

LINESTYLES = {
    "aggregate": "--",  # dashed, distinguishes summary overlays from individual series
}

CMAPS = {
    "sequential": "Reds",  # magnitude-only heatmap data (e.g. optimal metric value)
    "categorical": "tab10",  # qualitative palette for <=10 subgroups
    "categorical_large": "tab20",  # qualitative palette for >10 subgroups
}

FIGSIZE = {
    "single_panel": (5, 3),  # single 1D histogram/line panel
    "square_scatter": (7, 7),  # single square scatter panel
    "joint_grid": (10, 8),  # 2D joint distribution + marginal histograms
    "square_heatmap": (8, 8),  # single square heatmap panel with colorbar
}


def categorical_colors(n: int, cmap_name: str = CMAPS["categorical"]) -> list:
    """
    Returns n distinct qualitative colors for identifying arbitrary subgroups
    (e.g. individual off-target cell types, cell types in a scan comparison).

    :param n: Number of distinct colors needed
    :param cmap_name: Name of a qualitative matplotlib colormap to sample
    :return: List of n RGBA color tuples
    """
    cmap = plt.colormaps.get_cmap(cmap_name).resampled(n)
    return [cmap(i) for i in range(n)]


def apply_style() -> None:
    """
    Applies shared rcParams so figures intended for publication/posters share a
    common font, size hierarchy, line width, legend style, and grid style.
    """
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": [FONT_NAME],
            "svg.fonttype": "none",
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "legend.title_fontsize": FONT_SIZE_LEGEND,
            "legend.framealpha": LEGEND_FRAMEALPHA,
            "lines.linewidth": LINE_WIDTH,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": GRID_LINESTYLE,
        }
    )
