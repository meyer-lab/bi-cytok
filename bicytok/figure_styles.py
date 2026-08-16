import matplotlib.pyplot as plt

FONT_FAMILY = "serif"
FONT_NAME = "Times New Roman"  # matches manuscript mainfont in _quarto.yml

FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 18
FONT_SIZE_TICK = 14
FONT_SIZE_LEGEND = 14
FONT_SIZE_ANNOT = 14  # in-plot data annotations (heatmap cell labels, point/bar labels)

LINE_WIDTH = 1.5
LINE_WIDTH_AGGREGATE = (
    2.5  # emphasizes summary/aggregate overlays (e.g. "All off-target")
)
POINT_SIZE_BACKGROUND = 15  # bulk / non-emphasized scatter points
POINT_SIZE_FOREGROUND = 30  # highlighted points (outliers, top hits, star markers)
GRID_ALPHA = 0.3
GRID_LINESTYLE = "--"
LEGEND_FRAMEALPHA = 0.8
LEGEND_BORDERPAD = 0.3  # padding between legend border and its content (default 0.4)
LEGEND_LABELSPACING = 0.3  # vertical space between entries (default 0.5)
LEGEND_HANDLELENGTH = 1.5  # length of the marker/line handle sample (default 2.0)
LEGEND_HANDLETEXTPAD = 0.5  # space between handle and label text (default 0.8)
LEGEND_BORDERAXESPAD = 0.3  # space between legend and axes edge (default 0.5)

COLORS = {
    "target": "#CF4D6F",  # as currently used in raw_1D-hist.qmd
    "off_target": "#2774AE",  # anchor color
    "aggregate": "darkred",  # summary overlay across a group (e.g. "All off-target")
    "improved": "#2774AE",  # 2D metric >= 1D metric (pairing helps); anchor color
    "declined": "#C0392B",  # 2D metric < 1D metric (pairing hurts)
    "Treg": "#1e75bc",
    "CD8 T": "#f7941d",
    "NK": "#f0554f",
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
    "single_panel": (4, 3),  # single 1D histogram/line panel
    "square_scatter": (4, 4),  # single square scatter panel
    "joint_grid": (10, 8),  # 2D joint distribution + marginal histograms
    "square_heatmap": (8, 8),  # single square heatmap panel with colorbar
}


def gridspec_marginal_spacing(
    fig_width: float,
    fig_height: float,
    grid_bounds: tuple[float, float, float, float],
    n_rows: int,
    n_cols: int,
    gap_inches: float,
) -> tuple[float, float]:
    """
    Returns (hspace, wspace) for a 2D-joint-plus-marginals GridSpec such that
    the physical gap between the main panel and each marginal is the same in
    both directions. Matplotlib's hspace/wspace are fractions of the average
    row/column size, not of a shared physical unit, so equal hspace/wspace
    values (or a square figure's worth of intuition) produce visibly unequal
    gaps whenever the figure or the grid's row/column ratios aren't square.

    :param grid_bounds: (left, right, top, bottom) figure-fraction bounds passed to GridSpec
    :param gap_inches: desired physical gap between the main panel and each marginal
    :return: (hspace, wspace) to pass to GridSpec
    """
    left, right, top, bottom = grid_bounds
    tot_width = right - left
    tot_height = top - bottom

    def space_for_gap(gap: float, fig_dim: float, tot_frac: float, n: int) -> float:
        gap_frac = gap / fig_dim
        return gap_frac * n / (tot_frac - gap_frac * (n - 1))

    hspace = space_for_gap(gap_inches, fig_height, tot_height, n_rows)
    wspace = space_for_gap(gap_inches, fig_width, tot_width, n_cols)
    return hspace, wspace


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
            "legend.borderpad": LEGEND_BORDERPAD,
            "legend.labelspacing": LEGEND_LABELSPACING,
            "legend.handlelength": LEGEND_HANDLELENGTH,
            "legend.handletextpad": LEGEND_HANDLETEXTPAD,
            "legend.borderaxespad": LEGEND_BORDERAXESPAD,
            "lines.linewidth": LINE_WIDTH,
            "axes.grid": True,
            "grid.alpha": GRID_ALPHA,
            "grid.linestyle": GRID_LINESTYLE,
        }
    )
