import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

FONT_FAMILY = "serif"
FONT_NAME = "Times New Roman"  # matches manuscript mainfont in _quarto.yml

FONT_SIZE_TITLE = 14

DEFAULT_N_TICKS = 6  # default tick count per axis; override per-plot via set_tick_count

LINE_WIDTH = 1.5
LINE_WIDTH_AGGREGATE = (
    2.5  # emphasizes summary/aggregate overlays (e.g. "All off-target")
)
POINT_SIZE_XS = 7  # extra small scatter points (scatters containing entire scan)
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
    "outlier_neither": "lightgray",  # scan_outliers.qmd: neither metric is high
    "outlier_metric1": "#3A86FF",  # scan_outliers.qmd: metric 1 outlier
    "outlier_metric2": "#FF9F1C",  # scan_outliers.qmd: metric 2 outlier
    "outlier_both": "#E63946",  # scan_outliers.qmd: high on both metrics
    "outlier_user": "#9B5DE5",  # scan_outliers.qmd: user-specified pairs
}

LINESTYLES = {
    "aggregate": "--",  # dashed, distinguishes summary overlays from individual series
}

CMAPS = {
    "sequential": "Reds",  # magnitude-only heatmap data (e.g. optimal metric value)
    "categorical": "tab10",  # qualitative palette for <=10 subgroups
    "categorical_large": "tab20",  # qualitative palette for >10 subgroups
}

# CITE-seq CellType2 annotation values (see bicytok.imports.importCITE), fixed here so a
# given cell type always gets the same color everywhere it's plotted, regardless of which
# subset of cell types happens to be present in a particular scan or comparison. Without
# this, per-plot palettes built from `categorical_colors(len(cell_types_present), ...)`
# assign colors by *position in the current subset*, so the same cell type can land on a
# different color in every plot — or, worse, on the exact same color as an unrelated cell
# type whenever both subsets happen to have length 1 (confirmed happening in practice:
# Treg / CD8 TCM / pDC's single-cell-type panels in scan_comparison_scatter.qmd all drew
# the same color, since `categorical_colors(1, ...)` always returns the first sample).
CELL_TYPES = [
    "ASDC",
    "B intermediate",
    "B memory",
    "B naive",
    "CD14 Mono",
    "CD16 Mono",
    "CD4 CTL",
    "CD4 Naive",
    "CD4 Proliferating",
    "CD4 TCM",
    "CD4 TEM",
    "CD8 Naive",
    "CD8 Proliferating",
    "CD8 TCM",
    "CD8 TEM",
    "Doublet",
    "Eryth",
    "HSPC",
    "ILC",
    "MAIT",
    "NK",
    "NK Proliferating",
    "NK_CD56bright",
    "Plasmablast",
    "Platelet",
    "Treg",
    "cDC1",
    "cDC2",
    "dnT",
    "gdT",
    "pDC",
]

FIGSIZE = {
    "single_panel": (4, 3),  # single 1D histogram/line panel
    "square_scatter": (4, 4),  # single square scatter panel
    "joint_grid": (5, 4),  # 2D joint distribution + marginal histograms
    "square_heatmap": (5, 5),  # single square heatmap panel with colorbar
}

# fig.colorbar(mappable, ax=ax) shrinks the given ax to make room for the colorbar rather than
# growing the canvas, so a panel drawn at FIGSIZE["square_scatter"] with a colorbar attached ends
# up narrower than intended. Pass these explicitly to fig.colorbar() (fraction=COLORBAR_FRACTION,
# pad=COLORBAR_PAD) together with FIGSIZE["square_scatter_with_colorbar"], which is derived from
# square_scatter so the two stay in sync if that base size ever changes.
COLORBAR_FRACTION = 0.25
COLORBAR_PAD = 0.05
FIGSIZE["square_scatter_with_colorbar"] = (
    FIGSIZE["square_scatter"][0] / (1 - COLORBAR_FRACTION - COLORBAR_PAD),
    FIGSIZE["square_scatter"][1],
)


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


def set_tick_count(
    ax,
    n_x: int | None = DEFAULT_N_TICKS,
    n_y: int | None = DEFAULT_N_TICKS,
) -> None:
    """
    Caps the number of ticks on each axis via MaxNLocator, which rounds to "nice"
    values (1, 2, 5, 10, ...) rather than forcing an exact count.

    :param ax: Axes to apply tick locators to
    :param n_x: Max ticks on the x-axis, or None to leave matplotlib's default
    :param n_y: Max ticks on the y-axis, or None to leave matplotlib's default
    """
    if n_x is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n_x))
    if n_y is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n_y))


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


def discrete_categorical_colors(n: int, cmap_names: list[str]) -> list:
    """
    Returns n distinct qualitative colors by concatenating the *discrete* entries
    of one or more qualitative colormaps, without interpolating between entries
    the way `categorical_colors`' `resampled(n)` does.

    Interpolating a 20-color map down to, say, 12 categories blends adjacent
    entries into new in-between colors that can look muddier and less distinct
    than the original discrete set. Concatenating whole discrete palettes (e.g.
    tab20 + tab20b + tab20c) avoids that, at the cost of needing enough total
    entries across the given maps to cover n.

    :param n: Number of distinct colors needed
    :param cmap_names: Qualitative matplotlib colormaps to concatenate, in order
    :return: List of n RGBA color tuples
    """
    colors = [c for name in cmap_names for c in plt.colormaps.get_cmap(name).colors]
    assert len(colors) >= n, (
        f"Only {len(colors)} discrete colors available across {cmap_names}, need {n}"
    )
    return colors[:n]


# Canonical cell-type -> color mapping, built once from the fixed CELL_TYPES list above.
CELL_TYPE_COLORS = dict(
    zip(
        CELL_TYPES,
        discrete_categorical_colors(len(CELL_TYPES), ["tab20", "tab20b", "tab20c"]),
        strict=True,
    )
)


def standalone_legend_figure(
    handles: list, labels: list[str], ncol: int = 1, **legend_kwargs
):
    """
    Builds a Figure containing only a legend (its Axes hidden, not removed),
    sized to exactly the legend's own rendered extent.

    For layouts where one legend is shared across several subpanels rather than
    repeated in each (so it can be positioned independently, e.g. in Illustrator).

    Two implementation notes, both load-bearing:
    - The crop is done by resizing the Figure itself, not via
      `savefig(bbox_inches="tight")`: Quarto/Jupyter's inline display capture only
      reads `InlineBackend.print_figure_kwargs` once at kernel startup, so a
      mid-notebook `%config` change to request a tight bbox never takes effect.
    - The Axes is hidden (`set_axis_off`), not omitted: IPython's figure display
      silently produces no output at all for a Figure with zero Axes and zero
      Figure-level lines, so at least one (even if invisible) Axes must remain.

    :param handles: legend handles (e.g. Line2D proxies), one per category
    :param labels: legend label text, matched positionally to handles
    :param ncol: number of columns to arrange entries into (matplotlib fills
        columns top-to-bottom, so row count follows as ceil(len(handles) / ncol));
        default 1 keeps the original single-column layout
    :param legend_kwargs: forwarded to Axes.legend (e.g. title)
    :return: Figure containing only the legend, resized to its content
    """
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_position([0, 0, 1, 1])
    legend = ax.legend(
        handles=handles, labels=labels, loc="center", ncol=ncol, **legend_kwargs
    )
    fig.canvas.draw()  # force a render pass so the legend's extent is known
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(bbox.width, bbox.height)
    ax.set_position([0, 0, 1, 1])  # re-fill the resized canvas; recenters the legend
    return fig


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
