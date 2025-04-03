"""
Figure file to visualize the distribution of a single receptor across
target and off-target cell populations with histogram and KDE overlays.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- receptor: specific receptor whose distribution will be analyzed
- targCell: cell type whose receptor distributions will be compared to off-target cells
- sample_size: number of cells to sample for analysis
- cell_categorization: column name for cell type classification
- plot_cell_types: list of cell types to plot against the target cell type for
    comparison
- stat: statistic to use for histogram ('count', 'density', or 'probability')
- x_limit: whether or not to set x-axis limit to 99th percentile of target cell
    distribution
- normalize: whether or not to normalize receptor counts for comparison

Outputs:
- Single plot showing histogram with KDE overlay of receptor distributions:
  - Target cell distribution (with KDE)
  - Off-target cell distribution (with KDE)
- KL divergence and EMD values are displayed on the plot
- Scott bandwidth method is used for KDE
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_1D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    ax, f = getSetup((10, 6), (1, 1))
    ax = ax[0]

    receptor = "CD117"
    targCell = "ILC"
    sample_size = 100
    cell_categorization = "CellType2"
    plot_cell_types = [targCell, "other"]
    stat = "count"
    x_limit = False
    normalize = False

    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["Cell", "CellType1", "CellType2", "CellType3"]
    ]
    if receptor not in epitopes:
        raise ValueError(
            f"Receptor '{receptor}' not found. Available receptors: {epitopes}"
        )
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
        balance=True,
        rand_state=42,
    )

    # Calculate KL divergence and EMD for the specific receptor
    targ_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_targ_mask = ~targ_mask
    rec_abundances = sampleDF[[receptor]].to_numpy()
    KL_div_vals, EMD_vals = KL_EMD_1D(rec_abundances, targ_mask, off_targ_mask)
    kl_val = KL_div_vals[0]
    emd_val = EMD_vals[0]

    # Resample without balancing for histogram
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
        balance=False,
        rand_state=42,
        convert=False,
    )
    targ_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    sampleDF.loc[sampleDF["Cell Type"] != targCell, "Cell Type"] = "other"

    # Extract receptor abundances and normalize
    rec_abundance = sampleDF[receptor].values
    mean_abundance = np.mean(rec_abundance)
    targ_abundances = rec_abundance[targ_mask]

    all_cell_abundances = []
    for cell_type in plot_cell_types:
        if cell_type not in sampleDF["Cell Type"].unique():
            continue
        mask = (sampleDF["Cell Type"] == cell_type).to_numpy()
        if normalize:
            all_cell_abundances.append(rec_abundance[mask] / mean_abundance)
        else:
            all_cell_abundances.append(rec_abundance[mask])

        print(np.mean(rec_abundance[mask]), np.std(rec_abundance[mask]))

    # Plot histograms with KDE overlays
    colors = sns.color_palette("husl", len(all_cell_abundances))
    for i, abundances in enumerate(all_cell_abundances):
        if len(abundances) == 0:
            continue
        sns.histplot(
            abundances,
            ax=ax,
            color=colors[i],
            alpha=0.5,
            label=f"{plot_cell_types[i]} (n={len(abundances)})",
            stat=stat,
            kde=True,
            kde_kws={"bw_method": "scott"},
        )

    ax.set_title(
        f"{receptor} Distribution\nKL: {kl_val:.2f}, EMD: {emd_val:.2f}", fontsize=14
    )
    if normalize:
        ax.set_xlabel("Normalized receptor count", fontsize=12)
    else:
        ax.set_xlabel("Receptor count", fontsize=12)

    if stat == "density":
        ax.set_ylabel("Density", fontsize=12)
    elif stat == "count":
        ax.set_ylabel("Number of cells", fontsize=12)
    elif stat == "probability":
        ax.set_ylabel("Proportion of cells", fontsize=12)

    x_max = np.percentile(targ_abundances, 99)
    if x_limit:
        ax.set_xlim(0, x_max * 1.1)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return f
