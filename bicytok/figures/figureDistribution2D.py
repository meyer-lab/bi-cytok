"""
Figure file to visualize the 2D joint distribution of receptor pairs across
target and off-target cell populations with scatter plots and KDE contours.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- receptor1: first receptor whose distribution will be analyzed
- receptor2: second receptor whose distribution will be analyzed
- targCell: cell type whose receptor distributions will be compared to off-target cells
- sample_size: number of cells to sample for analysis
- cell_categorization: column name for cell type classification
- plot_cell_types: list of cell types to plot on the figure

Outputs:
- Single plot showing 2D distributions of receptor pairs:
  - Target cell distribution (with KDE contours)
  - Off-target cell distribution (with KDE contours)
- 2D KL divergence and EMD values are displayed on the plot
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_2D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((10, 8), (1, 1))
    ax = ax[0]

    receptor1 = "CD25"
    receptor2 = "CD146"
    targCell = "Treg"
    sample_size = 5000
    cell_categorization = "CellType2"
    plot_cell_types = ["Treg", "other"]

    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]

    for receptor in [receptor1, receptor2]:
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
    )

    # Calculate KL divergence and EMD for the receptor pair
    targ_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_targ_mask = ~targ_mask
    rec_abundances = sampleDF[[receptor1, receptor2]].to_numpy()
    KL_div_vals, EMD_vals = KL_EMD_2D(
        rec_abundances, targ_mask, off_targ_mask, calc_1D=False
    )
    kl_val = KL_div_vals[1, 0]
    emd_val = EMD_vals[1, 0]
    sampleDF.loc[sampleDF["Cell Type"] != targCell, "Cell Type"] = "other"

    # Extract receptor abundances and normalize
    rec1_abundance = sampleDF[receptor1].values
    rec2_abundance = sampleDF[receptor2].values
    mean_rec1 = np.mean(rec1_abundance)
    mean_rec2 = np.mean(rec2_abundance)

    # Create a dataframe with normalized values
    plot_df = pd.DataFrame(
        {
            "Cell Type": sampleDF["Cell Type"],
            f"{receptor1}_norm": rec1_abundance / mean_rec1,
            f"{receptor2}_norm": rec2_abundance / mean_rec2,
        }
    )

    # Generate statistics for each cell type
    stats_data = []
    for cell_type in plot_cell_types:
        if cell_type not in plot_df["Cell Type"].unique():
            continue
        cell_df = plot_df[plot_df["Cell Type"] == cell_type]

        stats_data.append(
            {
                "Cell Type": cell_type,
                f"{receptor1} Min": np.min(cell_df[f"{receptor1}_norm"]),
                f"{receptor1} Max": np.max(cell_df[f"{receptor1}_norm"]),
                f"{receptor1} Mean": np.mean(cell_df[f"{receptor1}_norm"]),
                f"{receptor1} Std": np.std(cell_df[f"{receptor1}_norm"]),
                f"{receptor2} Min": np.min(cell_df[f"{receptor2}_norm"]),
                f"{receptor2} Max": np.max(cell_df[f"{receptor2}_norm"]),
                f"{receptor2} Mean": np.mean(cell_df[f"{receptor2}_norm"]),
                f"{receptor2} Std": np.std(cell_df[f"{receptor2}_norm"]),
                "Count": len(cell_df),
            }
        )
    stats_df = pd.DataFrame(stats_data)
    print(
        f"Statistics for {receptor1} and {receptor2} receptor abundance by cell type:"
    )
    print(stats_df.to_string(index=False))

    # Create the 2D visualizations with both scatter and KDE contours
    colors = sns.color_palette("husl", len(plot_cell_types))
    for i, cell_type in enumerate(
        [t for t in plot_cell_types if t in plot_df["Cell Type"].unique()]
    ):
        cell_df = plot_df[plot_df["Cell Type"] == cell_type]

        ax.scatter(
            cell_df[f"{receptor1}_norm"],
            cell_df[f"{receptor2}_norm"],
            alpha=0.3,
            color=colors[i],
            s=10,
            label=f"{cell_type} (n={len(cell_df)})",
        )

        sns.kdeplot(
            x=cell_df[f"{receptor1}_norm"],
            y=cell_df[f"{receptor2}_norm"],
            ax=ax,
            color=colors[i],
            levels=5,
            linewidths=1.5,
        )

    # Set reasonable axis limits based on 99th percentile
    x_max = np.percentile(plot_df[f"{receptor1}_norm"], 99)
    y_max = np.percentile(plot_df[f"{receptor2}_norm"], 99)
    ax.set_xlim(0, x_max * 1.1)
    ax.set_ylim(0, y_max * 1.1)

    # Add title and labels
    ax.set_title(
        f"{receptor1} vs {receptor2} Distribution\nKL: {kl_val:.2f}, "
        f"EMD: {emd_val:.2f}",
        fontsize=14,
    )
    ax.set_xlabel(f"{receptor1} Normalized Expression", fontsize=12)
    ax.set_ylabel(f"{receptor2} Normalized Expression", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return f
