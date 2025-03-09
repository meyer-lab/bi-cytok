"""
Figure file to visualize receptor distributions with the highest and lowest
distance metrics between target and off-target cell populations.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose receptor distributions will be compared to off-target cells
- sample_size: number of cells to sample for analysis
- cell_categorization: column name for cell type classification

Outputs:
- Twenty subplots showing histograms of receptor distributions:
  - Top 5 receptors by KL divergence
  - Bottom 5 receptors by KL divergence
  - Top 5 receptors by EMD
  - Bottom 5 receptors by EMD
- Each plot shows target and off-target distributions
- KL divergence and EMD values are displayed on each plot
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_1D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    ax, f = getSetup((20, 16), (4, 5))

    # Parameters
    targCell = "Treg"
    sample_size = 100
    cell_categorization = "CellType2"

    # Load and prepare data
    CITE_DF = importCITE()

    # Ensure target cell exists in the dataset
    assert (
        targCell in CITE_DF[cell_categorization].unique()
    )

    # Sample cells for analysis
    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
    )

    # Create target and off-target masks
    targ_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_targ_mask = ~targ_mask
    n_targ = np.sum(targ_mask)
    n_off_targ = np.sum(off_targ_mask)

    # Filter out any columns with all zero values
    filtered_sampleDF = sampleDF[sampleDF.columns[~sampleDF.columns.isin(["Cell Type"])]]
    
    # Filter columns with all zeros in off-target cells
    off_target_zeros = filtered_sampleDF.loc[off_targ_mask].apply(lambda col: (col == 0).all())
    filtered_sampleDF = filtered_sampleDF.loc[:, ~off_target_zeros]
    receptor_columns = filtered_sampleDF.columns

    # Get receptor abundances
    rec_abundances = filtered_sampleDF.to_numpy()

    # Calculate KL divergence and EMD for all receptors
    KL_div_vals, EMD_vals = KL_EMD_1D(rec_abundances, targ_mask, off_targ_mask)

    # Create a DataFrame with the results
    metrics_df = pd.DataFrame(
        {"Receptor": receptor_columns, "KL_Divergence": KL_div_vals, "EMD": EMD_vals}
    )

    # Remove NaN values
    metrics_df = metrics_df.dropna()

    # Sort metrics to find highest and lowest values
    top_kl_df = metrics_df.nlargest(5, "KL_Divergence")
    bottom_kl_df = metrics_df.nsmallest(5, "KL_Divergence")
    top_emd_df = metrics_df.nlargest(5, "EMD")
    bottom_emd_df = metrics_df.nsmallest(5, "EMD")

    # Organize the plot data by category
    plot_data = []

    # Row 1: Top 5 KL divergence
    for i, (_, row) in enumerate(top_kl_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "category": "Top KL",
                "plot_idx": i,  # First row
            }
        )

    # Row 2: Bottom 5 KL divergence
    for i, (_, row) in enumerate(bottom_kl_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "category": "Bottom KL",
                "plot_idx": i + 5,  # Second row
            }
        )

    # Row 3: Top 5 EMD
    for i, (_, row) in enumerate(top_emd_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "category": "Top EMD",
                "plot_idx": i + 10,  # Third row
            }
        )

    # Row 4: Bottom 5 EMD
    for i, (_, row) in enumerate(bottom_emd_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "category": "Bottom EMD",
                "plot_idx": i + 15,  # Fourth row
            }
        )

    # Plot histograms for each selected receptor
    for plot_info in plot_data:
        receptor = plot_info["receptor"]
        category = plot_info["category"]
        kl_val = plot_info["kl_val"]
        emd_val = plot_info["emd_val"]
        i = plot_info["plot_idx"]

        # Extract receptor abundances and normalize
        rec_abundance = sampleDF[receptor].values
        mean_abundance = np.mean(rec_abundance)

        # Normalize by mean abundance
        targ_abundances = rec_abundance[targ_mask] / mean_abundance
        off_targ_abundances = rec_abundance[off_targ_mask] / mean_abundance

        # Plot histograms
        sns.histplot(
            targ_abundances,
            ax=ax[i],
            color="blue",
            alpha=0.5,
            label=f"{targCell} (n={n_targ})",
            stat="density",
            kde=False,
        )
        sns.histplot(
            off_targ_abundances,
            ax=ax[i],
            color="red",
            alpha=0.5,
            label=f"Off-target (n={n_off_targ})",
            stat="density",
            kde=False,
        )

        # Add title and labels
        ax[i].set_title(
            f"{receptor}\nKL: {kl_val:.2f}, EMD: {emd_val:.2f}", fontsize=10
        )
        ax[i].set_xlabel("Normalized Expression", fontsize=9)
        ax[i].set_ylabel("Density", fontsize=9)

        ax[i].set_xlim(0, 2)

        # Add legend only to first plot in each row
        if i % 5 == 0:
            ax[i].legend(loc="upper right", fontsize=8)

        # Add category label as text in corner
        label_color = {
            "Top KL": "darkgreen",
            "Bottom KL": "brown",
            "Top EMD": "purple",
            "Bottom EMD": "orange",
        }

        ax[i].text(
            0.05,
            0.95,
            category,
            transform=ax[i].transAxes,
            fontsize=9,
            fontweight="bold",
            color=label_color[category],
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor=label_color[category]),
        )

    # Add row titles
    row_titles = [
        "Top 5 KL Divergence",
        "Bottom 5 KL Divergence",
        "Top 5 EMD",
        "Bottom 5 EMD",
    ]

    for i, title in enumerate(row_titles):
        # Adding a text label for each row in the figure margin
        plt.figtext(
            0.01,
            0.87 - (i * 0.23),
            title,
            fontsize=12,
            fontweight="bold",
            rotation=90,
            ha="center",
        )

    # Add figure title with summary of metrics ranges
    kl_range = f"KL div range: [{metrics_df['KL_Divergence'].min():.2f}, {metrics_df['KL_Divergence'].mean():.2f}, {metrics_df['KL_Divergence'].max():.2f}]"
    emd_range = f"EMD range: [{metrics_df['EMD'].min():.2f}, {metrics_df['EMD'].mean():.2f}, {metrics_df['EMD'].max():.2f}]"
    plt.suptitle(
        f"Receptor Distribution Comparison: {targCell} vs Off-Target\n{kl_range}, {emd_range}",
        fontsize=14,
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98)

    return f
