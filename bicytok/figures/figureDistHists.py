"""
Figure file to visualize receptor distributions with the highest and lowest
distance metrics between target and off-target cell populations.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose receptor distributions will be compared to off-target cells
- sample_size: number of cells to sample for analysis
- cell_categorization: column name for cell type classification
- num_hists

Outputs:
- Subplots showing histograms of receptor distributions:
  - Top receptors by KL divergence
  - Bottom receptors by KL divergence
  - Top receptors by EMD
  - Bottom receptors by EMD
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
from ..selectivity_funcs import optimize_affs
from .common import getSetup

path_here = Path(__file__).parent.parent

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    num_hists = 10
    ax, f = getSetup((num_hists * 4, 20), (6, num_hists))

    # Parameters
    targCell = "Treg"
    sample_size = 200000
    cell_categorization = "CellType2"
    dose = 10e-2

    # Load and prepare data
    CITE_DF = importCITE()

    # Ensure target cell exists in the dataset
    assert targCell in CITE_DF[cell_categorization].unique()

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
    filtered_sampleDF = sampleDF[
        sampleDF.columns[~sampleDF.columns.isin(["Cell Type"])]
    ]

    # Filter columns with all zeros in off-target cells
    off_target_zeros = filtered_sampleDF.loc[off_targ_mask].apply(
        lambda col: (col == 0).all()
    )
    filtered_sampleDF = filtered_sampleDF.loc[:, ~off_target_zeros]
    receptor_columns = filtered_sampleDF.columns

    # Get receptor abundances
    rec_abundances = filtered_sampleDF.to_numpy()

    # Calculate KL divergence and EMD for all receptors
    KL_div_vals, EMD_vals = KL_EMD_1D(rec_abundances, targ_mask, off_targ_mask)

    selectivities = []
    for receptor in receptor_columns:
        receptor_abun = sampleDF[[receptor]].to_numpy()

        targ_abun = receptor_abun[targ_mask]
        off_targ_abun = receptor_abun[off_targ_mask]

        opt_selec, _ = optimize_affs(
            targ_abun, off_targ_abun, dose, valencies=np.array([[1]])
        )
        selectivities.append(1 / opt_selec)

    # Create a DataFrame with the results
    metrics_df = pd.DataFrame(
        {"Receptor": receptor_columns, "KL_Divergence": KL_div_vals, "EMD": EMD_vals, "Selectivity": selectivities}
    )

    # Remove NaN values
    metrics_df = metrics_df.dropna()

    # Sort metrics to find highest and lowest values
    top_kl_df = metrics_df.nlargest(num_hists, "KL_Divergence")
    bottom_kl_df = metrics_df.nsmallest(num_hists, "KL_Divergence")
    top_emd_df = metrics_df.nlargest(num_hists, "EMD")
    bottom_emd_df = metrics_df.nsmallest(num_hists, "EMD")
    top_selec_df = metrics_df.nlargest(num_hists, "Selectivity")
    bottom_selec_df = metrics_df.nsmallest(num_hists, "Selectivity")

    # Organize the plot data by category
    plot_data = []

    # Row 1: Top KL divergence
    for i, (_, row) in enumerate(top_kl_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "selec": row["Selectivity"],
                "category": "Top KL",
                "plot_idx": i,  # First row
            }
        )

    # Row 2: Bottom KL divergence
    for i, (_, row) in enumerate(bottom_kl_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "selec": row["Selectivity"],
                "category": "Bottom KL",
                "plot_idx": i + num_hists,  # Second row
            }
        )

    # Row 3: Top EMD
    for i, (_, row) in enumerate(top_emd_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "selec": row["Selectivity"],
                "category": "Top EMD",
                "plot_idx": i + 2 * num_hists,  # Third row
            }
        )

    # Row 4: Bottom EMD
    for i, (_, row) in enumerate(bottom_emd_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "selec": row["Selectivity"],
                "category": "Bottom EMD",
                "plot_idx": i + 3 * num_hists,  # Fourth row
            }
        )
    # Row 5: Top Selectivity
    for i, (_, row) in enumerate(top_selec_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "selec": row["Selectivity"],
                "category": "Top Selectivity",
                "plot_idx": i + 4 * num_hists,  # Fifth row
            }
        )
    # Row 6: Bottom Selectivity
    for i, (_, row) in enumerate(bottom_selec_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "selec": row["Selectivity"],
                "category": "Bottom Selectivity",
                "plot_idx": i + 5 * num_hists,  # Sixth row
            }
        )

    # Plot histograms for each selected receptor
    for plot_info in plot_data:
        receptor = plot_info["receptor"]
        category = plot_info["category"]
        kl_val = plot_info["kl_val"]
        emd_val = plot_info["emd_val"]
        selec_val = plot_info["selec"]
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
            f"{receptor}\nKL: {kl_val:.2f}, EMD: {emd_val:.2f}, Selectivity: {selec_val:.2f}", fontsize=10
        )
        ax[i].set_xlabel("Normalized Expression", fontsize=9)
        ax[i].set_ylabel("Density", fontsize=9)

        ax[i].set_xlim(0, 2)

        # Add legend only to first plot in each row
        if i % num_hists == 0:
            ax[i].legend(loc="upper right", fontsize=8)

        # Add category label as text in corner
        label_color = {
            "Top KL": "darkgreen",
            "Bottom KL": "brown",
            "Top EMD": "purple",
            "Bottom EMD": "orange",
            "Top Selectivity": "darkblue",
            "Bottom Selectivity": "red",
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

    # Add figure title with summary of metrics ranges
    kl_range = (
        f"KL div range: [{metrics_df['KL_Divergence'].min():.2f}, "
        f"{metrics_df['KL_Divergence'].mean():.2f}, "
        f"{metrics_df['KL_Divergence'].max():.2f}]"
    )
    emd_range = (
        f"EMD range: [{metrics_df['EMD'].min():.2f}, "
        f"{metrics_df['EMD'].mean():.2f}, {metrics_df['EMD'].max():.2f}]"
    )
    selec_range = (
        f"Selectivity range: [{metrics_df['Selectivity'].min():.2f}, "
        f"{metrics_df['Selectivity'].mean():.2f}, "
        f"{metrics_df['Selectivity'].max():.2f}]"
    )
    plt.suptitle(
        f"Receptor Distribution Comparison: {targCell} vs Off-Target\n{kl_range}, "
        f"{emd_range}, {selec_range}",
        fontsize=14,
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98)

    return f
