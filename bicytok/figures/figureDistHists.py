"""
Figure file to visualize receptor distributions with the highest and lowest
distance values between target and off-target cell populations.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- num_hists: number of histograms to display in each row
- targCell: cell type whose receptor distributions will be compared to off-target cells
- sample_size: number of cells to sample for analysis
- cell_categorization: column name for cell type classification
- dose: dosing parameter for selectivity optimization


Outputs:
- Subplots showing histograms of receptor distributions:
  - Top receptors by KL divergence
  - Bottom receptors by KL divergence
  - Top receptors by EMD
  - Bottom receptors by EMD
- Each plot shows target and off-target distributions
- KL divergence and EMD values are displayed on each plot
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_1D
from ..imports import filter_receptor_abundances, importCITE, sample_receptor_abundances
from ..selectivity_funcs import optimize_affs
from .common import getSetup

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    num_hists = 10
    ax, f = getSetup((num_hists * 4, 20), (6, num_hists))

    # Parameters
    targCell = "Treg"
    sample_size = 100
    cell_categorization = "CellType2"
    dose = 10e-2

    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["Cell", "CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})
    sampleDF = sample_receptor_abundances(
        CITE_DF=epitopesDF,
        numCells=min(sample_size, epitopesDF.shape[0]),
        targCellType=targCell,
    )
    filtered_sampleDF = filter_receptor_abundances(sampleDF, targCell)
    epitopes = list(filtered_sampleDF.columns[:-1])

    # Calculate distance metrics
    targ_mask = (filtered_sampleDF["Cell Type"] == targCell).to_numpy()
    off_targ_mask = ~targ_mask
    n_targ = np.sum(targ_mask)
    n_off_targ = np.sum(off_targ_mask)
    rec_abundances = filtered_sampleDF[epitopes].to_numpy()
    KL_div_vals, EMD_vals = KL_EMD_1D(rec_abundances, targ_mask, off_targ_mask)

    # Calculate selectivities
    selectivities = []
    for receptor in epitopes:
        receptor_abun = filtered_sampleDF[[receptor]].to_numpy()

        targ_abun = receptor_abun[targ_mask]
        off_targ_abun = receptor_abun[off_targ_mask]

        opt_selec, _ = optimize_affs(
            targ_abun, off_targ_abun, dose, valencies=np.array([[1]])
        )
        selectivities.append(1 / opt_selec)

    metrics_df = pd.DataFrame(
        {
            "Receptor": epitopes,
            "KL_Divergence": KL_div_vals,
            "EMD": EMD_vals,
            "Selectivity": selectivities,
        },
    )
    metrics_df = metrics_df.dropna()

    # Find highest and lowest values
    top_kl_df = metrics_df.nlargest(num_hists, "KL_Divergence")
    bottom_kl_df = metrics_df.nsmallest(num_hists, "KL_Divergence")
    top_emd_df = metrics_df.nlargest(num_hists, "EMD")
    bottom_emd_df = metrics_df.nsmallest(num_hists, "EMD")
    top_selec_df = metrics_df.nlargest(num_hists, "Selectivity")
    bottom_selec_df = metrics_df.nsmallest(num_hists, "Selectivity")

    # Row 1: Top KL divergence
    plot_data = []
    for i, (_, row) in enumerate(top_kl_df.iterrows()):
        plot_data.append(
            {
                "receptor": row["Receptor"],
                "kl_val": row["KL_Divergence"],
                "emd_val": row["EMD"],
                "selec": row["Selectivity"],
                "category": "Top KL",
                "plot_idx": i,
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
                "plot_idx": i + num_hists,
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
                "plot_idx": i + 2 * num_hists,
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
                "plot_idx": i + 3 * num_hists,
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
                "plot_idx": i + 4 * num_hists,
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
                "plot_idx": i + 5 * num_hists,
            }
        )

    # Plot histograms
    for plot_info in plot_data:
        receptor = plot_info["receptor"]
        category = plot_info["category"]
        kl_val = plot_info["kl_val"]
        emd_val = plot_info["emd_val"]
        selec_val = plot_info["selec"]
        i = plot_info["plot_idx"]

        # Extract receptor abundances and normalize
        rec_abundance = filtered_sampleDF[receptor].values
        mean_abundance = np.mean(rec_abundance)
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
        ax[i].set_title(
            f"{receptor}\nKL: {kl_val:.2f}, EMD: {emd_val:.2f}, "
            f"Selectivity: {selec_val:.2f}",
            fontsize=10,
        )
        ax[i].set_xlabel("Normalized Expression", fontsize=9)
        ax[i].set_ylabel("Density", fontsize=9)

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

    # Add figure title
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

    return f
