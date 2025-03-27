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
import pandas as pd
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_1D, KL_EMD_2D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent

plt.rcParams["svg.fonttype"] = "none"


def makeFigure():
    """
    Generate figure showing histogram with KDE overlay of on and off target distributions
    for a specified receptor.
    """
    ax, f = getSetup((10, 6), (1, 1))
    ax = ax[0]
    
    receptor="CD235ab"
    targCell="Treg"
    sample_size=5000
    cell_categorization="CellType2"
    signal = "CD25"
    # plot_cell_types = ['Mono', 'CD4 T', 'CD8 T', 'NK', 'B', 'other T', 'other', 'DC']
    # plot_cell_types = ['CD14 Mono', 'CD4 TCM', 'CD8 Naive', 'NK', 'CD8 TEM', 'CD16 Mono',
    #     'B intermediate', 'CD4 Naive', 'CD4 CTL', 'B naive', 'MAIT', 'gdT', 'CD8 TCM',
    #     'dnT', 'B memory', 'Doublet', 'pDC', 'CD8 Proliferating', 'Treg', 'Plasmablast',
    #     'CD4 TEM', 'cDC2', 'NK Proliferating', 'ASDC', 'HSPC', 'Platelet',
    #     'NK_CD56bright', 'CD4 Proliferating', 'Eryth', 'cDC1', 'ILC']
    # plot_cell_types = ['CD14 Mono', 'CD4 TCM_1', 'CD8 Naive', 'NK_2', 'CD8 TEM_1', 'CD16 Mono',
    #     'B intermediate lambda', 'CD4 Naive', 'CD4 CTL', 'B naive kappa', 'CD4 TCM_3',
    #     'MAIT', 'CD4 TCM_2', 'CD8 TEM_2', 'gdT_3', 'NK_1', 'CD8 TCM_1', 'dnT_2',
    #     'B intermediate kappa', 'B memory kappa', 'Doublet', 'pDC', 'CD8 TEM_5',
    #     'gdT_1', 'B naive lambda', 'NK_4', 'CD8 Proliferating', 'CD8 TCM_2',
    #     'Treg Naive', 'Plasma', 'CD4 TEM_1', 'Treg Memory', 'CD4 TEM_3', 'CD8 TCM_3',
    #     'cDC2_1', 'NK Proliferating', 'CD8 TEM_4', 'ASDC_pDC', 'CD4 TEM_2',
    #     'B memory lambda', 'dnT_1', 'HSPC', 'cDC2_2', 'Platelet', 'NK_CD56bright',
    #     'CD4 TEM_4', 'CD8 TEM_6', 'CD8 Naive_2', 'gdT_2', 'NK_3', 'CD8 TEM_3',
    #     'CD4 Proliferating', 'Eryth', 'gdT_4', 'Plasmablast', 'cDC1', 'ASDC_mDC', 'ILC']

    # plot_cell_types = ['CD14 Mono', 'CD4 TCM', 'CD8 Naive', 'NK', 'CD8 TEM', 'CD16 Mono',
    #     'B intermediate', 'CD4 Naive', 'CD4 CTL', 'B naive', 'CD8 TCM',
    #     'B memory', 'CD8 Proliferating', 'Treg', 'Platelet',
    #     'NK_CD56bright', 'CD4 Proliferating']
    plot_cell_types = ["Treg", "other"]

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
    
    # Ensure the requested receptor exists
    if receptor not in epitopes:
        raise ValueError(f"Receptor '{receptor}' not found. Available receptors: {epitopes}")
    
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
    
    # Calculate KL divergence and EMD for the specific receptor
    rec_abundances = sampleDF[[receptor] + [signal]].to_numpy()
    KL_div_vals, EMD_vals = KL_EMD_2D(rec_abundances, targ_mask, off_targ_mask)
    kl_val = KL_div_vals[1, 0]
    emd_val = EMD_vals[1, 0]

    sampleDF.loc[sampleDF["Cell Type"] != targCell, "Cell Type"] = "other"
    
    # Extract receptor abundances and normalize
    rec_abundance = sampleDF[receptor].values
    mean_abundance = np.mean(rec_abundance)
    
    # Normalize by mean abundance
    targ_abundances = rec_abundance[targ_mask] / mean_abundance

    all_cell_abundances = []
    for cell_type in plot_cell_types:
        if cell_type not in sampleDF["Cell Type"].unique():
            continue
        mask = (sampleDF["Cell Type"] == cell_type).to_numpy()
        all_cell_abundances.append(rec_abundance[mask] / mean_abundance)
    
    # Create a dataframe with statistics for each cell type
    stats_data = []

    # Add target cell stats
    stats_data.append({
        'Cell Type': targCell,
        'Min': np.min(targ_abundances),
        'Max': np.max(targ_abundances),
        'Mean': np.mean(targ_abundances),
        'Std': np.std(targ_abundances),
        'Count': len(targ_abundances)
    })

    # Add other cell types stats
    for i, abundances in enumerate(all_cell_abundances):
        if len(abundances) == 0:
            continue
        stats_data.append({
            'Cell Type': plot_cell_types[i],
            'Min': np.min(abundances),
            'Max': np.max(abundances),
            'Mean': np.mean(abundances),
            'Std': np.std(abundances),
            'Count': len(abundances)
        })

    # Create and display the statistics dataframe
    stats_df = pd.DataFrame(stats_data)
    print(f"\nStatistics for {receptor} receptor abundance by cell type:")
    print(stats_df.to_string(index=False))
    
    colors = sns.color_palette("husl", len(all_cell_abundances))
    # Plot histograms with KDE overlays
    for i, abundances in enumerate(all_cell_abundances):
        if len(abundances) == 0:
            continue
        sns.histplot(
            abundances,
            ax=ax,
            color=colors[i],
            alpha=0.5,
            label=f"{plot_cell_types[i]} (n={len(abundances)})",
            stat="density",
            kde=True,
            kde_kws={"bw_method": "scott"},
        )
    
    # Add title and labels
    ax.set_title(
        f"{receptor} Distribution\nKL: {kl_val:.2f}, EMD: {emd_val:.2f}", fontsize=14
    )
    ax.set_xlabel("Normalized Expression", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    
    # Set reasonable x-axis limits
    # x_max = max(np.percentile(targ_abundances, 99), np.percentile(off_targ_abundances, 99))
    # ax.set_xlim(0, min(x_max * 1.1, 3))
    
    # Add legend
    ax.legend(loc="upper right", fontsize=10)    
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    return f
