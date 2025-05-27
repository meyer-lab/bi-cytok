"""
Generates heatmaps visualizing the 2D EMD and KL Divergence of all receptor pairs

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose selectivity will be maximized
- sample_size: number of cells to sample for analysis
    (if greater than available cells, will use all)
- cell_categorization: column name in CITE-seq dataframe for cell type categorization

Outputs:
- Generates heatmaps of the EMD and KL divergences between all relevant receptor
    pairs. Filters out receptors with low average values based on percentiles
"""

import numpy as np
import pandas as pd
import seaborn as sns

from ..distance_metric_funcs import KL_EMD_2D
from ..imports import filter_receptor_abundances, importCITE, sample_receptor_abundances
from .common import getSetup


def makeFigure():
    ax, f = getSetup((10, 5), (1, 2))

    targCell = "Treg"
    sample_size = 100
    cell_categorization = "CellType2"

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

    on_target_mask = (filtered_sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask
    rec_abundances = filtered_sampleDF[epitopes].to_numpy()
    KL_div_vals, EMD_vals = KL_EMD_2D(
        rec_abundances, on_target_mask, off_target_mask, calc_1D=True
    )

    EMD_matrix = np.tril(EMD_vals, k=0)
    EMD_matrix = EMD_matrix + EMD_matrix.T - np.diag(np.diag(EMD_matrix))
    KL_matrix = np.tril(KL_div_vals, k=0)
    KL_matrix = KL_matrix + KL_matrix.T - np.diag(np.diag(KL_matrix))

    df_EMD = pd.DataFrame(EMD_matrix, index=epitopes, columns=epitopes)
    df_KL = pd.DataFrame(KL_matrix, index=epitopes, columns=epitopes)

    # Drop rows and columns with all NaN entries
    df_EMD = df_EMD.dropna(how="all").dropna(how="all", axis=1)
    df_KL = df_KL.dropna(how="all").dropna(how="all", axis=1)

    # Calculate average values for rows and columns
    emd_row_means = df_EMD.mean(axis=1)
    emd_col_means = df_EMD.mean(axis=0)
    kl_row_means = df_KL.mean(axis=1)
    kl_col_means = df_KL.mean(axis=0)

    # Set thresholds based on percentiles
    emd_threshold = np.percentile(np.concatenate([emd_row_means, emd_col_means]), 25)
    kl_threshold = np.percentile(np.concatenate([kl_row_means, kl_col_means]), 25)

    # Filter rows and columns with averages below threshold
    df_EMD = df_EMD.loc[emd_row_means >= emd_threshold, emd_col_means >= emd_threshold]
    df_KL = df_KL.loc[kl_row_means >= kl_threshold, kl_col_means >= kl_threshold]

    # Ensure both dataframes have matching indices for consistency
    common_receptors = sorted(set(df_EMD.index) & set(df_KL.index))
    df_EMD = df_EMD.loc[common_receptors, common_receptors]
    df_KL = df_KL.loc[common_receptors, common_receptors]

    # Visualize with heatmaps and smaller fontsize for labels
    sns.heatmap(
        df_EMD, cmap="bwr", ax=ax[0], cbar=True, xticklabels=True, yticklabels=True
    )
    sns.heatmap(
        df_KL, cmap="bwr", ax=ax[1], cbar=True, xticklabels=True, yticklabels=True
    )

    # Set smaller fontsize for tick labels on both axes
    for i in range(2):
        ax[i].tick_params(axis="x", labelsize=5)
        ax[i].tick_params(axis="y", labelsize=5)

    ax[0].set_title("EMD")
    ax[1].set_title("KL Divergence")

    return f
