"""
Generates horizontal bar charts to visualize the top 5 markers
    with the highest 1D KL Divergence and Earth Mover's Distance (EMD) values,
    comparing target and off-target cell distributions using CITE-seq data.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose selectivity will be maximized
- receptors_of_interest: list of receptors to be analyzed
- sample_size: number of cells to sample for analysis
    (if greater than available cells, will use all)
- cell_categorization: column name in CITE-seq dataframe for cell type categorization

Outputs:
- Plots horizontal bar charts for these top markers:
    - KL Divergence Plot: Top 5 markers sorted by KL divergence
    - EMD Plot: Top 5 markers sorted by EMD
- Each plot is labeled with marker names on the y-axis
    and their respective values (KL or EMD) on the x-axis
"""

from pathlib import Path

import numpy as np

from ..distance_metric_funcs import KL_EMD_1D
from ..imports import importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((7, 3.5), (1, 2))

    targCell = "Treg"
    receptors_of_interest = None
    sample_size = 5000
    cell_categorization = "CellType2"
    signal = "CD122"

    CITE_DF = importCITE()

    assert targCell in CITE_DF[cell_categorization].unique()

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
    if receptors_of_interest is not None:
        filtered_sampleDF = sampleDF.loc[
            :,
            sampleDF.columns.str.fullmatch("|".join(receptors_of_interest), case=False),
        ]
    else:
        filtered_sampleDF = sampleDF[
            sampleDF.columns[~sampleDF.columns.isin(["Cell Type"])]
        ]
    receptors_of_interest = filtered_sampleDF.columns

    on_target_mask = (sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask

    rec_abundances = filtered_sampleDF.to_numpy()

    KL_values, EMD_values = KL_EMD_1D(rec_abundances, on_target_mask, off_target_mask)

    top_5_KL_indices = np.argsort(np.nan_to_num(KL_values))[-10:]
    top_5_EMD_indices = np.argsort(np.nan_to_num(EMD_values))[-10:]

    # Plot KL values
    ax[0].bar(
        filtered_sampleDF.columns[top_5_KL_indices],
        KL_values[top_5_KL_indices],
        color="r",
    )
    ax[0].set_title("Top 10 KL Divergence Values")
    ax[0].set_xlabel("KL Divergence")
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right")

    # Plot EMD values
    ax[1].bar(
        filtered_sampleDF.columns[top_5_EMD_indices],
        EMD_values[top_5_EMD_indices],
        color="b",
    )
    ax[1].set_title("Top 10 EMD Values")
    ax[1].set_xlabel("EMD Value")
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha="right")

    return f
