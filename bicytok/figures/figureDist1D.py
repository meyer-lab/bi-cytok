"""
Generates horizontal bar charts to visualize the top 10 markers
    with the highest 1D KL Divergence and Earth Mover's Distance (EMD) values,
    comparing target and off-target cell distributions using CITE-seq data.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- targCell: cell type whose selectivity will be maximized
- receptors_of_interest: list of receptors to be analyzed
    (if None, all receptors will be used)
- sample_size: number of cells to sample for analysis
    (if greater than available cells, will use all)
- cell_categorization: column name in CITE-seq dataframe for cell type categorization

Outputs:
- Plots horizontal bar charts for these top markers:
    - KL Divergence Plot: Top 10 markers sorted by KL divergence
    - EMD Plot: Top 10 markers sorted by EMD
- Each plot is labeled with marker names on the y-axis
    and their respective values (KL or EMD) on the x-axis
"""

from pathlib import Path

import numpy as np

from ..distance_metric_funcs import KL_EMD_1D
from ..imports import filter_receptor_abundances, importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((7, 3.5), (1, 2))

    targCell = "Treg"
    receptors_of_interest = None
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
    filtered_sampleDF = filter_receptor_abundances(
        sampleDF, targ_cell_type=targCell, epitope_list=receptors_of_interest
    )
    receptors_of_interest = filtered_sampleDF.columns[
        ~filtered_sampleDF.columns.isin(["Cell Type"])
    ]

    on_target_mask = (filtered_sampleDF["Cell Type"] == targCell).to_numpy()
    off_target_mask = ~on_target_mask
    rec_abundances = filtered_sampleDF[receptors_of_interest].to_numpy()
    KL_values, EMD_values = KL_EMD_1D(rec_abundances, on_target_mask, off_target_mask)

    top_KL_indices = np.argsort(np.nan_to_num(KL_values))[-10:]
    top_EMD_indices = np.argsort(np.nan_to_num(EMD_values))[-10:]

    # Plot KL values
    ax[0].bar(
        filtered_sampleDF.columns[top_KL_indices],
        KL_values[top_KL_indices],
        color="r",
    )
    ax[0].set_title("Top 10 KL Divergence Values")
    ax[0].set_xlabel("KL Divergence")
    ax[0].tick_params(axis="x", labelrotation=45)

    # Plot EMD values
    ax[1].bar(
        filtered_sampleDF.columns[top_EMD_indices],
        EMD_values[top_EMD_indices],
        color="b",
    )
    ax[1].set_title("Top 10 EMD Values")
    ax[1].set_xlabel("EMD Value")
    ax[1].tick_params(axis="x", labelrotation=45)

    return f
