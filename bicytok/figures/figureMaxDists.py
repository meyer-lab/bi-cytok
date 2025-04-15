"""
Generates barplots of the top KL Divergence and EMD values for different cell types.

Data Import:
- The CITE-seq dataframe (`importCITE`)

Parameters:
- sample_size: Number of cells to sample for each cell type
- cell_categorization: Column name in CITE-seq dataframe for cell type categorization
- cell_types: List of cell types to analyze (if None, will use all unique cell types in
    the dataframe)

Outputs:
- Displays two bar plots:
    1. Top KL Divergence values for each cell type with the corresponding receptor
    2. Top EMD values for each cell type with the corresponding receptor
"""

from pathlib import Path

import numpy as np

from ..distance_metric_funcs import KL_EMD_1D
from ..imports import filter_receptor_abundances, importCITE, sample_receptor_abundances
from .common import getSetup

path_here = Path(__file__).parent.parent


def makeFigure():
    ax, f = getSetup((7, 3.5), (1, 2))

    sample_size = 100
    cell_categorization = "CellType2"
    cell_types = [
        "B memory",
        "B naive",
        "Treg",
        "NK",
        "CD8 Naive",
        "CD4 Naive",
        "CD8 TCM",
        "CD4 TCM",
        "CD14 Mono",
        "cDC1",
        "ILC",
        "CD16 Mono",
        "pDC",
        "NK_CD56bright",
    ]

    CITE_DF = importCITE()

    epitopes = [
        col
        for col in CITE_DF.columns
        if col not in ["Cell", "CellType1", "CellType2", "CellType3"]
    ]
    epitopesDF = CITE_DF[epitopes + [cell_categorization]]
    epitopesDF = epitopesDF.rename(columns={cell_categorization: "Cell Type"})

    if cell_types is None:
        cell_types = epitopesDF["Cell Type"].unique()

    top_EMD = []
    top_EMD_rec = []
    top_KL = []
    top_KL_rec = []
    for cell_type in cell_types:
        targCell = cell_type

        sampleDF = sample_receptor_abundances(
            CITE_DF=epitopesDF,
            numCells=min(sample_size, epitopesDF.shape[0]),
            targCellType=targCell,
            balance=True,
        )
        filtered_sampleDF = filter_receptor_abundances(sampleDF, targCell)
        epitopes = filtered_sampleDF.columns[:-1]

        on_target_mask = (filtered_sampleDF["Cell Type"] == targCell).to_numpy()
        off_target_mask = ~on_target_mask
        rec_abundances = filtered_sampleDF[epitopes].to_numpy()
        KL_values, EMD_values = KL_EMD_1D(
            rec_abundances, on_target_mask, off_target_mask
        )

        KL_values = np.nan_to_num(KL_values)
        EMD_values = np.nan_to_num(EMD_values)

        top_EMD_ind = np.argsort(EMD_values)[-1]
        top_EMD.append(EMD_values[top_EMD_ind])
        top_EMD_rec.append(filtered_sampleDF.columns[top_EMD_ind])
        top_KL_ind = np.argsort(KL_values)[-1]
        top_KL.append(KL_values[top_KL_ind])
        top_KL_rec.append(filtered_sampleDF.columns[top_KL_ind])

    EMD_labs = [
        f"{cell_type}: {rec}"
        for cell_type, rec in zip(cell_types, top_EMD_rec, strict=False)
    ]
    KL_labs = [
        f"{cell_type}: {rec}"
        for cell_type, rec in zip(cell_types, top_KL_rec, strict=False)
    ]

    # Sort by EMD values
    EMD_sort_ind = np.argsort(top_EMD)[::-1]
    top_EMD = np.array(top_EMD)[EMD_sort_ind]
    EMD_labs = np.array(EMD_labs)[EMD_sort_ind]

    # Sort by KL values
    KL_sort_ind = np.argsort(top_KL)[::-1]
    top_KL = np.array(top_KL)[KL_sort_ind]
    KL_labs = np.array(KL_labs)[KL_sort_ind]

    # Plot KL values
    ax[0].bar(
        KL_labs,
        top_KL,
        color="r",
    )
    ax[0].set_title("Top KL Divergence Values")
    ax[0].set_xlabel("KL Divergence")
    ax[0].set_xticks(KL_labs)
    ax[0].set_xticklabels(KL_labs, rotation=45, ha="right")

    # Plot EMD values
    ax[1].bar(
        EMD_labs,
        top_EMD,
        color="b",
    )
    ax[1].set_title("Top EMD Values")
    ax[1].set_xlabel("EMD Value")
    ax[1].set_xticks(EMD_labs)
    ax[1].set_xticklabels(EMD_labs, rotation=45, ha="right")

    return f
